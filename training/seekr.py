import os
import gc
import sys
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, get_constant_schedule_with_warmup

import deepspeed
from torch.optim import AdamW

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from data.data_utils import DistillAttnBudget2PromptDataset, SubPromptDataset, create_prompt_dataset
from data.data_collator import DistillAttn2DataCollator, DataCollator
from utils.utils import print_rank_0, to_device, set_random_seed, load_hf_tokenizer, get_optimizer_grouped_parameters
from utils.ds_utils import get_train_ds_config
from utils.model_utils import create_hf_model, save_model, load_task_model, get_start_task

from utils.llama_flash_att import replace_attn_distill_budget2_attn as replace_llama_distill_attn_qb
from utils.llama_flash_att import replace_attn_distill_budget_attn as replace_llama_distill_attn
from utils.llama_flash_att import replace_attn_output_attn as replace_llama_output_attn
from utils.llama_flash_att import replace_attn_grad_attn as replace_llama_grad_attn
from utils.llama_flash_att import replace_flash_attn as replace_llama_flash_attn

# add flash attention
from utils.llama_flash_att import replace_llama_attn_with_flash_attn


replace_llama_attn_with_flash_attn()


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path', type=str, default='Dahoas/rm-static', help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name', type=list_of_strings, default='all', help='Dataset to be used.')
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
    parser.add_argument('--data_output_path', type=str, default='/tmp/dataset_files/', help='Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True,)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--max_prompt_len", type=int, default=512, help="The maximum sequence length.",)
    parser.add_argument("--max_ans_len", type=int, default=512, help="The maximum sequence length.",)
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=list_of_strings, default=None, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--lr_scheduler_type", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout', action='store_true', help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload', action='store_true', help='Enable ZeRO Offload techniques.')
    parser.add_argument('--zero_stage', type=int, default=0, help='ZeRO optimization stage for Actor model (and clones).')
    # wandb logging
    parser.add_argument('--enable_wandb', action='store_true', help='Enable wandb logging')
    # SEEKR params
    parser.add_argument('--past_task_ratio', default=0.01, type=float, help='Replay ratio used for past task.')
    parser.add_argument('--replay_cycle', default=0, type=int, help='Control the sample frequency from the memory buffer.')
    parser.add_argument('--logits_distill_weight', default=0.5, type=float, help='Balance the logits_distill_loss with the replay_loss.')
    parser.add_argument('--attns_distill_weight', default=1e3, type=float, help='Weighting factor for the attn_distill_loss.')
    parser.add_argument('--attn_layer_budget', default=24, type=int, help='Layer budget for seekr.')
    parser.add_argument('--attn_head_budget', default=128, type=int, help='Head budget for seekr.')
    parser.add_argument('--attn_query_budget', default=100, type=int, help='Query budget for seekr.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def get_dataset(args, dataset, tokenizer):
    dataset_path = os.path.join(args.data_path, dataset)
    train_dataset, _, _ = create_prompt_dataset(
        args.local_rank,
        dataset_path,
        args.data_output_path,
        args.seed,
    )
    print_rank_0(f"Train dataset [{dataset}] length: {len(train_dataset)}", args.global_rank)
    past_task_num = int(len(train_dataset) * args.past_task_ratio) if args.past_task_ratio < 1 else int(args.past_task_ratio)
    indices = list(range(past_task_num))
    replay_dataset = SubPromptDataset(train_dataset, indices)
    print_rank_0(f"Replay dataset [{dataset}] length: {len(replay_dataset)}", args.global_rank)

    train_sampler = DistributedSampler(train_dataset)
    # The replay sampler is non-distributed and sequential to form logits-sample pairs for convenience
    replay_sampler = SequentialSampler(replay_dataset)

    data_collator = DataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=False
    )

    num_workers = 1 if dataset in ["MeetingBank", "Py150", "20Minuten"] else args.num_workers
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size,
                                  num_workers=num_workers)
    replay_dataloader = DataLoader(replay_dataset,
                                   collate_fn=data_collator,
                                   sampler=replay_sampler,
                                   batch_size=len(replay_dataset),
                                   num_workers=num_workers)
    return train_dataloader, replay_dataloader


def get_optimizer(args, model):
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps)

    return optimizer, lr_scheduler


def cal_replay_logits_distill_loss(logits: torch.Tensor, labels: torch.Tensor, teacher_logits: torch.Tensor):
    # only calculate loss for non-label and non-padding tokens
    loss_fct = torch.nn.CrossEntropyLoss()

    token_indices = torch.nonzero(labels != -100., as_tuple=True)
    logits_distill_loss = loss_fct(logits[token_indices], F.softmax(teacher_logits[token_indices], dim=-1))

    return logits_distill_loss


def get_replay_dataloader(args, replay_datasets, tokenizer):
    if len(replay_datasets) > 0:
        replay_dataset = ConcatDataset([replay_datasets[task] for task in replay_datasets])
        print_rank_0(f"Loaded replay dataset length: {len(replay_dataset)}", args.global_rank)
        replay_sampler = DistributedSampler(replay_dataset)
        replay_collator = DistillAttn2DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        workers_flag = any([task in ["MeetingBank", "Py150"] for task in replay_datasets])
        num_workers = 1 if workers_flag else args.num_workers
        replay_dataloader = DataLoader(replay_dataset,
                                       collate_fn=replay_collator,
                                       sampler=replay_sampler,
                                       batch_size=args.per_device_train_batch_size,
                                       num_workers=num_workers)
    else:
        replay_dataloader = None

    return replay_dataloader


def train_one_task(args, train_loader, replay_loader, epochs, model, device):
    start = datetime.now()
    total_steps = epochs * (len(train_loader) + len(replay_loader)) if replay_loader is not None else epochs * len(train_loader)
    progress_bar = tqdm(total=total_steps, leave=True)
    if replay_loader is None:
        replay_cycle = 0
    elif args.replay_cycle > 0:
        replay_cycle = args.replay_cycle
    else:
        replay_cycle = len(train_loader) // len(replay_loader)

    for epoch in range(epochs):
        epoch_start = datetime.now()
        replay_iter = iter(replay_loader) if replay_loader is not None else None
        print_rank_0(f"Beginning of Epoch {epoch + 1}/{epochs}, Total Micro Batches {len(train_loader)}", args.global_rank)
        model.train()

        for step, batch in enumerate(train_loader):

            if replay_cycle > 0 and (step + 1) % replay_cycle == 0:
                try:
                    replay_batch = next(replay_iter)
                except:
                    replay_iter = iter(replay_loader)
                    replay_batch = next(replay_iter)
                del replay_batch['sources']
                replay_batch = to_device(replay_batch, device)
                teacher_logits = replay_batch.pop('teacher_logits')
                teacher_attns = replay_batch.pop('teacher_attns')
                attn_query = replay_batch.pop('attn_query')
                select_heads = replay_batch.pop('select_heads')
                if args.attn_query_budget == 0:
                    replace_llama_distill_attn(model.module, teacher_attns, select_heads)
                else:
                    replace_llama_distill_attn_qb(model.module, teacher_attns, select_heads, attn_query)
                outputs = model(**replay_batch, use_cache=False)
                replay_loss = outputs.loss
                logits_distill_loss = cal_replay_logits_distill_loss(outputs.logits, replay_batch['labels'], teacher_logits)
                select_layers = set([l for sh in select_heads for l in sh.keys()])
                attns_distill_losses = [model.module.model.layers[l].self_attn.loss_attn_distill for l in select_layers]
                attns_distill_loss = sum(attns_distill_losses) / len(attns_distill_losses)
                loss = (1 - args.logits_distill_weight) * replay_loss \
                    + args.logits_distill_weight * logits_distill_loss \
                    + args.attns_distill_weight * attns_distill_loss
                if args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Epoch {epoch + 1}, Step {step}, " \
                                  f"Replay Loss: {replay_loss:.6f}, " \
                                  f"Logits Distill Loss: {logits_distill_loss:.6f}, " \
                                  f"Attns Distill Loss: {attns_distill_loss:.8f}"
                    progress_bar.set_description(description, refresh=False)
                model.backward(loss)
                model.step()
                replace_llama_flash_attn(model.module, select_layers)
                del teacher_logits, teacher_attns

            del batch['sources']
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if args.global_rank == 0:
                progress_bar.update(1)
                description = f"Epoch {epoch + 1}, Step {step}, New Task Loss: {loss.item():.6f}"
                progress_bar.set_description(description, refresh=False)
            model.backward(loss)
            model.step()

        print_rank_0(f"Finished one epoch. Time: {datetime.now()-epoch_start}", args.global_rank)
    print_rank_0(f"Finished one task. Time: {datetime.now()-start}", args.global_rank)


def attn_grad_hook(self: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
    self.attn_grad += (grad_output[0]**2).transpose(0, 1).mean((1,2,3))


class AttnGrad(nn.Module):
    def __init__(self, num_heads, device):
        super().__init__()
        self.register_buffer('attn_grad', torch.zeros(num_heads, device=device))

    def forward(self, attn_weights):
        return attn_weights


def get_replay_logits_and_attns(args, replay_loader, model, attn_queries, task, device):
    start = datetime.now()

    num_heads = model.module.model.config.num_attention_heads
    attn_grads = []
    for l in args.target_layers:
        attn_grad = AttnGrad(num_heads, device)
        attn_grad.register_full_backward_hook(attn_grad_hook)
        attn_grads.append(attn_grad)
    replace_llama_grad_attn(model.module, args.target_layers, attn_grads)

    mini_bs = 1
    replay_logits = []
    replay_attns = []
    idx = 0
    for batch in replay_loader:
        batch = to_device(batch, device)
        del batch['sources']
        bs = batch['labels'].shape[0]
        for start_i in range(0, bs, mini_bs):
            mini_batch = {k: v[start_i:start_i + mini_bs] for k, v in batch.items()}
            outputs = model(**mini_batch, use_cache=False)
            model.backward(outputs.loss)
            select_query = attn_queries[idx]
            attns = []
            for l in args.target_layers:
                layer_attn = model.module.model.layers[l].self_attn.attn_weights.detach().cpu()
                layer_attn = torch.stack([layer_attn[i].index_select(-2, query) for i, query in enumerate(select_query)])
                attns.append(layer_attn)
            attns = torch.stack(attns).transpose(0, 1)  # bs, n_layers, n_heads, seq_len, seq_len
            total_len = mini_batch['attention_mask'].shape[1]
            attn_queries[idx] -= total_len   # relative index to the rightmost
            idx = idx + 1
            replay_logits.append(outputs.logits.detach().cpu())
            replay_attns.append(attns)
    attn_grads = torch.stack([attn_grad.attn_grad / len(replay_attns) for attn_grad in attn_grads]).cpu()  # n_layers, n_heads
    attn_grads = attn_grads / attn_grads.sum(1, keepdim=True)
    replay_logits = torch.cat(replay_logits, dim=0)
    replay_attns = torch.cat(replay_attns, dim=0)   # bs, n_layers, n_heads, seq_len, seq_len
    attn_queries = torch.cat(attn_queries, dim=0)

    replace_llama_flash_attn(model.module, args.target_layers)
    print_rank_0(f"Finished calculate replay logits and attns. Time: {datetime.now()-start}", args.global_rank)

    # clear cuda cache
    for l in args.target_layers:
        del model.module.model.layers[l].self_attn.attn_weights
    torch.cuda.empty_cache()
    gc.collect()

    return replay_logits, replay_attns, attn_grads, attn_queries


def get_replay_attns(args, replay_loader, model, task, device):
    start = datetime.now()
    replace_llama_output_attn(model.module, args.target_layers)

    mini_bs = 1
    replay_attns = []
    attn_queries = []
    with torch.no_grad():
        for batch in replay_loader:
            batch = to_device(batch, device)
            del batch['sources']
            bs = batch['labels'].shape[0]
            for start_i in range(0, bs, mini_bs):
                mini_batch = {k: v[start_i:start_i + mini_bs] for k, v in batch.items()}
                total_len = mini_batch['attention_mask'].shape[1]
                seq_lens = mini_batch['attention_mask'].sum(1).cpu()
                if args.attn_query_budget == 0:
                    select_query = torch.arange(total_len).expand(mini_bs, total_len)
                else:
                    query_budget = args.attn_query_budget
                    select_query = torch.stack([torch.randperm(seq_len)[:query_budget] + total_len - seq_len if seq_len >= query_budget
                                                else torch.randint(seq_len, (query_budget,)) + total_len - seq_len
                                                for seq_len in seq_lens]).sort()[0]
                attn_queries.append(select_query)
                model(**mini_batch, use_cache=False)
                attns = []
                for l in args.target_layers:
                    layer_attn = model.module.model.layers[l].self_attn.attn_weights.cpu()
                    layer_attn = torch.stack([layer_attn[i].index_select(-2, query) for i, query in enumerate(select_query)])
                    attns.append(layer_attn)
                attns = torch.stack(attns).transpose(0, 1)  # bs, n_layers, n_heads, seq_len, seq_len
                replay_attns.append(attns)
    replay_attns = torch.cat(replay_attns, dim=0)   # bs, n_layers, n_heads, seq_len, seq_len

    replace_llama_flash_attn(model.module, args.target_layers)
    print_rank_0(f"Finished calculating replay attn. Time: {datetime.now()-start}", args.global_rank)

    dist.barrier()
    if args.global_rank <= 0:
        torch.save(replay_attns, os.path.join(args.output_dir, "tmp.bin"))

    # clear cuda cache
    del replay_attns
    for l in args.target_layers:
        del model.module.model.layers[l].self_attn.attn_weights
    torch.cuda.empty_cache()
    gc.collect()

    return attn_queries


def get_replay_attn_diff(args, replay_attn_after):
    dist.barrier()
    replay_attn_before = torch.load(os.path.join(args.output_dir, "tmp.bin"), map_location="cpu")
    attn_diffs = ((replay_attn_before - replay_attn_after) ** 2).transpose(0, 1).transpose(1, 2).mean((2,3,4))
    return attn_diffs


def select_layer_head(args, attn_ipt):
    start = datetime.now()

    # select layers
    select_layers = attn_ipt.sum(1).topk(args.attn_layer_budget)[1].sort()[0]
    select_attn_ipt = attn_ipt[select_layers]

    # select topk across all heads all layers
    select_pos = np.array(np.unravel_index(select_attn_ipt.flatten().topk(args.attn_head_budget)[1].numpy(), select_attn_ipt.shape)).transpose()
    select_pos = sorted(select_pos, key=lambda x: (x[0], x[1]))
    select_heads = {}   # layer-head dict
    for l, h in select_pos:
        layer = args.target_layers[select_layers[l]]
        if layer not in select_heads:
            select_heads[layer] = []
        select_heads[layer].append(h)
    select_heads = {l: torch.tensor(h) for l, h in select_heads.items()}

    print_rank_0(f"Selected heads: {select_heads}\nTime: {datetime.now()-start}", args.global_rank)

    return select_heads


def main():
    start_time = datetime.now()
    args = parse_args()
    set_random_seed(args.seed)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()

    args.global_rank = dist.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_wandb=args.enable_wandb)
    # set batch size
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * dist.get_world_size() * args.gradient_accumulation_steps

    dist.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config,
                            disable_dropout=args.disable_dropout)
    args.target_layers = list(range(len(model.model.layers)))
    start_i_task = get_start_task(args)

    train_task_list = {}
    replay_task_list = {}
    for dataset in args.dataset_name:
        train_dataloader, replay_dataloader = get_dataset(args, dataset, tokenizer)
        train_task_list[dataset] = train_dataloader
        replay_task_list[dataset] = replay_dataloader

    optimizer, lr_scheduler = get_optimizer(args, model)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print_rank_0(f"Finished preparing for training. Time: {datetime.now()-start_time}", args.global_rank)
    print_rank_0("***** Running training *****", args.global_rank)

    num_heads = model.module.model.config.num_attention_heads
    attn_ipt = torch.zeros((len(args.target_layers), num_heads))
    distill_replay_datasets = {}

    for i_task, task in enumerate(train_task_list):

        attn_queries = get_replay_attns(args, replay_task_list[task], model, task, device)

        if i_task > start_i_task:
            train_one_task(args,
                           train_task_list[task],
                           get_replay_dataloader(args, distill_replay_datasets, tokenizer),
                           int(args.num_train_epochs[i_task]),
                           model,
                           device)
            save_model(args, i_task, model, tokenizer)
        else:
            load_task_model(args, model, i_task)

        # get replay logits and replay attentions, calculate task sensitivity, update attn_queries to relative index to the rightmost
        replay_logits, replay_attns_after, new_attn_grads, attn_queries \
            = get_replay_logits_and_attns(args, replay_task_list[task], model, attn_queries, task, device)
        new_attn_diffs = get_replay_attn_diff(args, replay_attns_after)
        attn_ipt += new_attn_grads * new_attn_diffs
        select_heads = select_layer_head(args, attn_ipt)

        # save replay samples, replay logits and replay attentions of current task to the memory buffer
        distill_replay_datasets[task] = DistillAttnBudget2PromptDataset(replay_task_list[task].dataset,
                                                                        target_layers=args.target_layers,
                                                                        attn_query=attn_queries,
                                                                        select_heads=select_heads,
                                                                        teacher_logits=replay_logits,
                                                                        all_teacher_attns=tuple(attn for attn in replay_attns_after.transpose(0, 1)))
        del replay_logits, replay_attns_after

    print_rank_0(f"Finished continual training. Time: {datetime.now()-start_time}", args.global_rank)
    if args.global_rank <= 0:
        os.remove(os.path.join(args.output_dir, "tmp.bin"))


if __name__ == "__main__":
    main()
