import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, SchedulerType, get_constant_schedule_with_warmup

import deepspeed
from torch.optim import AdamW

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from data.data_utils import create_prompt_dataset, SubPromptDataset
from data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, set_random_seed, get_optimizer_grouped_parameters, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model_utils import create_hf_model, save_model, load_task_model, get_start_task

# add flash attention
from utils.llama_flash_att import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',type=str, default='Dahoas/rm-static', help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',type=list_of_strings, default='all', help='Dataset to be used.')
    parser.add_argument('--data_output_path',type=str, default='/tmp/dataset_files/', help='Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)')
    parser.add_argument('--num_workers',type=int, default=4)
    parser.add_argument("--model_name_or_path",type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--per_device_train_batch_size",type=int, default=16, help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--max_prompt_len",type=int, default=512, help="The maximum sequence length.",)
    parser.add_argument("--max_ans_len",type=int, default=512, help="The maximum sequence length.",)
    parser.add_argument("--learning_rate",type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--weight_decay",type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",type=list_of_strings, default=None, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps",type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--lr_scheduler_type",type=SchedulerType, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="The scheduler type to use.")
    parser.add_argument("--num_warmup_steps",type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",type=str, default=None, help="Where to store the model.")
    parser.add_argument("--seed",type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--local_rank",type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout', action='store_true', help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload', action='store_true', help='Enable ZeRO Offload techniques.')
    parser.add_argument('--zero_stage',type=int, default=0, help='ZeRO optimization stage for Actor model (and clones).')
    # wandb logging
    parser.add_argument('--enable_wandb', action='store_true', help='Enable wandb logging')
    # replay params
    parser.add_argument('--past_task_ratio', default=None,type=float, help='Replay ratio used for past task')
    parser.add_argument('--replay_dataset_name',type=str, default=None, help='Dataset to be used.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def get_dataset(args, dataset, replay_datasets, tokenizer):
    dataset_path = os.path.join(args.data_path, dataset)
    # Prepare the data
    train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
        args.local_rank,
        dataset_path,
        args.data_output_path,
        args.seed,
    )
    # mix train dataset with replay datasets
    train_dataset = ConcatDataset([*replay_datasets, train_dataset])
    print_rank_0(f"Train dataset {dataset} length: {len(train_dataset)}", args.global_rank)

    train_sampler = DistributedSampler(train_dataset)

    data_collator = DataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=False
    )

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size,
                                  num_workers=args.num_workers)

    return train_dataloader


def get_replay_dataset(args, dataset):
    if isinstance(dataset, str):
        dataset_path = os.path.join(args.data_path, dataset)
        # Prepare the data
        dataset, _, _ = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed,
        )
    past_task_num = int(len(dataset) * args.past_task_ratio) if args.past_task_ratio < 1 else int(args.past_task_ratio)
    indices = list(range(past_task_num))
    replay_dataset = SubPromptDataset(dataset, indices)
    return replay_dataset


def get_optimizer(args, model):
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    # total_train_dataloader_len = sum(len(train_task_list[task]) for task in list(train_task_list.keys()))
    # num_update_steps_per_epoch = math.ceil(total_train_dataloader_len / args.gradient_accumulation_steps)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps)

    return optimizer, lr_scheduler


def train_one_task(args, train_loder, epochs, model, device):

    total_steps = epochs * len(train_loder)
    progress_bar = tqdm(total=total_steps, leave=True, disable=(args.global_rank != 0))
    for epoch in range(epochs):
        print_rank_0(f"Beginning of Epoch {epoch + 1}/{epochs}, Total Micro Batches {len(train_loder)}", args.global_rank)
        model.train()

        for step, batch in enumerate(train_loder):
            del batch['sources']
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            # Update the description to include current step and loss, if needed
            if args.global_rank == 0:
                # Update the progress bar
                progress_bar.update(1)
                description = f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.6f}"
                progress_bar.set_description(description, refresh=False)

            model.backward(loss)
            # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
            model.step()

def main():
    args = parse_args()

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_wandb=args.enable_wandb)
    # set batch size
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config,
                            disable_dropout=args.disable_dropout)

    start_i_task = get_start_task(args)

    train_task_list = {}

    replay_datasets = [get_replay_dataset(args, args.replay_dataset_name)] if args.replay_dataset_name is not None else []
    for dataset in args.dataset_name:
        train_task_list[dataset] = get_dataset(args, dataset, replay_datasets, tokenizer)
        replay_datasets.append(get_replay_dataset(args, dataset))
        print_rank_0(f"Loaded replay dataset length: {len(replay_datasets[-1])}", args.global_rank)

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

    print_rank_0("***** Running training *****", args.global_rank)
    for i_task, task in enumerate(train_task_list):
        if i_task > start_i_task:
            print_rank_0(f"Begin training on {task}.", args.global_rank)
            train_one_task(args,
                           train_task_list[task],
                           int(args.num_train_epochs[i_task]),
                           model,
                           device)
            save_model(args, i_task, model, tokenizer)
        else:
            load_task_model(args, model, i_task)


if __name__ == "__main__":
    main()
