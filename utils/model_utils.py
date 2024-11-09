import os
import math
import torch
from transformers import AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig

from utils.utils import save_hf_format, save_zero_three_model, print_rank_0


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True)

    # llama use eos_token_id but not end_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    # compatible with OPT and llama2
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def save_model(args, round, model, tokenizer):
    if args.output_dir is not None:
        print_rank_0('saving model to ' + args.output_dir + "/" + str(round) + '...', args.global_rank)

    if args.global_rank == 0:
        save_hf_format(model, tokenizer, args, sub_folder=str(round))

    if args.zero_stage == 3:
        # For zero stage 3, each gpu only has a part of the model, so we need a special save function
        save_zero_three_model(model,
                              args.global_rank,
                              args.output_dir,
                              zero_stage=args.zero_stage,
                              sub_folder=str(round))
    print_rank_0('Sucessful saving model after round {}'.format(round), args.global_rank)

    torch.distributed.barrier()


def load_task_model(args, model, i_task=-1):
    if i_task > -1:
        task_model_path = os.path.join(args.output_dir, str(i_task), "pytorch_model.bin")
        task_model = torch.load(task_model_path, map_location='cpu')
        for name, param in model.named_parameters():
            name = name if name in task_model else name.replace("module.", "")
            param.data.copy_(task_model[name])
        del task_model
        print_rank_0(f"Loaded model from {task_model_path}", args.global_rank)


def get_start_task(args):
    i_task = -1
    for i in range(len(args.dataset_name)):
        if os.path.exists(os.path.join(args.output_dir, str(i), "pytorch_model.bin")):
            i_task = i
    return i_task

