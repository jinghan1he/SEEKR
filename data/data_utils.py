from typing import Tuple
import torch
from torch.utils.data import Dataset
import os
import hashlib
from . import raw_datasets


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, answer_dataset) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.answer_dataset = answer_dataset
        assert len(self.prompt_dataset) == len(self.answer_dataset)

    def __len__(self):
        return len(self.prompt_dataset)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompt_dataset[idx],
            "answer": self.answer_dataset[idx]
        }


class SubPromptDataset(Dataset):

    def __init__(self, dataset: PromptDataset, indices):
        self.prompt_dataset = []
        self.answer_dataset = []
        print(f"Selecting samples for replay.")
        for idx in indices:
            prompt_sentence = dataset.prompt_dataset[idx]
            answer_sentence = dataset.answer_dataset[idx]
            print(f"Prompt:\n{prompt_sentence}\nAnswer:\n{answer_sentence}")
            self.prompt_dataset.append(prompt_sentence)
            self.answer_dataset.append(answer_sentence)
        # init from replay dataset, which is a PromptDataset

    def __len__(self):
        return len(self.prompt_dataset)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompt_dataset[idx],
            "answer": self.answer_dataset[idx],
            "idx": idx,
        }


class DistillAttnBudget2PromptDataset(Dataset):
    def __init__(self, dataset: PromptDataset, teacher_logits: torch.Tensor, all_teacher_attns: Tuple[torch.Tensor],
                 attn_query: torch.Tensor, select_heads: dict, target_layers: list):
        self.prompt_dataset = dataset.prompt_dataset
        self.answer_dataset = dataset.answer_dataset
        self.teacher_logits = teacher_logits
        self.attn_query = attn_query
        self.select_heads = select_heads

        selected_teacher_attns = []
        for idx in range(len(self)):
            teacher_attns = []
            for l, l_attns in enumerate(all_teacher_attns):
                layer = target_layers[l]
                if layer in select_heads:
                    teacher_attns.append(l_attns[idx][select_heads[layer]])
            selected_teacher_attns.append(tuple(teacher_attns))
        self.teacher_attns = tuple(selected_teacher_attns)

    def __len__(self):
        return len(self.prompt_dataset)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompt_dataset[idx],
            "answer": self.answer_dataset[idx],
            "teacher_logits": self.teacher_logits[idx],
            "teacher_attns": self.teacher_attns[idx],  # nlayers, n_heads, seq_len, seq_len
            "attn_query": self.attn_query[idx],
            "select_heads": self.select_heads,    # layer-head dict
        }


# 根据传入的sampls，调用dataset object，获取数据想要的部分,tokenize
def get_prompt_dataset(current_dataset, raw_dataset, add_sys_prefix=False):
    prompt_dataset = []
    answer_dataset = []

    for i, tmp_data in enumerate(current_dataset):
        prompt_sentence = raw_dataset.get_prompt(tmp_data)  # the accept response
        if add_sys_prefix:
            prompt_sentence = f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{prompt_sentence}"
        answer_sentence = raw_dataset.get_answer(tmp_data)  # the reject response
        prompt_dataset.append(prompt_sentence)
        answer_dataset.append(answer_sentence)

    return PromptDataset(prompt_dataset, answer_dataset)


# step 2
def create_dataset(local_rank, dataset_name, output_path,
                   seed, add_sys_prefix=False, for_backbone=False):
    # 加载数据集，用datasets接口加载好返回，此外做了train,eval,test分片
    raw_dataset = raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank, dataset_name, for_backbone=for_backbone)

    train_dataset = raw_dataset.get_train_data()
    train_dataset = get_prompt_dataset(train_dataset, raw_dataset, add_sys_prefix=add_sys_prefix)

    eval_dataset = raw_dataset.get_eval_data()
    eval_dataset = get_prompt_dataset(eval_dataset, raw_dataset, add_sys_prefix=add_sys_prefix)

    test_dataset = raw_dataset.get_test_data()
    test_dataset = get_prompt_dataset(test_dataset, raw_dataset, add_sys_prefix=add_sys_prefix)

    return train_dataset, eval_dataset, test_dataset


# step 1
def create_prompt_dataset(local_rank,
                          data_path,
                          output_path,
                          seed,
                          reload=False,
                          add_sys_prefix=False,
                          for_backbone=False,
                          distributed=True,
                          ):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = data_path
    # 为什么单独要 sft data？
    fname = f"{fname}_seed{seed}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"
    test_fname = f"{output_path}/testdata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    # buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    # # 将不同进程的张量汇总sum
    # torch.distributed.all_reduce(buf_create_cache)

    # for debug
    # if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
    if local_rank <= 0:
        train_dataset, eval_dataset, test_dataset = create_dataset(
            local_rank, data_path, output_path,
            seed, add_sys_prefix=add_sys_prefix, for_backbone=for_backbone)

        # torch.save的数据格式可以是任意的
        # 提前准备好，可以加速预处理，torch.load 速度也会比较快
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
        torch.save(test_dataset, test_fname)

    if distributed:
        torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname), torch.load(test_fname)
