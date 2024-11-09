import argparse
import os
import sys
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from data.data_collator import DataCollator
from data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, set_random_seed, load_hf_tokenizer
from utils.model_utils import create_hf_model
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten, eval_rouge, eval_acc  # to be continued


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')

    def list_of_ints(arg):
        if ',' in arg:
            return list(map(lambda x: int(x), arg.split(',')))
        else:
            return list(map(lambda x: int(x), arg.split()))

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path', type=str, default='Dahoas/rm-static', help='Path to the training dataset. A single data path.')
    parser.add_argument('--data_output_path', type=str, default='/tmp/dataset_files/', help='Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)')
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--inference_model_path", type=str, required=True, help="Path to inference model.",)
    parser.add_argument("--max_prompt_len", type=int, default=512, help="The maximum sequence length.",)
    # inference params
    parser.add_argument("--max_ans_len", type=int, default=256, help="The maximum answer length.",)
    parser.add_argument("--temperature", type=float, default=0.1, help="Generate temperature params.",)
    parser.add_argument("--do_sample", action="store_false", help="Whether or not to use sampling.",)
    parser.add_argument("--num_return_sequences", type=int, default=2, help="num of return sequences")
    parser.add_argument("--inference_batch", type=int, default=4, help="Inference batch size.",)
    parser.add_argument("--inference_tasks", type=list_of_strings, default='all', help='Datasets to be used.')
    parser.add_argument("--test_rounds", type=list_of_ints, default=None, help="Total number of training epochs to perform.")
    parser.add_argument("--test_tasks", type=list_of_ints, default=None, help="Total number of training epochs to perform.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--inference_output_path', type=str, default=None, help="Where to store inference results.")

    args = parser.parse_args()

    return args


def prediction(args, model, infer_dataloader, tokenizer, device):
    progress_bar = tqdm(total=len(infer_dataloader), leave=True)
    predicted_sequences = []
    sources_sequences = []
    ground_truths = []
    model.eval()
    for step, batch in enumerate(infer_dataloader):
        sources_sequences += batch['sources']
        ground_truths += batch['gts']
        del batch['sources']
        del batch['gts']
        batch = to_device(batch, device)
        prompt_len = batch['input_ids'].shape[1]
        # update progress bar
        progress_bar.update(1)
        description = f"Step {step}"
        progress_bar.set_description(description, refresh=False)
        with torch.no_grad():
            generate_ids = model.generate(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask'],
                                          max_new_tokens=args.max_ans_len,
                                          bos_token_id=tokenizer.bos_token_id,
                                          eos_token_id=tokenizer.eos_token_id,
                                          pad_token_id=tokenizer.unk_token_id,
                                          temperature=args.temperature,
                                          do_sample=args.do_sample,
                                          num_return_sequences=args.num_return_sequences,
                                          use_cache=True
                                          )
        sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        predicted_sequences += sequences
    multi_predicted_sequences = [[] for i in range(args.num_return_sequences)]
    for i in range(0, len(predicted_sequences), args.num_return_sequences):
        for j in range(args.num_return_sequences):
            multi_predicted_sequences[j].append(predicted_sequences[i + j])
    return sources_sequences, multi_predicted_sequences, ground_truths


def save_inference_results(args, evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                           ground_truths: list, id: int, round: int, i_task: int, task: str):
    # save as a json file
    df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences, 'labels': ground_truths}
    if not os.path.exists(args.inference_output_path + str(id)):
        os.makedirs(args.inference_output_path + str(id))
    with open(args.inference_output_path + str(id) + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
        json.dump(df, file, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda")

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    inference_tasks = args.inference_tasks
    task_num = len(inference_tasks)
    for round in range(task_num):  # load models and adapters of a new round in continual learning
        if args.test_rounds is not None and round not in args.test_rounds:
            continue
        inference_model_path = os.path.join(args.inference_model_path, str(round))
        print_rank_0("Inference Model Path: " + inference_model_path, args.local_rank)

        model = create_hf_model(AutoModelForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config=None,
                                )

        inference_model = torch.load(os.path.join(inference_model_path, "pytorch_model.bin"))
        for name, param in model.named_parameters():
            param.data.copy_(inference_model[name])
        del inference_model

        model.to(device)

        for inference_task_id in range(round+1):    # evaluation for previous tasks in a single round
            if args.test_tasks is not None and inference_task_id not in args.test_tasks:
                continue
            inference_task = inference_tasks[inference_task_id]
            if os.path.exists(args.inference_output_path + str(args.num_return_sequences - 1) + "/results-" + str(round) + "-" + str(inference_task_id) + "-" + inference_task + ".json"):
                continue
            dataset_path = os.path.join(args.data_path, inference_task)
            # Prepare the data
            _, _, infer_dataset = create_prompt_dataset(
                args.local_rank,
                dataset_path,
                args.data_output_path,
                args.seed,
                distributed=False
            )

            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=args.max_prompt_len,
                max_ans_len=args.max_ans_len,
                pad_to_multiple_of=8,
                inference=True
            )
            infer_sampler = SequentialSampler(infer_dataset)
            if inference_task in ["Py150", "MeetingBank", "C-STANCE", "task181", "20Minuten"]:
                batch_size = 4
            else:
                batch_size = args.inference_batch
            infer_dataloader = DataLoader(infer_dataset,
                                          collate_fn=inf_data_collator,
                                          sampler=infer_sampler,
                                          batch_size=batch_size)

            print_rank_0(f"***** Start inference on {inference_task} *****", args.local_rank)
            sources_sequences, multi_predicted_sequences, ground_truths = prediction(args, model, infer_dataloader, tokenizer, device)

            for i in range(args.num_return_sequences):
                predicted_sequences = multi_predicted_sequences[i]
                assert len(predicted_sequences) == len(
                    ground_truths), f"Lengths unmatched: {len(predicted_sequences)} predicted_sequences vs. {len(ground_truths)} ground_truths"

                # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
                if inference_task == "ScienceQA":
                    evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
                elif inference_task == "MeetingBank":
                    evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
                elif inference_task == "C-STANCE":
                    evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
                elif inference_task == "Py150":
                    evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
                elif inference_task == "FOMC":
                    evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
                elif inference_task == "NumGLUE-cm":
                    evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
                elif inference_task == "NumGLUE-ds":
                    evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
                elif inference_task == "20Minuten":
                    evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
                elif inference_task in ["task639", "task1590", "task1729", "task181", "task748", "task1510", "task002",
                                        "task073", "task591", "task511", "task1290", "task1572"]:
                    evaluation_result = eval_rouge.eval(predicted_sequences, ground_truths)
                elif inference_task in ["task363", "task875", "task195"]:
                    evaluation_result = eval_acc.eval(predicted_sequences, ground_truths)
                else:
                    evaluation_result = {}

                print("***** Saving inference results *****")
                save_inference_results(args, evaluation_result, sources_sequences, predicted_sequences, ground_truths, i, round, inference_task_id, inference_task)


if __name__ == "__main__":
    main()
