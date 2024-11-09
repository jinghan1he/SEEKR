import os
import glob
import json
import argparse

from evaluations import eval_ScienceQA, eval_MeetingBank, eval_CStance, \
    eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten, \
    eval_rouge # to be continued


def main(result_dir):
    result_files = glob.glob(os.path.join(result_dir, "*.json"))
    for file in result_files:
        print(f"Re-evaluating {file}...")
        results = json.load(open(file, "r"))
        sources_sequences = results["prompts"]
        predicted_sequences = results["results"]
        ground_truths = results["labels"]

        answers = []
        for answer in predicted_sequences:
            answer = answer.replace("：", ":")
            if "\nReasoning:" in answer:
                short_answer = answer.split("\nReasoning:")[0].strip()  # Py150: answer first then reasoning
            elif "\nAnswer:" in answer:
                short_answer = answer.split("\nAnswer:")[1].strip()
            elif "\n回答:" in answer:
                short_answer = answer.split("\n回答:")[1].strip()
            elif "\n答案:" in answer:
                short_answer = answer.split("\n答案:")[1].strip()
            else:
                short_answer = answer.strip()

            if len(short_answer) == 0:
                short_answer = " "
            elif "C-STANCE" in file or "FOMC" in file or "ScienceQA" in file:
                short_answer = short_answer[0]

            answers.append(short_answer)

        if "ScienceQA" in file:
            evaluation_result = eval_ScienceQA.eval(answers, ground_truths)
        elif "MeetingBank" in file:
            evaluation_result = eval_MeetingBank.eval(answers, ground_truths)
        elif "C-STANCE" in file:
            evaluation_result = eval_CStance.eval(answers, ground_truths)
        elif "Py150" in file:
            evaluation_result = eval_Py150.eval(answers, ground_truths)
        elif "FOMC" in file:
            evaluation_result = eval_FOMC.eval(answers, ground_truths)
        elif "NumGLUE-cm" in file:
            evaluation_result = eval_NumGLUE_cm.eval(answers, ground_truths)
        elif "NumGLUE-ds" in file:
            evaluation_result = eval_NumGLUE_ds.eval(answers, ground_truths)
        elif "20Minuten" in file:
            evaluation_result = eval_20Minuten.eval(sources_sequences, answers, ground_truths)
        elif "task" in file:
            evaluation_result = eval_rouge.eval(answers, ground_truths)
        else:
            evaluation_result = {}

        save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, file)


def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list, ground_truths: list, file: str):
    # save as a json file
    df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences, 'labels': ground_truths}
    with open(file, "w+", encoding='utf-8') as file:
        json.dump(df, file, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=''
    )
    parser.add_argument(
        '--predict',
        type=str,
        default='predictions'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(os.path.join(args.output_dir, args.predict))
