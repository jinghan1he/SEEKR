import os
import json
import numpy as np
import pandas as pd
import argparse

task_metrics = {"C-STANCE": "accuracy",
                "FOMC": "accuracy",
                "MeetingBank": "rouge-L",
                "Py150": "similarity",
                "ScienceQA": "accuracy",
                "NumGLUE-cm": "accuracy",
                "NumGLUE-ds": "accuracy",
                "20Minuten": "sari",
                "task639": "rouge-L",
                "task1590": "rouge-L",
                "task1729": "rouge-L",
                "task181": "rouge-L",
                "task748": "rouge-L",
                "task1510": "rouge-L",
                "task002": "rouge-L",
                "task073": "rouge-L",
                "task591": "rouge-L",
                "task511": "rouge-L",
                "task1290": "rouge-L",
                "task1572": "rouge-L",
                "task363": "accuracy",
                "task875": "accuracy",
                "task195": "accuracy"
}


def read(file):
    return json.load(open(file, "r"))


def get_task_name(file):
    for task_name in task_metrics:
        if task_name in file:
            return task_name


def get_round(file):
    return int(file.split("-")[1])


def get_task(file):
    return int(file.split("-")[2])


def get_task_acc(output_file, task_name):
    acc = read(output_file)['eval'][task_metrics[task_name]]
    if isinstance(acc, list):
        # sari metric
        acc = acc[0]["sari"]
    acc = float(acc)
    acc = acc * 0.01 if acc > 1 else acc
    return acc


def cl_results(output_dir):
    result_files = os.listdir(output_dir)
    num_tasks = max([get_round(file) for file in result_files]) + 1
    tasks = [""] * num_tasks
    results = np.zeros((num_tasks, num_tasks))
    for file in result_files:
        r, t, n = get_round(file), get_task(file), get_task_name(file)
        acc = get_task_acc(os.path.join(output_dir, file), n)
        results[t, r] = acc * 100
        tasks[t] = n
    # compute bwt
    avg = np.mean(results[:, -1])
    bwt = np.mean([results[i, -1] - results[i, i] for i in range(num_tasks)])
    results = pd.DataFrame(results, columns=tasks, index=tasks)
    pd.options.display.float_format = '{:.2f}'.format
    print(results)
    print(f"avg {avg}")
    print(f"bwt {bwt}")
    results2 = pd.DataFrame(np.zeros((1, num_tasks+2)), columns=tasks+["avg", "bwt"])
    results2.iloc[0, :-2] = results.T.iloc[-1, :]
    results2.iloc[0, -2] = avg
    results2.iloc[0, -1] = bwt
    print(results2)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cl_results(os.path.join(args.output_dir, args.predict))

