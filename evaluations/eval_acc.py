from utils.metrics import calculate_accuracy


def resolve(dataset: list):
    answers = []
    for datium in dataset:
        answers.append(datium.split()[0])
    return answers


def eval(predicted_sequences, ground_truths):
    accuracy = calculate_accuracy(resolve(predicted_sequences), ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
