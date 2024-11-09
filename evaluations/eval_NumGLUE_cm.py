from utils.metrics import calculate_accuracy


def eval(predicted_sequences, ground_truths):
    accuracy = calculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
