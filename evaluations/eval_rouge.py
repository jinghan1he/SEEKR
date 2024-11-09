from utils.metrics import calculate_rouge, calculate_accuracy


def eval(predicted_sequences, ground_truths):
    rouge = calculate_rouge(predicted_sequences, ground_truths)
    accuracy = calculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result
