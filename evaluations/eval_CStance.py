from utils.metrics import calculate_accuracy

# resolving answer and reasoning
def resolve(dataset: list):
    answers = []
    for datium in dataset:
        if len(datium) > 0:
            answers.append(datium[0]) # the first char is the answer. e.g. A, B,...
        else:
            answers.append(" ")
    return answers

def eval(predicted_sequences, ground_truths):
    predicted_sequences = resolve(predicted_sequences)
    accuracy = calculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result