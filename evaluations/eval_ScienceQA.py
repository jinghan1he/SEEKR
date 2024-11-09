from utils.metrics import calculate_bleu, calculate_rouge, calculate_accuracy


# resolving answer and reasoning
def resolve(dataset: list):
    answers = []
    reasonings = []
    for datium in dataset:
        if len(datium) > 0:
            answers.append(datium[0]) # the first char is the answer. e.g. A, B,...
            reasonings.append(datium[2:]) # A/nBecause...
        else:
            answers.append(" ")
            reasonings.append(" ")
    outputs = {"answers": answers, "reasonings": reasonings}
    return outputs


def eval(predicted_sequences, ground_truths):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    bleu_1 = calculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    bleu_4 = calculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    rouge = calculate_rouge(outputs["reasonings"], gts["reasonings"])
    accuracy = calculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge, "accuracy": accuracy}
    return evaluation_result
