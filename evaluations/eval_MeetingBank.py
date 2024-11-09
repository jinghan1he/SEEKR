from utils.metrics import calculate_bleu, calculate_rouge


def eval(predicted_sequences, ground_truths):
    bleu_1 = calculate_bleu(predicted_sequences, ground_truths, 1)
    bleu_4 = calculate_bleu(predicted_sequences, ground_truths, 4)
    rouge = calculate_rouge(predicted_sequences, ground_truths)
    evaluation_result = {"bleu-1": bleu_1, "bleu-4": bleu_4, "rouge-L": rouge}
    return evaluation_result
