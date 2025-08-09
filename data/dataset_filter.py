import csv 
def extract_dot_and_question_positions(sample, tokenizer):
    encoded = tokenizer.encode(sample, add_special_tokens=False)
    dot_id = tokenizer.encode(".", add_special_tokens=False)[0]
    q_id = tokenizer.encode("?", add_special_tokens=False)[0]

    dot_positions = [i for i, tid in enumerate(encoded) if tid == dot_id]
    q_positions = [i for i, tid in enumerate(encoded) if tid == q_id]
    return dot_positions, q_positions


def filter_samples(tokenizer):
    with open("data/rule_taker_full_sentence.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        filtered = []
        for idx, sample in enumerate(data):
            if idx == 0:
                continue
            if idx % 3 == 1:
                Base, Source, Question, Base_Answer, Expected_Answer = sample
                text = Base + " Question: " + Question + "?"
                output = Base_Answer
                dot_pos, q_pos = extract_dot_and_question_positions(text, tokenizer)
                filtered.append([text, output, dot_pos, q_pos])
    return filtered
