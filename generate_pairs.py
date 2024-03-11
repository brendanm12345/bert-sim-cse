import csv
import json

def generate_pairs_and_write_to_json(file_path, output_json_path):
    pairs = {"positive_pairs": [], "negative_pairs": []}

    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if len(row) == 3:  # Ensure row has sentence, entailment, and contradiction
                sentence, entailment, contradiction = row
                pairs["positive_pairs"].append({"sentence": sentence, "entailment": entailment})
                pairs["negative_pairs"].append({"sentence": sentence, "contradiction": contradiction})

    with open(output_json_path, mode='w', encoding='utf-8') as json_file:
        json.dump(pairs, json_file, ensure_ascii=False, indent=4)

generate_pairs_and_write_to_json("nli_for_simcse.csv", "pairs.json")
