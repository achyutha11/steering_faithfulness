import json
import re
import os
import argparse
from utils import MODEL_MAP, HINT_MAP
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="deepseek-llama3-8b")
    parser.add_argument("--dataset", default="MMLU-Pro-Math")
    args = parser.parse_args()

    results = {}

    directory = f"../results/steered_gens/{args.model}/{args.dataset}/"

    for filename in os.listdir(directory):

        if filename.endswith(".json"):

            filepath = os.path.join(directory, filename)

            with open(filepath, 'r') as file:
                steered_gen = json.load(file)

            faithful_count = 0

            for data in steered_gen:
                faithful_count += bool(re.search(HINT_MAP[data['hint']], data['response']))

            faithfulness_rate = faithful_count / len(steered_gen)

            name = os.path.splitext(filename)[0]
            results[name] = faithfulness_rate

    os.makedirs(f"../results/data/{args.model}", exist_ok=True)
    with open(f"../results/data/{args.model}/steering_results_{args.dataset}.pkl", "wb") as f:
        pickle.dump(results, f)
