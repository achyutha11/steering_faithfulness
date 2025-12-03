import json
import torch
import argparse
from model import load_model
import re
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import HINT_MAP, MODEL_MAP
import os

class ResponseDataset(Dataset):
    """
    Class to store text data from a particular trait.
    """
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}


def collate_fn(batch):
    return {
        key: torch.stack([example[key] for example in batch], dim=0)
        for key in batch[0]
    }


def get_mean_acts(dataloader, layer, model):
    """
    Get the mean activations of the model on a set of prompts at a particular layer's residual stream.

    Inputs:
        - dataloader (DataLoader): DataLoader object containing set of prompts for analysis
        - layer (int): Layer to be analyzed
        - model: Model from which activations should be collected

    Outputs:
        - mean_acts (tensor): Mean last-token activations from the model at the specified layer on the provided set of prompts
    """

    final_acts = []

    # Process data in batches
    for batch in tqdm(dataloader):

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Run inputs through the model, and save hidden states
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Get activations at specified layer
        hidden = outputs.hidden_states[layer].detach()

        # Retrieve last-token activations specifically from each of the inputs in the batch
        final_token_acts = hidden[torch.arange(hidden.size(0)), -1]
        final_acts.append(final_token_acts.cpu())

    # Concatenate activations from all inputs, then take mean across inputs
    # Results in a 1 x D vector, where D is the dimension of the residual stream
    final_acts = torch.cat(final_acts, dim=0)
    mean_acts = final_acts.mean(dim=0)

    torch.cuda.empty_cache()

    return mean_acts


def get_window_mean_acts(dataloader, layer, model):
    all_acts = []

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden = outputs.hidden_states[layer]

        mean_seq_acts = hidden.mean(dim=1)

        all_acts.append(mean_seq_acts.cpu())

    final_acts = torch.cat(all_acts, dim=0)
    return final_acts.mean(dim=0)


def get_steering_vec(layer, dl1, dl2, model):
    """
    Get steering vector for a desired trait for a particular layer.

    Inputs:
        - layer (int): Layer from which the steering vector is extracted
        - dl1 (DataLoader): DataLoader object containing positive examples of the trait
        - dl2 (DataLoader): DataLoader object containing negative examples of the trait
        - model: Model for which we need a steering vector

    Outputs:
        - tensor: Steering vector for faithfulness
    """

    # Retrieve mean activations for faithful and unfaithful data, and return the difference
    dl1_acts = get_window_mean_acts(dl1, layer, model)
    dl2_acts = get_window_mean_acts(dl2, layer, model)
    return dl1_acts - dl2_acts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, help='Layers at which to test.')
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="deepseek-llama3-8b")
    args = parser.parse_args()

    hint_filtered = []

    # Go through results with normal and hinted prompting
    # Collect all questions where the presence of the hint changes the model answer from incorrect to correct
    for dataset in ['MATH-500', 'gsm8k', 'AIME2024', 'gpqa', 'AIME2025']: #, 'MMLU-Pro-math']:
        with open(f"../src/normal_results/{dataset}/{args.model}/1_runs.json", "r") as f:
            normal_results = json.load(f)

        with open(f"../src/hint_results/{dataset}/{args.model}/1_runs.json", "r") as f:
            hint_results = json.load(f)

        incor_to_cor = []
        normal_recs = normal_results['runs'][0]['records']
        hint_recs = hint_results['runs'][0]['records']
        reasoning_length = 15000

        # Filtering for reasoning length to ensure we don't just include questions where the model never completed its answer
        for index, question in enumerate(normal_recs):
            if not question['correct'] and hint_recs[index]['correct'] and question['reasoning_length'] < reasoning_length and str(question['prediction']).split("\\%")[0] != question['gold']:
                incor_to_cor.append(index)

        for index in incor_to_cor:
            hint_filtered.append(hint_recs[index])

    faithful = []
    unfaithful = []

    # Simple regular expression check to see if the hint was cited in model responses
    # Keeping track of the index at which the hint was cited (to be used later to create a faithful text dataset)
    for data in hint_filtered:
        hint_cited = bool(re.search(HINT_MAP[data['hint']], data['full_response'][:16000]))
        data['index'] = re.search(HINT_MAP[data['hint']], data['full_response']).span()[0] if hint_cited else 0
        faithful.append(data) if hint_cited else unfaithful.append(data)

    # Faithful data obtained by taking the 100 characters either side of the hint citation index
    # faithful_responses = [i['full_response'][i['index'] - 100: i['index'] + 100] for i in faithful]
    # faithful_responses = [i['full_response'][: i['index'] + 100] for i in faithful]
    # faithful_responses = [i['full_response'][:16000] for i in faithful]
    faithful_responses = [i['full_response'][:500] for i in faithful]


    # Unfaithful data obtained by taking the full response
    unfaithful_responses = [i['full_response'][:500] for i in unfaithful]

    # Load model
    model_id = MODEL_MAP[args.model]
    model, tokenizer = load_model(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure intermediate hidden states are tracked
    model.config.output_hidden_states = True

    # Create faithful and unfaithful dataloader objects
    faithful_ds = ResponseDataset(faithful_responses, tokenizer)
    faithful_dl = DataLoader(faithful_ds, batch_size=1, collate_fn=collate_fn)

    unfaithful_ds = ResponseDataset(unfaithful_responses, tokenizer)
    unfaithful_dl = DataLoader(unfaithful_ds, batch_size=1, collate_fn=collate_fn)

    layer_list = args.layers

    for layer in layer_list:

        # Obtain steering vector
        steering_vec = get_steering_vec(layer, faithful_dl, unfaithful_dl, model)

        directions_dict = {
            "post": {
                "direction": {
                    f"model.layers[{layer}].mlp.down_proj": steering_vec
                }
            }
        }

        os.makedirs(f"../results/steering_vecs/{args.model}", exist_ok=True)
        torch.save(directions_dict, f"../results/steering_vecs/{args.model}/sv_{layer}.pt")
