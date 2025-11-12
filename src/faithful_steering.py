import json
import torch
import argparse
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm
from vllm import LLM, SamplingParams
from utils import HINT_MAP, MODEL_MAP
import os
import re
from functools import partial
import torch.distributed as dist


DEFAULT_BUDGET = 4096

def apply_chat(prompt: str, tokenizer):
    """
    Wraps a user prompt in the vLLM chat template.
    """
    conversations = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True
    )


def make_params(n: int, budget: int, cfg) -> SamplingParams:
    """
    Build SamplingParams from model config and given budget.
    """
    kw = {"n": n, "max_tokens": budget}
    if hasattr(cfg, "temperature") and cfg.temperature is not None:
        kw["temperature"] = cfg.temperature
    if hasattr(cfg, "top_k") and cfg.top_k is not None:
        kw["top_k"] = cfg.top_k
    if hasattr(cfg, "top_p") and cfg.top_p is not None:
        kw["top_p"] = cfg.top_p
    return SamplingParams(**kw)


class LinearInterventionHook():
    def __init__(self, direction, weight):
        self.direction = direction
        self.weight = weight
    def __call__(self, module, input, output):
        # align dtype/device
        ref = output[0] if isinstance(output, tuple) else output
        self.direction = self.direction.type_as(ref)
        if isinstance(output, tuple):
            # reconstruct tuple: (modified_first, *rest)
            modified = ref + self.direction.to(ref.device) * self.weight
            return (modified,) + output[1:]
        else:
            return output + self.direction.to(output.device) * self.weight


def add_steering(model, directions, weight, components=None):
    if not components:
        return
    for component in components:
        if component not in directions:
            # silently skip if direction not provided for this component
            continue
        steering_vector = directions[component]
        hook = LinearInterventionHook(steering_vector, weight)
        eval(f"model.{component}.register_forward_hook(hook)")


def parse_components_spec(spec: str, num_layers: int):
    """
    Parse strings like:
      - 'attn0-1' (layers [0])
      - 'mlp1-3'  (layers [1,2])
      - 'attn'    (all attn layers)
      - 'mlp'     (all mlp layers)
      - 'attn0-1mlp1-3' (both)
      - 'attnmlp' (everything)
    Returns: sorted unique list of component paths:
      'model.layers[i].self_attn.o_proj' and 'model.layers[i].mlp.down_proj'
    """
    if not spec:
        return []

    pattern = re.compile(r'(attn|mlp)(\d+(?:-\d+)?)?')
    matches = list(pattern.finditer(spec))
    if not matches:
        raise ValueError(f"Unrecognized component spec: {spec}")

    layers_to_attn = set()
    layers_to_mlp  = set()

    def expand_range(tok):
        # tok: None | 'k' | 'k-m' (end exclusive)
        if tok is None:
            return list(range(num_layers))
        if '-' in tok:
            a, b = tok.split('-', 1)
            s = int(a); e = int(b)
        else:
            s = int(tok); e = s + 1
        s = max(0, s); e = min(num_layers, e)
        if s >= e:
            return []
        return list(range(s, e))

    for m in matches:
        kind = m.group(1)
        rng  = m.group(2)
        layer_idxs = expand_range(rng)
        if kind == 'attn':
            layers_to_attn.update(layer_idxs)
        else:
            layers_to_mlp.update(layer_idxs)

    components = []
    for i in sorted(layers_to_attn):
        components.append(f"model.layers[{i}].self_attn.o_proj")
    for i in sorted(layers_to_mlp):
        components.append(f"model.layers[{i}].mlp.down_proj")
    return components


def register_steering(hf_model, *, direction_path, alpha, components_spec):
    inner = getattr(hf_model, "model", None)
    if inner is None or not hasattr(inner, "layers"):
        raise RuntimeError("Unexpected model structure: missing .model.layers")
    num_layers = len(inner.layers)

    directions = torch.load(direction_path, map_location="cpu")["post"]["direction"]
    comps = parse_components_spec(components_spec, num_layers)
    add_steering(
        hf_model,
        directions=directions,
        weight=alpha,
        components=comps,
    )


# def make_hook(alpha, steering_vec):
#     """
#     Build hook to steer model generation.

#     Inputs:
#         - alpha (float): Scaling factor for steering vector
#         - steering_vec (tensor): Steering vector to be added during generation
#     Outputs:
#         - steering_hook (function): Function to be used for steering
#     """
#     def steering_hook(module, input, output):
#         if isinstance(output, torch.Tensor):
#             return output + alpha * steering_vec.to(output.device)

#         elif isinstance(output, tuple):
#             hidden = output[0]
#             hidden = hidden + alpha * steering_vec.to(hidden.device)
#             # return new tuple with modified first element
#             return (hidden,) + output[1:]

#     return steering_hook


# def run_steering_exp(name, model_name, model, layer, alpha, steering_vec, question_data):
#     """
#     Run the steering experiment at a particular layer given a vector and alpha value.

#     Inputs:
#         name (str): Name for the experiment
#         model_name (str): Name for the model
#         model: Model to be used for generation
#         layer (int): Layer at which steering should be done
#         alpha (float): Amount by which steering vector should be scaled
#         steering_vec (tensor): Vector encoding a particular trait
#         question_data (lst): List of question prompt data

#     Outputs:
#         faithful_rate (float): Proportion of responses that are faithful when steering is applied with the specified parameters.
#     """

#     # List to record all steered generation text
#     all_decoded = []
#     batch_size = 8

#     questions = ["Problem: " + i['question'] + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}. " + i['hint'] + " " + i['gold'] for i in question_data]

#     # Add hook for steering generation
#     handle = model.model.layers[layer].register_forward_hook(make_hook(alpha, steering_vec))

#     # Iterate over prompts, generate in batches
#     for i in tqdm(range(0, len(questions), batch_size)):
#         batch_data = question_data[i:i+batch_size]
#         batch_prompts = questions[i:i+batch_size]

#         # Get prompts into the correct format (same as original generation setting)
#         batch_prompts_formatted = [
#             tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
#                                         tokenize=False, add_generation_prompt=True)
#             for prompt in batch_prompts
#         ]

#         input_ids = tokenizer(batch_prompts_formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)

#         with torch.no_grad():
#             outputs = model.generate(
#                 **input_ids,
#                 max_new_tokens=4096
#             )

#         # Decode generation
#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         # Filter out prompt to avoid overcounting faithful responses
#         responses = [{"response": x.split("<think>")[1], "prompt": y, "hint": HINT_MAP[z['hint']], "prediction": z['prediction'], "answer": z["gold"]} for x, y, z in zip(decoded, batch_prompts, batch_data)]
#         # Track generated responses
#         all_decoded.extend(responses)

#     handle.remove()

#     # Save steered text generations
#     os.makedirs(f"../results/steered_gens/{model_name}", exist_ok=True)
#     with open(f"../results/steered_gens/{model_name}/{name}_gen.json", "w") as f:
#         json.dump(all_decoded, f)

#     return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, help='Layers at which to test.')
    parser.add_argument("--alphas", nargs="+", type=float, help="Alpha values to test.")
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="deepseek-llama3-8b")
    args = parser.parse_args()

    hint_filtered = []

    # Go through results with normal and hinted prompting
    # Collect all questions where the presence of the hint changes the model answer from incorrect to correct
    for dataset in ['MMLU-Pro-math']: #, 'MATH-500', 'AIME2024', 'gpqa', 'AIME2025', 'MMLU-Pro-math']:
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

    # Load model
    model_id = MODEL_MAP[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
    cfg = GenerationConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm = LLM(
        model=model_id,
        max_model_len=min(DEFAULT_BUDGET + 256, max_pos),
        dtype='half'
    )

    steering_configs = []

    # for alpha in args.alphas:
    #     tup = (f"l{args.layers[0]}_{args.layers[1]}_{args.layers[2]}_{alpha}", args.layers, alpha, f"../results/steering_vecs/{args.model}/")
    #     steering_configs.append(tup)

    for layer in args.layers:
        for alpha in args.alphas:
            tup = (f"l{layer}_{alpha}", layer, alpha, f"../results/steering_vecs/{args.model}/sv_{layer}.pt")
            steering_configs.append(tup)

    count = 0

    for name, layer, alpha, path in steering_configs:

        # for layer in layers:
        steering = partial(
            register_steering,
            direction_path=path,
            alpha=alpha,
            components_spec=f"mlp{layer}",
        )

        hf_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

        handles = []
        # for layer in layers:
        # sv_path = path + f"sv_{layer}.pt"
        # register hook for this layer only
        comps = [f"model.layers[{layer}].mlp.down_proj"]
        directions = torch.load(path, map_location="cpu")["post"]["direction"]
        for comp in comps:
            if comp in directions:
                hook = LinearInterventionHook(directions[comp], alpha)
                h = eval(f"hf_model.{comp}.register_forward_hook(hook)")
                handles.append(h)

        # ADJUST LATER
        sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=min(DEFAULT_BUDGET, max_pos - 512))

        prompts = ["Problem: " + i['question'] + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}. " + i['hint'] + " " + i['gold'] + ".\n\n" for i in hint_filtered]

        results = llm.generate(prompts=prompts, sampling_params=sampling_params)

        for h in handles:
            h.remove()

        # Only one run needed when generation is deterministic
        runs = {rid: [] for rid in range(10)}

        responses = [{"response": i.outputs[0].text, "prompt": i.prompt, "hint": j['hint'], "prediction": j['prediction'], "answer": j["gold"]} for i, j in zip(results, hint_filtered)]

        # Save steered text generations
        os.makedirs(f"../results/steered_gens/{args.model}", exist_ok=True)
        with open(f"../results/steered_gens/{args.model}/{name}_gen.json", "w") as f:
            json.dump(responses, f)

        count += 1

        print(f"Completed {name}. {len(steering_configs) - count} configs remaining.\n")


    if dist.is_initialized():
        dist.destroy_process_group()
