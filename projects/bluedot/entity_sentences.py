# requirements: anthropic python-dotenv datasets

from dotenv import load_dotenv
import json
import re
import os
import sys
from types import SimpleNamespace

import anthropic
from datasets import load_dataset


PROMPT_BATCH_SIZE = 10
with open("entity-prompt.md", "rt") as fp:
    PROMPT_TEMPLATE = fp.read()


config_defaults = {
    "subset": None,
    "split": "train",
    "max_length": 2048,
    "ignore_buffer": 0,
    "default_ignore": False,
    "last_span_token": False,
    "pos_weight": 1.0,
    "neg_weight": 1.0,
    "shuffle": True,
    "seed": 42,
    "process_on_the_fly": False,
    "max_num_samples": None,
}

train_dataset_config = [
    {
        "dataset_id": "llama3_1_8b_longfact_train",
        "hf_repo": "./longfact-annotations/",  # "obalcells/longfact-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "train",
        "max_length": 1536,
        "default_ignore": False,
        "last_span_token": False,
        "ignore_buffer": 0,
        "pos_weight": 10.0,
        "neg_weight": 10.0,
        "shuffle": True,
        "seed": 42,
        "process_on_the_fly": False,
    },
]
"""
    {
        "dataset_id": "llama3_1_8b_longfact_augmented_train",
        "hf_repo": "./longfact-augmented-annotations",  # "obalcells/longfact-augmented-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "train",
        "max_length": 1536,
        "default_ignore": False,
        "last_span_token": False,
        "ignore_buffer": 0,
        "pos_weight": 10.0,
        "neg_weight": 10.0,
        "shuffle": True,
        "seed": 42,
        "process_on_the_fly": False,
    },
]
"""

configs = []
for i, entry in enumerate(train_dataset_config):
    cfg = SimpleNamespace(**dict(config_defaults, **entry))
    configs.append(cfg)


def parse_json_response(text):
    """
    Parse JSON from a Claude response that may or may not be fenced with backticks.
    Handles: raw JSON, ```json ... ```, ``` ... ```, and leading/trailing whitespace.
    """
    text = text.strip()

    # Strip fenced code block if present (```json or ``` or any lang tag)
    fenced = re.match(r"^```(?:\w+)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    return json.loads(text)


def get_claims(text, entities):
    entities_md = "\n".join(
        f"{i+1}. `{entity['span']}`" for i, entity in enumerate(entities)
    )
    prompt = PROMPT_TEMPLATE.format(text=text, entities=entities_md)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    response_raw = message.content[0].text
    try:
        response = parse_json_response(response_raw)
    except json.decoder.JSONDecodeError:
        sys.stderr.write("JSON decode error\nInput:" + repr(response_raw))
        raise

    assert len(response) == len(entities)
    assert all(response[i]["index"] == i + 1 for i in range(len(response)))
    assert all(
        response[i]["entity"] == entities[i]["span"] for i in range(len(response))
    )

    return [
        {
            "span": entry["entity"],
            "claim": entry["output"],
        }
        for entry in response
    ]


if __name__ == "__main__":
    load_dotenv()

    datasets = []
    dataset_claims = []
    for i, cfg in enumerate(configs):
        if cfg.subset:
            dataset = load_dataset(cfg.hf_repo, cfg.subset, split=cfg.split)
        else:
            dataset = load_dataset(cfg.hf_repo, split=cfg.split)
        datasets.append(dataset)

        try:
            fp = open(os.path.join(cfg.hf_repo, "_claims.jsonl"), "rt")
        except FileNotFoundError:
            claims = []  # [{'claims': [{'span': '', 'claim': ''}]}]
        else:
            claims = [json.loads(line.strip()) for line in fp]
            fp.close()
        dataset_claims.append(claims)

        # Sanity checks
        for i, claim in enumerate(claims):
            if len(dataset[i]["annotations"]) != len(claim["claims"]):
                print(
                    f'ERROR: In {cfg.hf_repo} at index {i}, there are {len(dataset[i]["annotations"])} annotations and {len(claim["claims"])} claims'
                )

    for dataset_index, dataset in enumerate(datasets):
        cfg = configs[dataset_index]
        claims = dataset_claims[dataset_index]

        prompts_total = sum(
            [(len(d["annotations"]) - 1) // PROMPT_BATCH_SIZE + 1 for d in dataset]
        )
        prompts_completed = sum(
            [(len(c["claims"]) - 1) // PROMPT_BATCH_SIZE + 1 for c in claims]
        )
        print(
            "Found",
            f"{prompts_completed}/{prompts_total}",
            "prompts for dataset",
            cfg.hf_repo,
        )

        for datum_index in range(len(claims), len(dataset)):
            datum = dataset[datum_index]
            new_claims = {"claims": []}
            assert len(datum["conversation"]) == 2
            assert datum["conversation"][1]["role"] == "assistant"

            for i in range(0, len(datum["annotations"]), PROMPT_BATCH_SIZE):
                new_claims["claims"].extend(
                    get_claims(
                        datum["conversation"][1]["content"],
                        datum["annotations"][i : i + PROMPT_BATCH_SIZE],
                    )
                )
            claims.append(new_claims)

            prompts_completed += (
                len(new_claims["claims"]) - 1
            ) // PROMPT_BATCH_SIZE + 1
            print(
                "Completed",
                f"{prompts_completed}/{prompts_total}",
                "prompts for dataset",
                cfg.hf_repo,
            )

            with open(os.path.join(cfg.hf_repo, "_claims.jsonl"), "at") as fp:
                fp.write(json.dumps(new_claims))
                fp.write("\n")
