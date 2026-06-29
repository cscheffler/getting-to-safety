# requirements: anthropic python-dotenv datasets

from dotenv import load_dotenv
import subprocess
import itertools
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

dataset_configs = [
    {
        "hf_repo": "./longfact-annotations/",  # "obalcells/longfact-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "train",
    },
    {
        "hf_repo": "./longfact-augmented-annotations",  # "obalcells/longfact-augmented-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "train",
    },
    {
        "hf_repo": "./longfact-annotations/",  # "obalcells/longfact-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "test",
    },
    {
        "hf_repo": "./longfact-augmented-annotations",  # "obalcells/longfact-augmented-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "test",
    },
    {
        "hf_repo": "./longfact-annotations/",  # "obalcells/longfact-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "validation",
    },
    {
        "hf_repo": "./longfact-augmented-annotations",  # "obalcells/longfact-augmented-annotations",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "validation",
    },
]

configs = []
for i, entry in enumerate(dataset_configs):
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


class ClaimApi:
    def query(self, prompt):
        raise NotImplemented


class ClaudeClaimApi(ClaimApi):
    def __init__(self, model="claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic()

    def query(self, prompt):
        message = self.client.messages.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return message.content[0].text


class ClaudeCliClaimApi(ClaimApi):
    def __init__(self, model="claude-sonnet-4-6"):
        self.model = model

    def query(self, prompt):
        # claude -p --model claude-sonnet-4-6 < prompt.txt 2>/dev/null
        p = subprocess.Popen(
            ["claude", "--allowedTools", "", "-p", "--model", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdout, stderr = p.communicate(prompt.encode("utf-8"))
        return stdout.decode("utf-8")


class CodexCliClaimApi(ClaimApi):
    def __init__(self, model="gpt-5.4"):
        self.model = model

    def query(self, prompt):
        # codex exec --model gpt-5.4 - < prompt.txt 2>/dev/null
        p = subprocess.Popen(
            ["codex", "exec", "--model", self.model, "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        stdout, stderr = p.communicate(prompt.encode("utf-8"))
        return stdout.decode("utf-8")


def get_claims(text, entities, api, max_attempts=3):
    entities_md = "\n".join(
        f"{i+1}. `{entity['span']}`" for i, entity in enumerate(entities)
    )
    prompt = PROMPT_TEMPLATE.format(text=text, entities=entities_md)

    class AttemptFailed(Exception):
        pass

    for attempt in range(max_attempts):
        try:
            response_raw = api.query(prompt)
            try:
                response = parse_json_response(response_raw)
            except json.decoder.JSONDecodeError:
                sys.stderr.write(
                    "JSON decode error\nInput: " + repr(response_raw) + "\n"
                )
                raise AttemptFailed

            if False:  # DEBUG
                print(len(response) == len(entities))
                _ = [response[i]["index"] == i + 1 for i in range(len(response))]
                print(all(_), _)
                _ = [
                    response[i]["entity"] == entities[i]["span"]
                    for i in range(len(response))
                ]
                print(all(_), _)
                import pdb

                pdb.set_trace()

            if len(response) != len(entities):
                print(prompt)
                print(entities)
                print(response)
                raise AttemptFailed
                sys.stderr.write(
                    f"Mismatch between response list length ({len(response)}) and entities list length ({len(entities)}).\n"
                )
            for i in range(len(response)):
                if response[i]["index"] != i + 1:
                    print(prompt)
                    print(entities)
                    print(response)
                    sys.stderr.write(
                        f"WARNING: Mismatch between index at index {i}, {response[i]['index']} ≠ {i+1}.\n"
                    )
                    # raise AttemptFailed  # Downgraded to a warning
            for i in range(len(response)):
                if response[i]["entity"] != entities[i]["span"]:
                    print(prompt)
                    print(entities)
                    print(response)
                    sys.stderr.write(
                        f"Mismatch between entity text at index {i}, {response[i]['entity']} ≠ {entities[i]['span']}.\n"
                    )
                    raise AttemptFailed
        except AttemptFailed:
            if attempt + 1 < max_attempts:
                continue
            else:
                raise AttemptFailed("Reached maximum number of attempts")
        else:
            return [
                {
                    "span": entry["entity"],
                    "claim": entry["output"],
                }
                for entry in response
            ]


def get_claims_for_datum(datum, api):
    datum = dataset[datum_index]
    assert len(datum["conversation"]) == 2
    assert datum["conversation"][1]["role"] == "assistant"

    new_claims = {"claims": []}
    for i in range(0, len(datum["annotations"]), PROMPT_BATCH_SIZE):
        new_claims["claims"].extend(
            get_claims(
                datum["conversation"][1]["content"],
                datum["annotations"][i : i + PROMPT_BATCH_SIZE],
                api,
            )
        )
    return new_claims


if __name__ == "__main__":
    available_apis = ["codex-cli", "claude-cli", "claude"]
    try:
        index = sys.argv.index("--api")
    except ValueError:
        api_name = available_apis[0]
    else:
        api_name = sys.argv[index + 1]
        assert api_name in available_apis

    print(f"Using API {api_name}.")

    # Set the Anthropic API key if and only if we're not using the CLI
    if api_name == "claude":
        load_dotenv()

    datasets = []
    dataset_claims = []
    rows_to_redo = {}
    for i, cfg in enumerate(configs):
        if cfg.subset:
            dataset = load_dataset(cfg.hf_repo, cfg.subset, split=cfg.split)
        else:
            dataset = load_dataset(cfg.hf_repo, split=cfg.split)
        datasets.append(dataset)

        claims_path = os.path.join(
            cfg.hf_repo, cfg.subset or "", f"{cfg.split}_claims.jsonl"
        )
        try:
            fp = open(claims_path, "rt")
        except FileNotFoundError:
            claims = []  # [{'claims': [{'span': '', 'claim': ''}]}]
        else:
            claims = [json.loads(line.strip()) for line in fp]
            fp.close()
        dataset_claims.append(claims)

        # Sanity checks
        rows_to_redo[cfg.hf_repo] = []
        for i, claim in enumerate(claims):
            if len(dataset[i]["annotations"]) != len(claim["claims"]):
                print(
                    f'ERROR: In {cfg.hf_repo} at index {i}, there are {len(dataset[i]["annotations"])} annotations and {len(claim["claims"])} claims'
                )
                rows_to_redo[cfg.hf_repo].append(i)

    api = {
        "claude": ClaudeClaimApi,
        "claude-cli": ClaudeCliClaimApi,
        "codex-cli": CodexCliClaimApi,
    }[api_name]()

    for dataset_index, dataset in enumerate(datasets):
        cfg = configs[dataset_index]
        claims = dataset_claims[dataset_index]

        prompts_total = sum(
            [
                (len(d.get("annotations", [])) - 1) // PROMPT_BATCH_SIZE + 1
                for d in dataset
            ]
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

        claims_path = os.path.join(
            cfg.hf_repo, cfg.subset or "", f"{cfg.split}_claims.jsonl"
        )

        if len(rows_to_redo[cfg.hf_repo]) > 0:
            for datum_index in rows_to_redo[cfg.hf_repo]:
                new_claims = get_claims_for_datum(dataset[datum_index], api)
                claims[datum_index] = new_claims
                print("Redid row", datum_index)

            with open(claims_path, "wt") as fp:
                for c in claims:
                    fp.write(json.dumps(c))
                    fp.write("\n")

        for datum_index in range(len(claims), len(dataset)):
            new_claims = get_claims_for_datum(dataset[datum_index], api)
            claims.append(new_claims)
            with open(claims_path, "at") as fp:
                fp.write(json.dumps(new_claims))
                fp.write("\n")

            prompts_completed += (
                len(new_claims["claims"]) - 1
            ) // PROMPT_BATCH_SIZE + 1
            print(
                "Completed",
                f"{prompts_completed}/{prompts_total}",
                "prompts for dataset",
                cfg.hf_repo,
            )
