"""
subset: string (e.g. "Meta-Llama-3.1-8B-Instruct")
split: string (one of "train", "validation", "test")

claim: string (e.g. "William James was born on January 11, 1842.")
affirm_probs: list of float (e.g. [0.9, 0.88, 0.75, 0.92]) This will be NULL until we run the claim through the target model

source_repo: string (e.g. "obalcells/triviaqa-balanced")
source_revision: sha256 (commit hash)
source_conversation_user_sha256: sha256 (hash of user turn in `conversation`)
source_annotations_index: int (e.g. 15)
source_annotations_span: string (e.g. "January 11, 1842")
source_label: string (one of "Supported", "Not Nupported", "Insufficient Information", "None")


_claims.jsonl: one line per row in the source data set {"claims": [{"span": string, "claim": string} for each entry in `annotations` column]}


For triviaqa-balanced, use `gt_completion` as the claim, and use `label` in ["S", "NS"] as the label. Annotation index and span can be null.
"""
from datasets import load_dataset, Dataset
import json
import hashlib
from types import SimpleNamespace
import os
import dotenv


dotenv.load_dotenv()


destination_hf_repo = "carlscheffler/bsclaims"
model_id = "Meta-Llama-3.1-8B-Instruct"
split = "train"


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
        "hf_repo_local": "./longfact-annotations/",
        "hf_repo_revision": "949fd2f71b3efd403350e38539340d9d9ceefc9d",
        "dataset_id": "llama3_1_8b_longfact_train",
        "hf_repo": "obalcells/longfact-annotations",
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
    {
        "hf_repo_local": "./triviaqa-balanced/",
        "hf_repo_revision": "620117a4967fd4a45f3536c46579830fc98e5e22",
        "dataset_id": "llama3_1_8b_trivia_qa_train",
        "hf_repo": "obalcells/triviaqa-balanced",
        "subset": "Meta-Llama-3.1-8B-Instruct",
        "split": "train",
        "max_length": 1536,
        "default_ignore": True,
        "last_span_token": False,
        "ignore_buffer": 0,
        "pos_weight": 10.0,
        "neg_weight": 10.0,
        "shuffle": True,
        "seed": 42,
        "process_on_the_fly": False,
    },
]

configs = []
for i, entry in enumerate(train_dataset_config):
    cfg = SimpleNamespace(**dict(config_defaults, **entry))
    configs.append(cfg)


def source_id(row):
    assert row["conversation"][0]["role"] == "user"
    key = row["conversation"][0]["content"]
    return hashlib.sha256(key.encode()).hexdigest()


if __name__ == "__main__":
    new_dataset = []
    for i, cfg in enumerate(configs):
        assert cfg.subset == model_id
        assert cfg.split == split

        # Load source data set and the extracted claims
        dataset = load_dataset(cfg.hf_repo_local, cfg.subset, split=cfg.split)

        if "triviaqa-balanced" in cfg.hf_repo:
            # For triviaqa-balanced, use `gt_completion` as the claim, and use
            # `label` in ["S", "NS"] as the label. Annotation index and span
            # can be null.
            for i in range(len(dataset)):
                source_row = dataset[i]
                assert source_row["label"] in ["S", "NS"]
                label = {"S": "Supported", "NS": "Not Supported"}[source_row["label"]]
                new_row = {
                    "claim": source_row["gt_completion"],
                    "affirm_probs": None,
                    "source_repo": cfg.hf_repo,
                    "source_revision": cfg.hf_repo_revision,
                    "source_row_hash": source_id(source_row),
                    "source_annotations_index": None,
                    "source_annotations_span": None,
                    "source_label": label,
                }
                new_dataset.append(new_row)

        else:
            claims_path = os.path.join(
                cfg.hf_repo_local, cfg.subset or "", f"{cfg.split}_claims.jsonl"
            )
            with open(claims_path, "rt") as fp:
                claims = [json.loads(line.strip()) for line in fp]

            # Construct rows for new data set
            assert len(claims) == len(dataset)
            for i in range(len(claims)):
                source_row = dataset[i]
                source_annotations = source_row["annotations"]
                claims_row = claims[i]["claims"]

                assert len(source_annotations) == len(claims_row)
                for j in range(len(claims_row)):
                    annotation = source_annotations[j]
                    claim = claims_row[j]

                    # We skip a few bad rows in the source data where the `span` text does not appear in the conversation text
                    if annotation["label"] is None and annotation["index"] is None:
                        continue

                    assert annotation["span"] == claim["span"]
                    assert annotation["label"] in [
                        "Supported",
                        "Not Supported",
                        "Insufficient Information",
                        None,  # There are a small number of source rows with no label even though the entity exists in the text
                    ]

                    if annotation["label"] is None:
                        print(source_row, annotation, claim)

                    new_row = {
                        "claim": claim["claim"],
                        "affirm_probs": None,
                        "source_repo": cfg.hf_repo,
                        "source_revision": cfg.hf_repo_revision,
                        "source_row_hash": source_id(source_row),
                        "source_annotations_index": annotation["index"],
                        "source_annotations_span": annotation["span"],
                        "source_label": annotation["label"],
                    }
                    new_dataset.append(new_row)

    print(len(new_dataset))
    new_dataset = Dataset.from_list(new_dataset)
    # print("Not pushing live data set")
    # """
    new_dataset.push_to_hub(
        destination_hf_repo, config_name=model_id, split=split, private=True
    )
    # """
