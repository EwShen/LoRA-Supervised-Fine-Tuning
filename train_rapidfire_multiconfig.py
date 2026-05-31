import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import Dataset
from rapidfireai import Experiment


def normalize_db_id_to_filename(db_id: str) -> str:
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


@lru_cache(maxsize=64)
def load_schema_cached(db_id: str, schemas_dir: str) -> dict[str, list[str]]:
    p = Path(schemas_dir) / normalize_db_id_to_filename(db_id)
    raw = json.loads(p.read_text(encoding="utf-8"))
    tables = raw["table_names_original"]
    schema = {t: [] for t in tables}
    for tidx, cname in raw["column_names_original"]:
        if tidx == -1:
            continue
        schema[tables[tidx]].append(cname)
    return schema


def schema_to_text(schema: dict[str, list[str]]) -> str:
    lines = []
    for t in sorted(schema):
        lines.append(f"- {t}: [{', '.join(schema[t])}]")
    return "\n".join(lines)


def filter_schema_for_question(question: str, schema: dict[str, list[str]], schema_top_k: int) -> dict[str, list[str]]:
    q_tokens = tokenize(question)
    scored = []
    for table_name, cols in schema.items():
        score = 0
        table_tokens = tokenize(table_name.replace("-", "_"))
        score += len(table_tokens & q_tokens) * 3
        for col in cols:
            col_tokens = tokenize(col)
            score += len(col_tokens & q_tokens)
        scored.append((score, table_name))

    scored.sort(reverse=True)
    selected = [t for s, t in scored if s > 0][:schema_top_k]
    if not selected:
        selected = [t for _s, t in scored[:schema_top_k]]
    return {t: schema[t] for t in selected}


def build_prompt(question: str, schema: dict[str, list[str]]) -> str:
    return (
        "You are a schema linking model.\n"
        "Task: Given a question and DB schema, output ONLY a JSON object mapping table names to lists of referenced column names.\n"
        "Rules:\n"
        "1) Use only table/column identifiers present in schema.\n"
        "2) If a table is referenced but no specific columns are referenced, use an empty list.\n"
        "3) No extra keys or text. Output valid JSON only.\n\n"
        f"Schema:\n{schema_to_text(schema)}\n\n"
        f"Question:\n{question}\n\n"
        "JSON:"
    )


def sample_create_model(model_config: dict[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model_name"]
    model_kwargs = model_config.get("model_kwargs", {})
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_text_dataset(items: list[dict[str, Any]], schemas_dir: str, schema_top_k: int) -> Dataset:
    rows = []
    for it in items:
        schema = load_schema_cached(it["db_id"], schemas_dir)
        filtered = filter_schema_for_question(it["question"], schema, schema_top_k=schema_top_k)
        prompt = build_prompt(it["question"], filtered)
        completion = json.dumps(it["schema_links"], ensure_ascii=False)
        rows.append({"text": f"{prompt}\n{completion}"})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="train.json")
    ap.add_argument("--schemas_dir", default="./schemas")
    ap.add_argument("--experiment_name", default="p2-rapidfire-multiconfig")
    ap.add_argument("--base_models", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--learning_rates", default="1e-4,2e-4,3e-4")
    ap.add_argument("--top_ks", default="16,24")
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_models = [x.strip() for x in args.base_models.split(",") if x.strip()]
    learning_rates = parse_csv_floats(args.learning_rates)
    top_ks = parse_csv_ints(args.top_ks)

    items = json.loads(Path(args.train_json).read_text(encoding="utf-8"))
    all_config_payloads = []
    for top_k in top_ks:
        for model_name in base_models:
            for lr in learning_rates:
                payload = {
                    "model_name": model_name,
                    "peft_config": {
                        "r": 16,
                        "lora_alpha": 32,
                        "lora_dropout": 0.05,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "bias": "none",
                    },
                    "training_args": {
                        "learning_rate": lr,
                        "lr_scheduler_type": "linear",
                        "per_device_train_batch_size": args.per_device_train_batch_size,
                        "per_device_eval_batch_size": args.per_device_train_batch_size,
                        "num_train_epochs": args.num_train_epochs,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "logging_steps": 10,
                        "eval_strategy": "epoch",
                        "save_strategy": "epoch",
                        "bf16": True,
                        "fp16": False,
                        "report_to": "none",
                    },
                    "model_type": "causal_lm",
                    "model_kwargs": {"device_map": "auto", "torch_dtype": "bfloat16", "use_cache": False},
                    "run_tag": f"{model_name.split('/')[-1]}__lr{lr}__topk{top_k}",
                    "schema_top_k": top_k,
                }
                all_config_payloads.append(payload)

    summary_out = Path(f"{args.experiment_name}_config_summary.json")
    summary_out.write_text(
        json.dumps(
            [
                {
                    "run_tag": c["run_tag"],
                    "model_name": c["model_name"],
                    "learning_rate": c["training_args"]["learning_rate"],
                    "top_k": c["schema_top_k"],
                }
                for c in all_config_payloads
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(all_config_payloads)} configs. Wrote summary to {summary_out}.")

    experiment = Experiment(experiment_name=args.experiment_name, mode="fit")
    for top_k in top_ks:
        ds = build_text_dataset(items, args.schemas_dir, schema_top_k=top_k).shuffle(seed=args.seed)
        split = ds.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        config_payloads = [c for c in all_config_payloads if c["schema_top_k"] == top_k]
        print(f"Submitting {len(config_payloads)} configs for top_k={top_k}...")
        experiment.run_fit(config_payloads, sample_create_model, train_dataset, eval_dataset, num_chunks=1, seed=args.seed)
    print("RapidFire multi-config run submitted.")


if __name__ == "__main__":
    main()
