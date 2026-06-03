import argparse
import json
import re
from collections import Counter
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
def load_schema_bundle(db_id: str, schemas_dir: str) -> dict[str, Any]:
    p = Path(schemas_dir) / normalize_db_id_to_filename(db_id)
    raw = json.loads(p.read_text(encoding="utf-8"))
    tables = raw["table_names_original"]
    col_names = raw["column_names_original"]
    col_types = raw.get("column_types", [])
    primary_keys = set(raw.get("primary_keys", []))
    foreign_keys = raw.get("foreign_keys", [])

    fk_cols = set()
    for pair in foreign_keys:
        if len(pair) == 2:
            fk_cols.add(pair[0])
            fk_cols.add(pair[1])

    col_refs = []
    schema = {t: [] for t in tables}
    for col_idx, (tidx, cname) in enumerate(col_names):
        if tidx == -1:
            col_refs.append(None)
            continue
        col_refs.append((tables[tidx], cname))
        schema[tables[tidx]].append({
            "name": cname,
            "type": col_types[col_idx].upper() if col_idx < len(col_types) else "TEXT",
            "is_pk": col_idx in primary_keys,
            "is_fk": col_idx in fk_cols,
        })

    fk_pairs = []
    for left_idx, right_idx in foreign_keys:
        left = col_refs[left_idx] if 0 <= left_idx < len(col_refs) else None
        right = col_refs[right_idx] if 0 <= right_idx < len(col_refs) else None
        if left and right:
            fk_pairs.append((left[0], left[1], right[0], right[1]))

    return {"columns": schema, "fk_pairs": fk_pairs}


def schema_to_text(schema_bundle: dict[str, Any]) -> str:
    schema = schema_bundle["columns"]
    lines = []
    for t in sorted(schema):
        col_strs = []
        for c in sorted(schema[t], key=lambda x: x["name"]):
            ann = []
            if c["is_pk"]:
                ann.append("PK")
            if c["is_fk"]:
                ann.append("FK")
            ann_str = f" [{','.join(ann)}]" if ann else ""
            col_strs.append(f"{c['name']} ({c['type']}){ann_str}")
        lines.append(f"- {t}: [{', '.join(col_strs)}]")

    if schema_bundle["fk_pairs"]:
        lines.append("Foreign Keys:")
        for lt, lc, rt, rc in sorted(schema_bundle["fk_pairs"]):
            lines.append(f"  {lt}.{lc} -> {rt}.{rc}")

    return "\n".join(lines)


def filter_schema_for_question(
    question: str, schema_bundle: dict[str, Any], schema_top_k: int
) -> dict[str, Any]:
    schema = schema_bundle["columns"]
    q_tokens = tokenize(question)
    scored = []
    for table_name, cols in schema.items():
        score = 0
        table_tokens = tokenize(table_name.replace("-", "_"))
        score += len(table_tokens & q_tokens) * 3
        for c in cols:
            col_tokens = tokenize(c["name"])
            score += len(col_tokens & q_tokens)
        scored.append((score, table_name))

    scored.sort(reverse=True)
    selected = [t for s, t in scored if s > 0][:schema_top_k]
    if not selected:
        selected = [t for _s, t in scored[:schema_top_k]]

    selected_set = set(selected)
    filtered_cols = {t: schema[t] for t in selected}
    filtered_fks = [
        (lt, lc, rt, rc)
        for lt, lc, rt, rc in schema_bundle["fk_pairs"]
        if lt in selected_set and rt in selected_set
    ]

    return {"columns": filtered_cols, "fk_pairs": filtered_fks}


def build_prompt(question: str, schema_bundle: dict[str, Any]) -> str:
    return (
        "You are a schema linking model.\n"
        "Task: Given a question and DB schema, output ONLY a JSON object mapping table names to lists of referenced column names.\n"
        "Rules:\n"
        "1) Use only table/column identifiers present in schema.\n"
        "2) If a table is referenced but no specific columns are referenced, use an empty list.\n"
        "3) No extra keys or text. Output valid JSON only.\n\n"
        f"Schema:\n{schema_to_text(schema_bundle)}\n\n"
        f"Question:\n{question}\n\n"
        "JSON:"
    )


def build_text_dataset(
    items: list[dict[str, Any]],
    schemas_dir: str,
    schema_top_k: int,
    balance: bool = True,
) -> Dataset:
    db_counts = Counter(it["db_id"] for it in items)
    max_count = max(db_counts.values())

    rows = []
    for it in items:
        schema_bundle = load_schema_bundle(it["db_id"], schemas_dir)
        filtered = filter_schema_for_question(it["question"], schema_bundle, schema_top_k)
        prompt = build_prompt(it["question"], filtered)
        completion = json.dumps(it["schema_links"], ensure_ascii=False)
        rows.append({
            "text": f"{prompt}\n{completion}",
            "db_id": it["db_id"],
        })

    if balance:
        db_rows = {}
        for row in rows:
            db_rows.setdefault(row["db_id"], []).append(row)

        balanced_rows = []
        for db_id, db_row_list in db_rows.items():
            count = len(db_row_list)
            repeats = max(1, round(max_count / count))
            balanced_rows.extend((db_row_list * repeats)[:max_count])
        rows = balanced_rows

    return Dataset.from_list([{"text": r["text"]} for r in rows])


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json",   default="train_combined.json")
    ap.add_argument("--schemas_dir",  default="./schemas")
    ap.add_argument("--experiment_name", default="p2-rapidfire-multiconfig")
    ap.add_argument("--base_models",  default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--learning_rates", default="2e-4,3e-4")
    ap.add_argument("--top_ks",       default="16,24")
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--no_balance",   action="store_true")
    args = ap.parse_args()

    base_models    = [x.strip() for x in args.base_models.split(",") if x.strip()]
    learning_rates = parse_csv_floats(args.learning_rates)
    top_ks         = parse_csv_ints(args.top_ks)

    items = json.loads(Path(args.train_json).read_text(encoding="utf-8"))
    print(f"Loaded {len(items)} training examples from {args.train_json}")

    all_config_payloads = []
    for top_k in top_ks:
        for model_name in base_models:
            for lr in learning_rates:
                payload = {
                    "model_name": model_name,
                    "peft_config": {
                        "r": 32,
                        "lora_alpha": 64,
                        "lora_dropout": 0.05,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "bias": "none",
                    },
                    "training_args": {
                        "learning_rate": lr,
                        "lr_scheduler_type": "cosine",
                        "per_device_train_batch_size": args.per_device_train_batch_size,
                        "per_device_eval_batch_size": args.per_device_train_batch_size,
                        "num_train_epochs": args.num_train_epochs,
                        "gradient_accumulation_steps": args.gradient_accumulation_steps,
                        "logging_steps": 10,
                        "eval_strategy": "epoch",
                        "save_strategy": "epoch",
                        "bf16": False,
                        "fp16": False,
                        "no_cuda": True,
                        "report_to": "none",
                    },
                    "model_type": "causal_lm",
                    "model_kwargs": {
                        "device_map": "cpu",
                        "torch_dtype": "float32",
                        "use_cache": False,
                    },
                    "run_tag": f"{model_name.split('/')[-1]}__lr{lr}__topk{top_k}",
                    "schema_top_k": top_k,
                }
                all_config_payloads.append(payload)

    summary_out = Path(f"{args.experiment_name}_config_summary.json")
    summary_out.write_text(
        json.dumps(
            [{"run_tag": c["run_tag"], "model_name": c["model_name"],
              "learning_rate": c["training_args"]["learning_rate"],
              "top_k": c["schema_top_k"]} for c in all_config_payloads],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(all_config_payloads)} configs → {summary_out}")

    experiment = Experiment(experiment_name=args.experiment_name, mode="fit")

    for top_k in top_ks:
        ds = build_text_dataset(
            items, args.schemas_dir, schema_top_k=top_k,
            balance=not args.no_balance,
        ).shuffle(seed=args.seed)

        split         = ds.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset  = split["test"]

        config_payloads = [c for c in all_config_payloads if c["schema_top_k"] == top_k]
        print(f"Submitting {len(config_payloads)} configs for top_k={top_k} "
              f"({len(train_dataset)} train / {len(eval_dataset)} eval examples)...")

        experiment.run_fit(
            config_payloads, sample_create_model,
            train_dataset, eval_dataset,
            num_chunks=1, seed=args.seed,
        )

    print("RapidFire multi-config run complete.")


if __name__ == "__main__":
    main()