import argparse
import inspect
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def normalize_db_id_to_filename(db_id: str) -> str:
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def load_schema_bundle(db_id: str, schemas_dir: str) -> dict[str, Any]:
    p = Path(schemas_dir) / normalize_db_id_to_filename(db_id)
    raw = json.loads(p.read_text(encoding="utf-8"))
    tables = raw["table_names_original"]
    column_names = raw["column_names_original"]
    column_types = raw.get("column_types", [])

    schema = {t: [] for t in tables}
    col_types_by_table = {t: {} for t in tables}
    col_refs = []
    for col_idx, (tidx, cname) in enumerate(column_names):
        if tidx == -1:
            col_refs.append(None)
            continue
        schema[tables[tidx]].append(cname)
        col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
        col_types_by_table[tables[tidx]][cname] = col_type
        col_refs.append((tables[tidx], cname))

    pks = {t: [] for t in tables}
    for col_idx in raw.get("primary_keys", []):
        ref = col_refs[col_idx] if 0 <= col_idx < len(col_refs) else None
        if ref is not None:
            t, c = ref
            pks[t].append(c)

    fks = []
    for left_idx, right_idx in raw.get("foreign_keys", []):
        left_ref = col_refs[left_idx] if 0 <= left_idx < len(col_refs) else None
        right_ref = col_refs[right_idx] if 0 <= right_idx < len(col_refs) else None
        if left_ref is not None and right_ref is not None:
            fks.append((left_ref[0], left_ref[1], right_ref[0], right_ref[1]))

    return {
        "columns": schema,
        "column_types": col_types_by_table,
        "primary_keys": pks,
        "foreign_keys": fks,
    }


def schema_to_text(
    schema_bundle: dict[str, Any],
    schema_format: str,
    include_column_types: bool,
    sort_schema_columns: bool,
) -> str:
    schema = schema_bundle["columns"]
    lines = []
    for t in sorted(schema):
        cols = sorted(schema[t]) if sort_schema_columns else schema[t]
        col_texts = []
        for col in cols:
            if include_column_types:
                col_type = schema_bundle.get("column_types", {}).get(t, {}).get(col, "text")
                col_texts.append(f"{col} ({str(col_type).upper()})")
            else:
                col_texts.append(col)
        if schema_format == "pk_fk":
            pk_cols = schema_bundle["primary_keys"].get(t, [])
            pk_txt = ", ".join(pk_cols) if pk_cols else "-"
            lines.append(f"- {t}: cols=[{', '.join(col_texts)}] PK=[{pk_txt}]")
        else:
            lines.append(f"- {t}: [{', '.join(col_texts)}]")

    if schema_format == "pk_fk":
        fk_lines = []
        for lt, lc, rt, rc in schema_bundle.get("foreign_keys", []):
            fk_lines.append(f"{lt}.{lc} -> {rt}.{rc}")
        if fk_lines:
            lines.append("FKs:")
            for fk in sorted(fk_lines):
                lines.append(f"- {fk}")
    return "\n".join(lines)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


SCHEMA_ALIAS_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "NTSB": {
        "CDC": ("crush", "collision speed", "barrier", "depth of crush", "fixed stationary obstacle"),
        "AVOID": ("avoidance", "crash avoidance"),
        "ADAPT": ("adaptive", "adaptive driving", "after-marked"),
        "GV": ("lighting condition", "different vehicles", "vin", "vehicle count"),
        "EDREVENT": ("event data recorder", "events recorded", "event data recording"),
        "EDRPRECRASH": ("event data recorder", "road speed limit", "recorded speed"),
        "ICS": ("energy source", "unknown energy"),
        "TIREPLAC": ("recommended tire", "recommended front tire", "manufacturer"),
        "TIRE": ("left front tire", "tire size", "mismatched front"),
    },
    "SBODemoUS-General": {
        "OPMG": ("project", "open issue"),
        "PMG2": ("open issue", "remarks", "solution"),
    },
    "SBODemoUS-Service": {
        "OSCL": ("service call", "assigned date", "response date", "resolved date", "originated", "business partner shipping", "billing address"),
        "OSCO": ("originated", "origin"),
    },
    "SBODemoUS-Banking": {
        "OCHO": ("checks for payment", "check", "account number"),
        "CHO1": ("checks for payment", "more than one line", "check line"),
    },
    "SBODemoUS-Sales Opportunities": {
        "OPR1": ("row-level sales opportunity", "weighted amount", "sequence number", "percentage rate"),
        "OOPR": ("sales opportunity", "card code"),
    },
    "SBODemoUS-Reports": {
        "RDOC": ("extension error", "repetetive areas", "repetitive areas", "conversion font", "email"),
        "OUQR": ("queries", "query category"),
    },
    "SBODemoUS-Business Partners": {
        "OCRD": ("business partner", "payment code", "closing date procedure"),
        "CRD1": ("tax code", "business partner"),
    },
    "SBODemoUS-Inventory and Production": {
        "OITM": ("commited items", "committed items", "item group", "item called"),
        "ITM1": ("price-per-item", "price per item"),
        "ITM4": ("volume", "average volume"),
        "OITB": ("item group",),
    },
    "SBODemoUS-Finance": {
        "OACT": ("general ledger", "account name", "travel expense"),
        "BGT1": ("budget debit", "budget credit", "monthly budget"),
    },
    "ATBI": {
        "tbl_Deadwood": ("decay", "stage of decay"),
        "tbl_Locations": ("quad name", "topographic position"),
        "tbl_Overstory": ("mature overstory",),
        "tbl_Seedlings": ("saplings",),
        "tbl_Nests": ("nested subplots", "not been observed within nested"),
    },
    "NYSED_SRC2022": {
        "BOCES_and_N/RC": ("board of cooperative educational services", "boces"),
        "Postsecondary_Enrollment": ("postsecondary", "graduates", "out of state", "two-year", "four-year"),
    },
    "CratersWildlifeObservations": {
        "VERTEBRATES": ("vertebrates", "dusky flycatcher"),
    },
}


def alias_score(db_id: str, table_name: str, question: str) -> int:
    rules = SCHEMA_ALIAS_RULES.get(db_id, {}).get(table_name, ())
    question_lc = question.lower()
    score = 0
    for phrase in rules:
        if phrase in question_lc:
            score += 8
    return score


def filter_schema_for_question(
    question: str,
    schema: dict[str, list[str]],
    schema_mode: str,
    schema_top_k: int,
    db_id: str = "",
    use_schema_aliases: bool = False,
) -> dict[str, list[str]]:
    if schema_mode == "full":
        return schema

    q_tokens = tokenize(question)
    scored = []
    for table_name, cols in schema.items():
        score = 0
        table_tokens = tokenize(table_name.replace("-", "_"))
        score += len(table_tokens & q_tokens) * 3
        for col in cols:
            col_tokens = tokenize(col)
            score += len(col_tokens & q_tokens)
        if use_schema_aliases:
            score += alias_score(db_id, table_name, question)
        scored.append((score, table_name))

    scored.sort(reverse=True)
    selected = [t for s, t in scored if s > 0][:schema_top_k]
    if not selected:
        selected = [t for _s, t in scored[:schema_top_k]]
    return {t: schema[t] for t in selected}


def apply_table_subset(schema_bundle: dict[str, Any], subset_schema: dict[str, list[str]]) -> dict[str, Any]:
    selected_tables = set(subset_schema.keys())
    col_types = {
        t: {
            c: schema_bundle.get("column_types", {}).get(t, {}).get(c, "text")
            for c in subset_schema[t]
        }
        for t in subset_schema
    }
    pks = {t: [c for c in schema_bundle["primary_keys"].get(t, []) if c in subset_schema[t]] for t in subset_schema}
    fks = []
    for lt, lc, rt, rc in schema_bundle.get("foreign_keys", []):
        if lt in selected_tables and rt in selected_tables:
            fks.append((lt, lc, rt, rc))
    return {"columns": subset_schema, "column_types": col_types, "primary_keys": pks, "foreign_keys": fks}


def build_prompt(
    question: str,
    schema_bundle: dict[str, Any],
    schema_format: str,
    include_column_types: bool,
    sort_schema_columns: bool,
) -> str:
    return (
        "You are a schema linking model.\n"
        "Task: Given a question and DB schema, output ONLY a JSON object mapping table names to lists of referenced column names.\n"
        "Rules:\n"
        "1) Use only table/column identifiers present in schema.\n"
        "2) If a table is referenced but no specific columns are referenced, use an empty list.\n"
        "3) No extra keys or text. Output valid JSON only.\n\n"
        f"Schema:\n{schema_to_text(schema_bundle, schema_format=schema_format, include_column_types=include_column_types, sort_schema_columns=sort_schema_columns)}\n\n"
        f"Question:\n{question}\n\n"
        "JSON:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", default="train.json")
    ap.add_argument("--schemas_dir", default="./schemas")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output_dir", default="./adapter")
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_seq_length", type=int, default=1536)
    ap.add_argument("--schema_mode", choices=["full", "lexical"], default="full")
    ap.add_argument("--schema_top_k", type=int, default=24)
    ap.add_argument("--schema_alias_boost", action="store_true")
    ap.add_argument("--schema_format", choices=["basic", "pk_fk"], default="pk_fk")
    ap.add_argument("--schema_include_types", action="store_true")
    ap.add_argument("--sort_schema_columns", action="store_true")
    ap.add_argument("--lr_scheduler_type", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--multitable_threshold", type=int, default=2)
    ap.add_argument("--multitable_factor", type=int, default=1)
    ap.add_argument(
        "--sbod_factor",
        type=int,
        default=1,
        help="Oversampling factor for SBODemoUS-* db_id examples.",
    )
    ap.add_argument(
        "--sap_balance_factor",
        type=int,
        default=1,
        help="Additional oversampling factor for rare SBODemoUS/SAP examples.",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = json.loads(Path(args.train_json).read_text(encoding="utf-8"))

    if args.sap_balance_factor > 1:
        db_counts = Counter(it.get("db_id", "") for it in items)
        balanced_items = []
        for it in items:
            db_id = it.get("db_id", "")
            is_rare_sap = db_id.startswith("SBODemoUS-") and db_counts[db_id] <= 24
            factor = args.sap_balance_factor if is_rare_sap else 1
            balanced_items.extend([it] * factor)
        items = balanced_items
        print(f"Applied rare SAP curriculum balancing: factor={args.sap_balance_factor}, size={len(items)}")

    if args.sbod_factor > 1:
        balanced_items = []
        for it in items:
            db_id = it.get("db_id", "")
            factor = args.sbod_factor if db_id.startswith("SBODemoUS-") else 1
            balanced_items.extend([it] * factor)
        items = balanced_items
        print(f"Applied SBOD oversampling: factor={args.sbod_factor}, size={len(items)}")

    if args.multitable_factor > 1:
        balanced_items = []
        for it in items:
            n_tables = len(it.get("schema_links", {}))
            factor = args.multitable_factor if n_tables >= args.multitable_threshold else 1
            balanced_items.extend([it] * factor)
        items = balanced_items
        print(
            f"Applied multitable oversampling: threshold>={args.multitable_threshold}, "
            f"factor={args.multitable_factor}, size={len(items)}"
        )

    examples = []
    for it in items:
        schema_bundle = load_schema_bundle(it["db_id"], args.schemas_dir)
        filtered_schema = filter_schema_for_question(
            question=it["question"],
            schema=schema_bundle["columns"],
            schema_mode=args.schema_mode,
            schema_top_k=args.schema_top_k,
            db_id=it["db_id"],
            use_schema_aliases=args.schema_alias_boost,
        )
        filtered_bundle = apply_table_subset(schema_bundle, filtered_schema)
        prompt = build_prompt(
            it["question"],
            filtered_bundle,
            schema_format=args.schema_format,
            include_column_types=args.schema_include_types,
            sort_schema_columns=args.sort_schema_columns,
        )
        target = json.dumps(it["schema_links"], ensure_ascii=False)
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = f"{prompt}\n{target}"
        examples.append({"text": text})

    ds = Dataset.from_list(examples)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    train_args = TrainingArguments(
        output_dir="./runs/lora_baseline",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        bf16=True,
        fp16=False,
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": ds,
        "peft_config": peft_config,
        "args": train_args,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "dataset_text_field": "text",
    }
    sig = inspect.signature(SFTTrainer.__init__)
    filtered_kwargs = {k: v for k, v in trainer_kwargs.items() if k in sig.parameters}
    trainer = SFTTrainer(**filtered_kwargs)

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
