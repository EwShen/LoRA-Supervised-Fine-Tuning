"""
repair_predictions.py -- conservative column add-back for schema-link predictions.

This script does not run the model. It post-processes an existing predictions JSON:

  python repair_predictions.py \
    --predictions preds_exp1_aug_pkfk.json \
    --questions_input validation_input.json \
    --schemas_dir schemas \
    --output preds_exp1_aug_pkfk_repaired.json

Then evaluate the repaired file with eval.py.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


COLUMN_ALIAS_RULES: dict[str, dict[tuple[str, str], tuple[str, ...]]] = {
    "NTSB": {
        ("INJURY", "AIS"): ("severity", "injury severity", "critical", "ais"),
        ("INJURY", "REGION"): ("body region", "region"),
        ("CRASH", "YEAR"): ("year",),
        ("CRASH", "MONTH"): ("month",),
        ("CRASH", "DAY"): ("day",),
        ("VEHICLE", "VIN"): ("vin", "vehicle identification"),
    },
    "NYSED_SRC2022": {
        ("Postsecondary_Enrollment", "Enrollment"): ("enrollment", "enrolled"),
        ("Postsecondary_Enrollment", "Graduates"): ("graduates", "graduate"),
        ("Postsecondary_Enrollment", "Institution_Type"): ("institution type", "two-year", "four-year"),
    },
    "SBODemoUS-Business Partners": {
        ("OCRD", "CardCode"): ("business partner code", "card code", "customer code", "vendor code"),
        ("OCRD", "CardName"): ("business partner name", "card name", "customer name", "vendor name"),
    },
    "SBODemoUS-Finance": {
        ("OACT", "AcctCode"): ("account code", "account number"),
        ("OACT", "AcctName"): ("account name", "general ledger"),
    },
    "SBODemoUS-Inventory and Production": {
        ("OITM", "ItemCode"): ("item code", "item number"),
        ("OITM", "ItemName"): ("item name", "item called"),
        ("OITB", "ItmsGrpNam"): ("item group", "group name"),
    },
}


def normalize_db_id_to_filename(db_id: str) -> str:
    return db_id.replace(" ", "_").replace("/", "_") + ".json"


def tokenize(text: str) -> set[str]:
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", spaced)
    spaced = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", spaced)
    spaced = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", spaced)
    raw_tokens = re.findall(r"[A-Za-z0-9]+", spaced.replace("_", " ").replace("-", " "))
    tokens = set()
    for raw in raw_tokens:
        token = raw.lower()
        if token in STOPWORDS:
            continue
        tokens.add(token)
        if token.endswith("ies") and len(token) > 3:
            tokens.add(token[:-3] + "y")
        elif token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def load_schema(schemas_dir: str, db_id: str) -> dict[str, list[str]]:
    path = Path(schemas_dir) / normalize_db_id_to_filename(db_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    tables = raw["table_names_original"]
    schema = {table: [] for table in tables}
    for table_idx, col_name in raw["column_names_original"]:
        if table_idx == -1:
            continue
        schema[tables[table_idx]].append(col_name)
    return schema


def canonicalize_links(links: Any, schema: dict[str, list[str]]) -> dict[str, list[str]]:
    if not isinstance(links, dict):
        return {}

    table_lookup = {table.lower(): table for table in schema}
    col_lookup = {table: {col.lower(): col for col in cols} for table, cols in schema.items()}
    output: dict[str, list[str]] = {}

    for raw_table, raw_cols in links.items():
        table_key = str(raw_table).lower()
        if table_key not in table_lookup:
            continue
        table = table_lookup[table_key]
        selected = set()
        if isinstance(raw_cols, list):
            for raw_col in raw_cols:
                col_key = str(raw_col).lower()
                if col_key in col_lookup[table]:
                    selected.add(col_lookup[table][col_key])
        output[table] = sorted(selected)
    return output


def alias_matches(db_id: str, table: str, col: str, question_lc: str) -> bool:
    for phrase in COLUMN_ALIAS_RULES.get(db_id, {}).get((table, col), ()):
        if phrase in question_lc:
            return True
    return False


def repair_links(
    db_id: str,
    question: str,
    links: dict[str, list[str]],
    schema: dict[str, list[str]],
    max_added_per_table: int,
) -> dict[str, list[str]]:
    q_tokens = tokenize(question)
    question_lc = question.lower()
    repaired: dict[str, list[str]] = {}

    for table, cols in links.items():
        existing = set(cols)
        added = 0
        for col in schema.get(table, []):
            if col in existing:
                continue
            col_tokens = tokenize(col)
            exact_token_hit = bool(col_tokens) and col_tokens.issubset(q_tokens)
            alias_hit = alias_matches(db_id, table, col, question_lc)
            if exact_token_hit or alias_hit:
                existing.add(col)
                added += 1
                if added >= max_added_per_table:
                    break
        repaired[table] = sorted(existing)

    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions_input", required=True)
    parser.add_argument("--schemas_dir", default="schemas")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_added_per_table", type=int, default=2)
    args = parser.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    questions = {
        item["question_id"]: item
        for item in json.loads(Path(args.questions_input).read_text(encoding="utf-8"))
    }

    schema_cache: dict[str, dict[str, list[str]]] = {}
    repaired_predictions = []
    total_added = 0

    for pred in predictions:
        qid = pred["question_id"]
        question_item = questions[qid]
        db_id = question_item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema(args.schemas_dir, db_id)
        schema = schema_cache[db_id]

        before = canonicalize_links(pred.get("schema_links", {}), schema)
        after = repair_links(
            db_id=db_id,
            question=question_item["question"],
            links=before,
            schema=schema,
            max_added_per_table=args.max_added_per_table,
        )
        total_added += sum(len(after.get(t, [])) - len(before.get(t, [])) for t in after)
        repaired_predictions.append({"question_id": qid, "schema_links": after})

    Path(args.output).write_text(json.dumps(repaired_predictions, indent=2), encoding="utf-8")
    print(f"Wrote {len(repaired_predictions)} repaired predictions to {args.output}")
    print(f"Added {total_added} column(s) total.")


if __name__ == "__main__":
    main()

