"""
main.py -- Baseline Project 2 inference script.

CLI:
    python3 main.py --input input.json --output output.json [--schemas_dir ./schemas]
"""
import argparse
import json
import re
from pathlib import Path


def normalize_db_id_to_filename(db_id: str) -> str:
    # Release packet maps spaces to underscores in schema filenames.
    safe = db_id.replace(" ", "_").replace("/", "_")
    return f"{safe}.json"


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def load_schema(db_id: str, schemas_dir: str) -> dict[str, list[str]]:
    schema_path = Path(schemas_dir) / normalize_db_id_to_filename(db_id)
    with schema_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    tables = raw["table_names_original"]
    links = {t: [] for t in tables}

    # Spider format: column_names_original entries are [table_idx, col_name].
    # Skip synthetic wildcard entry [-1, "*"].
    for table_idx, col_name in raw["column_names_original"]:
        if table_idx == -1:
            continue
        links[tables[table_idx]].append(col_name)

    return links


def predict_schema_links(question: str, db_id: str, schemas_dir: str) -> dict[str, list[str]]:
    """
    Lightweight lexical baseline:
    - links a table if table-name token appears in question
    - links a column if column-name token appears in question
    - if any column for a table matched, include that table
    """
    schema = load_schema(db_id, schemas_dir)
    q_tokens = tokenize(question)
    output: dict[str, list[str]] = {}

    for table_name, cols in schema.items():
        table_tokens = tokenize(table_name.replace("-", "_"))
        matched_cols = []

        for col in cols:
            col_tokens = tokenize(col)
            if col_tokens and col_tokens.issubset(q_tokens):
                matched_cols.append(col)

        table_mentioned = bool(table_tokens and table_tokens.issubset(q_tokens))
        if matched_cols or table_mentioned:
            output[table_name] = sorted(set(matched_cols))

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input questions JSON")
    parser.add_argument("--output", required=True, help="Path to output predictions JSON")
    parser.add_argument("--schemas_dir", default="./schemas", help="Path to schemas folder")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    preds = []
    for item in items:
        links = predict_schema_links(item["question"], item["db_id"], args.schemas_dir)
        preds.append({"question_id": item["question_id"], "schema_links": links})

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    print(f"Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()

