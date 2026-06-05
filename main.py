"""
main.py -- Project 2 inference script (SLM + optional LoRA adapter).

Required CLI:
    python3 main.py --input input.json --output output.json
"""
import argparse
import ast
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


# Convert a db_id into the released schema filename convention
def normalize_db_id_to_filename(db_id: str) -> str:
    safe = db_id.replace(" ", "_").replace("/", "_")
    return f"{safe}.json"


# Tokenize natural language and schema identifiers for lexical matching
def tokenize(text: str) -> set[str]:
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", spaced)
    spaced = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", spaced)
    spaced = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", spaced)
    tokens = re.findall(r"[A-Za-z0-9]+", spaced.replace("_", " ").replace("-", " "))

    expanded = set()
    for token in tokens:
        t = token.lower()
        expanded.add(t)
        if t.endswith("ies") and len(t) > 3:
            expanded.add(t[:-3] + "y")
        elif t.endswith("s") and len(t) > 3:
            expanded.add(t[:-1])
    return expanded


# Load schema columns, primary keys, and foreign keys for one database
def load_schema_bundle(db_id: str, schemas_dir: str) -> dict[str, Any]:
    schema_path = Path(schemas_dir) / normalize_db_id_to_filename(db_id)
    with schema_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    tables = raw["table_names_original"]
    column_names = raw["column_names_original"]
    links = {t: [] for t in tables}
    col_refs = []
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            col_refs.append(None)
            continue
        links[tables[table_idx]].append(col_name)
        col_refs.append((tables[table_idx], col_name))

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

    return {"db_id": db_id, "columns": links, "primary_keys": pks, "foreign_keys": fks}


# Fallback used if model loading or JSON parsing fails
def lexical_fallback(question: str, schema: dict[str, list[str]]) -> dict[str, list[str]]:
    q_tokens = tokenize(question)
    output: dict[str, list[str]] = {}
    for table_name, cols in schema.items():
        table_tokens = tokenize(table_name.replace("-", "_"))
        matched_cols = []
        for col in cols:
            col_tokens = tokenize(col)
            if col_tokens and col_tokens.issubset(q_tokens):
                matched_cols.append(col)
        if matched_cols or (table_tokens and table_tokens.issubset(q_tokens)):
            output[table_name] = sorted(set(matched_cols))
    return output


# Serialize the selected schema into the prompt
def schema_to_text(schema_bundle: dict[str, Any], schema_format: str) -> str:
    schema = schema_bundle["columns"]
    lines = []
    for table_name in sorted(schema):
        cols = ", ".join(schema[table_name])
        if schema_format == "pk_fk":
            pks = schema_bundle["primary_keys"].get(table_name, [])
            pk_txt = ", ".join(pks) if pks else "-"
            lines.append(f"- {table_name}: cols=[{cols}] PK=[{pk_txt}]")
        else:
            lines.append(f"- {table_name}: [{cols}]")

    if schema_format == "pk_fk":
        fk_lines = []
        for lt, lc, rt, rc in schema_bundle.get("foreign_keys", []):
            fk_lines.append(f"{lt}.{lc} -> {rt}.{rc}")
        if fk_lines:
            lines.append("FKs:")
            for fk in sorted(fk_lines):
                lines.append(f"- {fk}")
    return "\n".join(lines)


# Alias boosts help lexical retrieval find abbreviated domain tables
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


# Score one table using hand-built natural-language aliases
def alias_score(db_id: str, table_name: str, question: str) -> int:
    rules = SCHEMA_ALIAS_RULES.get(db_id, {}).get(table_name, ())
    question_lc = question.lower()
    score = 0
    for phrase in rules:
        if phrase in question_lc:
            score += 8
    return score


# Select tables for the prompt using lexical overlap plus optional aliases
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
        # Table-name matches are weighted higher than individual column matches
        score = 0
        table_tokens = tokenize(table_name.replace("-", "_"))
        score += len(table_tokens & q_tokens) * 3
        for col in cols:
            col_tokens = tokenize(col)
            score += len(col_tokens & q_tokens)
        if use_schema_aliases:
            score += alias_score(db_id, table_name, question)
        scored.append((score, table_name))

    # Keep only positively matched tables unless nothing matched at all
    scored.sort(reverse=True)
    selected = [t for s, t in scored if s > 0][:schema_top_k]
    if not selected:
        selected = [t for _s, t in scored[:schema_top_k]]
    return {t: schema[t] for t in selected}


# Keep PK/FK metadata aligned with the filtered table subset
def apply_table_subset(schema_bundle: dict[str, Any], subset_schema: dict[str, list[str]]) -> dict[str, Any]:
    selected_tables = set(subset_schema.keys())
    pks = {t: [c for c in schema_bundle["primary_keys"].get(t, []) if c in subset_schema[t]] for t in subset_schema}
    fks = []
    for lt, lc, rt, rc in schema_bundle.get("foreign_keys", []):
        if lt in selected_tables and rt in selected_tables:
            fks.append((lt, lc, rt, rc))
    return {"columns": subset_schema, "primary_keys": pks, "foreign_keys": fks}


# Prompt contract: JSON object only, using identifiers from the schema
def build_prompt(question: str, schema_bundle: dict[str, Any], schema_format: str) -> str:
    schema_text = schema_to_text(schema_bundle, schema_format=schema_format)
    return (
        "You are a schema linking model.\n"
        "Task: Given a question and DB schema, output ONLY a JSON object mapping table names "
        "to lists of referenced column names.\n"
        "Rules:\n"
        "1) Use only table/column identifiers present in schema.\n"
        "2) If a table is referenced but no specific columns are referenced, use an empty list.\n"
        "3) No extra keys or text. Output valid JSON only.\n\n"
        "4) Prefer the most specific table whose columns match the question; do not output related lookup or metadata tables unless their columns are directly needed.\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Question:\n{question}\n\n"
        "JSON:"
    )


# Remove Markdown fences sometimes produced around JSON
def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# Remove trailing commas before closing braces/brackets
def trim_trailing_commas(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


# Repair simple truncation by closing open strings/brackets/braces
def close_truncated_json(text: str) -> str:
    stack = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    repaired = text.rstrip()
    while repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    if in_string:
        repaired += '"'
    repaired += "".join(reversed(stack))
    return trim_trailing_commas(repaired)


# Try strict JSON first, then repaired JSON, then Python-style literals
def parse_json_object_candidate(candidate: str) -> dict[str, Any]:
    candidate = strip_code_fences(candidate)
    attempts = [
        candidate,
        trim_trailing_commas(candidate),
        close_truncated_json(candidate),
    ]

    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    for attempt in attempts:
        try:
            parsed = ast.literal_eval(attempt)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed

    return {}


# Extract the first JSON-like object from generated text
def extract_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        return {}

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        # Track quoted strings so braces inside strings do not affect depth
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                parsed = parse_json_object_candidate(candidate)
                if parsed:
                    return parsed
                break

    return parse_json_object_candidate(text[start:])


# Validate tables/columns against the active schema and restore original casing
def canonicalize_prediction(raw_pred: Any, schema: dict[str, list[str]]) -> dict[str, list[str]]:
    if not isinstance(raw_pred, dict):
        return {}

    table_lookup = {t.lower(): t for t in schema}
    col_lookup = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}
    out: dict[str, list[str]] = {}

    for raw_table, raw_cols in raw_pred.items():
        table_key = str(raw_table).lower()
        if table_key not in table_lookup:
            # Drop hallucinated tables before evaluation sees them
            continue
        table_name = table_lookup[table_key]
        valid_cols = []
        if isinstance(raw_cols, list):
            for col in raw_cols:
                ckey = str(col).lower()
                if ckey in col_lookup[table_name]:
                    valid_cols.append(col_lookup[table_name][ckey])
                # Invalid columns are ignored to preserve schema validity
        out[table_name] = sorted(set(valid_cols))

    return out


# Limit very long outputs to reduce false-positive tables
def cap_output_tables(links: dict[str, list[str]], max_tables: int) -> dict[str, list[str]]:
    if max_tables <= 0:
        return links

    capped: dict[str, list[str]] = {}
    for i, (table_name, cols) in enumerate(links.items()):
        if i >= max_tables:
            break
        capped[table_name] = cols
    return capped


# Encapsulates model loading, prompting, generation, and output validation
class SchemaLinkingModel:
    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str],
        checkpoint_path: Optional[str],
        device: str,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        schema_mode: str,
        schema_top_k: int,
        schema_format: str,
        use_schema_aliases: bool,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.schema_mode = schema_mode
        self.schema_top_k = schema_top_k
        self.schema_format = schema_format
        self.use_schema_aliases = use_schema_aliases
        self.enabled = False

        try:
            # Import lazily so the script can still fall back if dependencies are missing
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            print(f"[WARN] transformers/torch not available ({exc}); using lexical fallback.")
            return

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        if checkpoint_path:
            # Full checkpoint path, if supplied
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch_dtype,
                device_map=device,
            )
        else:
            # Default path: load public base model and attach local LoRA adapter
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                device_map=device,
            )
            if adapter_path:
                try:
                    from peft import PeftModel
                except Exception as exc:
                    print(f"[WARN] peft not available ({exc}); using base model without adapter.")
                else:
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self.enabled = True

    def predict(self, question: str, schema_bundle: dict[str, Any]) -> dict[str, list[str]]:
        # Build the filtered prompt schema for this question
        prompt_schema = filter_schema_for_question(
            question=question,
            schema=schema_bundle["columns"],
            schema_mode=self.schema_mode,
            schema_top_k=self.schema_top_k,
            db_id=schema_bundle.get("db_id", ""),
            use_schema_aliases=self.use_schema_aliases,
        )
        prompt_bundle = apply_table_subset(schema_bundle, prompt_schema)
        if not self.enabled:
            # Dependency/model-load failure path
            return lexical_fallback(question, prompt_schema)

        prompt = build_prompt(question, prompt_bundle, schema_format=self.schema_format)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Qwen uses a chat template, so format the prompt as a user message
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            # Fallback for tokenizers without chat-template support
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with self._torch.no_grad():
            # Deterministic generation when temperature is zero
            if isinstance(inputs, Mapping):
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            else:
                output_ids = self.model.generate(input_ids=inputs, **gen_kwargs)

        if isinstance(inputs, Mapping):
            input_len = inputs["input_ids"].shape[1]
        else:
            input_len = inputs.shape[1]
        new_tokens = output_ids[0, input_len:]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Parse, repair, and validate generated JSON before returning
        raw_json = extract_first_json_object(raw_text)
        parsed = canonicalize_prediction(raw_json, schema_bundle["columns"])
        if not parsed:
            # If generation is unusable, still return a valid schema-link object
            return lexical_fallback(question, prompt_schema)
        return parsed


# CLI entry point used by the grader and local validation runs
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input questions JSON")
    parser.add_argument("--output", required=True, help="Path to output predictions JSON")
    parser.add_argument("--schemas_dir", default="./schemas", help="Path to schemas folder")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default="./adapter")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--device", default="auto", help='Transformers device_map value (e.g. "auto", "cuda:0")')
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--schema_mode", choices=["full", "lexical"], default="lexical")
    parser.add_argument("--schema_top_k", type=int, default=25)
    parser.add_argument("--schema_alias_boost", action="store_true", default=True)
    parser.add_argument("--no_schema_alias_boost", action="store_false", dest="schema_alias_boost")
    parser.add_argument("--schema_format", choices=["basic", "pk_fk"], default="pk_fk")
    parser.add_argument("--max_output_tables", type=int, default=5)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    with in_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    adapter_path = args.adapter_path if Path(args.adapter_path).exists() else None
    checkpoint_path = args.checkpoint_path if args.checkpoint_path and Path(args.checkpoint_path).exists() else None
    if args.adapter_path and adapter_path is None and checkpoint_path is None:
        print(f"[WARN] Adapter path not found: {args.adapter_path}; inference will use base model or lexical fallback.")

    predictor = SchemaLinkingModel(
        base_model=args.base_model,
        adapter_path=adapter_path,
        checkpoint_path=checkpoint_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        schema_mode=args.schema_mode,
        schema_top_k=args.schema_top_k,
        schema_format=args.schema_format,
        use_schema_aliases=args.schema_alias_boost,
    )

    schema_cache: dict[str, dict[str, Any]] = {}
    preds = []
    for item in items:
        # Cache schemas because many questions share the same db_id
        db_id = item["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_schema_bundle(db_id, args.schemas_dir)
        links = predictor.predict(item["question"], schema_cache[db_id])
        links = cap_output_tables(links, args.max_output_tables)
        preds.append({"question_id": item["question_id"], "schema_links": links})

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()

