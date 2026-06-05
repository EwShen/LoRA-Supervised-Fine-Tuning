import subprocess
from pathlib import Path


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, command)


def run_experiment(
    name: str,
    train_json: str,
    base_model: str,
    learning_rate: str,
    train_top_k: str,
    infer_top_k: str = "25",
    num_train_epochs: str = "3.0",
    lora_r: str = "32",
    lora_alpha: str = "64",
    lora_dropout: str = "0.05",
    sbod_factor: str = "3",
    multitable_factor: str = "3",
) -> None:
    out_dir = f"./adapter_{name}"
    pred_path = f"preds_{name}.json"
    per_q_path = f"per_q_{name}.csv"
    log_dir = Path("logs") / "manual"

    train_cmd = [
        "python",
        "train_lora.py",
        "--train_json",
        train_json,
        "--output_dir",
        out_dir,
        "--base_model",
        base_model,
        "--num_train_epochs",
        num_train_epochs,
        "--learning_rate",
        learning_rate,
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "8",
        "--max_seq_length",
        "1536",
        "--schema_mode",
        "lexical",
        "--schema_top_k",
        train_top_k,
        "--schema_format",
        "pk_fk",
        "--lora_r",
        lora_r,
        "--lora_alpha",
        lora_alpha,
        "--lora_dropout",
        lora_dropout,
        "--sbod_factor",
        sbod_factor,
        "--multitable_factor",
        multitable_factor,
    ]

    # Current pushed main supports alias-aware training. Older local copies do not,
    # so this flag is intentionally optional at runtime.
    train_lora_text = Path("train_lora.py").read_text(encoding="utf-8")
    if "--schema_alias_boost" in train_lora_text:
        train_cmd.insert(train_cmd.index("--schema_format"), "--schema_alias_boost")

    infer_cmd = [
        "python",
        "main.py",
        "--input",
        "validation_input.json",
        "--output",
        pred_path,
        "--adapter_path",
        out_dir,
        "--schema_mode",
        "lexical",
        "--schema_top_k",
        infer_top_k,
        "--schema_format",
        "pk_fk",
        "--schema_alias_boost",
        "--base_model",
        base_model,
        "--max_new_tokens",
        "192",
        "--temperature",
        "0.0",
    ]

    eval_cmd = [
        "python",
        "eval.py",
        "--predictions",
        pred_path,
        "--gold",
        "validation_gold_schema_links.json",
        "--schemas_dir",
        "schemas",
        "--questions_input",
        "validation_input.json",
        "--per_question_out",
        per_q_path,
    ]

    run_command(train_cmd, log_dir / f"{name}_train.log")
    run_command(infer_cmd, log_dir / f"{name}_infer.log")
    run_command(eval_cmd, log_dir / f"{name}_eval.log")

