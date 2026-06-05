from run_utils import run_experiment


if __name__ == "__main__":
    run_experiment(
        name="manual_c2_qwen15_lr1e4_top24",
        train_json="train.json",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        learning_rate="1e-4",
        train_top_k="24",
    )

