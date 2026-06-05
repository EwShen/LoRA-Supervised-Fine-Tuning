from run_utils import run_experiment


if __name__ == "__main__":
    run_experiment(
        name="exp1_aug_pkfk",
        train_json="train_augmented.json",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        learning_rate="2e-4",
        train_top_k="24",
        sbod_factor="1",
        multitable_factor="1",
    )

