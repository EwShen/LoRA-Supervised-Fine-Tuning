from run_utils import run_experiment


if __name__ == "__main__":
    run_experiment(
        name="exp5_combined_cosine_balanced",
        train_json="train_combined.json",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        learning_rate="2e-4",
        train_top_k="24",
        sbod_factor="1",
        multitable_factor="2",
        sap_balance_factor="2",
        lr_scheduler_type="cosine",
        schema_include_types=True,
        sort_schema_columns=True,
    )

