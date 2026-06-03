0.5030 score with adapter_sbod_mt3x_k24_e300_r32

Run the commands below for reproducibility purposes (Click on code above for proper commands):

python main.py \
  --input validation_input.json \
  --output preds_default_baked.json \
  --adapter_path ./adapter_sbod_mt3x_k24_e300_r32 \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --temperature 0.0

python eval.py \
  --predictions preds_default_baked.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas \
  --questions_input validation_input.json
