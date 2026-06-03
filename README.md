0.48 score with adapter_sbod_mt3x_k24_e300_r32

Run the commands below for reproducibility purposes:

python main.py \
  --input validation_input.json \
  --output preds_bettertok_bestadapter_k32.json \
  --adapter_path ./adapter_sbod_mt3x_k24_e300_r32 \
  --schema_mode lexical \
  --schema_top_k 32 \
  --schema_format pk_fk \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --max_new_tokens 192 \
  --temperature 0.0

python eval.py \
  --predictions preds_bettertok_bestadapter_k32.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas \
  --questions_input validation_input.json
