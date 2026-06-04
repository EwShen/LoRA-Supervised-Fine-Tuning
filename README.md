0.5219 score with adapter

Run the commands below for reproducibility purposes (Click on code above for proper commands):

python main.py \
  --input validation_input.json \
  --output preds_final_adapter_test.json \
  --adapter_path ./adapter \
  --schema_mode lexical \
  --schema_top_k 25 \
  --schema_format pk_fk \
  --schema_alias_boost \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --max_new_tokens 192 \
  --temperature 0.0

python eval.py \
  --predictions preds_final_adapter_test.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas \
  --questions_input validation_input.json \
  --per_question_out per_q_final_adapter_test.csv
