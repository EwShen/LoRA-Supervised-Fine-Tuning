# Same config, lower LR, 4 epochs
python train_lora.py \
  --train_json train.json \
  --schemas_dir ./schemas \
  --output_dir ./adapter_sbod_mt3x_k24_e400_r32 \
  --schema_mode lexical \
  --schema_top_k 24 \
  --schema_format pk_fk \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --num_train_epochs 4.0 \
  --learning_rate 5e-5 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --multitable_threshold 2 \
  --multitable_factor 3 \
  --sbod_factor 3

  python main.py \
  --input validation_input.json \
  --output preds_sbod_mt3x_k32_e300_r32.json \
  --adapter_path ./adapter_sbod_mt3x_k24_e400_r32 \
  --schema_mode lexical \
  --schema_top_k 32 \
  --schema_format pk_fk \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --max_new_tokens 192 \
  --temperature 0.0

python eval.py \
  --predictions preds_sbod_mt3x_k32_e400_r32.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas \
  --questions_input validation_input.json
