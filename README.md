# CSE 234 Project 2: Schema Linking SFT

## Overview

This repository contains our final schema-linking inference pipeline for Project 2. The system uses a LoRA-adapted Qwen2.5-1.5B-Instruct model to produce schema-link JSON objects from natural-language questions and database schemas.

## Final Model

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning method: LoRA
- Adapter location: `./adapter`
- Schema format: `pk_fk`
- Schema filtering: lexical top-k with alias boosting
- Inference schema top-k: `25`
- Max new tokens: `192`
- Temperature: `0.0`

The base model is loaded from Hugging Face at inference time. The LoRA adapter is included in this repo under `adapter/`.
