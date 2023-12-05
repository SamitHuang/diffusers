MODEL_PATH=models/sdxl_base
DATA_PATH=datasets/chinese_art_blip/train
#DATA_PATH=datasets/pokemon-blip-captions
#DATA_PATH=datasets/pokemon_blip/train

python train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir=$DATA_PATH  \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --max_train_steps=12000 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-cnart-model-lora-sdxl-bs4" \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=4 \
  --max_grad_norm=0. \
  # --image_column="image" \
  # --caption_column="label" \
  # --enable_xformers_memory_efficient_attention=True \
