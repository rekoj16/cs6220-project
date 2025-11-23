# Description
This folder contains code to 
1. finetune the `densenet121-res224-pc` model from `torchxrayvision` through LoRA
2. run the inference on `ChestXpert` validation set with either pre-trained or finetuned model
3. generate heatmap on each inferenced xray image
4. use OpenAI model to interterpret the model performance with the heatmap

## Steps to use
** Assuming you have installed all required dependencies and have access to a GPU
1. Change any config in `main()` of `lora_finetune.py` if you needed.
2. `python lora_finetune.py` - finetune the `densenet121-res224-pc` model with `LoRA`
3. `python inference.py --use-lora --visualize  --limit 220 --output-dir viz --batch_size 8 --lora-checkpoint '/home/hice1/ymai8/scratch/cs6220-project/densenet121-res224-pc/result(4)_batch_32_epoch_10_lr_1e-4_rank_16_alpha_32_dropout_0.1_all-conv-layers/best_lora_model.pth'` - run the inference on the validation set with both the `pretrained` and `finetuned` models and also generate heatmaps for each xray images
    - `--limit <x>`: generate x number of heatmaps
    - `--output-dir <dir name>`: generate a folder to hold all heatmaps
    - `--batch_size <x>`: run the inference with `x` batch size
    - `--lora-checkpoint <path>`: load your best LoRA finetuned model checkpoint
4. `python interpret_img_with_llm.py /home/hice1/ymai8/scratch/cs6220-project/densenet121-res224-pc/viz/sample_0062_comparison.png --task-type hard` - upload a heatmap image for OpenAI model to evaluate both baseline and finetuned models' performance in natural language
    - `--task-type <[easy, average, hard]>` - task type to anaylze; if not provided, it will infer from the matched prediction labels