# Description
This folder contains code to 
1. finetune the `clip-vit-base-patch32` model from `OpenAi` through LoRA
2. run the inference on `ChestXpert` validation set with either pre-trained or finetuned model
3. generate heatmap on each inferenced xray image
4. use the OpenAI model to interterpret the model performance with the heatmap

## Steps to use
** Assuming you have installed all required dependencies and have access to a GPU
1. Change any config in `main()` of `finetune.py` if you needed.
2. `python finetune.py` - finetune the `clip-vit-base-patch32` model with `LoRA`
    - assumes dataset is in same folder with name `CheXpert-v1.0-small/`
4. `python heatmap.py` - generate heatmaps based off of validation set
    - assumes set is in same folder with name `CheXpert-v1.0-small/`
6. `python explain.py <target img path>` - generate interpretability for target img

Baseline can be run with `python baseline.py`
- assumes dataset is in same folder with name `CheXpert-v1.0-small/`
