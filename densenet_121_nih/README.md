# Description
This folder contains code to 
1. Finetune the `densenet121-res224-nih` model from `torchxrayvision` through full-parameter finetuning
2. Run the inference on `CheXpert` validation set with either pre-trained or finetuned model
3. Generate heatmap on each inferenced x-ray image
4. Use OpenAI model to interpret the model performance with the heatmap

## Steps to use
** Assuming you have installed all required dependencies and have access to a GPU
1. Change any config in the beginning config section of `densenet121nih_full.py` if you need, like num epochs, learning rate, batch size, and output dir.
2. `python densenet121nih_full.py` - finetune the `densenet121-res224-nih` model with full-parameter finetuning
3. `python gradcam.py` - generate heatmaps of select data inferred by both baseline and saved finetuned model (change config to point to it)
4. `python densenet121nih_heatmap_interpreter.py` - generate AI interpretations of select heatmaps, edit input panel to specific png in gradcam folder from previous script, change class name to class you are investigating, edit output dir name, and change any other variables as you desire