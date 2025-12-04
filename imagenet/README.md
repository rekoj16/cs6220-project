Description:
The folder contains:
1. Fine-tuned model of Efficient-B0 (pretrained on ImageNet) using LoRA on the CheXpert dataset.
2. Run inference on the CheXpert validation set to compare with the baseline vs fine-tuned
3. Grad-Cam heatmaps generations
4. Useing OpenAi GPT-4o-mini model to explain the model's performance and provide a diagnostic.

Steps to run:
You need to install all the reqiured dependencies (torch, torchvision, peft, grad-cam, openai, pandas, etc.)
1. chexpert_loader.py: cleaning the raw CheXpert labels
2. python model.py - runs the zero-shot baseline evaluation to see the performance before finetuning.
3. python lora.py - perfoms the fine-tuning for the model use LoRA, and saves the resutls to the adpaters folder "imagenet/results/lora5"
4. python run_evaluation.py - evaluate the fine-tuned model and generates the auroc score
5. python generate_heatmap.py - generate the Grad-CAM heatmaps for the validation set
6. python prompt.py - generates an AI interpretation of the heatmaps