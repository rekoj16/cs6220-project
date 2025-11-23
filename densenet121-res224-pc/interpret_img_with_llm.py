import argparse
import base64
import json
import os
import sys
from pathlib import Path
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interpret_xray_heatmap(image_path, task_type):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    client = OpenAI()
    base64_image = encode_image(image_path)
    task_type = task_type.lower()
    if task_type and task_type not in ['easy', 'average', 'hard']:
        print(f"Error: Invalid task type '{task_type}'. Must be one of 'easy', 'average', or 'hard'.")
        sys.exit(1)


    prompt = """
        You are an expert radiology AI explainer.
        Your task is to interpret two model-generated Grad-CAM heatmaps from the same chest X-ray:
        (1) a baseline model’s heatmap and prediction
        (2) a fine-tuned model’s heatmap and prediction.

        You will be given one image, which contains three parts:
            – The original X-ray image with ground truth labels at the top
            – The baseline model’s heatmap overlaid on the original image with top 5 predicted labels and probabilities at the top
            – The fine-tuned model’s heatmap overlaid on the original image with top 5 predicted labels and probabilities at the top

        There are three task types you need to perform:
        1. easy: both models are performing well
        2. average: the finetune model performs well in some cases, and the baseline model performs well on cases where the fine–tuned model struggles
        3. hard: both models are struggling

        Note that if no task type is specified, you should infer the task type based on the heatmaps and model predictions compared to the ground truth.

        Base on the type task, please perform the following steps clearly and systematically:

        1. **Extract the ground truth labels and the top predicted labels with probabilities from both models.**

        2. **If the task is 'easy', describe both models’ behavior**
            – Where is their attention concentrated?
            – Are these regions anatomically relevant to the ground truth pathology?

        3. **If the task is 'average', describe both models’ behavior**
            – Where is their attention concentrated?
            – How does one model perform better than the other in terms of attention focus and relevance to the ground truth?
            – How does one model perform worse than the other in terms of attention focus and relevance to the ground truth?

        4. **If the task is 'hard', describe both models’ behavior**
            – Where is their attention concentrated?
            – Are these regions anatomically relevant to the ground truth pathology?
            – Why might both models be struggling to focus on the relevant areas?

        Please analyze the attached heatmap-overlaid X-ray image with task type {task_type} and provide a structured explanation.
        You should summarize your findings in at most 5 sentences.
        """.format(task_type=task_type)

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }},
                ]}
            ]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Interpret chest X-ray heatmap visualizations using OpenAI's vision model"
    )
    parser.add_argument(
        "image_path", 
        type=str, 
        help="Path to the heatmap visualization image (containing original, baseline, and finetuned panels)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="interpretations",
        help="Directory to save TXT interpretations (default: interpretations/)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        help="Task type for interpretation: 'easy', 'average', or 'hard'"
    )
    
    args = parser.parse_args()
    
    interpretation = interpret_xray_heatmap(args.image_path, args.task_type)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    image_path = Path(args.image_path)
    txt_filename = f"{image_path.stem}_interpretation.txt"
    txt_path = output_dir / txt_filename
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Image: {image_path.name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(interpretation)
        print(f"Interpretation saved to: {txt_path}")
    except Exception as e:
        print(f"Error saving TXT file: {e}")
        print(interpretation)
        sys.exit(1)
    print(interpretation)


if __name__ == "__main__":
    main()
