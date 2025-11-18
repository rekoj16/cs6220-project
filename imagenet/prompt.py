import os
import base64
from openai import OpenAI
import glob

API_KEY = "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" 
MODEL_ID = "gpt-4o-mini" 
# easy: "/home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/heatmaps_refined_batch1/patient64740/study1/view1_frontal/Enlarged Cardiomediastinum_comparison.png"
# avg: "/home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/heatmaps_refined_batch1/patient64720/study1/view1_frontal/Pleural Effusion_comparison.png"
# hard: /home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/heatmaps_refined_batch1/patient64726/study1/view1_frontal/Enlarged Cardiomediastinum_comparison.png
TEST_IMAGE_PATH = "/home/hice1/achen448/scratch/CS6220/cs6220-project/imagenet/heatmaps_refined_batch1/patient64720/study1/view1_frontal/Pleural Effusion_comparison.png"

def encode_image(image_path):
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# send the image and prompt to GPT-4o-mini for analysis
def generate_analysis(client, image_path):
    base64_image = encode_image(image_path)
    filename = os.path.basename(image_path)
    pathology_name = filename.split("_comparison")[0]
    prompt = f"""
    You are an expert Radiologist and AI Researcher evaluating a deep learning experiment.
    You are comparing two models trained to detect pathologies in Chest X-rays:
    1.Baseline Model: Pre-trained on ImageNet (generic images).
    2.Finetuned Model: The same architecture finetuned on Chest X-rays using LoRA.
    Please analyze the attached comparison image for the pathology: **{pathology_name}**.

    Image Layout:
    - Left: Original X-ray with Ground Truth labels.
    - Middle: Baseline Model Heatmap (Grad-CAM) + Top 3 Predictions.
    - Right: Finetuned Model Heatmap (Grad-CAM) + Top 3 Predictions.

    Please provide a structured report in the following format:

    1. Image Quality & Ground Truth:
    - Is the X-ray image clear enough for diagnosis?
    - Does the Ground Truth label confirm the presence of {pathology_name}?

    2. Baseline Model Analysis:
    - Heatmap Focus: Where is the model looking? Is it focused on relevant anatomy (lungs/heart) or background artifacts (text, bones, empty space)?
    - Prediction: Does {pathology_name} appear in its Top 3 predictions?

    3. Finetuned Model Analysis:
    - Heatmap Focus: Has the focus shifted? Does the heatmap now align with the *radiological indicators* of {pathology_name}? (e.g., lung opacities for Pneumonia, pleural edge for Pneumothorax).
    - Prediction: Is {pathology_name} in the Top 3? Did the confidence score increase compared to the Baseline?

    4. Final Assessment:
    - Verdict: [IMPROVED / WORSE / NO CHANGE]
    - Summary: Briefly explain why based on the visual evidence. Did the model learn the correct features, or is it hallucinating?
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" 
                            },
                        },
                    ],
                }
            ],
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling API: {e}"
if __name__ == "__main__":
    
    client = OpenAI(api_key=API_KEY)
    
    print(f"--- Analyzing image: {os.path.basename(TEST_IMAGE_PATH)} ---")
    print(f"--- Using Model: {MODEL_ID} ---")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: File not found at {TEST_IMAGE_PATH}")
        print("Please update TEST_IMAGE_PATH variable.")
        exit()

    analysis = generate_analysis(client, TEST_IMAGE_PATH)
    
    print("\n--- Diagnosis & Reasoning ---\n")
    print(analysis)
    print("\n")
    
    # Save the analysis to a text file next to the image
    output_txt_path = TEST_IMAGE_PATH.replace(".png", "_analysis.txt")
    with open(output_txt_path, "w") as f:
        f.write(analysis)
    print(f"Analysis saved to: {output_txt_path}")