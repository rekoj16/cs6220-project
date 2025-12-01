import os
import base64
import json
from openai import OpenAI
from pathlib import Path
import time
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_heatmap(image_path, model="gpt-4o"):
    base64_image = encode_image(image_path)
    
    prompt = """You are an expert radiologist analyzing chest X-ray interpretations and AI model behavior.
This image shows three panels:
1. LEFT: Original chest X-ray with ground truth labels
2. MIDDLE: Baseline CLIP model's heatmap and predictions
3. RIGHT: Fine-tuned model's heatmap and predictions

Please provide a detailed clinical and technical analysis:

**Fine-tuned Model Analysis**:
- Compare predictions to actual radiographic findings
- For CORRECT predictions: What clinically relevant features is the model focusing on?
- For INCORRECT predictions: What is the model misinterpreting and why?
- Evaluate if the attention pattern demonstrates appropriate clinical reasoning
- Compare attention strategy to baseline: Is it more anatomically informed?

Focus on explaining the "WHY" behind both the radiographic appearances and the model behaviors, connecting visual features to clinical knowledge."""


    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "image_path": image_path,
            "model": model,
            "analysis": analysis,
            "tokens_used": response.usage.total_tokens,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {str(e)}")
        return {
            "image_path": image_path,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


def process_heatmap_directory(input_dir="heatmaps", output_dir="explanations", model="gpt-4o"):
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*_comparison.png"))
    
    print(f"Found {len(image_files)} images to analyze")
    
    all_results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Analyzing {image_path.name}...")
        
        result = analyze_heatmap(str(image_path), model=model)
        
        if "error" not in result:
            print(f"✓ Analysis complete ({result['tokens_used']} tokens)")
            
            output_name = image_path.stem.replace("_comparison", "")
            txt_path = Path(output_dir) / f"{output_name}_explanation.txt"
            
            with open(txt_path, "w") as f:
                f.write(f"Image: {image_path.name}\n")
                f.write(f"Analyzed: {result['timestamp']}\n")
                f.write(f"Model: {result['model']}\n")
                f.write("="*80 + "\n\n")
                f.write(result['analysis'])
            
            print(f"✓ Saved to {txt_path}")
        else:
            print(f"✗ Error: {result['error']}")
        
        all_results.append(result)
        
        if idx < len(image_files):
            time.sleep(1)
    
    json_path = Path(output_dir) / "all_explanations.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n All results saved to {json_path}")
    
    successful = sum(1 for r in all_results if "error" not in r)
    total_tokens = sum(r.get("tokens_used", 0) for r in all_results)
    
    print(f"\nSummary:")
    print(f"  Successfully analyzed: {successful}/{len(image_files)}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Estimated cost (gpt-4o): ${total_tokens * 0.000005:.4f}")


def analyze_single_image(image_path, output_path=None, model="gpt-4o"):
    print(f"Analyzing {image_path}...")
    result = analyze_heatmap(image_path, model=model)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Analyzed: {result['timestamp']}\n")
            f.write(f"Model: {result['model']}\n")
            f.write("="*80 + "\n\n")
            f.write(result['analysis'])
        print(f"✓ Saved to {output_path}")
    else:
        print("\n" + "="*80)
        print(result['analysis'])
        print("="*80)
        print(f"\nTokens used: {result['tokens_used']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze chest X-ray heatmaps using OpenAI API")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch",
                        help="Process single image or entire directory")
    parser.add_argument("--input", default="heatmaps",
                        help="Input directory (batch mode) or image path (single mode)")
    parser.add_argument("--output", default="explanations",
                        help="Output directory (batch mode) or file path (single mode)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)")
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("\nPlease create a .env file in the same directory with:")
        print("OPENAI_API_KEY=your-api-key-here")
        exit(1)
    
    if args.mode == "single":
        analyze_single_image(args.input, args.output, args.model)
    else:
        process_heatmap_directory(args.input, args.output, args.model)