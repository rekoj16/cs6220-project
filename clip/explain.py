import os
import base64
import json
from openai import OpenAI
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_heatmap(image_path, model="gpt-4o"):
    """
    Send heatmap image to OpenAI API for analysis.
    
    Args:
        image_path: Path to the heatmap comparison image
        model: OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)
    
    Returns:
        Dictionary with analysis results
    """
    
    # Encode the image
    base64_image = encode_image(image_path)
    
    # Create the prompt
    prompt = """You are a medical imaging AI expert analyzing chest X-ray interpretations.

This image shows three panels:
1. LEFT: Original chest X-ray with ground truth labels
2. MIDDLE: Baseline CLIP model's heatmap and predictions
3. RIGHT: Fine-tuned model's heatmap and predictions

Please provide a detailed analysis covering:

1. **Ground Truth Assessment**: What pathologies are actually present?

2. **Baseline Model Performance**: 
   - How accurate are its predictions?
   - Where is the attention focused (heatmap interpretation)?
   - What errors or biases do you observe?

3. **Fine-tuned Model Performance**:
   - How accurate are its predictions?
   - Where is the attention focused (heatmap interpretation)?
   - What improvements over baseline do you see?

4. **Attention Pattern Comparison**:
   - Compare the heatmap patterns between baseline and fine-tuned
   - Are the models looking at clinically relevant regions?
   - Does the attention align with where these pathologies typically appear?

5. **Clinical Relevance**:
   - Would the fine-tuned model be more trustworthy for clinical use?
   - Any concerning false positives or false negatives?

6. **Overall Assessment**: Brief summary of which model performs better and why.

Be specific about anatomical regions and pathology locations."""

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
                                "detail": "high"  # Use "high" for detailed analysis
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.3  # Lower temperature for more consistent analysis
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
    """
    Process all heatmap images in a directory and save explanations.
    
    Args:
        input_dir: Directory containing heatmap comparison images
        output_dir: Directory to save explanation JSON and text files
        model: OpenAI model to use
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all comparison images
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*_comparison.png"))
    
    print(f"Found {len(image_files)} images to analyze")
    
    all_results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Analyzing {image_path.name}...")
        
        # Analyze the image
        result = analyze_heatmap(str(image_path), model=model)
        
        if "error" not in result:
            print(f"✓ Analysis complete ({result['tokens_used']} tokens)")
            
            # Save individual text file
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
        
        # Rate limiting: sleep between requests to avoid hitting API limits
        if idx < len(image_files):
            time.sleep(1)  # Adjust based on your API rate limits
    
    # Save all results to JSON
    json_path = Path(output_dir) / "all_explanations.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ All results saved to {json_path}")
    
    # Print summary
    successful = sum(1 for r in all_results if "error" not in r)
    total_tokens = sum(r.get("tokens_used", 0) for r in all_results)
    
    print(f"\nSummary:")
    print(f"  Successfully analyzed: {successful}/{len(image_files)}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Estimated cost (gpt-4o): ${total_tokens * 0.000005:.4f}")  # Rough estimate


def analyze_single_image(image_path, output_path=None, model="gpt-4o"):
    """
    Analyze a single heatmap image.
    
    Args:
        image_path: Path to the image
        output_path: Optional path to save explanation (if None, prints to console)
        model: OpenAI model to use
    """
    
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
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("\nPlease create a .env file in the same directory with:")
        print("OPENAI_API_KEY=your-api-key-here")
        exit(1)
    
    if args.mode == "single":
        analyze_single_image(args.input, args.output, args.model)
    else:
        process_heatmap_directory(args.input, args.output, args.model)