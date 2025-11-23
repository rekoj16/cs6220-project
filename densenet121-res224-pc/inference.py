import sys
import os
import argparse
import torch
import torchvision
import torchxrayvision as xrv
from torch.utils.data import DataLoader
from label_matcher import LabelMatcher
from compute_metrics import ComputeMetrics
from lora_tuned_model import load_lora_finetuned_model
from heatmap_visualization import visualize_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained or LoRA finetuned DenseNet models and optionally generate CAM visualizations.")
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA finetuned model for metric evaluation.')
    parser.add_argument('--lora-checkpoint', type=str, default="/home/hice1/ymai8/scratch/cs6220-project/densenet121-res224-pc/result(4)_batch_32_epoch_10_lr_1e-4_rank_16_alpha_32_dropout_0.1_all-conv-layers/best_lora_model.pth", help='Path to LoRA finetuned model checkpoint.')
    parser.add_argument('--visualize', action='store_true', help='Generate heatmap visualizations comparing baseline and finetuned models.')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of samples to visualize.')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Directory to save visualization images.')
    parser.add_argument('--prediction_threshold', type=float, default=0.6, help='Threshold for positive predictions in metrics.')
    parser.add_argument('--batch_size', type=int, default=16, help='Evaluation dataloader batch size.')
    return parser.parse_args()


def main():
    args = parse_args()
    USE_LORA_MODEL = args.use_lora
    CHECKPOINT_PATH = args.lora_checkpoint
    try:
        # 1. Setup transforms and load dataset
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        
        d_chex = xrv.datasets.CheX_Dataset(
            imgpath="/home/hice1/ymai8/scratch/cs6220-project/CheXpert-v1.0-small",
            csvpath="/home/hice1/ymai8/scratch/cs6220-project/CheXpert-v1.0-small/valid.csv",
            views=["PA", "AP"], 
            unique_patients=False, 
            transform=transform
        )
        
        # Load baseline and optionally LoRA finetuned models
        baseline_model = xrv.models.DenseNet(weights="densenet121-res224-pc")
        if USE_LORA_MODEL:
            finetuned_model = load_lora_finetuned_model(CHECKPOINT_PATH)
            print(f"Loaded LoRA finetuned model from {CHECKPOINT_PATH}")
        else:
            finetuned_model = baseline_model  # For uniform handling later
            print("LoRA model not requested; using baseline for metrics.")
        
        # Use LabelMatcher to match pathologies
        baseline_matcher = LabelMatcher()
        pathology_mapping = baseline_matcher.match_pathologies(baseline_model, d_chex)
        print("Printing baseline model pathology mapping summary:")
        baseline_matcher.print_summary()
        
        if USE_LORA_MODEL:
            finetuned_model_matcher = LabelMatcher()
            finetuned_pathology_mapping = finetuned_model_matcher.match_pathologies(finetuned_model, d_chex)
            print("Printing LoRA finetuned model pathology mapping summary:")
            finetuned_model_matcher.print_summary()
        
        # Run Eval
        print("---SETTING UP EVALUATION---")
        batch_size = args.batch_size 
        dataloader = DataLoader(
            d_chex, # torchxrayvision helper dataset class
            batch_size=batch_size, # 16
            shuffle=False, 
            num_workers=2
        )
        
        print(f"Created DataLoader with batch_size={batch_size}")
        print(f"Total samples: {len(d_chex)}")
        
        # Initialize ComputeMetrics and run evaluation
        print("---RUNNING MODEL EVALUATION---")
        prediction_threshold = args.prediction_threshold
        baseline_metrics_computer = ComputeMetrics(prediction_threshold=prediction_threshold)
        if USE_LORA_MODEL:
            finetuned_metrics_computer = ComputeMetrics(prediction_threshold=prediction_threshold)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Move models to device to avoid dtype/device mismatch
        baseline_model = baseline_model.to(device)
        if USE_LORA_MODEL:
            finetuned_model = finetuned_model.to(device)
        print(f"Using device {device} with prediction threshold: {prediction_threshold}")
        
        # Visualization Section (optional)
        if args.visualize:
            print("---GENERATING HEATMAP VISUALIZATIONS (Original + Baseline + Finetuned)---")
            os.makedirs(args.output_dir, exist_ok=True)
            vis_count = 0
            max_vis = args.limit 
            dataset_pathologies = getattr(d_chex, 'pathologies', [])
            
            # Helper function to extract patient ID from path
            def extract_patient_id(dataset_idx):
                try:
                    if hasattr(d_chex, 'csv') and 'Path' in d_chex.csv.columns:
                        path = d_chex.csv.iloc[dataset_idx]['Path']
                        # Extract patient ID from path like "CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
                        path_parts = path.split('/')
                        for part in path_parts:
                            if part.startswith('patient'):
                                return part
                    return None
                except:
                    return None
            
            current_idx = 0  # Track current dataset index
            for batch in dataloader:
                if isinstance(batch, dict):
                    imgs = batch.get('img')
                    labs = batch.get('lab')
                else:
                    imgs = batch[0]
                    labs = batch[1] if len(batch) > 1 else None
                if imgs is None:
                    print("Skipping batch without 'img'.")
                    continue
                if imgs.ndim == 3:
                    imgs = imgs.unsqueeze(1)  # (B,1,H,W)
                for i in range(imgs.shape[0]):
                    if vis_count >= max_vis:
                        break
                    img = imgs[i].unsqueeze(0)  # (1,1,H,W)
                    gt_lab = labs[i] if labs is not None else torch.zeros(len(dataset_pathologies))
                    sample_id = f"{vis_count:04d}"
                    
                    # Get patient ID for current sample
                    patient_id = extract_patient_id(current_idx + i)
                    
                    out_path = visualize_sample(
                        baseline_model=baseline_model,
                        finetuned_model=finetuned_model,
                        img=img,
                        device=device,
                        sample_id=sample_id,
                        output_dir=args.output_dir,
                        ground_truth=gt_lab,
                        dataset_pathologies=dataset_pathologies,
                        patient_id=patient_id,
                        positive_label_values=(1.0,),
                        top_k=5,
                    )
                    if patient_id:
                        print(f"Saved visualization for {patient_id}: {out_path}")
                    else:
                        print(f"Saved visualization: {out_path}")
                    vis_count += 1
                
                # Update current index for next batch
                current_idx += imgs.shape[0]
                
                if vis_count >= max_vis:
                    break
            print(f"\nGenerated {vis_count} visualization image(s) to directory: {args.output_dir}")

        # Run evaluation
        baseline_results = baseline_metrics_computer.evaluate_model(
            model=baseline_model,
            dataloader=dataloader,
            pathology_mapping=pathology_mapping,
            device=device
        )

        if USE_LORA_MODEL:
            finetuned_results = finetuned_metrics_computer.evaluate_model(
                model=finetuned_model,
                dataloader=dataloader,
                pathology_mapping=finetuned_pathology_mapping,
                device=device
            )
        
        
        # 8. Print and save results
        print("---EVALUATION COMPLETE---")
        print("\nBASELINE MODEL RESULTS:")
        baseline_metrics_computer.print_results()
        csv_path = "baseline_inference_result.csv"
        baseline_metrics_computer.save_results_to_csv(csv_path)
        plot_path = "baseline_auroc_scores.png"
        baseline_metrics_computer.plot_auroc_scores(figsize=(14, 8), save_path=plot_path)
        print("BASELINE TOP PERFORMING PATHOLOGIES (AUROC)")
        top_auroc = baseline_metrics_computer.get_best_performing_pathologies(metric='auroc', top_k=5)
        for i, (pathology, score) in enumerate(top_auroc, 1):
            print(f"{i}. {pathology}: {score:.4f}")
        
        if 'summary_stats' in baseline_results:
            mean_auroc = baseline_results['summary_stats']['mean_auroc']
            print(f"\n🎯 Overall Performance: Mean AUROC = {mean_auroc:.4f}")

        if USE_LORA_MODEL:
            print("\nLoRA FINETUNED MODEL RESULTS:")
            finetuned_metrics_computer.print_results()
            csv_path = "lora_finetuned_inference_result.csv"
            finetuned_metrics_computer.save_results_to_csv(csv_path)
            plot_path = "lora_finetuned_auroc_scores.png"
            finetuned_metrics_computer.plot_auroc_scores(figsize=(14, 8), save_path=plot_path)
            print("LoRA FINETUNED TOP PERFORMING PATHOLOGIES (AUROC)")
            top_auroc_finetuned = finetuned_metrics_computer.get_best_performing_pathologies(metric='auroc', top_k=5)
            for i, (pathology, score) in enumerate(top_auroc_finetuned, 1):
                print(f"{i}. {pathology}: {score:.4f}")
            
            if 'summary_stats' in finetuned_results:
                mean_auroc_finetuned = finetuned_results['summary_stats']['mean_auroc']
                print(f"\n🎯 Overall Performance: Mean AUROC = {mean_auroc_finetuned:.4f}")
        
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()