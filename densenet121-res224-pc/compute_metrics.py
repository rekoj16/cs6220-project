import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
)
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class ComputeMetrics:
    """
    A class to compute various evaluation metrics for multi-label classification.
    
    This class provides methods to:
    - Compute AUROC scores per label and overall
    - Generate comprehensive evaluation reports
    - Visualize performance metrics
    """
    
    def __init__(self, prediction_threshold: float = 0.5):
        """
        Initialize the ComputeMetrics class.
        
        Args:
            prediction_threshold: Threshold for converting predictions to binary (default: 0.5)
        """
        self.prediction_threshold = prediction_threshold
        self.results = {}
    
    def evaluate_model(self, 
                      model: Any, 
                      dataloader: DataLoader, 
                      pathology_mapping: Dict[int, Tuple[int, str]],
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, Any]:
        """
        Evaluate model on given dataloader and compute metrics.
        
        Args:
            model: The trained model to evaluate
            dataloader: DataLoader containing validation/test data
            pathology_mapping: Mapping from model indices to (dataset_idx, pathology_name)
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing all computed metrics
        """
        print(f"Starting model evaluation on {device}")
        print(f"Evaluating {len(pathology_mapping)} matched pathologies")
        
        model.eval()
        model = model.to(device)
        
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    print(f"Processing batch {batch_idx}/{len(dataloader)}")
                
                images = batch['img'].to(device)
                labels = batch['lab'].to(device)
                outputs = model(images)
                predictions = torch.sigmoid(outputs).detach().cpu().numpy()
                
                all_predictions.append(predictions)
                all_labels.append(labels.detach().cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        results = self._compute_all_metrics(
            all_predictions, 
            all_labels, 
            pathology_mapping
        )
        self.results = results
        
        return results
    
    def _compute_all_metrics(self, 
                           predictions: np.ndarray, 
                           labels: np.ndarray,
                           pathology_mapping: Dict[int, Tuple[int, str]]) -> Dict[str, Any]:
        """
        Compute all metrics for the given predictions and labels.
        
        Args:
            predictions: Model predictions (N, num_classes)
            labels: Ground truth labels (N, num_classes)
            pathology_mapping: Mapping from model indices to dataset info
            
        Returns:
            Dictionary containing all computed metrics
        """
        results = {
            'auroc_per_label': {},
            'f1_per_label': {},
            'recall_per_label': {},
            'precision_per_label': {},
            'accuracy_per_label': {},
            'pathology_names': [],
            'summary_stats': {}
        }
        
        auroc_scores = []
        f1_scores = []
        recall_scores = []
        precision_scores = []
        accuracy_scores = []
        
        for model_idx, (dataset_idx, pathology_name) in pathology_mapping.items():
            # Get predictions and labels for this pathology
            y_pred = predictions[:, model_idx]
            y_true = labels[:, dataset_idx]
            
            # Handle NaN values in labels (common in medical datasets)
            valid_mask = ~np.isnan(y_true)
            if not np.any(valid_mask):
                print(f"WARNING: No valid labels found for {pathology_name}, skipping")
                continue
                
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            # Convert to binary for non-AUROC metrics using configurable threshold
            y_pred_binary = (y_pred_valid > self.prediction_threshold).astype(int)
            y_true_binary = y_true_valid.astype(int)
            
            try:
                # Compute AUROC (only if we have both classes)
                if len(np.unique(y_true_binary)) > 1:
                    auroc = roc_auc_score(y_true_binary, y_pred_valid)
                    results['auroc_per_label'][pathology_name] = auroc
                    auroc_scores.append(auroc)
                else:
                    print(f"WARNING: Only one class present for {pathology_name}, cannot compute AUROC")
                    results['auroc_per_label'][pathology_name] = np.nan
                
                # Compute other metrics
                f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
                precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
                accuracy = accuracy_score(y_true_binary, y_pred_binary)
                
                results['f1_per_label'][pathology_name] = f1
                results['recall_per_label'][pathology_name] = recall
                results['precision_per_label'][pathology_name] = precision
                results['accuracy_per_label'][pathology_name] = accuracy
                
                f1_scores.append(f1)
                recall_scores.append(recall)
                precision_scores.append(precision)
                accuracy_scores.append(accuracy)
                
                results['pathology_names'].append(pathology_name)
                
                print(f"{pathology_name}: AUROC={auroc:.4f}, F1={f1:.4f}, "
                      f"Recall={recall:.4f}, Precision={precision:.4f}, Accuracy={accuracy:.4f}")
                
            except Exception as e:
                print(f"ERROR: Error computing metrics for {pathology_name}: {e}")
                continue
        
        # Compute summary statistics
        if auroc_scores:
            results['summary_stats'] = {
                'prediction_threshold': self.prediction_threshold,
                'mean_auroc': np.mean(auroc_scores),
                'std_auroc': np.std(auroc_scores),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'mean_recall': np.mean(recall_scores),
                'std_recall': np.std(recall_scores),
                'mean_precision': np.mean(precision_scores),
                'std_precision': np.std(precision_scores),
                'mean_accuracy': np.mean(accuracy_scores),
                'std_accuracy': np.std(accuracy_scores),
                'num_pathologies': len(auroc_scores)
            }
        
        return results
    
    def print_results(self):
        """Print a comprehensive summary of evaluation results."""
        if not self.results:
            print("WARNING: No results to print. Run evaluate_model first.")
            return
        
        print("MODEL EVALUATION RESULTS")
        print("="*100)
        
        # Print per-label results
        print("-" * 100)
        print(f"{'Pathology':<25} {'AUROC':<8} {'F1':<8} {'Recall':<8} {'Precision':<10} {'Accuracy':<8}")
        print("-" * 100)
        
        for pathology in self.results['pathology_names']:
            auroc = self.results['auroc_per_label'].get(pathology, np.nan)
            f1 = self.results['f1_per_label'].get(pathology, np.nan)
            recall = self.results['recall_per_label'].get(pathology, np.nan)
            precision = self.results['precision_per_label'].get(pathology, np.nan)
            accuracy = self.results['accuracy_per_label'].get(pathology, np.nan)
            
            print(f"{pathology:<25} {auroc:<8.4f} {f1:<8.4f} {recall:<8.4f} {precision:<10.4f} {accuracy:<8.4f}")
        
        # Print summary statistics
        if 'summary_stats' in self.results:
            stats = self.results['summary_stats']
            print("SUMMARY STATISTICS")
            print("="*100)
            print(f"Prediction threshold used: {stats['prediction_threshold']}")
            print(f"Number of pathologies evaluated: {stats['num_pathologies']}")
            print(f"Mean AUROC: {stats['mean_auroc']:.4f} ± {stats['std_auroc']:.4f}")
            print(f"Mean F1 Score: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
            print(f"Mean Recall: {stats['mean_recall']:.4f} ± {stats['std_recall']:.4f}")
            print(f"Mean Precision: {stats['mean_precision']:.4f} ± {stats['std_precision']:.4f}")
            print(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    
    def save_results_to_csv(self, filepath: str):
        if not self.results or not self.results['pathology_names']:
            print("WARNING: No results to save.")
            return
        data = []
        for pathology in self.results['pathology_names']:
            row = {
                'Pathology': pathology,
                'AUROC': self.results['auroc_per_label'].get(pathology, np.nan),
                'F1_Score': self.results['f1_per_label'].get(pathology, np.nan),
                'Recall': self.results['recall_per_label'].get(pathology, np.nan),
                'Precision': self.results['precision_per_label'].get(pathology, np.nan),
                'Accuracy': self.results['accuracy_per_label'].get(pathology, np.nan)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def plot_auroc_scores(self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None):
        """
        Plot AUROC scores for each pathology, sorted ascending.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the plot
        """
        if not self.results or not self.results['auroc_per_label']:
            print("WARNING: No AUROC results to plot.")
            return
        
        # Extract and sort AUROC values
        auroc_dict = self.results['auroc_per_label']
        sorted_items = sorted(auroc_dict.items(), key=lambda x: x[1])  # ascending
        
        pathologies = [item[0] for item in sorted_items]
        auroc_scores = [item[1] for item in sorted_items]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(pathologies)), auroc_scores, alpha=0.7)
        plt.xlabel('Pathology')
        plt.ylabel('AUROC Score')
        plt.title('AUROC Scores per Pathology - DenseNet121-res224-pc on CheXpert')
        plt.xticks(range(len(pathologies)), pathologies, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, auroc_scores):
            plt.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{score:.3f}',
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_auroc_scores(self) -> Dict[str, float]:
        return self.results.get('auroc_per_label', {})
    
    def get_best_performing_pathologies(self, metric: str = 'auroc', top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k best performing pathologies for a given metric.
        
        Args:
            metric: Metric to rank by ('auroc', 'f1', 'recall', 'precision', 'accuracy')
            top_k: Number of top pathologies to return
            
        Returns:
            List of (pathology_name, score) tuples sorted by score
        """
        metric_key = f'{metric}_per_label'
        if metric_key not in self.results:
            print(f"WARNING: Metric {metric} not found in results.")
            return []
        
        scores = self.results[metric_key]
        sorted_pathologies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_pathologies[:top_k]
    
    def set_prediction_threshold(self, new_threshold: float):
        old_threshold = self.prediction_threshold
        self.prediction_threshold = new_threshold
        print(f"Prediction threshold changed from {old_threshold} to {new_threshold}")
    
    def get_prediction_threshold(self) -> float:
        return self.prediction_threshold