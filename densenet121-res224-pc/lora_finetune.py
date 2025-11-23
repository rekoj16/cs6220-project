import os
import sys
from typing import Dict, Optional, List
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchxrayvision as xrv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from peft import LoraConfig, get_peft_model


class LoRAModel(nn.Module):
    """
    Wrapper for applying LoRA to pretrained models.
    Assumes the base_model's classifier head has already been replaced
    and warm-started with matching weights.
    """
    
    def __init__(self, base_model, num_classes: int, lora_config: Optional[Dict] = None):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # Default LoRA configuration - target Conv layers only
        if lora_config is None:
            lora_config = {
                'r': 16,  # the rank of two matrices
                'lora_alpha': 32, # the scaling factor to the product of two matrices => 2x weights with r=16, a = 32.
                'target_modules': [],  # apply adapter to all cov feature layers
                'lora_dropout': 0.1, 
            }

        # Apply LoRA to backbone (Conv layers) only
        self._apply_lora(lora_config)
        self._freeze_parameters()
    
    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA adapters to selected Conv2d layers in backbone."""
        try:
            conv_leaf_names: List[str] = []
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Conv2d) and name.startswith('features'):
                    conv_leaf_names.append(name)

            if not conv_leaf_names:
                print("No Conv2d layers found under features; skipping LoRA.")
                return

            conv1_layers = [n for n in conv_leaf_names if n.endswith('conv1')]
            conv2_layers = [n for n in conv_leaf_names if n.endswith('conv2')]

            # Pick last 6 conv2 (3x3) and last 2 conv1 (1x1) as default
            # selected = conv2_layers[-6:] + conv1_layers[-2:]
            # Pick all conv layers
            selected = conv2_layers + conv1_layers

            lora_config['target_modules'] = selected

            peft_config = LoraConfig(**lora_config)
            self.base_model = get_peft_model(self.base_model, peft_config)

            print(f"LoRA applied to {len(selected)} Conv2d layers:")
            for layer_name in selected:
                print(f"  - {layer_name}")
        except Exception as e:
            print(f"Error applying LoRA: {e}")
    
    def _freeze_parameters(self):
        """Freeze backbone parameters except LoRA and classifier."""
        for name, param in self.base_model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            # Keep classifier parameters trainable because peft has frozen the classifer too
            # and we need to fine-tune it for the new CheXpert labels
            elif 'classifier' in name:
                param.requires_grad = True
            # Freeze everything else (backbone)
            else:
                param.requires_grad = False
    
    def forward(self, x):
        return self.base_model(x)
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class CheXpertFineTuner:
    """Main class for LoRA fine-tuning."""
    
    def __init__(self, data_path: str, results_dir: str):
        self.data_path = data_path
        self.results_dir = Path(results_dir)        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # CheXpert pathology labels (all 13)
        self.chexpert_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
            'Lung Opacity', 'Effusion', 'Pleural Other', 'Pneumonia', 
            'Pneumothorax', 'Support Devices'
        ]
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
    
    def load_pretrained_model(self, model_name: str = "densenet121-res224-pc"):
        """Load pretrained model, transfer matching weights, and set up for CheXpert fine-tuning."""
        
        original_model = xrv.models.DenseNet(weights=model_name)
        
        # Get original classifier and its labels
        original_classifier = original_model.classifier
        in_features = original_classifier.in_features
        
        # Get the 18 labels from the original pretrained model
        original_labels = original_model.pathologies
        
        # Create the new 13-class classifier for CheXpert
        num_chexpert_classes = len(self.chexpert_labels)
        new_classifier = nn.Linear(in_features, num_chexpert_classes)
        
        print("TRANSFERRING CLASSIFIER WEIGHTS (HEAD-SWAP)")
        copied_weights = 0
        with torch.no_grad():
            for new_idx, chexpert_label in enumerate(self.chexpert_labels):
                try:
                    # Find this label in the *original* 18-label list
                    old_idx = original_labels.index(chexpert_label)
                    
                    # Copy weights and bias
                    new_classifier.weight.data[new_idx] = original_classifier.weight.data[old_idx]
                    new_classifier.bias.data[new_idx] = original_classifier.bias.data[old_idx]
                    print(f"Copied: {chexpert_label} (Old index {old_idx} -> New index {new_idx})")
                    copied_weights += 1
                except ValueError:
                    # This CheXpert label wasn't in the original model
                    print(f"New: {chexpert_label} (New index {new_idx}) - will be randomly initialized.")
        
        print(f"Weight transfer complete. Copied {copied_weights} matching pathologies.")
        print(f"{num_chexpert_classes - copied_weights} new pathologies will be learned from scratch.")

        # Now, replace the head on the original model
        original_model.classifier = new_classifier

        original_model.op_threshs = None
        print("Disabled op_threshs to prevent output shape mismatch during training.")
        
        # Count original parameters (for reference, before LoRA)
        total_params = sum(p.numel() for p in original_model.parameters())
        print(f"\nOriginal model: {model_name}")
        print(f"Total parameters (after head-swap): {total_params:,}")
        print(f"Training on all {len(self.chexpert_labels)} CheXpert pathologies")
        
        lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
        }
        
        # Create LoRA model (it will now wrap the model with the *new* head)
        self.model = LoRAModel(
            base_model=original_model,
            num_classes=num_chexpert_classes,
            lora_config=lora_config
        )
        
        # Count trainable parameters
        total_lora_params, trainable_lora_params = self.model.get_trainable_parameters()
        
        print(f"\nAfter LoRA adaptation:")
        print(f"Total parameters: {total_lora_params:,}")
        print(f"Trainable parameters: {trainable_lora_params:,}")
        print(f"Trainable percentage: {100 * trainable_lora_params / total_lora_params:.2f}%")
        
        self.model.to(self.device)
    
    def prepare_dataset(self, train_split: float = 0.8, batch_size: int = 16):
        """Prepare CheXpert dataset with train/val splits."""
        print("PREPARING CHEXPERT DATASET")
        print(f"{'='*80}")

        # Define transforms
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        
        # Load full dataset
        full_dataset = xrv.datasets.CheX_Dataset(
            imgpath=self.data_path,
            csvpath=os.path.join(self.data_path, "train.csv"),
            views=["PA", "AP"],
            unique_patients=False,
            transform=transform
        )
        
        print(f"Total samples in dataset: {len(full_dataset)}")
        print(f"Training on all {len(self.chexpert_labels)} CheXpert pathologies")
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        print(f"Train size: {train_size}")
        print(f"Validation size: {val_size}")
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        print(f"Data loaders created with batch_size={batch_size}")
    
    def fine_tune_model(self, num_epochs: int = 3, learning_rate: float = 1e-4):
        """Fine-tune the model using LoRA on backbone + new classifier for all CheXpert pathologies."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate, 
            weight_decay=1e-5
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3)
        
        history = {'train_loss': [], 'val_loss': [], 'batch_train_loss': [], 'batch_val_loss': []}
        best_val_loss = float('inf')
        best_model_path = self.results_dir / "best_lora_model.pth"
        
        print(f"Training for {num_epochs} epochs with learning rate {learning_rate}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            train_loss, batch_train_losses = self._train_epoch(criterion, optimizer)
            val_loss, batch_val_losses = self._validate_epoch(criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['batch_train_loss'].append(batch_train_losses)
            history['batch_val_loss'].append(batch_val_losses)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_model_path)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(best_model_path))
        print(f"\nBest model loaded with validation loss: {best_val_loss:.4f}")
        
        # Save training history
        self._save_training_history(history)
        # Plot and save loss curve
        self._plot_loss_curve(history)

        # Evaluate AUROC on validation set and plot
        auroc_scores = self._evaluate_auroc()
        self._plot_auroc(auroc_scores)
        self._save_auroc_scores(auroc_scores)
        return history
    
    def _train_epoch(self, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        batch_losses: List[float] = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['img'].to(self.device)
            targets = batch['lab'].to(self.device)

            # convert nan and -1 to 0 for binary targets
            binary_targets = self._convert_targets_to_binary(targets)
            
            optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = criterion(outputs, binary_targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, batch_losses
    
    def _validate_epoch(self, criterion):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        batch_losses: List[float] = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['img'].to(self.device)
                targets = batch['lab'].to(self.device)
                
                binary_targets = self._convert_targets_to_binary(targets)
                
                outputs = self.model(images)
                loss = criterion(outputs, binary_targets)
                
                total_loss += loss.item()
                batch_losses.append(loss.item())
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, batch_losses
    
    def _convert_targets_to_binary(self, targets):
        """Convert XRV targets to binary format for all 13 CheXpert pathologies."""
        batch_size = targets.shape[0]
        num_chexpert_labels = len(self.chexpert_labels)
        
        # Ensure we have the right number of classes (13)
        if targets.shape[1] >= num_chexpert_labels:
            binary_targets = targets[:, :num_chexpert_labels].float()
        else:
            # Pad with zeros if dataset has fewer labels
            binary_targets = torch.zeros(batch_size, num_chexpert_labels, device=self.device)
            binary_targets[:, :targets.shape[1]] = targets.float()
        
        # Treat all -1.0 (uncertain) as 0.0 (negative)
        binary_targets[binary_targets == -1] = 0.0
        # Treat all 'nan' (not mentioned) as 0.0 (negative)
        binary_targets = torch.nan_to_num(binary_targets, nan=0.0)

        return binary_targets
    
    def _save_training_history(self, history):
        history_path = self.results_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")

    def _plot_loss_curve(self, history: Dict[str, List[float]]):
        try:
            num_epochs = len(history['train_loss'])
            # If only one epoch, plot batch-level losses instead of a single point
            if num_epochs == 1 and history.get('batch_train_loss') and history['batch_train_loss'] and history.get('batch_val_loss') and history['batch_val_loss']:
                batch_train_loss = history['batch_train_loss'][0]
                batch_val_loss = history['batch_val_loss'][0]
                plt.figure(figsize=(8,5))
                plt.plot(range(1, len(batch_train_loss)+1), batch_train_loss, marker='o', linewidth=1.2, label='Batch Train Loss')
                plt.plot(range(1, len(batch_val_loss)+1), batch_val_loss, marker='s', linewidth=1.2, label='Batch Val Loss')
                plt.xlabel('Batch')
                plt.ylabel('BCEWithLogits Loss')
                plt.title('Training Loss (Single Epoch)')
                plt.grid(alpha=0.3)
                plt.legend()
                loss_curve_path = self.results_dir / 'loss_curve_per_batch_one_epoch.png'
                plt.tight_layout()
                plt.savefig(loss_curve_path)
                plt.close()
                print(f"Batch-level loss curve saved to {loss_curve_path}")
            else:
                # Plot the mean loss in each epoch
                plt.figure(figsize=(8,5))
                epochs = range(1, num_epochs + 1)
                plt.plot(epochs, history['train_loss'], marker='o', label='Train Loss')
                plt.plot(epochs, history['val_loss'], marker='s', label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('BCEWithLogits Loss')
                plt.title('Training & Validation Loss')
                plt.legend()
                plt.grid(alpha=0.3)
                loss_curve_path = self.results_dir / 'loss_curve.png'
                plt.tight_layout()
                plt.savefig(loss_curve_path)
                plt.close()
                print(f"Loss curve saved to {loss_curve_path}")
        except Exception as e:
            print(f"Failed to plot loss curve: {e}")

    def _evaluate_auroc(self):
        self.model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['img'].to(self.device)
                targets = batch['lab'].to(self.device)
                binary_targets = self._convert_targets_to_binary(targets)
                outputs = self.model(images)
                all_outputs.append(outputs.detach().cpu())
                all_targets.append(binary_targets.detach().cpu())

        outputs_tensor = torch.cat(all_outputs, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        probs = torch.sigmoid(outputs_tensor).numpy()
        labels = targets_tensor.numpy()

        auroc_scores = {}
        valid_class_scores = []
        for i, pathology in enumerate(self.chexpert_labels):
            y_true = labels[:, i]
            y_score = probs[:, i]
            # Need at least one positive and one negative
            if np.sum(y_true == 1) > 0 and np.sum(y_true == 0) > 0:
                try:
                    score = roc_auc_score(y_true, y_score)
                    auroc_scores[pathology] = score
                    valid_class_scores.append(score)
                except ValueError:
                    auroc_scores[pathology] = None
            else:
                auroc_scores[pathology] = None

        if valid_class_scores:
            auroc_scores['macro_average'] = float(np.mean(valid_class_scores))
        else:
            auroc_scores['macro_average'] = None

        for k, v in auroc_scores.items():
            if k == 'macro_average':
                print(f"  Macro Average AUROC: {v if v is not None else 'N/A'}")
            else:
                print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

        return auroc_scores

    def _plot_auroc(self, auroc_scores: Dict[str, Optional[float]]):
        try:
            pathologies = []
            scores = []
            for p in self.chexpert_labels:
                s = auroc_scores.get(p)
                pathologies.append(p)
                scores.append(s if s is not None else 0.0)

            plt.figure(figsize=(10,6))
            bars = plt.bar(range(len(pathologies)), scores, color='#4c72b0')
            plt.xticks(range(len(pathologies)), pathologies, rotation=45, ha='right')
            plt.ylabel('AUROC')
            plt.title('Validation AUROC per Pathology')
            plt.ylim(0, 1.0)
            for idx, bar in enumerate(bars):
                original = auroc_scores.get(pathologies[idx])
                if original is not None:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{original:.2f}",
                             ha='center', va='bottom', fontsize=8)
                else:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, "N/A",
                             ha='center', va='bottom', fontsize=8, color='red')

            macro = auroc_scores.get('macro_average')
            if macro is not None:
                plt.axhline(macro, color='green', linestyle='--', linewidth=1.2, label=f"Macro Avg: {macro:.3f}")
                plt.legend()

            plt.tight_layout()
            auroc_path = self.results_dir / 'auroc.png'
            plt.savefig(auroc_path)
            plt.close()
            print(f"AUROC plot saved to {auroc_path}")
        except Exception as e:
            print(f"Failed to plot AUROC: {e}")

    def _save_auroc_scores(self, auroc_scores: Dict[str, Optional[float]]):
        out_path = self.results_dir / 'auroc_scores.json'
        try:
            with open(out_path, 'w') as f:
                json.dump(auroc_scores, f, indent=2)
            print(f"AUROC scores saved to {out_path}")
        except Exception as e:
            print(f"Failed to save AUROC scores: {e}")


def main():
    DATA_PATH = "/home/hice1/ymai8/scratch/cs6220-project/CheXpert-v1.0-small"
    RESULTS_DIR = "result(4)_batch_32_epoch_10_lr_1e-4_rank_16_alpha_32_dropout_0.1_all-conv-layers"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # Initialize fine-tuner
        fine_tuner = CheXpertFineTuner(
            data_path=DATA_PATH,
            results_dir=RESULTS_DIR
        )
        
        # Step 1: Load pretrained model
        fine_tuner.load_pretrained_model("densenet121-res224-pc")
        
        # Step 2: Prepare dataset
        fine_tuner.prepare_dataset(
            train_split=0.8,
            batch_size=32
        )
        
        # Step 3: Fine-tune model
        history = fine_tuner.fine_tune_model(
            num_epochs=10,
            learning_rate=1e-4
        )
        
        # Final summary
        print("FINE-TUNING COMPLETE - SUMMARY")
        print(f"Model successfully fine-tuned with LoRA")
        print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"All results saved to: {RESULTS_DIR}/")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()