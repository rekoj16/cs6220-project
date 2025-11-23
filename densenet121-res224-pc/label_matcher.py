from typing import Dict, List, Tuple, Any
import sys


class LabelMatcher:
    """
    A class to handle pathology label matching between models and datasets.
    
    This class provides methods to:
    - Match pathology labels between model and dataset
    - Print matching results and analysis
    - Provide comprehensive reporting of matched/unmatched labels
    """
    
    def __init__(self):
        self.pathology_mapping = {}
        self.model_pathologies = []
        self.dataset_pathologies = []
        self.model_pathologies_clean = []
        self.dataset_pathologies_clean = []
    
    def match_pathologies(self, model: Any, dataset: Any) -> Dict[int, Tuple[int, str]]:
        """
        Match pathologies between model and dataset.
        
        Args:
            model: The torchxrayvision model with pathologies attribute
            dataset: The dataset with pathologies attribute
            
        Returns:
            Dictionary mapping model indices to (dataset_index, pathology_name) tuples
        """
        # Store original pathologies
        self.model_pathologies = model.pathologies
        self.dataset_pathologies = dataset.pathologies
        print(f"Model pathologies ({len(self.model_pathologies)}): {self.model_pathologies}")
        print(f"Dataset pathologies ({len(self.dataset_pathologies)}): {self.dataset_pathologies}")
        
        # Clean pathologies
        self.model_pathologies_clean = [p for p in self.model_pathologies if p.strip()]
        self.dataset_pathologies_clean = [str(p) for p in self.dataset_pathologies]
        print(f"Cleaned model pathologies ({len(self.model_pathologies_clean)}): {self.model_pathologies_clean}")
        print(f"Cleaned dataset pathologies ({len(self.dataset_pathologies_clean)}): {self.dataset_pathologies_clean}")
        
        # Create mapping
        self.pathology_mapping = self._create_pathology_mapping()
        
        # Print and analyze results
        self._print_mapping_results()
        self._analyze_unmapped_pathologies()
        
        return self.pathology_mapping
    
    def _create_pathology_mapping(self) -> Dict[int, Tuple[int, str]]:
        """Create mapping between model indices and dataset indices for matching pathologies."""
        pathology_mapping = {}
        
        for model_idx, model_pathology in enumerate(self.model_pathologies):
            if model_pathology.strip():  # Skip empty pathologies
                for dataset_idx, dataset_pathology in enumerate(self.dataset_pathologies_clean):
                    if model_pathology.strip().lower() == str(dataset_pathology).strip().lower():
                        pathology_mapping[model_idx] = (dataset_idx, model_pathology)
                        break
        
        return pathology_mapping
    
    def _print_mapping_results(self):
        """Print the pathology mapping results."""
        print(f"Pathology mapping found {len(self.pathology_mapping)} matches:")
        for model_idx, (dataset_idx, name) in self.pathology_mapping.items():
            print(f"  Model[{model_idx}] -> Dataset[{dataset_idx}]: {name}")
    
    def _analyze_unmapped_pathologies(self):
        """Analyze and log unmapped pathologies from both model and dataset."""
        mapped_model_indices = set(self.pathology_mapping.keys())
        mapped_dataset_indices = set(idx for idx, _ in self.pathology_mapping.values())
        
        # Find unmapped model pathologies
        unmapped_model_pathologies = []
        for i, p in enumerate(self.model_pathologies):
            if p.strip() and i not in mapped_model_indices:
                unmapped_model_pathologies.append((i, p))
        
        # Find unmapped dataset pathologies
        unmapped_dataset_pathologies = []
        for i, p in enumerate(self.dataset_pathologies):
            if i not in mapped_dataset_indices:
                unmapped_dataset_pathologies.append((i, str(p)))
        
        # Print unmapped pathologies
        print(f"Unmapped model pathologies ({len(unmapped_model_pathologies)}):")
        for idx, name in unmapped_model_pathologies:
            print(f"  Model[{idx}]: {name}")
        
        print(f"Unmapped dataset pathologies ({len(unmapped_dataset_pathologies)}):")
        for idx, name in unmapped_dataset_pathologies:
            print(f"  Dataset[{idx}]: {name}")
    
    def get_matched_labels(self) -> List[Tuple[int, int, str]]:
        matched_labels = []
        for model_idx, (dataset_idx, name) in self.pathology_mapping.items():
            matched_labels.append((model_idx, dataset_idx, name))
        return matched_labels
    
    def get_matched_pathology_names(self) -> List[str]:
        return [name for _, name in self.pathology_mapping.values()]
    
    def get_model_indices_for_matched_pathologies(self) -> List[int]:
        return list(self.pathology_mapping.keys())
    
    def get_dataset_indices_for_matched_pathologies(self) -> List[int]:
        return [dataset_idx for dataset_idx, _ in self.pathology_mapping.values()]
    
    def validate_matching_success(self, min_matches: int = 1) -> bool:
        success = len(self.pathology_mapping) >= min_matches
        
        if not success:
            print(f"WARNING: Matching failed: Only {len(self.pathology_mapping)} matches found, "
                  f"minimum required: {min_matches}")
        else:
            print(f"Matching successful: {len(self.pathology_mapping)} pathologies matched")
        
        return success
    
    def print_summary(self):
        print("PATHOLOGY MATCHING SUMMARY")
        print("="*80)
        
        print(f"Total model pathologies: {len(self.model_pathologies)}")
        print(f"Total dataset pathologies: {len(self.dataset_pathologies)}")
        print(f"Successfully matched pathologies: {len(self.pathology_mapping)}")
        
        if self.pathology_mapping:
            print(f"\nMatched pathologies:")
            for model_idx, (dataset_idx, name) in self.pathology_mapping.items():
                print(f"  • {name} (Model[{model_idx}] ↔ Dataset[{dataset_idx}])")
        
        # Calculate coverage
        model_coverage = len(self.pathology_mapping) / len([p for p in self.model_pathologies if p.strip()]) * 100
        dataset_coverage = len(set(idx for idx, _ in self.pathology_mapping.values())) / len(self.dataset_pathologies) * 100
        
        print(f"\nCoverage:")
        print(f" 1. Model pathology coverage: {model_coverage:.1f}%")
        print(f" 2. Dataset pathology coverage: {dataset_coverage:.1f}%")