"""MHIST PyTorch Dataset with train/val/test split support."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MHISTDataset(Dataset):
    """MHIST Histopathology Dataset.
    
    Binary classification: HP (Hyperplastic Polyp) vs SSA (Sessile Serrated Adenoma).
    Labels are majority vote from 7 board-certified GI pathologists.
    """
    
    LABEL_MAP = {"HP": 0, "SSA": 1}
    LABEL_NAMES = ["HP", "SSA"]
    
    def __init__(
        self,
        annotations_csv: str,
        image_dir: str,
        split: str = "train",
        transform=None,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            annotations_csv: Path to annotations.csv
            image_dir: Path to images/ directory
            split: One of 'train', 'val', 'test'
            transform: torchvision transforms to apply
            val_ratio: Fraction of official train to hold out for validation
            seed: Random seed for reproducible val split
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.split = split
        
        if not os.path.exists(annotations_csv):
            raise FileNotFoundError(f"Annotations file not found: {annotations_csv}")
        df = pd.read_csv(annotations_csv)
        
        required_cols = ["Image Name", "Majority Vote Label", 
                         "Number of Annotators who Selected SSA (Out of 7)", "Partition"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if split == "test":
            self.df = df[df["Partition"] == "test"].reset_index(drop=True)
        else:
            train_df = df[df["Partition"] == "train"].reset_index(drop=True)
            labels = train_df["Majority Vote Label"].values
            
            train_idx, val_idx = train_test_split(
                np.arange(len(train_df)),
                test_size=val_ratio,
                stratify=labels,
                random_state=seed,
            )
            
            if split == "train":
                self.df = train_df.iloc[train_idx].reset_index(drop=True)
            elif split == "val":
                self.df = train_df.iloc[val_idx].reset_index(drop=True)
            else:
                raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")

        missing = []
        for fname in self.df["Image Name"]:
            if not (self.image_dir / fname).exists():
                missing.append(fname)
        if missing:
            print(f"WARNING: {len(missing)} images missing from disk: {missing[:5]}...")
        
        self.labels = [self.LABEL_MAP[lbl] for lbl in self.df["Majority Vote Label"]]
        self.annotator_counts = self.df[
            "Number of Annotators who Selected SSA (Out of 7)"
        ].values.astype(int)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["Image Name"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "label": self.labels[idx],
            "filename": row["Image Name"],
            "annotator_ssa_count": int(self.annotator_counts[idx]),
        }
    
    def get_class_counts(self) -> Dict[str, int]:
        """Return class distribution."""
        labels = self.df["Majority Vote Label"].value_counts().to_dict()
        return labels
    
    def get_class_weights(self) -> List[float]:
        """Compute inverse-frequency class weights for loss function."""
        counts = self.df["Majority Vote Label"].value_counts()
        total = len(self.df)
        weights = [total / (len(counts) * counts.get(name, 1)) 
                   for name in self.LABEL_NAMES]
        return weights
    
    def get_stats_summary(self) -> Dict:
        """Return dataset statistics."""
        counts = self.get_class_counts()
        total = len(self.df)
        return {
            "split": self.split,
            "total": total,
            "class_counts": counts,
            "class_percentages": {k: f"{v/total*100:.1f}%" for k, v in counts.items()},
            "annotator_agreement_mean": float(np.mean(self.annotator_counts)),
            "annotator_agreement_std": float(np.std(self.annotator_counts)),
        }
