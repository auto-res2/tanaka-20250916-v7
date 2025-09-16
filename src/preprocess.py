"""Data loading and preprocessing logic for AIS-KD experiments."""
from __future__ import annotations

import random
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["load_dataset_for_kd", "prepare_datasets", "FederatedDataSplit"]


class KnowledgeDistillationDataset(Dataset):
    """Dataset wrapper for knowledge distillation training."""
    
    def __init__(self, data_list: List[Dict], tokenizer=None, max_length: int = 128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        if "sentence" in example:
            text = example["sentence"]
        elif "text" in example:
            text = example["text"]
        else:
            text = str(example)
        
        if self.tokenizer is None:
            words = text.split()[:self.max_length-2]  # Leave room for special tokens
            input_ids = [101] + [hash(word) % 30000 + 1000 for word in words] + [102]  # [CLS] + tokens + [SEP]
            
            while len(input_ids) < self.max_length:
                input_ids.append(0)  # PAD token
            
            input_ids = input_ids[:self.max_length]
            attention_mask = [1 if token != 0 else 0 for token in input_ids]
            
            result = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        else:
            try:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                result = {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                }
            except Exception as e:
                print(f"Warning: Tokenization failed, using fallback: {e}")
                input_ids = torch.randint(1000, 30000, (self.max_length,))
                attention_mask = torch.ones(self.max_length)
                result = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
        
        if "label" in example and example["label"] is not None:
            result["labels"] = torch.tensor(example["label"], dtype=torch.long)
        
        return result


class FederatedDataSplit:
    """Utility for splitting data across federated clients."""
    
    def __init__(self, data_list: List[Dict], num_clients: int, seed: int = 42):
        self.data = data_list
        self.num_clients = num_clients
        self.seed = seed
        random.seed(seed)
    
    def split_iid(self) -> List[List[Dict]]:
        """Split data IID across clients."""
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        client_size = len(indices) // self.num_clients
        client_datasets = []
        
        for i in range(self.num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size if i < self.num_clients - 1 else len(indices)
            client_indices = indices[start_idx:end_idx]
            client_data = [self.data[idx] for idx in client_indices]
            client_datasets.append(client_data)
        
        return client_datasets


def load_glue_dataset_simple(task: str, max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """Load GLUE dataset with fallback to dummy data."""
    try:
        from datasets import load_dataset
        
        if task == "all":
            tasks = ["sst2", "mrpc"]
            all_train = []
            all_val = []
            
            for t in tasks:
                try:
                    train_ds = load_dataset("glue", t, split="train")
                    val_ds = load_dataset("glue", t, split="validation")
                    
                    if max_samples:
                        train_size = min(max_samples // len(tasks), len(train_ds))
                        val_size = min(max_samples // len(tasks), len(val_ds))
                        train_ds = train_ds.select(range(train_size))
                        val_ds = val_ds.select(range(val_size))
                    
                    all_train.extend(list(train_ds))
                    all_val.extend(list(val_ds))
                    
                except Exception as e:
                    print(f"Warning: Could not load GLUE task {t}: {e}")
                    continue
            
            if all_train and all_val:
                return all_train, all_val
            
        else:
            train_dataset = load_dataset("glue", task, split="train")
            val_dataset = load_dataset("glue", task, split="validation")
            
            if max_samples:
                train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
                val_dataset = val_dataset.select(range(min(max_samples, len(val_dataset))))
            
            return list(train_dataset), list(val_dataset)
            
    except Exception as e:
        print(f"Warning: Could not load GLUE dataset {task}: {e}")
    
    print("Using dummy data for training...")
    dummy_train = [
        {"sentence": f"This is a positive example {i}.", "label": 1} for i in range(50)
    ] + [
        {"sentence": f"This is a negative example {i}.", "label": 0} for i in range(50)
    ]
    
    dummy_val = [
        {"sentence": f"Validation positive {i}.", "label": 1} for i in range(10)
    ] + [
        {"sentence": f"Validation negative {i}.", "label": 0} for i in range(10)
    ]
    
    if max_samples:
        dummy_train = dummy_train[:max_samples]
        dummy_val = dummy_val[:min(max_samples//5, len(dummy_val))]
    
    return dummy_train, dummy_val


def load_dataset_for_kd(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Load and prepare datasets for knowledge distillation training."""
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["student"]["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded tokenizer for {cfg['student']['model_name']}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        tokenizer = None
    
    dataset_name = cfg["data"]["dataset_name"]
    
    if dataset_name == "glue":
        subset = cfg["data"].get("subset", "sst2")
        max_samples = cfg["data"].get("max_samples", None)
        train_data, val_data = load_glue_dataset_simple(subset, max_samples)
    else:
        print("Using fallback dummy dataset...")
        train_data = [{"sentence": f"Sample training text {i}", "label": i % 2} for i in range(100)]
        val_data = [{"sentence": f"Sample validation text {i}", "label": i % 2} for i in range(20)]
    
    print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
    
    train_kd_dataset = KnowledgeDistillationDataset(train_data, tokenizer)
    val_kd_dataset = KnowledgeDistillationDataset(val_data, tokenizer)
    
    train_loader = DataLoader(
        train_kd_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_kd_dataset,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def prepare_datasets(*args, **kwargs):
    """Legacy interface for backward compatibility."""
    if args and isinstance(args[0], dict):
        return load_dataset_for_kd(args[0])
    return None
