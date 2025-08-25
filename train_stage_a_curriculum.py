# -*- coding: utf-8 -*-
"""
Enhanced Stage A training script with curriculum learning support.

This script demonstrates how to use the new curriculum learning capabilities
in the SAFE Stage A trainer.
"""

import torch
from pathlib import Path
import argparse

from safe.models.safe_model import SAFEModel
from safe.training.stage_a import StageATrainer
from safe.data.datasets import create_safe_dataloader
from configs.model_configs import get_config


def create_datasets(use_dummy=True, data_path="./data"):
    """Create datasets for training."""
    
    if use_dummy:
        print("Creating dummy datasets for demonstration...")
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        # Ensure project root is in path for tests import
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from tests.fixtures.mock_datasets import create_curriculum_test_datasets
        
        # Create curriculum-aware test datasets
        curriculum_datasets = create_curriculum_test_datasets()
        
        # Combine datasets for training
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([
            curriculum_datasets["easy"],
            curriculum_datasets["medium"], 
            curriculum_datasets["hard"]
        ])
        
        val_dataset = curriculum_datasets["mixed"]
        
        return train_dataset, val_dataset
    
    else:
        print("Creating real datasets...")
        from safe.data.datasets import AVQADataset, AudioCapsDataset, VQADataset
        from torch.utils.data import ConcatDataset
        
        # Create real datasets (skip AVQA for now, use AudioCaps + VQA)
        try:
            # Load available datasets
            audiocaps_dataset = AudioCapsDataset(data_path=data_path, split="train")
            vqa_train_dataset = VQADataset(data_path=data_path, split="train")
            vqa_val_dataset = VQADataset(data_path=data_path, split="val")
            
            # Use AudioCaps for training, VQA val for validation
            # train_dataset = ConcatDataset([audiocaps_dataset, vqa_train_dataset])
            train_dataset = audiocaps_dataset
            val_dataset = vqa_val_dataset
            
            print(f"✓ Real datasets loaded:")
            print(f"  - Train samples: {len(train_dataset)}")
            print(f"  - Val samples: {len(val_dataset)}")
            
            # Validate dataset sizes
            if len(train_dataset) < 10:
                print(f"Warning: Very small training dataset ({len(train_dataset)} samples)")
                print("Consider downloading more data or using dummy datasets for testing")
                
            if len(val_dataset) < 5:
                print(f"Warning: Very small validation dataset ({len(val_dataset)} samples)")
                print("Validation metrics may not be reliable")
                
            # Minimum size check
            if len(train_dataset) == 0:
                print("Error: No training samples found")
                raise ValueError("Cannot train with empty dataset")
            
        except Exception as e:
            print(f"Warning: Could not load real datasets ({e})")
            print("\n" + "="*60)
            print("REAL DATA SETUP INSTRUCTIONS:")
            print("="*60)
            print("To use real datasets, you need to:")
            print("1. Download AudioCaps dataset to ./data/audiocaps/") 
            print("2. Download VQA v2 dataset to ./data/vqa/")
            print("3. (Optional) Download MUSIC-AVQA dataset to ./data/avqa/")
            print("\nFor demo purposes, falling back to dummy datasets...")
            print("="*60 + "\n")
            return create_datasets(use_dummy=True, data_path=data_path)
        
        return train_dataset, val_dataset


def setup_model(config_name="demo"):
    """Setup SAFE model for training using specified configuration."""
    print("Setting up SAFE model with {} configuration...".format(config_name))
    
    # Get model configuration
    config = get_config(config_name)
    
    print("Configuration: {}".format(config['description']))
    print("Expected VRAM: {}GB".format(config['expected_vram_gb']))
    print("LLM: {}".format(config['llm_model_name']))
    print("Vision: {}".format(config['vision_model_name']))
    print("Audio: {}".format(config['audio_encoder_type']))
    
    # Create SAFE model with configuration
    model = SAFEModel(
        # Base VL config
        llm_model_name=config["llm_model_name"],
        vision_model_name=config["vision_model_name"],
        
        # Audio config
        audio_encoder_type=config["audio_encoder_type"],
        audio_encoder_config=config.get("audio_encoder_config", {}),
        
        # Projector config
        projector_type=config["projector_type"],
        num_audio_tokens=config["num_audio_tokens"],
        projector_config=config.get("projector_config", {}),
        
        # Fusion config
        fusion_type=config["fusion_type"],
        fusion_layer_indices=config.get("fusion_layer_indices"),
        lora_rank=config["lora_rank"],
        fusion_config=config.get("fusion_config", {}),
        
        # Training config
        freeze_base_vl=config["freeze_base_vl"],
        freeze_audio_encoder=config["freeze_audio_encoder"]
    )
    
    print("✓ SAFE model created")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Frozen parameters: {frozen_params:,}")
    print(f"  - Trainable ratio: {trainable_params/total_params:.1%}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="SAFE Stage A Training with Curriculum Learning")
    parser.add_argument("--config", type=str, default="full", 
                       choices=["demo", "full", "multimodal"],
                       help="Model configuration to use")
    parser.add_argument("--curriculum", type=str, 
                       default="configs/curriculum/default_curriculum.yaml",
                       help="Path to curriculum configuration file")
    parser.add_argument("--no-curriculum", action="store_true",
                       help="Disable curriculum learning (use traditional training)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs for traditional training")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/stage_a_curriculum",
                       help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Path to real datasets")
    parser.add_argument("--use-real-data", action="store_true",
                       help="Use real datasets instead of dummy data")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model
    model = setup_model(args.config)
    model.to(device)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(
        use_dummy=not args.use_real_data,
        data_path=args.data_path
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Simple collate function for mock datasets."""
        from safe.data.datasets import _collate_multimodal_batch
        return _collate_multimodal_batch(batch)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for mock datasets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    
    # Training configuration
    training_config = {
        "learning_rate_projector": 1e-4,
        "learning_rate_adapter": 5e-5,
        "weight_decay": 0.01,
        "num_epochs": args.epochs,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "save_steps": 500,
        "eval_steps": 100,
        "logging_steps": 50,
        "output_dir": args.output_dir,
        "early_stopping_patience": 5,
        "retention_tolerance": 0.01  # 1% tolerance for demo
    }
    
    # Initialize trainer
    print("Initializing Stage A trainer...")
    
    curriculum_config = None if args.no_curriculum else args.curriculum
    
    trainer = StageATrainer(
        safe_model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=training_config,
        curriculum_config=curriculum_config
    )
    
    print("✓ Trainer initialized")
    print()
    
    if args.no_curriculum:
        print(f"🚀 Starting traditional Stage A training...")
        print(f"This will train for {args.epochs} epoch(s)")
    else:
        print(f"🎓 Starting Stage A training with curriculum learning...")
        stages = trainer.curriculum_manager.config.get_num_stages()
        print(f"This will progress through {stages} curriculum stages")
        
    print(f"Total training steps: {len(train_loader) * args.epochs if args.no_curriculum else 'Variable (curriculum-controlled)'}")
    
    try:
        # Start training
        final_metrics = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"Final metrics:")
        print(f"  - Audio Accuracy: {final_metrics['audio_accuracy']:.4f}")
        print(f"  - VL Retention: {final_metrics['retention_score']:.4f}")
        print(f"  - Total Loss: {final_metrics['total_loss']:.4f}")
        
        if trainer.use_curriculum:
            summary = trainer.curriculum_manager.get_progress_summary()
            print(f"\nCurriculum Summary:")
            print(f"  - Stages Completed: {summary['current_stage_idx']}/{summary['total_stages']}")
            print(f"  - Total Epochs: {summary['total_epochs']}")
            print(f"  - Final Stage: {summary['current_stage_name']}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print("\nTraining failed: {}".format(str(e)))
        raise


if __name__ == "__main__":
    main()