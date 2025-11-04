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
from configs.config_manager import ConfigManager, print_experiment_summary


def create_datasets_with_config(config: dict, data_path="./data"):
    """Create datasets for training based on configuration."""
    dataset_config = config.get("dataset", {})
    use_dummy = dataset_config.get("use_dummy_data", True)
    
    if use_dummy:
        print("Creating configured dummy datasets...")
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        # Ensure project root is in path for tests import
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from tests.fixtures.mock_datasets import MockSAFEDataset
        
        # Get dataset sizes from config
        train_size = dataset_config.get("train_size", 100)
        val_size = dataset_config.get("val_size", 50)
        
        # Get data composition settings
        difficulty_dist = dataset_config.get("difficulty_distribution", {
            "easy": 0.4, "medium": 0.4, "hard": 0.2
        })
        modality_dist = dataset_config.get("modality_distribution", {
            "audio_visual": 0.4, "audio_only": 0.3, "visual_only": 0.25, "text_only": 0.05
        })
        
        print(f"  - Train size: {train_size}")
        print(f"  - Val size: {val_size}")
        print(f"  - Difficulty distribution: {difficulty_dist}")
        
        # Create datasets with configured sizes
        from safe.data.curriculum import DifficultyLevel
        
        # Convert string keys to DifficultyLevel enum
        difficulty_enum_dist = {}
        for key, val in difficulty_dist.items():
            if key == "easy":
                difficulty_enum_dist[DifficultyLevel.EASY] = val
            elif key == "medium":
                difficulty_enum_dist[DifficultyLevel.MEDIUM] = val
            elif key == "hard":
                difficulty_enum_dist[DifficultyLevel.HARD] = val
        
        train_dataset = MockSAFEDataset(
            size=train_size,
            difficulty_distribution=difficulty_enum_dist,
            modality_distribution=modality_dist,
            seed=42
        )
        
        val_dataset = MockSAFEDataset(
            size=val_size,
            difficulty_distribution=difficulty_enum_dist,
            modality_distribution=modality_dist,
            seed=43
        )
        
        return train_dataset, val_dataset
    
    else:
        print("Creating real datasets...")
        from safe.data.datasets import AudioCapsDataset, VQADataset, WavCapsDataset
        from torch.utils.data import ConcatDataset

        audio_train_split = dataset_config.get("audio_train_split", "train")
        audio_eval_split = dataset_config.get("audio_eval_split", "val")
        use_vqa_in_train = dataset_config.get("include_vqa_in_train", True)
        use_wavcaps_in_train = dataset_config.get("include_wavcaps_in_train", True)
        wavcaps_split = dataset_config.get("wavcaps_train_split", "train")

        # Create real datasets (skip AVQA for now, prioritize AudioCaps for eval)
        try:
            # Load available datasets
            audiocaps_train = AudioCapsDataset(data_path=data_path, split=audio_train_split)
            audiocaps_eval = AudioCapsDataset(data_path=data_path, split=audio_eval_split)
            vqa_train_dataset = VQADataset(data_path=data_path, split="train")
            vqa_val_dataset = VQADataset(data_path=data_path, split="val")
            wavcaps_train = None
            if use_wavcaps_in_train:
                try:
                    wavcaps_train = WavCapsDataset(data_path=data_path, split=wavcaps_split)
                except Exception as wav_exc:
                    print(f"Warning: Failed to load WavCaps ({wav_exc}). Continuing without it.", flush=True)

            train_parts = [audiocaps_train]
            if wavcaps_train is not None:
                train_parts.append(wavcaps_train)
            if use_vqa_in_train:
                train_parts.append(vqa_train_dataset)

            train_dataset = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]

            # Validation combines AudioCaps (audio metrics) + VQA (VL metrics)
            val_dataset = ConcatDataset([audiocaps_eval, vqa_val_dataset])

            print(f"✓ Real datasets loaded:")
            print(f"  - AudioCaps train samples: {len(audiocaps_train)} (split='{audio_train_split}')")
            if wavcaps_train is not None:
                print(f"  - WavCaps train samples: {len(wavcaps_train)} (split='{wavcaps_split}')")
            if use_vqa_in_train:
                print(f"  - VQA train samples: {len(vqa_train_dataset)}")
            print(f"  - Combined train samples: {len(train_dataset)}")
            print(f"  - AudioCaps eval samples: {len(audiocaps_eval)} (split='{audio_eval_split}')")
            print(f"  - VQA val samples: {len(vqa_val_dataset)}")
            print(f"  - Combined val samples: {len(val_dataset)}")

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
            fallback_dataset_cfg = dict(dataset_config)
            fallback_dataset_cfg["use_dummy_data"] = True
            fallback_config = dict(config)
            fallback_config["dataset"] = fallback_dataset_cfg
            return create_datasets_with_config(fallback_config, data_path=data_path)
        
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
    parser = argparse.ArgumentParser(description="SAFE Stage A Training with Comprehensive Configuration System")
    
    # New unified experiment configuration system
    parser.add_argument("--experiment", type=str, 
                       help="Experiment configuration to use (e.g., 'debug', 'overfit_test', 'retention_baseline')")
    
    # Legacy configuration options (for backward compatibility)
    parser.add_argument("--config", type=str, default="demo", 
                       choices=["demo", "full", "multimodal"],
                       help="Model configuration to use (legacy)")
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
    
    # Configuration utilities
    parser.add_argument("--list-experiments", action="store_true",
                       help="List all available experiment configurations")
    parser.add_argument("--show-config", action="store_true",
                       help="Show configuration details and exit")
    
    args = parser.parse_args()
    
    # Handle configuration utilities
    if args.list_experiments:
        manager = ConfigManager()
        configs = manager.list_available_configs()
        print("\n📋 Available Experiment Configurations:")
        for config_type, names in configs.items():
            if names:
                print(f"  {config_type.title()}: {', '.join(names)}")
        print("\nUsage: python train_stage_a_curriculum.py --experiment <name>")
        return
    
    # Load experiment configuration or use legacy mode
    if args.experiment:
        print(f"🧪 Loading experiment configuration: {args.experiment}")
        manager = ConfigManager()
        config = manager.compose_config(args.experiment)
        warnings = manager.validate_config(config)
        
        if args.show_config:
            manager.print_config_summary(config)
            if warnings:
                print("⚠️  Configuration Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            return
            
        if warnings:
            print("⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
            response = input("Continue with training? (y/N): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return
                
    else:
        print("⚡ Using legacy configuration mode")
        # Use legacy configuration system
        config = {
            "experiment": {
                "name": "legacy",
                "description": "Legacy configuration mode",
                "experiment_type": "legacy"
            },
            "training": {
                "num_epochs": args.epochs,
                "batch_size": args.batch_size,
                "output_dir": args.output_dir,
                "max_eval_batches": None  # No limit in legacy mode
            },
            "dataset": {
                "use_dummy_data": not args.use_real_data,
                "train_size": 500,  # Default legacy size
                "val_size": 250
            },
            "model": get_config(args.config)
        }
    
    args.config = config  # Store config for later use
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model using config
    model_config = config["model"]
    print(f"\n🤖 Setting up model: {model_config['name']}")
    model = SAFEModel(
        # Base VL config
        llm_model_name=model_config["llm_model_name"],
        vision_model_name=model_config["vision_model_name"],
        
        # Audio config
        audio_encoder_type=model_config["audio_encoder_type"],
        audio_encoder_config=model_config.get("audio_encoder_config", {}),
        
        # Projector config
        projector_type=model_config["projector_type"],
        num_audio_tokens=model_config["num_audio_tokens"],
        projector_config=model_config.get("projector_config", {}),
        
        # Fusion config
        fusion_type=model_config["fusion_type"],
        fusion_layer_indices=model_config.get("fusion_layer_indices"),
        lora_rank=model_config["lora_rank"],
        fusion_config=model_config.get("fusion_config", {}),
        
        # Training config
        freeze_base_vl=model_config["freeze_base_vl"],
        freeze_audio_encoder=model_config["freeze_audio_encoder"]
    )
    model.to(device)
    
    print("✓ Model initialized")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable ratio: {trainable_params/total_params:.1%}")
    
    # Create datasets using config
    print(f"\n📊 Creating datasets...")
    train_dataset, val_dataset = create_datasets_with_config(
        config=config,
        data_path=args.data_path
    )
    
    # Create data loaders using config
    from torch.utils.data import DataLoader
    training_config = config["training"]
    batch_size = training_config.get("batch_size", 4)
    
    def collate_fn(batch):
        """Simple collate function for mock datasets."""
        from safe.data.datasets import _collate_multimodal_batch
        return _collate_multimodal_batch(batch)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for mock datasets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"✓ Train loader: {len(train_loader)} batches")
    print(f"✓ Val loader: {len(val_loader)} batches")
    if training_config.get("max_eval_batches"):
        print(f"✓ Evaluation limited to {training_config['max_eval_batches']} batches")
    
    # Use training configuration from config
    training_config = config["training"]
    
    # Initialize trainer
    print("Initializing Stage A trainer...")
    
    # Determine curriculum configuration
    experiment_config = config.get("experiment", {})
    enable_curriculum = experiment_config.get("enable_curriculum", False)
    
    if enable_curriculum and not args.no_curriculum:
        curriculum_config = experiment_config.get("curriculum_config", args.curriculum)
    else:
        curriculum_config = None
    
    trainer = StageATrainer(
        safe_model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=training_config,
        curriculum_config=curriculum_config
    )
    
    print("✓ Trainer initialized")
    print()
    
    # Print experiment summary
    experiment_type = experiment_config.get("experiment_type", "traditional")
    expected_runtime = experiment_config.get("expected_runtime_minutes", "Unknown")
    
    print(f"🚀 Starting {experiment_type} training...")
    print(f"Expected runtime: {expected_runtime} minutes")
    
    if curriculum_config is None:
        num_epochs = training_config.get("num_epochs", 5)
        print(f"Training for {num_epochs} epoch(s)")
    else:
        print(f"Using curriculum learning")
        if hasattr(trainer, 'curriculum_manager') and trainer.curriculum_manager:
            stages = trainer.curriculum_manager.config.get_num_stages()
            print(f"Progressing through {stages} curriculum stages")
    
    # Calculate training steps
    if curriculum_config is None:
        total_steps = len(train_loader) * training_config.get("num_epochs", 5)
        print(f"Total training steps: {total_steps}")
    else:
        print(f"Total training steps: Variable (curriculum-controlled)")
    
    try:
        # Start training
        final_metrics = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print(f"Final metrics:")
        print(f"  - Audio Accuracy: {final_metrics.get('audio_accuracy', 'N/A'):.4f}")
        print(f"  - VL Retention: {final_metrics.get('retention_score', 'N/A'):.4f}")
        print(f"  - Total Loss: {final_metrics.get('total_loss', 'N/A'):.4f}")
        
        if trainer.use_curriculum:
            summary = trainer.curriculum_manager.get_progress_summary()
            print(f"\nCurriculum Summary:")
            print(f"  - Stages Completed: {summary.get('current_stage_idx', 0)}/{summary.get('total_stages', 0)}")
            print(f"  - Total Epochs: {summary.get('total_epochs', 0)}")
            print(f"  - Final Stage: {summary.get('current_stage_name', 'N/A')}")
            
        # Save experiment results
        if args.experiment:
            print(f"\n📝 Experiment '{args.experiment}' completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        if args.experiment:
            print(f"❌ Experiment '{args.experiment}' failed")
        raise


if __name__ == "__main__":
    main()
