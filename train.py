"""
Training script for BldgMF (Building Mesh MeanFlow) - Face-Centric Version.

Usage:
    # Train with real data
    python train.py --data_root /path/to/data --output_dir ./checkpoints

    # Train with synthetic data (for testing)
    python train.py --synthetic --output_dir ./checkpoints

    # Resume training
    python train.py --data_root /path/to/data --resume ./checkpoints/latest.pt
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model.model import FaceCentricMeshMeanFlowNet
from model.loss import MeshPerceptualLoss, MeshGeometricLoss
from trainer import FaceCentricMeshMeanFlowTrainer
from data.data_loading import (
    FaceCentricMeshDataset,
    SyntheticFaceCentricMeshDataset,
    create_dataloader,
)

# Default data paths (Kyoto dataset)
DEFAULT_DATA_ROOT = '../BldgMF_data'


def parse_args():
    parser = argparse.ArgumentParser(description='Train BldgMF Face-Centric model')

    # Data
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory containing obj/tiff/geojson subdirectories')
    parser.add_argument('--mesh_root', type=str, default=None,
                        help='Override mesh directory (default: data_root/obj_unlabeled_2025/100k/2022/kyoto)')
    parser.add_argument('--image_root', type=str, default=None,
                        help='Override image directory (default: data_root/tiff_kyoto_buf1_2025)')
    parser.add_argument('--footprint_root', type=str, default=None,
                        help='Override footprint directory (default: data_root/geojson_kyoto_2025)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--num_synthetic', type=int, default=1000,
                        help='Number of synthetic samples')

    # Model
    parser.add_argument('--max_faces', type=int, default=512,
                        help='Maximum faces per mesh')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--condition_type', type=str, default='footprint',
                        choices=['none', 'footprint', 'image', 'both'],
                        help='Type of conditioning')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps for learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')

    # Loss weights
    parser.add_argument('--lambda_perceptual', type=float, default=0.4,
                        help='Perceptual loss weight')
    parser.add_argument('--lambda_geometric', type=float, default=0.1,
                        help='Geometric loss weight')

    # MeanFlow
    parser.add_argument('--noise_schedule', type=str, default='logit_normal',
                        choices=['logit_normal', 'uniform'],
                        help='Noise schedule')
    parser.add_argument('--r_ratio', type=float, default=0.5,
                        help='Ratio of r != t samples')

    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval (steps)')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save interval (epochs)')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval (epochs)')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    return parser.parse_args()


def setup_experiment(args):
    """Setup experiment directory and logging."""
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(exp_dir / 'logs'))

    print(f"Experiment directory: {exp_dir}")

    return exp_dir, writer


def create_model(args):
    """Create face-centric model based on args."""
    model = FaceCentricMeshMeanFlowNet(
        max_faces=args.max_faces,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        condition_type=args.condition_type,
        dropout=args.dropout,
    )
    return model


def create_dataloaders(args):
    """Create train and validation dataloaders."""
    use_footprint = args.condition_type in ['footprint', 'both']
    use_image = args.condition_type in ['image', 'both']

    if args.synthetic:
        print("Using synthetic data")
        train_dataset = SyntheticFaceCentricMeshDataset(
            num_samples=args.num_synthetic,
            max_faces=args.max_faces,
            use_footprint=use_footprint,
            use_image=use_image,
        )
        val_dataset = SyntheticFaceCentricMeshDataset(
            num_samples=args.num_synthetic // 10,
            max_faces=args.max_faces,
            use_footprint=use_footprint,
            use_image=use_image,
            seed=123,  # Different seed for val
        )
    else:
        if args.data_root is None:
            raise ValueError("--data_root is required when not using --synthetic")

        dataset_kwargs = dict(
            data_root=args.data_root,
            mesh_root=args.mesh_root,
            image_root=args.image_root,
            footprint_root=args.footprint_root,
            max_faces=args.max_faces,
            use_footprint=use_footprint,
            use_image=use_image,
        )

        train_dataset = FaceCentricMeshDataset(
            **dataset_kwargs,
            split='train',
            augment=True,
        )
        val_dataset = FaceCentricMeshDataset(
            **dataset_kwargs,
            split='val',
            augment=False,
        )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def create_optimizer_and_scheduler(model, args, num_training_steps):
    """Create optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Linear warmup then cosine decay
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, num_training_steps - args.warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def train_epoch(trainer, train_loader, scheduler, writer, epoch, global_step, args):
    """Train for one epoch."""
    trainer.model.train()

    epoch_losses = {'total_loss': 0, 'velocity_loss': 0}
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Training step
        loss_dict = trainer.training_step(batch)

        # Update scheduler
        scheduler.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0
            epoch_losses[k] += v
        num_batches += 1
        global_step += 1

        # Logging
        if global_step % args.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            log_str = f"Epoch {epoch} | Step {global_step} | LR {lr:.2e}"
            for k, v in loss_dict.items():
                log_str += f" | {k}: {v:.4f}"
                writer.add_scalar(f'train/{k}', v, global_step)
            writer.add_scalar('train/lr', lr, global_step)
            print(log_str)

    # Average losses
    epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}

    return epoch_losses, global_step


@torch.no_grad()
def validate(trainer, val_loader, writer, epoch, args):
    """Run validation."""
    trainer.model.eval()

    val_losses = {'total_loss': 0, 'velocity_loss': 0}
    num_batches = 0

    for batch in val_loader:
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass only (no optimization)
        face_vertices = batch['face_vertices']  # [B, F, 3, 3]
        face_mask = batch['face_mask']          # [B, F]
        B, F, _, _ = face_vertices.shape

        # Normalize
        face_vertices_flat = face_vertices.reshape(B, -1, 3)
        flat_mask = face_mask.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)
        face_vertices_norm, norm_stats = trainer.normalizer.normalize(face_vertices_flat, flat_mask)
        face_vertices_norm = face_vertices_norm.reshape(B, F, 3, 3)

        # Sample time
        r, t = trainer.sample_time(B, args.device)

        # Create noisy vertices
        epsilon = torch.randn_like(face_vertices_norm)
        z_t = (1 - t.unsqueeze(-1).unsqueeze(-1)) * face_vertices_norm + t.unsqueeze(-1).unsqueeze(-1) * epsilon
        v_gt = epsilon - face_vertices_norm

        # Prepare conditions
        condition_kwargs = {}
        if 'footprint' in batch:
            condition_kwargs['footprint'] = batch['footprint']
            condition_kwargs['footprint_mask'] = batch.get('footprint_mask')
        if 'image' in batch:
            condition_kwargs['image'] = batch['image']

        # Compute loss
        velocity_loss, _ = trainer.compute_meanflow_loss(
            trainer.model, z_t, r, t, v_gt, face_mask, condition_kwargs
        )

        val_losses['velocity_loss'] += velocity_loss.item()
        val_losses['total_loss'] += velocity_loss.item()
        num_batches += 1

    # Average
    val_losses = {k: v / num_batches for k, v in val_losses.items()}

    # Log
    for k, v in val_losses.items():
        writer.add_scalar(f'val/{k}', v, epoch)

    print(f"Validation | Epoch {epoch} | " +
          " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()]))

    return val_losses


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss,
                    exp_dir, is_best=False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }

    # Save latest
    latest_path = exp_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)

    # Save periodic checkpoint
    epoch_path = exp_dir / f'checkpoint_epoch{epoch:04d}.pt'
    torch.save(checkpoint, epoch_path)

    # Save best
    if is_best:
        best_path = exp_dir / 'best.pt'
        torch.save(checkpoint, best_path)
        print(f"Saved best model (val_loss: {best_val_loss:.4f})")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['global_step'], checkpoint.get('best_val_loss', float('inf'))


def main():
    args = parse_args()

    # Setup
    exp_dir, writer = setup_experiment(args)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = create_model(args)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    num_training_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, num_training_steps)

    # Create losses (optional - can be None for basic training)
    perceptual_loss = None  # FaceCentricMeshPerceptualLoss would need separate implementation
    geometric_loss = None   # FaceCentricMeshGeometricLoss would need separate implementation

    # Create trainer
    trainer = FaceCentricMeshMeanFlowTrainer(
        model=model,
        optimizer=optimizer,
        perceptual_loss=perceptual_loss,
        geometric_loss=geometric_loss,
        lambda_perceptual=args.lambda_perceptual,
        lambda_geometric=args.lambda_geometric,
        noise_schedule=args.noise_schedule,
        r_ratio=args.r_ratio,
    )

    # Resume if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()

        # Train
        train_losses, global_step = train_epoch(
            trainer, train_loader, scheduler, writer, epoch, global_step, args
        )

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_losses = validate(trainer, val_loader, writer, epoch, args)

            # Check if best
            is_best = val_losses['total_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total_loss']

            # Save checkpoint
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, best_val_loss,
                    exp_dir, is_best
                )

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, args.num_epochs - 1, global_step, best_val_loss,
        exp_dir, is_best=False
    )

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {exp_dir}")

    writer.close()


if __name__ == '__main__':
    main()
