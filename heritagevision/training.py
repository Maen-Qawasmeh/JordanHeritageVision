"""Shared, reproducible training engine for all four independently launched CNNs.

Only training/validation manifests are used for optimization and model selection.
Test data is never evaluated by this module.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LandmarkDataset
from .models import build_model, freeze_backbone, unfreeze_last_blocks, DEFAULT_IMAGE_SIZES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled):
    is_training = optimizer is not None
    model.train(is_training)
    if is_training:
        # Small batches + pretrained CNNs: do not overwrite ImageNet BN stats.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    for images, labels in tqdm(loader, desc='train' if is_training else 'val', leave=False):
        images = images.to(device, non_blocking=(device.type == 'cuda'))
        labels = labels.to(device, non_blocking=(device.type == 'cuda'))
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if is_training:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        amount = labels.size(0)
        total_loss += float(loss.item()) * amount
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        seen += amount
    return total_loss / seen, correct / seen


def save_curves(history, folder):
    for metric in ('accuracy', 'loss'):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history['epoch'], history[f'train_{metric}'], label='Train')
        ax.plot(history['epoch'], history[f'val_{metric}'], label='Validation')
        ax.set(xlabel='Epoch', ylabel=metric.title(), title=f'{metric.title()} per epoch')
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / f'{metric}_curve.png', dpi=150)
        plt.close(fig)


def train_cli(model_name, argv=None):
    parser = argparse.ArgumentParser(description=f'{model_name} supervised transfer learning')
    parser.add_argument('--split-dir', default='data/splits_final')
    parser.add_argument('--output-dir', default=f'outputs/{model_name}')
    parser.add_argument('--epochs', type=int, default=6, help='Head-training epochs. Set to 0 only with --finetune-from.')
    parser.add_argument('--finetune-epochs', type=int, default=0)
    parser.add_argument('--finetune-from', type=Path, default=None,
                        help='Existing best_model.pt. Skip head training and fine-tune from these weights.')
    parser.add_argument('--batch-size', type=int, default=8, help='Start with 8; try 16 if VRAM permits')
    parser.add_argument('--image-size', type=int, default=DEFAULT_IMAGE_SIZES[model_name])
    parser.add_argument('--lr', type=float, default=0.001, help='Head-training learning rate')
    parser.add_argument('--finetune-lr', type=float, default=1e-5)
    parser.add_argument('--last-blocks', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0, help='0 is safest on Windows')
    parser.add_argument('--no-amp', action='store_true', help='Disable CUDA mixed precision')
    args = parser.parse_args(argv)

    if args.epochs < 0 or args.finetune_epochs < 0 or args.epochs + args.finetune_epochs < 1:
        parser.error('Use a positive number of training epochs')
    if args.finetune_from and (args.epochs != 0 or args.finetune_epochs < 1):
        parser.error('With --finetune-from use --epochs 0 --finetune-epochs N')
    if not args.finetune_from and args.epochs < 1:
        parser.error('--epochs must be positive unless --finetune-from is supplied')
    if args.image_size != DEFAULT_IMAGE_SIZES[model_name]:
        parser.error(f'{model_name} expects --image-size {DEFAULT_IMAGE_SIZES[model_name]} in this comparison')
    if args.lr <= 0 or args.finetune_lr <= 0 or args.last_blocks < 1 or args.batch_size < 1:
        parser.error('learning rates and last-blocks must be positive')

    planned_out = Path(args.output_dir)
    if (planned_out / 'checkpoints' / 'best_model.pt').exists() or (planned_out / 'reports' / 'history.json').exists():
        parser.error(f'Output directory already has an experiment: {planned_out}. Use a NEW --output-dir to preserve results.')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_enabled = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    print(f'Model: {model_name} | Device: {device} | AMP: {amp_enabled} | Batch size: {args.batch_size}')
    if device.type != 'cuda':
        print('WARNING: CUDA unavailable. Training on CPU may be slow.')

    split_dir = Path(args.split_dir)
    meta_file = split_dir / 'metadata.json'
    if not meta_file.exists():
        raise SystemExit(f'Missing {meta_file}. Prepare splits before training.')
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    classes = meta['classes']
    manifest_fingerprints = {
        name: fingerprint(split_dir / f'{name}.csv')
        for name in ('train', 'val', 'test')
    }
    loaders = {}
    for name in ('train', 'val'):
        dataset = LandmarkDataset(split_dir / f'{name}.csv', meta['data_dir'], classes,
                                  training=(name == 'train'), image_size=args.image_size)
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=(name == 'train'), num_workers=args.num_workers,
                                   pin_memory=(device.type == 'cuda'))
        print(f'{name}: {len(dataset)} images')

    # When resuming, pretrained=False prevents downloading/reinitializing
    # ImageNet weights; the full pretrained+fine-tuned state comes from ckpt.
    print('Building model ...')
    model = build_model(model_name, len(classes), pretrained=not bool(args.finetune_from))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    out = Path(args.output_dir)
    ckpt_dir, report_dir = out / 'checkpoints', out / 'reports'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'best_model.pt'
    if args.finetune_from and args.finetune_from.resolve() == ckpt_path.resolve():
        parser.error('finetune-from and output checkpoint must be different paths')

    history = {name: [] for name in ('epoch', 'train_loss', 'train_accuracy',
                                     'val_loss', 'val_accuracy', 'stage')}
    best_acc = -1.0
    best_loss = float('inf')
    best_stage = None

    def save_best(stage):
        torch.save({
            'model_state': model.state_dict(),
            'classes': classes,
            'model_name': model_name,
            'image_size': args.image_size,
            'best_val_accuracy': best_acc,
            'best_val_loss': best_loss,
            'stage': stage,
            'manifest_fingerprints': manifest_fingerprints,
            'seed': args.seed,
        }, ckpt_path)

    if args.finetune_from:
        source_ckpt = torch.load(args.finetune_from, map_location=device, weights_only=True)
        if source_ckpt['model_name'] != model_name or source_ckpt['classes'] != classes:
            raise SystemExit('Checkpoint model/class mapping does not match this training run')
        if source_ckpt['image_size'] != args.image_size:
            raise SystemExit('Checkpoint image_size does not match')
        for split in ('train', 'val', 'test'):
            recorded = source_ckpt.get('manifest_fingerprints', {}).get(split)
            if recorded and recorded != manifest_fingerprints[split]:
                raise SystemExit(f'Checkpoint {split}.csv differs from the current split. Stop.')
        if 'manifest_fingerprints' not in source_ckpt:
            print('NOTE: Legacy checkpoint has no split fingerprint. Verify you used data/splits_final.')
        model.load_state_dict(source_ckpt['model_state'])
        best_acc = float(source_ckpt['best_val_accuracy'])
        best_loss = float(source_ckpt.get('best_val_loss', float('inf')))
        best_stage = 'imported_best_checkpoint'
        save_best(best_stage)
        print(f'Imported prior validation best: {best_acc:.3%}. Saved independent copy in {ckpt_path}.')
    else:
        freeze_backbone(model, model_name)

    def run_stage(stage, epochs, lr):
        nonlocal best_acc, best_loss, best_stage
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=1e-4
        )
        print(f'\n== {stage} | epochs={epochs} | learning rate={lr:g} ==')
        for _ in range(epochs):
            number = len(history['epoch']) + 1
            train_loss, train_acc = run_epoch(model, loaders['train'], optimizer, loss_fn,
                                               device, scaler, amp_enabled)
            val_loss, val_acc = run_epoch(model, loaders['val'], None, loss_fn,
                                           device, scaler, amp_enabled)
            print(f'Epoch {number} ({stage}): train acc={train_acc:.3f}, '
                  f'val acc={val_acc:.3f}, val loss={val_loss:.3f}')
            for field, value in [
                ('epoch', number), ('train_loss', train_loss), ('train_accuracy', train_acc),
                ('val_loss', val_loss), ('val_accuracy', val_acc), ('stage', stage),
            ]:
                history[field].append(value)
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                best_acc, best_loss, best_stage = val_acc, val_loss, stage
                save_best(stage)
                print('Saved best validation checkpoint.')
            (report_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            save_curves(history, report_dir)

    if args.epochs:
        run_stage('frozen_backbone', args.epochs, args.lr)
    if args.finetune_epochs:
        # Fine-tune from best head weights, never simply the final epoch.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        unfreeze_last_blocks(model, model_name, last_n=args.last_blocks)
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Unfrozen trainable parameters: {count:,}')
        run_stage('fine_tuning', args.finetune_epochs, args.finetune_lr)

    (report_dir / 'config.json').write_text(json.dumps({
        'model_name': model_name, 'classes': classes, 'split_dir': str(split_dir.resolve()),
        'manifest_fingerprints': manifest_fingerprints, 'image_size': args.image_size,
        'batch_size': args.batch_size, 'head_epochs': args.epochs,
        'finetune_epochs': args.finetune_epochs,
        'finetune_from': str(args.finetune_from) if args.finetune_from else None,
        'head_lr': args.lr, 'finetune_lr': args.finetune_lr,
        'amp': amp_enabled, 'seed': args.seed, 'best_stage': best_stage,
        'best_val_accuracy': best_acc,
        'best_val_loss': best_loss if math.isfinite(best_loss) else None,
    }, indent=2), encoding='utf-8')
    print(f'\nBest validation accuracy: {best_acc:.3%} | stage: {best_stage}')
    print(f'Checkpoint: {ckpt_path.resolve()}')
