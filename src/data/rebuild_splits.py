"""
Rebuild dataset splits with stratified sampling to ensure all classes 
appear in train/val/test sets. This improves recall by ensuring the 
model sees all classes during training AND can be evaluated on them.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

from src.config import get_settings

# Configuration defaults (resolved per-call from centralized settings)
runtime = get_settings().runtime
_DEFAULT_SOURCE = Path(runtime.source_dataset_dir)
_DEFAULT_OUTPUT = Path(runtime.stratified_output_dir)

# Module-level (overridden per-call in main())
SOURCE_DIR = _DEFAULT_SOURCE
OUTPUT_DIR = _DEFAULT_OUTPUT

def get_classes_in_label(label_path):
    """Extract unique class IDs from a label file."""
    classes = set()
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return classes


def collect_all_images():
    """Collect all images from train/val/test splits."""
    all_images = []
    seen = set()
    
    for split in ['train', 'val', 'valid', 'test']:
        candidates = [
            (SOURCE_DIR / 'images' / split, SOURCE_DIR / 'labels' / split),
            (SOURCE_DIR / split / 'images', SOURCE_DIR / split / 'labels'),
        ]
        
        for img_dir, label_dir in candidates:
            if not img_dir.exists():
                continue
            
            for img_path in img_dir.glob('*'):
                if img_path in seen or img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue

                seen.add(img_path)
                label_path = label_dir / f"{img_path.stem}.txt"
                classes = get_classes_in_label(label_path)
                all_images.append({
                    'img_path': img_path,
                    'label_path': label_path,
                    'classes': classes,
                    'name': img_path.name
                })
    
    return all_images


def stratified_split(images, train_ratio, val_ratio, test_ratio):
    """
    Stratified split ensuring each class appears in all splits.
    Priority: ensure at least 1 sample per class in each split.
    """
    # Group images by their "primary" class (first class, or rarest class they contain)
    class_to_images = defaultdict(list)
    
    # Count class frequencies
    class_counts = defaultdict(int)
    for img in images:
        for c in img['classes']:
            class_counts[c] += 1
    
    # Assign each image to its rarest class for stratification
    for img in images:
        if img['classes']:
            rarest_class = min(img['classes'], key=lambda c: class_counts[c])
            class_to_images[rarest_class].append(img)
        else:
            # Images with no labels go to a special bucket
            class_to_images[-1].append(img)
    
    train_set = []
    val_set = []
    test_set = []
    
    # For each class, distribute images
    for class_id, class_images in class_to_images.items():
        random.shuffle(class_images)
        n = len(class_images)
        
        if n >= 3:
            # Ensure at least 1 in each split
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * test_ratio))
            n_train = n - n_val - n_test
            
            # If we don't have enough for train, adjust
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - 1) // 2)
                n_test = n - n_train - n_val
        elif n == 2:
            # Put 1 in train, 1 in val (can't cover test)
            n_train, n_val, n_test = 1, 1, 0
        else:
            # Single image goes to train
            n_train, n_val, n_test = 1, 0, 0
        
        train_set.extend(class_images[:n_train])
        val_set.extend(class_images[n_train:n_train + n_val])
        test_set.extend(class_images[n_train + n_val:n_train + n_val + n_test])
    
    return train_set, val_set, test_set


def copy_files(images, split_name, output_dir):
    """Copy image and label files to output directory."""
    img_out = output_dir / split_name / 'images'
    label_out = output_dir / split_name / 'labels'
    
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    
    for img in images:
        # Copy image
        shutil.copy2(img['img_path'], img_out / img['img_path'].name)
        
        # Copy label if exists
        if img['label_path'].exists():
            shutil.copy2(img['label_path'], label_out / img['label_path'].name)


def analyze_split(images, name):
    """Analyze class distribution in a split."""
    class_counts = defaultdict(int)
    for img in images:
        for c in img['classes']:
            class_counts[c] += 1
    
    print(f"\n{name}: {len(images)} images, {len(class_counts)} classes")
    
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        print(f"  Object counts: min={min_count}, max={max_count}, ratio={max_count/max(min_count,1):.1f}x")
    
    return set(class_counts.keys())


def create_data_yaml(output_dir, original_yaml_path):
    """Create data.yaml for the new dataset."""
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    data['test'] = 'test/images'
    data.pop('path', None)
    
    # Remove roboflow metadata
    if 'roboflow' in data:
        del data['roboflow']
    
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def main(base_dir=None):
    """Rebuild splits. *base_dir* is prepended to the default source/output paths."""
    global SOURCE_DIR, OUTPUT_DIR

    runtime = get_settings().runtime
    random.seed(runtime.random_seed)

    if base_dir is not None:
        base = Path(base_dir)
        SOURCE_DIR = base / Path(runtime.source_dataset_dir)
        OUTPUT_DIR = base / Path(runtime.stratified_output_dir)
    else:
        SOURCE_DIR = Path(runtime.source_dataset_dir)
        OUTPUT_DIR = Path(runtime.stratified_output_dir)

    print("=" * 60)
    print("REBUILDING DATASET WITH STRATIFIED SPLITS")
    print(f"  SOURCE_DIR = {SOURCE_DIR}")
    print(f"  OUTPUT_DIR = {OUTPUT_DIR}")
    print("=" * 60)

    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all images
    print("\nCollecting images from all splits...")
    all_images = collect_all_images()
    print(f"Total images collected: {len(all_images)}")
    
    # Count total classes
    all_classes = set()
    for img in all_images:
        all_classes.update(img['classes'])
    print(f"Total unique classes: {len(all_classes)}")
    
    # Perform stratified split
    print("\nPerforming stratified split...")
    train_set, val_set, test_set = stratified_split(
        all_images,
        runtime.train_ratio,
        runtime.val_ratio,
        runtime.test_ratio,
    )
    
    # Analyze splits
    train_classes = analyze_split(train_set, "TRAIN")
    val_classes = analyze_split(val_set, "VAL")
    test_classes = analyze_split(test_set, "TEST")
    
    # Check coverage
    print("\n" + "=" * 60)
    print("CLASS COVERAGE CHECK")
    print("=" * 60)
    
    missing_in_val = train_classes - val_classes
    missing_in_test = train_classes - test_classes
    
    if missing_in_val:
        print(f"\nWARNING: {len(missing_in_val)} classes missing from val: {sorted(missing_in_val)}")
    else:
        print("\nOK: All training classes present in validation set!")
    
    if missing_in_test:
        print(f"WARNING: {len(missing_in_test)} classes missing from test: {sorted(missing_in_test)}")
    else:
        print("OK: All training classes present in test set!")
    
    # Copy files
    print("\n" + "=" * 60)
    print("COPYING FILES")
    print("=" * 60)
    
    copy_files(train_set, 'train', OUTPUT_DIR)
    print(f"Copied {len(train_set)} images to train/")
    
    copy_files(val_set, 'valid', OUTPUT_DIR)
    print(f"Copied {len(val_set)} images to valid/")
    
    copy_files(test_set, 'test', OUTPUT_DIR)
    print(f"Copied {len(test_set)} images to test/")
    
    # Create data.yaml
    create_data_yaml(OUTPUT_DIR, SOURCE_DIR / 'data.yaml')
    print(f"\nCreated data.yaml at {OUTPUT_DIR / 'data.yaml'}")
    
    print("\n" + "=" * 60)
    print("DONE! New dataset at:", OUTPUT_DIR)
    print("=" * 60)
    print("\nTo use the new dataset, update train.py:")
    print(f"  data='{OUTPUT_DIR}/data.yaml'")


if __name__ == '__main__':
    main()
