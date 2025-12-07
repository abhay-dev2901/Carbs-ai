#!/usr/bin/env python3
"""
Script to help expand the dataset by organizing images from downloaded sources
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse

# Mapping of common dataset folder names to your dish names
DISH_MAPPING = {
    'biryani': 'biryani',
    'dal': 'dal',
    'halwa': 'halwa',
    'poha': 'poha',
    'roti': 'roti',
    # Add variations you might find in other datasets
    'Biryani': 'biryani',
    'Dal': 'dal',
    'Halwa': 'halwa',
    'Poha': 'poha',
    'Roti': 'roti',
    'chapati': 'roti',
    'Chapati': 'roti',
    'dal_tadka': 'dal',
    'dal_makhani': 'dal',
    'Dal Tadka': 'dal',
    'Dal Makhani': 'dal',
    'sohan_halwa': 'halwa',
    'gajar_ka_halwa': 'halwa',
    'Sohan Halwa': 'halwa',
    'Gajar Ka Halwa': 'halwa',
    'rasgulla': 'rasgulla',
    'Rasgulla': 'rasgulla',
}

def validate_image(img_path):
    """Validate that image is readable and not corrupted"""
    try:
        img = Image.open(img_path)
        img.verify()
        # Reopen for actual use (verify closes the file)
        img = Image.open(img_path)
        # Check minimum size
        if img.size[0] < 100 or img.size[1] < 100:
            return False, "Image too small"
        return True, None
    except Exception as e:
        return False, str(e)

def organize_images(source_dir, target_dir, dish_mapping=None, min_size=224):
    """
    Organize images from downloaded dataset into your structure
    
    Args:
        source_dir: Directory containing downloaded images (may have subfolders)
        target_dir: Your data/images directory
        dish_mapping: Dict mapping source names to target dish names
        min_size: Minimum image dimension
    """
    if dish_mapping is None:
        dish_mapping = DISH_MAPPING
    
    source = Path(source_dir)
    target = Path(target_dir)
    
    if not source.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    target.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    
    # Check if source has subdirectories (organized by dish)
    subdirs = [d for d in source.iterdir() if d.is_dir()]
    
    if subdirs:
        # Organized structure: source/dish_name/images
        print("Found organized structure (subdirectories)")
        for source_dish_dir in subdirs:
            source_dish_name = source_dish_dir.name
            target_dish = dish_mapping.get(source_dish_name, None)
            
            if target_dish is None:
                print(f"  Skipping unknown dish: {source_dish_name}")
                continue
            
            target_path = target / target_dish
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Count existing images
            existing_count = len(list(target_path.glob("*.jpg")))
            
            # Copy and validate images
            added = 0
            skipped = 0
            for img_file in source_dish_dir.rglob("*.jpg"):
                is_valid, error = validate_image(img_file)
                
                if not is_valid:
                    skipped += 1
                    if skipped <= 5:  # Show first 5 errors
                        print(f"    Skipping {img_file.name}: {error}")
                    continue
                
                # Generate unique filename
                new_name = f"{img_file.stem}_{existing_count + added}.jpg"
                dest_path = target_path / new_name
                
                # Avoid duplicates
                if dest_path.exists():
                    skipped += 1
                    continue
                
                shutil.copy2(img_file, dest_path)
                added += 1
            
            stats[target_dish] = {'added': added, 'skipped': skipped, 'total': existing_count + added}
            print(f"  {target_dish}: Added {added} images, skipped {skipped}, total now: {existing_count + added}")
    
    else:
        # Flat structure: all images in one folder
        print("Found flat structure (all images in one folder)")
        print("Note: You'll need to manually organize or use image classification to sort")
        print(f"Found {len(list(source.rglob('*.jpg')))} images in {source_dir}")
        print("Consider using your trained classifier to auto-organize:")
        print("  python scripts/auto_organize.py <source_dir>")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dish, stat in stats.items():
        print(f"{dish:15} Total: {stat['total']:4d}  Added: {stat['added']:4d}  Skipped: {stat['skipped']:3d}")
    print("="*60)
    
    # Check if targets are met
    print("\nTarget: 200+ images per dish")
    for dish, stat in stats.items():
        if stat['total'] >= 200:
            print(f"✅ {dish}: {stat['total']} images (target met)")
        else:
            print(f"⚠️  {dish}: {stat['total']} images (need {200 - stat['total']} more)")

def count_images(data_dir):
    """Count images in each dish folder"""
    data_path = Path(data_dir)
    print("\nCurrent Dataset Status:")
    print("="*60)
    total = 0
    for dish_dir in sorted(data_path.iterdir()):
        if dish_dir.is_dir():
            count = len(list(dish_dir.glob("*.jpg")))
            total += count
            status = "✅" if count >= 200 else "⚠️"
            print(f"{status} {dish_dir.name:15} {count:4d} images")
    print("="*60)
    print(f"Total: {total} images")
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize images from downloaded dataset")
    parser.add_argument("source_dir", help="Directory containing downloaded images")
    parser.add_argument("--target", default="data/images", help="Target directory (default: data/images)")
    parser.add_argument("--count-only", action="store_true", help="Just count current images")
    
    args = parser.parse_args()
    
    if args.count_only:
        count_images(args.target)
    else:
        print(f"Organizing images from {args.source_dir} to {args.target}")
        organize_images(args.source_dir, args.target)

