import os
import argparse

def add_dcm_extension(root_dir, dry_run=False):
    """
    Recursively adds .dcm extension to files in the specified directory 
    if they don't already have it.
    """
    print(f"Scanning directory: {root_dir}")
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Skip if already has .dcm extension (case insensitive)
            if filename.lower().endswith('.dcm'):
                continue
                
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, filename + '.dcm')
            
            if dry_run:
                print(f"[DRY RUN] Would rename: {old_path} -> {new_path}")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Error renaming {old_path}: {e}")
            count += 1
    
    if dry_run:
        print(f"Found {count} files to rename.")
    else:
        print(f"Renamed {count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively add .dcm extension to files.")
    parser.add_argument("directory", help="Root directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without renaming")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.directory):
        add_dcm_extension(args.directory, args.dry_run)
    else:
        print(f"Error: Directory not found: {args.directory}")
