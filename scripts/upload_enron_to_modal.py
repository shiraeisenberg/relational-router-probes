"""One-time upload of Enron corpus to Modal volume.

This script creates a tarball of executive maildir folders only (~50MB vs 450MB full corpus)
and uploads it to a Modal volume for use during extraction.

Usage:
    python scripts/upload_enron_to_modal.py

Prerequisites:
    - Enron corpus extracted to data/enron/maildir/
    - Modal CLI configured (modal setup)
"""

import modal
from pathlib import Path
import subprocess
import os

app = modal.App("enron-upload")


# List of executive folder names to include
EXECUTIVE_FOLDERS = [
    "lay-k", "skilling-j", "fastow-a",  # C-Suite
    "delainey-d", "lavorato-j", "kitchen-l", "buy-r", "haedicke-m",  # VPs
    "kean-s", "shankman-j", "shapiro-r", "whalley-g", "horton-s",
    "beck-s", "bass-e", "kaminski-v", "taylor-m", "jones-t",  # Directors
    "sager-e", "shackleton-s",
    "campbell-l", "arnold-j", "farmer-d", "germany-c", "nemec-g", "heard-m",  # Managers
]


@app.local_entrypoint()
def main():
    """Create tarball of executive folders and upload to Modal volume."""
    
    local_maildir = Path("data/enron/maildir")
    if not local_maildir.exists():
        print("Error: data/enron/maildir not found")
        print("Download the Enron corpus first:")
        print("  curl -O https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz")
        print("  tar -xzf enron_mail_20150507.tar.gz -C data/enron/")
        return
    
    # Check which executive folders exist
    existing_folders = []
    for folder in EXECUTIVE_FOLDERS:
        if (local_maildir / folder).exists():
            existing_folders.append(folder)
    
    print(f"Found {len(existing_folders)}/{len(EXECUTIVE_FOLDERS)} executive folders")
    
    if not existing_folders:
        print("Error: No executive folders found!")
        return
    
    # Create tarball of executive folders only (or use existing)
    tar_path = Path("data/enron/executives_maildir.tar.gz")
    
    if tar_path.exists():
        tar_size_mb = tar_path.stat().st_size / 1024 / 1024
        print(f"Using existing tarball: {tar_path} ({tar_size_mb:.1f} MB)")
    else:
        print(f"Creating tarball of {len(existing_folders)} executive folders...")
        
        # Build tar command (--no-xattrs to skip extended attributes on macOS)
        tar_cmd = ["tar", "--no-xattrs", "-czf", str(tar_path), "-C", str(local_maildir)] + existing_folders
        
        result = subprocess.run(tar_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Try without --no-xattrs flag (for non-BSD tar)
            tar_cmd = ["tar", "-czf", str(tar_path), "-C", str(local_maildir)] + existing_folders
            result = subprocess.run(tar_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error creating tarball: {result.stderr}")
                return
        
        tar_size_mb = tar_path.stat().st_size / 1024 / 1024
        print(f"Tarball created: {tar_path} ({tar_size_mb:.1f} MB)")
    
    # Upload to Modal volume
    print("Uploading to Modal volume 'enron-corpus'...")
    
    try:
        # Create volume if it doesn't exist
        vol = modal.Volume.from_name("enron-corpus", create_if_missing=True)
        
        # Read tarball and upload
        with open(tar_path, 'rb') as f:
            tar_data = f.read()
        
        # Upload to volume
        with vol.batch_upload() as batch:
            batch.put_file(tar_path, "/executives_maildir.tar.gz")
        
        print(f"✓ Uploaded {tar_size_mb:.1f} MB to Modal volume 'enron-corpus'")
        print("")
        print("Next steps:")
        print("  # Run Enron extraction on Modal")
        print("  modal run src/routing/modal_app.py --extract --dataset enron --split train --max-samples 5000")
        print("")
        print("  # You can now delete the local corpus to save space:")
        print("  rm -rf data/enron/maildir")
        print("  rm data/enron/enron_mail_20150507.tar.gz")
        
    except Exception as e:
        print(f"Error uploading to Modal: {e}")
        print("Make sure Modal is configured: modal setup")
        return


if __name__ == "__main__":
    # Allow running without Modal for testing
    import sys
    if "--local-only" in sys.argv:
        # Just create the tarball, don't upload
        local_maildir = Path("data/enron/maildir")
        if not local_maildir.exists():
            print("Error: data/enron/maildir not found")
            sys.exit(1)
        
        existing_folders = [f for f in EXECUTIVE_FOLDERS if (local_maildir / f).exists()]
        print(f"Found {len(existing_folders)} executive folders")
        
        tar_path = Path("data/enron/executives_maildir.tar.gz")
        tar_cmd = ["tar", "-czf", str(tar_path), "-C", str(local_maildir)] + existing_folders
        subprocess.run(tar_cmd, check=True)
        
        tar_size_mb = tar_path.stat().st_size / 1024 / 1024
        print(f"Created: {tar_path} ({tar_size_mb:.1f} MB)")
    else:
        # Run via Modal
        print("Run with: modal run scripts/upload_enron_to_modal.py")

