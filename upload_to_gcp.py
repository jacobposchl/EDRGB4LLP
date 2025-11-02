#!/usr/bin/env python3
"""
Upload preprocessed DSEC data from local Windows machine to GCP Storage.

Usage:
    python upload_to_gcp.py --local-root ./processed/v1 --gcp-bucket cmpm-bucket --gcp-prefix event-rgb/processed/v1

Before running:
    1. Install: pip install google-cloud-storage
    2. Authenticate: gcloud auth application-default login
    3. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account key JSON
"""
import os
import argparse
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm


def get_all_files(root: Path):
    """Recursively get all files under root directory."""
    files = []
    for item in root.rglob('*'):
        if item.is_file():
            files.append(item)
    return files


def upload_directory_to_gcs(local_root: Path, bucket_name: str, gcp_prefix: str, dry_run: bool = False):
    """
    Upload entire directory tree to GCS, preserving structure.
    
    Args:
        local_root: Local directory to upload (e.g., ./processed/v1)
        bucket_name: GCS bucket name (e.g., cmpm-bucket)
        gcp_prefix: Prefix in bucket (e.g., event-rgb/processed/v1)
        dry_run: If True, only print what would be uploaded
    """
    if not local_root.exists():
        raise FileNotFoundError(f"Local root does not exist: {local_root}")
    
    # Get all files to upload
    all_files = get_all_files(local_root)
    
    if not all_files:
        print(f"[warn] No files found in {local_root}")
        return
    
    print(f"[info] Found {len(all_files)} files to upload")
    
    if dry_run:
        print("[info] DRY RUN - showing what would be uploaded:")
        for local_file in all_files[:10]:  # Show first 10
            rel_path = local_file.relative_to(local_root)
            gcs_path = f"{gcp_prefix}/{rel_path}".replace("\\", "/")
            print(f"  {local_file} -> gs://{bucket_name}/{gcs_path}")
        if len(all_files) > 10:
            print(f"  ... and {len(all_files) - 10} more files")
        return
    
    # Initialize GCS client
    print(f"[info] Connecting to GCS bucket: {bucket_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Upload files with progress bar
    uploaded = 0
    skipped = 0
    failed = 0
    
    with tqdm(total=len(all_files), desc="Uploading", unit="file") as pbar:
        for local_file in all_files:
            try:
                # Calculate relative path and GCS destination
                rel_path = local_file.relative_to(local_root)
                # Convert Windows backslashes to forward slashes for GCS
                gcs_path = f"{gcp_prefix}/{rel_path}".replace("\\", "/")
                
                # Create blob
                blob = bucket.blob(gcs_path)
                
                # Check if already exists (optional: add --force flag to skip this)
                if blob.exists():
                    # Compare file sizes to see if we need to re-upload
                    local_size = local_file.stat().st_size
                    blob.reload()  # Refresh metadata
                    if blob.size == local_size:
                        skipped += 1
                        pbar.set_postfix({"uploaded": uploaded, "skipped": skipped, "failed": failed})
                        pbar.update(1)
                        continue
                
                # Upload file
                blob.upload_from_filename(str(local_file))
                uploaded += 1
                pbar.set_postfix({"uploaded": uploaded, "skipped": skipped, "failed": failed})
                
            except Exception as e:
                failed += 1
                print(f"\n[error] Failed to upload {local_file}: {e}")
                pbar.set_postfix({"uploaded": uploaded, "skipped": skipped, "failed": failed})
            
            pbar.update(1)
    
    print(f"\n[done] Upload complete!")
    print(f"  Uploaded: {uploaded} files")
    print(f"  Skipped:  {skipped} files (already exist with same size)")
    print(f"  Failed:   {failed} files")
    print(f"\n[info] View in GCS: https://console.cloud.google.com/storage/browser/{bucket_name}/{gcp_prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload preprocessed DSEC data to Google Cloud Storage"
    )
    parser.add_argument(
        "--local-root",
        type=str,
        default="./processed/v1",
        help="Local directory containing processed data (default: ./processed/v1)"
    )
    parser.add_argument(
        "--gcp-bucket",
        type=str,
        default="cmpm-bucket",
        help="GCS bucket name (default: cmpm-bucket)"
    )
    parser.add_argument(
        "--gcp-prefix",
        type=str,
        default="event-rgb/processed/v1",
        help="Prefix path in GCS bucket (default: event-rgb/processed/v1)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if files already exist (not yet implemented)"
    )
    
    args = parser.parse_args()
    
    local_root = Path(args.local_root)
    
    print("=" * 60)
    print("GCP Upload Configuration")
    print("=" * 60)
    print(f"Local root:  {local_root.absolute()}")
    print(f"GCS bucket:  {args.gcp_bucket}")
    print(f"GCS prefix:  {args.gcp_prefix}")
    print(f"Dry run:     {args.dry_run}")
    print("=" * 60)
    print()
    
    if not args.dry_run:
        response = input("Proceed with upload? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("[info] Upload cancelled")
            return
    
    upload_directory_to_gcs(
        local_root=local_root,
        bucket_name=args.gcp_bucket,
        gcp_prefix=args.gcp_prefix,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
