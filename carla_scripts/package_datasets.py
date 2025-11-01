"""
Package CARLA Speed Sweep Dataset
==================================
Creates a distributable ZIP archive with:
- All PNG frames (organized by speed)
- MP4 videos (if generated)
- Metadata CSV files
- Summary JSONs
- README with dataset info
"""

import pathlib
import zipfile
import json
import datetime

OUT_BASE = pathlib.Path("_out")
ARCHIVE_NAME = "carla_speed_sweep_dataset"  # Will add timestamp

def create_readme(speed_folders):
    """Generate README with dataset information"""
    
    readme = f"""# CARLA Speed Sweep Dataset

## Overview

This dataset contains controlled-speed driving scenes captured in CARLA simulator.
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Purpose

RGB baseline for event+RGB fusion research. Focus on temporal metrics:
- Frame-accurate timestamps
- Deterministic vehicle motion
- Clean speed control for latency/reaction time measurements

## Dataset Structure

```
"""
    
    for folder in speed_folders:
        speed = folder.name
        frames = list(folder.glob("*.png"))
        video = list(folder.glob("*.mp4"))
        
        readme += f"{speed}/\n"
        readme += f"  ├── metadata.csv          # Frame-by-frame telemetry\n"
        readme += f"  ├── run_summary.json      # QA metrics\n"
        if video:
            readme += f"  ├── {video[0].name}    # MP4 video\n"
        readme += f"  └── *.png                 # {len(frames)} frames (000000-{len(frames)-1:06d})\n\n"
    
    readme += """```

## Specifications

- **Simulator:** CARLA 0.9.15
- **Map:** Town04 (long straights)
- **Vehicle:** Tesla Model 3
- **Camera:** RGB, 1280x720, 90° FOV, front-mounted (x=1.5, z=2.2)
- **Framerate:** 20 FPS (fixed_delta_seconds = 0.05)
- **Duration:** 12 seconds per scene (240 frames)
- **Speeds:** 20, 40, 60 km/h (controlled via pure-pursuit + P-controller)

## Files

### PNG Frames
- Sequential numbering: `000000.png` → `000239.png`
- Format: 1280x720 RGB
- Synchronized to simulation ticks (deterministic)

### metadata.csv
Columns:
- `frame`: Frame index (0-239)
- `sim_time_s`: Simulation time in seconds
- `speed_mps`: Actual vehicle speed (m/s)
- `speed_kmh`: Actual vehicle speed (km/h)
- `target_kmh`: Target speed (20/40/60)
- `error_kmh`: Speed tracking error

### run_summary.json
QA metrics:
- `frames_expected`: Should be 240
- `frames_saved`: Actual frame count
- `match`: Boolean (frames_saved == frames_expected)
- `mean_speed_error_kmh`: Average speed tracking error
- `max_speed_error_kmh`: Maximum speed tracking error

### MP4 Videos (if present)
- Encoded with H.264 (CRF 23)
- 20 FPS (matches source framerate)
- YUV420p pixel format (wide compatibility)

## Use Cases

1. **Event Camera Fusion:** Use as RGB baseline to compare with event streams
2. **Temporal Analysis:** Study frame-accurate reaction times
3. **Latency Benchmarking:** Measure Δt between object appearance and detection
4. **Speed Control Validation:** Analyze controller performance from metadata

## Reproducibility

Dataset generated using queue-based synchronous capture in CARLA:
- Deterministic physics (fixed time steps)
- Blocking image retrieval (no dropped frames)
- Seed-controlled randomness (seed=42)

To reproduce:
1. CARLA 0.9.15 on Town04
2. Use provided `speed_sweeps_QUEUE_FIX.py` script
3. Same configuration parameters

## Citation

If you use this dataset, please cite:
- CARLA Simulator: Dosovitskiy et al., "CARLA: An Open Urban Driving Simulator"
- [Your research project/paper when available]

## Contact

[Add your contact info / project repository]

## License

[Specify license - e.g., CC BY 4.0, MIT, etc.]
Dataset generated using CARLA (MIT License).
"""
    
    return readme

def get_folder_size(folder):
    """Calculate total size of folder in MB"""
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return total / (1024 * 1024)

def main():
    print("=" * 70)
    print("CARLA Speed Sweep → ZIP Packager")
    print("=" * 70)
    
    # Check output directory
    if not OUT_BASE.exists():
        print(f"❌ ERROR: {OUT_BASE} not found!")
        print("Run speed_sweeps_QUEUE_FIX.py first to generate data.")
        return
    
    # Find speed folders
    speed_folders = sorted([d for d in OUT_BASE.iterdir() if d.is_dir() and d.name.endswith("kmh")])
    
    if not speed_folders:
        print(f"❌ ERROR: No speed folders found in {OUT_BASE}")
        return
    
    print(f"✓ Found {len(speed_folders)} speed folders\n")
    
    # Show what will be archived
    print("Dataset contents:")
    total_size = 0
    for folder in speed_folders:
        size = get_folder_size(folder)
        total_size += size
        frames = len(list(folder.glob("*.png")))
        videos = len(list(folder.glob("*.mp4")))
        print(f"  {folder.name}: {frames} frames" + (f", {videos} video" if videos else "") + f" ({size:.1f} MB)")
    
    print(f"\nTotal size: {total_size:.1f} MB")
    
    # Create archive name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = pathlib.Path(f"{ARCHIVE_NAME}_{timestamp}.zip")
    
    print(f"\n📦 Creating archive: {archive_path}")
    
    # Create README
    readme_content = create_readme(speed_folders)
    
    # Create ZIP
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add README
        zipf.writestr("README.md", readme_content)
        print("  ✓ Added README.md")
        
        # Add each speed folder
        for folder in speed_folders:
            speed = folder.name
            print(f"  📁 Adding {speed}/...")
            
            # Add all files
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    arcname = f"{speed}/{file_path.name}"
                    zipf.write(file_path, arcname)
            
            # Count what was added
            frames = len(list(folder.glob("*.png")))
            videos = len(list(folder.glob("*.mp4")))
            print(f"    ✓ {frames} frames, {videos} video(s), metadata")
    
    # Final stats
    archive_size = archive_path.stat().st_size / (1024 * 1024)
    compression_ratio = (1 - archive_size / total_size) * 100
    
    print(f"\n{'=' * 70}")
    print("ARCHIVE CREATED")
    print(f"{'=' * 70}")
    print(f"File: {archive_path.absolute()}")
    print(f"Size: {archive_size:.1f} MB (compressed from {total_size:.1f} MB)")
    print(f"Compression: {compression_ratio:.1f}%")
    print(f"\n✓ Dataset ready for distribution!")

if __name__ == "__main__":
    main()