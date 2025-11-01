"""
Convert CARLA Speed Sweep Scenes to MP4 Videos
==============================================
Converts each speed folder (20kmh, 40kmh, 60kmh) into an MP4 video.
Requires ffmpeg to be installed and in PATH.
"""

import subprocess
import pathlib
import json
import sys

OUT_BASE = pathlib.Path("_out")
VIDEO_FPS = 20  # Match your FIXED_DT (0.05s = 20 FPS)
VIDEO_QUALITY = 23  # CRF value: 18=high quality, 23=good, 28=lower (lower number = better quality)

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def convert_to_video(run_dir, output_name):
    """
    Convert PNG sequence to MP4 using ffmpeg.
    
    Args:
        run_dir: Path to folder with PNG sequence (000000.png, 000001.png, ...)
        output_name: Name of output MP4 file
    """
    input_pattern = str(run_dir / "%06d.png")  # 000000.png, 000001.png, etc.
    output_path = run_dir / output_name
    
    # ffmpeg command
    # -framerate: input framerate (20 FPS)
    # -i: input pattern
    # -c:v libx264: H.264 codec
    # -crf: quality (18-28, lower = better)
    # -pix_fmt yuv420p: compatibility format
    # -y: overwrite output
    cmd = [
        "ffmpeg",
        "-framerate", str(VIDEO_FPS),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-crf", str(VIDEO_QUALITY),
        "-pix_fmt", "yuv420p",
        "-y",
        str(output_path)
    ]
    
    print(f"  Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Get video file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Created: {output_path.name} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ✗ ffmpeg failed:")
        print(result.stderr[-500:])  # Last 500 chars of error
        return False

def main():
    print("=" * 70)
    print("CARLA Speed Sweep → MP4 Converter")
    print("=" * 70)
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("\n❌ ERROR: ffmpeg not found!")
        print("\nInstall ffmpeg:")
        print("  1. Download from: https://ffmpeg.org/download.html")
        print("  2. Add to PATH, or:")
        print("  3. Install via chocolatey: choco install ffmpeg")
        print("  4. Or scoop: scoop install ffmpeg")
        sys.exit(1)
    
    print("✓ ffmpeg found\n")
    
    # Check output directory
    if not OUT_BASE.exists():
        print(f"❌ ERROR: {OUT_BASE} not found!")
        print("Run speed_sweeps_QUEUE_FIX.py first to generate data.")
        sys.exit(1)
    
    # Find speed folders
    speed_folders = sorted([d for d in OUT_BASE.iterdir() if d.is_dir() and d.name.endswith("kmh")])
    
    if not speed_folders:
        print(f"❌ ERROR: No speed folders found in {OUT_BASE}")
        sys.exit(1)
    
    print(f"Found {len(speed_folders)} speed folders:\n")
    
    success_count = 0
    
    for run_dir in speed_folders:
        speed = run_dir.name  # e.g., "20kmh"
        
        # Check for frames
        frames = sorted(run_dir.glob("*.png"))
        if not frames:
            print(f"⚠️  {speed}: No PNG frames found, skipping")
            continue
        
        print(f"📹 {speed}: Converting {len(frames)} frames...")
        
        # Load metadata for info
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
                print(f"  Target speed: {summary.get('target_speed_kmh')} km/h")
                print(f"  Frames: {summary.get('frames_saved')}/{summary.get('frames_expected')}")
        
        # Convert to video
        video_name = f"{speed}_scene.mp4"
        if convert_to_video(run_dir, video_name):
            success_count += 1
        
        print()
    
    print("=" * 70)
    print(f"COMPLETE: {success_count}/{len(speed_folders)} videos created")
    print("=" * 70)
    
    if success_count == len(speed_folders):
        print("\n✓ All videos created successfully!")
        print(f"\nVideos are in: {OUT_BASE.absolute()}")
        print("  - 20kmh/20kmh_scene.mp4")
        print("  - 40kmh/40kmh_scene.mp4")
        print("  - 60kmh/60kmh_scene.mp4")
    else:
        print(f"\n⚠️  Some videos failed to convert")

if __name__ == "__main__":
    main()