import os
import shutil
import argparse

def reorganize_demucs(source_dir, output_dir):
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            subdir_path = os.path.join(root, dir_name)
            vocals_path = os.path.join(subdir_path, "vocals.wav")
            if os.path.exists(vocals_path):
                original_name = os.path.basename(subdir_path)
                output_path = os.path.join(output_dir, f"{original_name}.wav")
                shutil.copy(vocals_path, output_path)
                print(f"Copied {vocals_path} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize DMuCS outputs.")
    parser.add_argument("-s", "--source-dir", help="Source directory (htdemucs).", required=True)
    parser.add_argument("-o", "--output-dir", help="Output directory.", required=True)
    args = parser.parse_args()

    reorganize_demucs(args.source_dir, args.output_dir)
