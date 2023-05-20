import os
import argparse


def denoise(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filename_spec = '"{track}.{ext}"'
    cmd = f"demucs -o {out_dir} --filename {filename_spec} --int24 {src_dir}/*.wav"
    print(f"> {cmd}")
    os.system(cmd)
    cmd = f"mv {out_dir}/htdemucs/*.wav {out_dir}/"
    print(f"> {cmd}")
    os.system(cmd)
    cmd = f"rm -r {out_dir}/htdemucs"
    print(f"> {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saves the voice as a separate audio track from non-vocals.")
    parser.add_argument("-s", "--source_dir", help="Source directory.", required=True)
    parser.add_argument("-o", "--output_dir", help="Output directory.", required=True)

    args = parser.parse_args()

    src_dir = args.source_dir
    out_dir = args.output_dir

    denoise(src_dir, out_dir)