import os
import argparse

from pydub import AudioSegment

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def save_wave_image_as_png(audio_file_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_save_path = output_path[:-3] + "png"

    # Load audio file
    audio, sr = sf.read(audio_file_path)

    # Get time axis
    duration = len(audio) / sr
    time = np.linspace(0, duration, len(audio))

    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.grid(True)

    # Save waveform as image
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return img_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Removes the first & last 350ms from audio track.")
    parser.add_argument("-s", "--source_dir", help="Source directory (htdemucs).", required=True)
    parser.add_argument("-o", "--output_dir", help="Output directory.", required=True)

    args = parser.parse_args()

    src_dir = args.source_dir
    out_dir = args.output_dir

    idx = 1
    for filename in os.listdir(src_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(src_dir, filename)
            output_path = os.path.join(out_dir, filename)
            save_wave_image_as_png(file_path, output_path)
            print(f"{idx} processed: {output_path}")
            idx += 1
