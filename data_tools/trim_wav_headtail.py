import os
import argparse

from pydub import AudioSegment

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def save_wave_image_as_png(audio_file_path):
    img_save_path = audio_file_path[:-3] + "png"

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


def silence_audio_head_and_tail(file_path, head_pos, tail_pos):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)
    modified_audio = audio[head_pos:-tail_pos]  # Remove first and last 0.35 seconds
    return modified_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trims the first & last specified ms from audio track.")
    parser.add_argument("-s", "--source_dir", help="Source directory.", required=True)
    parser.add_argument("-o", "--output_dir", help="Output directory.", required=True)
    parser.add_argument("-hp", "--head_pos", default=250, type=int, help="Head position.")
    parser.add_argument("-tp", "--tail_pos", default=100, type=int, help="Tail position.")
    parser.add_argument("-swi", "--save_wav_img", action="store_true", help="Create wav images for visual audio inspection.")

    args = parser.parse_args()

    src_dir = args.source_dir
    out_dir = args.output_dir
    head_pos = args.head_pos
    tail_pos = args.tail_pos

    print(f"stripping {head_pos}ms from head; {tail_pos}ms from tail")
    idx = 1
    for filename in os.listdir(src_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(src_dir, filename)
            modified_audio = silence_audio_head_and_tail(file_path, head_pos, tail_pos)
            output_path = os.path.join(out_dir, filename)
            modified_audio.export(output_path, format='wav')
            if (args.save_wav_img):
                save_wave_image_as_png(output_path)
            print(f"{idx} processed: {output_path}")
            idx += 1
