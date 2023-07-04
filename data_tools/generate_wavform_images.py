import os
import argparse

from pydub import AudioSegment
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def save_spectrogram_as_png(audio_file_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_save_path = output_path[:-3] + "png"

    # Load audio file
    audio, sr = librosa.load(audio_file_path)

    # Generate spectrogram
    spectrogram = np.abs(librosa.stft(audio))

    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()

    # Save spectrogram as image
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return img_save_path


def save_wave_image_as_png(audio_file_path, output_path, spec=False):
    if spec:
        return save_spectrogram_as_png(audio_file_path, output_path)
    else:
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
    parser = argparse.ArgumentParser(description="Generates image wavforms or spectrograms for each audio track for inspection.")
    parser.add_argument("-s", "--source_dir", help="Source directory.", required=True)
    parser.add_argument("-o", "--output_dir", help="Output directory.", required=True)
    parser.add_argument("--spec", help="Generate spectrograms instead of waveform images.", action="store_true")

    args = parser.parse_args()

    src_dir = args.source_dir
    out_dir = args.output_dir
    generate_spec = args.spec

    idx = 1
    for filename in os.listdir(src_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(src_dir, filename)
            output_path = os.path.join(out_dir, filename)
            save_wave_image_as_png(file_path, output_path, spec=generate_spec)
            print(f"{idx} processed: {output_path}")
            idx += 1
