import argparse
import os
import time
import sys
import numpy as np
import readchar
import librosa
import sounddevice as sd
import soundfile as sf

import readchar

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def save_wave_image_as_png(audio_file_path):
    img_save_path = os.path.dirname(audio_file_path)

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
    waveform_image_path = img_save_path + "/waveform.png"
    plt.savefig(waveform_image_path, dpi=300, bbox_inches='tight')
    plt.close()

    return waveform_image_path


def display_available_devices():
    print(sd.query_devices())


def read_metadata_file(file_name):
    lines = []
    with open(file_name, "r") as f:
        lines = f.readlines()
    return lines


def play_audio(audio_file_path):
    save_wave_image_as_png(audio_file_path)
    y, sr = librosa.load(audio_file_path, sr=None)
    sd.play(y, sr)
    sd.wait()


def record_audio(recording_device, sample_rate, audio_file_path):
    print("\nPress 's' to stop recording\n")
    with sf.SoundFile(audio_file_path, mode="w", samplerate=sample_rate, channels=1, subtype='PCM_16') as file:
        def stream_callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            file.write(indata.copy())

        stream = sd.InputStream(device=recording_device, channels=1, samplerate=sample_rate, callback=stream_callback)
        stream.start()
        print("Recording audio...")
        try:
            while True:
                key = readchar.readkey()
                if key == 's':
                    print("Stopped recording.")
                    stream.stop()
                    break
                time.sleep(0.1)
        finally:
            stream.close()
    save_wave_image_as_png(audio_file_path)



def check_if_overwrite_intended(audio_file_path):
    overwrite = True
    if os.path.exists(audio_file_path):
        response = input("File already exists. Overwrite? (y/n): ")
        if response.lower() == 'y':
            os.remove(audio_file_path)
        else:
            overwrite = False
    else:
        os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
    return overwrite


def initiate_transcript_recording(recording_device, sample_rate, audio_file_path):
    overwrite_check = check_if_overwrite_intended(audio_file_path)
    if (not overwrite_check):
        return
    record_audio(recording_device, sample_rate, audio_file_path)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Metadata player recorder tool for extended LJSpeech dataset.", add_help=False)
    parser.add_argument("-l", "--list_devices", action="store_true", help="list to devices for audio recording.")
    args, _ = parser.parse_known_args()

    if (args.list_devices):
        display_available_devices()
        parser.exit(0)

    parser = argparse.ArgumentParser(description="Metadata player recorder tool for extended LJSpeech dataset.")
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to metadata file.")
    parser.add_argument("-s", "--start", type=int, default=1, help="Line number to start at.")
    parser.add_argument("-d", "--device", default=-1, type=int, help="Device # from -l|--list-devices to use for audio recording.")
    parser.add_argument('-r', '--sample_rate', type=int, default=44100, help='Sampling rate to use in recording.')
    args = parser.parse_args()

    recording_device = args.device
    sample_rate = args.sample_rate

    # Open metadata file and read lines
    lines = read_metadata_file(args.filename)

    # Iterate through lines
    i = args.start - 1  # Subtract 1 because list indices start at 0
    while i < len(lines):
        line = lines[i]
        audio_file_path, speaker, transcript = line.strip().split("|")

        # Display line number and transcript
        print(f"Line {i+1}")
        print(f"\n{transcript}\n\n")

        # Check if audio file exists
        if not os.path.exists(audio_file_path):
            print(f"Audio file: {audio_file_path} missing or needs to be recorded!")
        else:
            # Play audio and handle keyboard input
            play_audio(audio_file_path)
        while True:
            key = readchar.readkey()
            if key == ' ':
                if sd.is_playing():
                    sd.stop()
                else:
                    play_audio(audio_file_path)
            elif key == 'n':
                i += 1
                break
            elif key == 'p':
                i -= 1
                break
            elif key == 'r':
                play_audio(audio_file_path)
            elif key == 'x':
                if (recording_device == -1):
                    print("Restart the script and specify the audio device number to use for recording at the command line.")
                else:
                    initiate_transcript_recording(recording_device, sample_rate, audio_file_path)
            elif key == 'l':
                lines = read_metadata_file(args.filename)
                break

        # Clear console
        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
