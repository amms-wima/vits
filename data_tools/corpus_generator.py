import os
import argparse
import whisper

def transcribe_audio_files(source_directory, output_corpus, whisper_size):
    audio_files = [file for file in os.listdir(source_directory) if file.endswith(".wav")]
    audio_files = sorted(audio_files)
    model = whisper.load_model(whisper_size)

    with open(output_corpus, 'w', encoding='utf-8') as f:
        for audio_file in audio_files:
            file_path = os.path.join(source_directory, audio_file)
            transcribe_options = dict(task="transcribe")
            json = model.transcribe(file_path, **transcribe_options)
            text = json['text'].strip()
            print(f"{audio_file}: {text}")
            corpus_entry = f"{file_path}|sid|{text}\n"
            f.write(corpus_entry)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Whisper ASR Audio Transcription")
    parser.add_argument("-s", "--source_directory", default="./", help="Path to the directory containing the WAV files")
    parser.add_argument("-o", "--output_corpus", default="./rev-eng-corpus.csv", help="Path to the directory containing the WAV files")
    parser.add_argument("--whisper_size", default="small.en", help="Whisper ASR model size")
    args = parser.parse_args()

    # Call the function to transcribe audio files
    transcribe_audio_files(args.source_directory, args.output_corpus, args.whisper_size)
