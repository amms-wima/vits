from pathlib import Path
import utils
from models import SynthesizerTrn
import torch
from torch import no_grad, LongTensor
from text import text_to_sequence
import commons
import scipy.io.wavfile as wavf
import argparse
import re
import numpy as np

import pysbd
from text.symbols import symbols
import os
import glob
import fnmatch
from tqdm import tqdm

class TextToSpeech():
    _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    _REMOVE_LIST = ["\"", "'", "“", "”"]
    _REMAIN_PUNC_REGEX = r'(?<=[^A-Z].[;:…])\s+'

    _HPS: utils.HParams = None
    _CLI_CFG: any = None
    _SEGMENTER: pysbd.Segmenter = None
    _NET_G: SynthesizerTrn = None

    _SID: LongTensor = None


    def __init__(self):
        if (TextToSpeech._HPS is None or TextToSpeech._CLI_CFG is None):
            raise Exception("TextToSpeech.init() should be called prior to instantiation!")


    def synthesize(self, text, concat_audio=None):
        text_segments = TextToSpeech._split_into_segments(text) 
        for i, text_seg in enumerate(text_segments):
            cleaned_seg = TextToSpeech._clean_segment(text_seg.strip())
            seg_audio = self._synthesize_segment(cleaned_seg)
            pause_dur = TextToSpeech._query_pause_duration(text_seg[-1])
            if (TextToSpeech._CLI_CFG["verbose"]):
                print(f"{text_seg} [pause: {pause_dur}]")
            concat_audio = TextToSpeech._concat_audio_segment(concat_audio, seg_audio, pause_dur)
        return concat_audio


    @staticmethod
    def init(config: any):
        hps = utils.get_hparams_from_file(args.config_path)
        if (TextToSpeech._HPS is None):
            TextToSpeech._HPS = hps
        if (TextToSpeech._CLI_CFG is None):
            TextToSpeech._CLI_CFG = config
        
        if (TextToSpeech._SEGMENTER is None):
            TextToSpeech._prepare_segmenter()
        if (TextToSpeech._NET_G is None):
            TextToSpeech._prepare_model()

        TextToSpeech._SID = LongTensor([TextToSpeech._CLI_CFG["sid"]]).to(TextToSpeech._DEVICE)


    @staticmethod
    def save_audio_file(wav, path: str, save_as_mp3: bool = False):
        orig_path = path
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        if save_as_mp3:
            path = "/tmp/temp.wav"
        wavf.write(path, TextToSpeech._HPS.data.sampling_rate, wav.astype(np.int16))
        if save_as_mp3:
            orig_path = os.path.splitext(orig_path)[0] + ".mp3"
            os.system(f"ffmpeg -loglevel panic -y -i {path} {orig_path}")
            os.remove(path)


    def _text_to_tensor(self, text):
        text_norm = text_to_sequence(text, TextToSpeech._HPS.data.text_cleaners)
        if (TextToSpeech._HPS.data.add_blank):
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm


    def _synthesize_segment(self, text):
        audio = None
        stn_tst = self._text_to_tensor(text)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(TextToSpeech._DEVICE)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(TextToSpeech._DEVICE)
            audio = TextToSpeech._NET_G.infer(x_tst, x_tst_lengths, sid=TextToSpeech._SID, 
                    noise_scale=TextToSpeech._CLI_CFG["noise_scale"], 
                    noise_scale_w=TextToSpeech._CLI_CFG["noise_scale_w"],
                    length_scale=1.0 / TextToSpeech._CLI_CFG["length_scale"]
                )[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths
        return audio


    @staticmethod
    def _prepare_segmenter():
        TextToSpeech._SEGMENTER = pysbd.Segmenter(language="en", clean=False)
        TextToSpeech._SEGMENTER.language_module.Abbreviation.ABBREVIATIONS.append('ven')
        TextToSpeech._SEGMENTER.language_module.Abbreviation.PREPOSITIVE_ABBREVIATIONS.append('ven')


    @staticmethod
    def _prepare_model():
        TextToSpeech._NET_G = SynthesizerTrn(
                len(symbols),
                TextToSpeech._HPS.data.filter_length // 2 + 1,
                TextToSpeech._HPS.train.segment_size // TextToSpeech._HPS.data.hop_length,
                n_speakers=TextToSpeech._HPS.data.n_speakers,
                **TextToSpeech._HPS.model
            ).to(TextToSpeech._DEVICE)
        _ = TextToSpeech._NET_G.eval()
        _ = utils.load_checkpoint(TextToSpeech._CLI_CFG["model_path"], TextToSpeech._NET_G, None)


    @staticmethod
    def _clean_segment(text):
        for punc in TextToSpeech._REMOVE_LIST:
            ret = text.replace(punc, "")
        return ret


    @staticmethod
    def _split_into_segments(text: str) -> list[str]:
        sentences = TextToSpeech._SEGMENTER.segment(text)
        segments = []
        for sentence in sentences:
            sen_trimmed = sentence.strip()
            subs = re.split(TextToSpeech._REMAIN_PUNC_REGEX, sen_trimmed)
            segments += subs
        return segments


    @staticmethod
    def _query_pause_duration(punctuation):
        if punctuation == '.' or punctuation == ':' or punctuation == ';' or punctuation == '—':
            pause_duration = 0.5
        elif punctuation == '?' or punctuation == '!':
            pause_duration = 1
        elif punctuation == '…':
            pause_duration = 1.25
        else:
            pause_duration = 0        
        return pause_duration


    @staticmethod
    def _concat_audio_segment(concat_audio, seg_audio, pause_duration):
        pause_samples = int(pause_duration * TextToSpeech._HPS.data.sampling_rate)
        pause_audio = np.zeros(pause_samples)
        if concat_audio is None:
            concat_audio = np.concatenate((seg_audio, pause_audio))
        else:
            concat_audio = np.concatenate((concat_audio, seg_audio, pause_audio))
        return concat_audio


class TextFileToSpeech():
    src_file: str = None
    output_file: str = None
    tts_synthesizer: TextToSpeech = None

    def __init__(self, src_file, output_file):
        self.src_file = src_file
        self.output_file = output_file
        self.tts_synthesizer = TextToSpeech()


    def synthesize(self):
        file_audio = None
        with open(self.src_file, 'r') as f:
            for i, text in enumerate(tqdm(f)):
                text = text.strip()
                if (text == ''):
                    continue
                file_audio = self.tts_synthesizer.synthesize(text, file_audio)
        save_as_mp3 = TextToSpeech._CLI_CFG["save_as_mp3"]
        if (TextToSpeech._CLI_CFG["prepend_sid_in_filename"]):
            dir, fn  = os.path.split(self.output_file)
            sid = TextToSpeech._CLI_CFG["sid"]
            self.output_file = os.path.join(dir, f"s{sid}-{fn}")
        TextToSpeech.save_audio_file(file_audio, self.output_file, save_as_mp3)


class TextDirectoryToSpeech():
    def __init__(self, root_src_dir, filter, recurse_dirs, root_output_dir):
        self.root_src_dir = root_src_dir
        self.filter = filter
        self.recurse_dirs = recurse_dirs
        self.root_output_dir = root_output_dir

    def synthesize(self):
        if self.recurse_dirs:
            for dirpath, dirnames, filenames in os.walk(self.root_src_dir):
                rel_dirpath = os.path.relpath(dirpath, self.root_src_dir)
                output_dir = os.path.join(self.root_output_dir, rel_dirpath)
                os.makedirs(output_dir, exist_ok=True)

                for filename in filenames:
                    if fnmatch.fnmatch(filename, self.filter):
                        src_file = os.path.join(dirpath, filename)
                        output_file = self._get_output_filename(src_file)
                        self._synthesize_file(src_file, output_file)
        else:
            files = glob.glob(os.path.join(self.root_src_dir, self.filter))
            for src_file in files:
                output_file = self._get_output_filename(src_file)
                self._synthesize_file(src_file, output_file)


    def _synthesize_file(self, src_file, output_file):
        tts_app = TextFileToSpeech(src_file, output_file)
        tts_app.synthesize()


    def _get_output_filename(self, src_file):
        rel_file = os.path.relpath(src_file, self.root_src_dir)
        filename = os.path.basename(rel_file)
        output_file = os.path.join(self.root_output_dir, rel_file)
        output_file = os.path.splitext(output_file)[0] + ".wav"
        return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vits tts')
    parser.add_argument('-m', '--model_path', type=str, default="./build/G^latest.pth")
    parser.add_argument('-c', '--config_path', type=str, default="./build/config.json")
    parser.add_argument('-s', '--sid', type=int)
    parser.add_argument('-t', '--text_file', type=str, help="Source file to TTS processing.")
    parser.add_argument('-d', '--root_dir', type=str, help="Source root directory to start file TTS processing.")
    parser.add_argument('-f', '--filter', type=str, default="*.txt", help="Filter to use for text file selections in directories")
    parser.add_argument('-r', '--recurse_dirs', action="store_true", help="Process directories recursively when -d is specified.")
    parser.add_argument('-o', '--output_name', type=str, required=True, help="Either a save-as filename or a root output directory if -d is specified.")
    parser.add_argument('-p', '--prepend_sid_in_filename', action="store_true")
    parser.add_argument('-mp3', '--save_as_mp3', action="store_true")
    parser.add_argument('-ns', '--noise_scale', type=float,default=.667)
    parser.add_argument('-nsw', '--noise_scale_w', type=float,default=0.6)
    parser.add_argument('-ls', '--length_scale', type=float,default=1)
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    app = None
    if (((args.text_file is None) and (args.root_dir is None)) or ((args.text_file is not None) and (args.root_dir is not None))):
        raise Exception("Specify either text_file or root_dir but not both")
    output_name = Path(args.output_name)
    if (args.text_file is not None):
        output_name.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_name.mkdir(parents=True, exist_ok=True)

    config = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "sid": args.sid,
        "text_file": args.text_file,
        "output_name": output_name,
        "prepend_sid_in_filename": args.prepend_sid_in_filename,
        "save_as_mp3": args.save_as_mp3,
        "noise_scale": args.noise_scale,
        "noise_scale_w": args.noise_scale_w,
        "length_scale": args.length_scale,
        "verbose": args.verbose,
    }
    TextToSpeech.init(config)

    if (args.text_file):
        app = TextFileToSpeech(args.text_file, output_name)
    else:
        app = TextDirectoryToSpeech(args.root_dir, args.filter, args.recurse_dirs, output_name)
    app.synthesize()
