from pathlib import Path
import utils
from models import SynthesizerTrn
import torch
from torch import no_grad, LongTensor
from text import text_to_sequence, cleaned_text_to_sequence
import commons
import scipy.io.wavfile as wavf
import argparse
import re
import numpy as np

import io
import os
import sys
import logging

import glob
import fnmatch
from tqdm import tqdm

import pysbd
from text.symbols import symbols


logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger()


class TextToSpeech():
    _DEFAULT_CLEANERS = ["en_training_clean_and_phonemize"]
    _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    _REMOVE_LIST = ["\"", "'", "“", "”"]
    _REMAIN_PUNC_REGEX = r'(?<=[^A-Z].[;:…])\s+'

    _HPS: utils.HParams = None
    _CLI_CFG: any = None
    _SEGMENTER: pysbd.Segmenter = None
    _NET_G: SynthesizerTrn = None

    _SID: LongTensor = None


    def __init__(self):
        if (TextToSpeech._CLI_CFG is None):
            raise Exception("TextToSpeech.init() should be called prior to instantiation!")


    def synthesize(self, text, concat_audio=None):
        text_segments = TextToSpeech._split_into_segments(text) 
        ipa_text = ""
        for i, text_seg in enumerate(text_segments):
            cleaned_seg = TextToSpeech._clean_segment(text_seg.strip())
            seg_audio, ipa_seg = self._synthesize_segment(cleaned_seg)
            pause_dur = TextToSpeech._query_pause_duration(text_seg[-1])
            if (TextToSpeech._CLI_CFG.verbose):
                logger.debug(f"{text_seg} [pause: {pause_dur}]")
            concat_audio = TextToSpeech._concat_audio_segment(concat_audio, seg_audio, pause_dur)
            ipa_text += ("" if i == 0 else "\n") + ipa_seg
        return concat_audio, ipa_text


    @staticmethod
    def init(config: any):
        TextToSpeech._CLI_CFG = config
        hps = None
        if (config.config_path is None):
            if (config.verbose):
                logger.debug(f"Using default cleaners: {TextToSpeech._DEFAULT_CLEANERS}")
        else:
            hps = utils.get_hparams_from_file(config.config_path)
        if (TextToSpeech._HPS is None and hps is not None):
            TextToSpeech._HPS = hps
        if (TextToSpeech._SEGMENTER is None):
            TextToSpeech._prepare_segmenter()
        if (TextToSpeech._NET_G is None and not TextToSpeech._CLI_CFG.no_infer):
            TextToSpeech._prepare_model()
            TextToSpeech._SID = LongTensor([TextToSpeech._CLI_CFG.sid]).to(TextToSpeech._DEVICE)


    def _text_to_tensor(self, text):
        if (TextToSpeech._CLI_CFG.read_as_ipa):
            ipa_seg = text
            text_norm = cleaned_text_to_sequence(text)
        else:
            cleaners =  TextToSpeech._DEFAULT_CLEANERS if (TextToSpeech._HPS is None) else TextToSpeech._HPS.data.text_cleaners
            text_norm, ipa_seg = text_to_sequence(text, cleaners)
        if ((TextToSpeech._HPS is not None) and (TextToSpeech._HPS.data.add_blank)):
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm, ipa_seg

    
    def _synthesize_segment(self, text):
        audio = None
        stn_tst, ipa_seg = self._text_to_tensor(text)
        if (not TextToSpeech._CLI_CFG.no_infer):
            with no_grad():
                x_tst = stn_tst.unsqueeze(0).to(TextToSpeech._DEVICE)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).to(TextToSpeech._DEVICE)
                audio = TextToSpeech._NET_G.infer(x_tst, x_tst_lengths, sid=TextToSpeech._SID, 
                        noise_scale=TextToSpeech._CLI_CFG.noise_scale, 
                        noise_scale_w=TextToSpeech._CLI_CFG.noise_scale_w,
                        length_scale=1.0 / TextToSpeech._CLI_CFG.length_scale
                    )[0][0, 0].data.cpu().float().numpy()
            del x_tst, x_tst_lengths
        del stn_tst
        return audio, ipa_seg


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


    @staticmethod
    def save_ipa_file(text: str, path: str):
        path = os.path.splitext(path)[0] + ".ipa"
        with open(path, "w") as f:
            f.write(text)


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
        _ = utils.load_checkpoint(TextToSpeech._CLI_CFG.model_path, TextToSpeech._NET_G, None)


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
        if (TextToSpeech._CLI_CFG.no_infer):
            return concat_audio
        pause_samples = int(pause_duration * TextToSpeech._HPS.data.sampling_rate)
        pause_audio = np.zeros(pause_samples)
        if concat_audio is None:
            concat_audio = np.concatenate((seg_audio, pause_audio))
        else:
            concat_audio = np.concatenate((concat_audio, seg_audio, pause_audio))
        return concat_audio


class AbstractSourceTextToSpeech():
    tts_synthesizer: TextToSpeech = None
    output_file: str = None

    def __init__(self, output_file):
        self.output_file = output_file
        self.tts_synthesizer = TextToSpeech()


    def synthesize(self):
        file_audio = None
        ipa_text = ""
        with self._open_src_stream_as_iterable() as itr:
            for i, text in enumerate(tqdm(itr)):
                text = text.strip()
                if (text == ''):
                    continue
                file_audio, ipa_line = self.tts_synthesizer.synthesize(text, file_audio)
                ipa_text += ("" if i == 0 else "\n") + ipa_line
        if (TextToSpeech._CLI_CFG.verbose and not TextToSpeech._CLI_CFG.read_as_ipa):
            logger.debug(f"Resultant IPA\n:[{ipa_text}]\n")
        mp3 = TextToSpeech._CLI_CFG.mp3
        if (TextToSpeech._CLI_CFG.prepend_sid_in_filename):
            dir, fn  = os.path.split(self.output_file)
            sid = TextToSpeech._CLI_CFG.sid
            self.output_file = os.path.join(dir, f"s{sid}-{fn}")
        if (TextToSpeech._CLI_CFG.save_ipa_file):
            TextToSpeech.save_ipa_file(ipa_text, self.output_file)
        if ((not TextToSpeech._CLI_CFG.no_infer) and (not TextToSpeech._CLI_CFG.test)):
            TextToSpeech.save_audio_file(file_audio, self.output_file, mp3)
        return self.output_file


    def _open_src_stream_as_iterable(self): 
        pass


class TextFileToSpeech(AbstractSourceTextToSpeech):
    src_file: str = None

    def __init__(self, src_file, output_file):
        super().__init__(output_file)
        self.src_file = src_file


    def _open_src_stream_as_iterable(self): 
        return open(self.src_file, 'r')


class StandardInputToSpeech(AbstractSourceTextToSpeech):
    def __init__(self, output_file):
        super().__init__(output_file)


    def _open_src_stream_as_iterable(self): 
        text = sys.stdin.read().strip()
        ret = io.StringIO(text)
        return ret


class TextDirectoryToSpeech():
    def __init__(self, root_src_dir, filter, recurse_dirs, root_output_dir):
        self.root_src_dir = root_src_dir
        self.filter = filter
        self.recurse_dirs = recurse_dirs
        self.root_output_dir = root_output_dir

    def synthesize(self):
        last_ret_val = None
        if self.recurse_dirs:
            for dirpath, dirnames, filenames in os.walk(self.root_src_dir):
                rel_dirpath = os.path.relpath(dirpath, self.root_src_dir)
                output_dir = os.path.join(self.root_output_dir, rel_dirpath)
                os.makedirs(output_dir, exist_ok=True)

                for filename in filenames:
                    if fnmatch.fnmatch(filename, self.filter):
                        src_file = os.path.join(dirpath, filename)
                        output_file = self._get_output_filename(src_file)
                        last_ret_val = self._synthesize_file(src_file, output_file)
        else:
            files = glob.glob(os.path.join(self.root_src_dir, self.filter))
            for src_file in files:
                output_file = self._get_output_filename(src_file)
                last_ret_val = self._synthesize_file(src_file, output_file)
        return last_ret_val
    

    def _synthesize_file(self, src_file, output_file):
        tts_app = TextFileToSpeech(src_file, output_file)
        return tts_app.synthesize()


    def _get_output_filename(self, src_file):
        rel_file = os.path.relpath(src_file, self.root_src_dir)
        output_file = os.path.join(self.root_output_dir, rel_file)
        output_file = os.path.splitext(output_file)[0] + ".wav"
        return output_file


def tts_cli(args):
    app = None
    if (((args.text_file is None) and (args.root_dir is None) and (not args.stdin)) or ((args.text_file is not None) and (args.root_dir is not None) and (args.stdin))):
        raise Exception("Specify one of either text_file or root_dir, stdin")
    if ((not args.no_infer) and (args.config_path is None)):
        raise Exception("Specify one of either no_infer or config_path")

    output_name = None
    if (args.output_file is not None):
        output_name = Path(args.output_file)
        output_name.parent.mkdir(parents=True, exist_ok=True)
    elif (args.output_dir is not None):
        output_name = Path(args.output_dir)
        output_name.mkdir(parents=True, exist_ok=True)

    TextToSpeech.init(args) 

    if (args.verbose):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if (args.stdin):
        app = StandardInputToSpeech(output_name)    
    elif (args.text_file):
        app = TextFileToSpeech(args.text_file, output_name)
    else:
        app = TextDirectoryToSpeech(args.root_dir, args.filter, args.recurse_dirs, output_name)
    return app.synthesize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vits tts')
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-c', '--config_path', type=str)
    parser.add_argument('-s', '--sid', type=int)
    parser.add_argument('-t', '--text_file', type=str, help="Source file to TTS processing.")
    parser.add_argument('-d', '--root_dir', type=str, help="Source root directory to start file TTS processing.")
    parser.add_argument('-f', '--filter', type=str, default="*.txt", help="Filter to use for text file selections in directories")
    parser.add_argument('-r', '--recurse_dirs', action="store_true", help="Process directories recursively when -d is specified.")
    parser.add_argument('-of', '--output_file', type=str, help="Audio save-as filename if -t or stdin is specified.")
    parser.add_argument('-od', '--output_dir', type=str, help="Root output directory if -d is specified.")
    parser.add_argument('-ns', '--noise_scale', type=float,default=.667)
    parser.add_argument('-nsw', '--noise_scale_w', type=float,default=0.6)
    parser.add_argument('-ls', '--length_scale', type=float,default=1)
    parser.add_argument('--prepend_sid_in_filename', action="store_true")
    parser.add_argument('--test', action="store_true", help="Test everything except saving audio.")
    parser.add_argument('--mp3', action="store_true", help="Save as mp3 rather than wav file.")
    parser.add_argument('--no_infer', action="store_true", help="Dont run the inference; used for phonemization only tasks.")
    parser.add_argument('--save_ipa_file', action="store_true", help="Save IPA text file to stage processing.")
    parser.add_argument('--read_as_ipa', action="store_true", help="Process the source as IPA and by-pass cleaning & phonemization.")
    parser.add_argument('--stdin', action="store_true", help="Use standard input instead of file source.")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()

    tts_cli(args)
