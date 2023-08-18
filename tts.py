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
import json

import glob
import fnmatch
from tqdm import tqdm

from text.symbols import symbols


logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger()


class TextToSpeech():
    _DEFAULT_CLEANERS = ["en_training_clean_and_phonemize"]
    _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    _REMAIN_PUNC_REGEX = r'(?<=[.:;?!…—\[\]\(\)])\s*'

    _MODELS_CACHE = {}

    _hps: utils.HParams = None
    _config: any = None
    _net_g: SynthesizerTrn = None

    _sid: LongTensor = None


    def __init__(self, config: any):
        self._config = config
        if (self._config.config_path is None):
            if (self._config.verbose):
                logger.debug(f"Using default cleaners: {TextToSpeech._DEFAULT_CLEANERS}")
        else:
            self._hps = utils.get_hparams_from_file(self._config.config_path)
        if (not self._config.no_infer):
            self._prepare_model()
            if (self._config.sid is not None):
                self._sid = LongTensor([self._config.sid]).to(TextToSpeech._DEVICE)


    def synthesize(self, text, concat_audio=None):
        text_segments = self._split_into_segments(text) 
        ipa_text = ""
        for i, text_seg in enumerate(text_segments):
            trimmed_seg = text_seg.strip()
            if (trimmed_seg == ''):
                continue
            seg_audio, ipa_seg = self._synthesize_segment(trimmed_seg)
            pause_dur = TextToSpeech._query_pause_duration(trimmed_seg[-1])
            if (self._config.verbose):
                logger.debug(f"\nsegment: {trimmed_seg}")
                logger.debug(f"\tipa: {ipa_seg}")
                logger.debug(f"\tpause: {pause_dur}")
            concat_audio = self._concat_audio_segment(concat_audio, seg_audio, pause_dur)
            ipa_text += ("" if i == 0 else "\n") + ipa_seg
        return concat_audio, ipa_text


    def _text_to_tensor(self, text):
        if (self._config.read_as_ipa):
            ipa_seg = text
            text_norm = cleaned_text_to_sequence(text)
        else:
            cleaners =  TextToSpeech._DEFAULT_CLEANERS if (self._hps is None) else self._hps.data.text_cleaners
            text_norm, ipa_seg = text_to_sequence(text, cleaners, self._config.ph_backend, self._config.ph_lang)
        if ((self._hps is not None) and (self._hps.data.add_blank)):
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm, ipa_seg

    
    def _synthesize_segment(self, text):
        audio = None
        stn_tst, ipa_seg = self._text_to_tensor(text)
        if (not self._config.no_infer):
            with no_grad():
                x_tst = stn_tst.unsqueeze(0).to(TextToSpeech._DEVICE)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).to(TextToSpeech._DEVICE)
                audio = self._net_g.infer(x_tst, x_tst_lengths, sid=self._sid, 
                        noise_scale=self._config.noise_scale, 
                        noise_scale_w=self._config.noise_scale_w,
                        length_scale=1.0 / self._config.length_scale
                    )[0][0, 0].data.to(TextToSpeech._DEVICE).cpu().numpy()
            del x_tst, x_tst_lengths
        del stn_tst
        return audio, ipa_seg


    def _prepare_model(self):
        cache_key = (self._config.model_path, self._config.config_path)
        self._net_g = TextToSpeech._MODELS_CACHE.get(cache_key)
        if (self._net_g is not None):
            return

        self._net_g = SynthesizerTrn(
                len(symbols),
                self._hps.data.filter_length // 2 + 1,
                self._hps.train.segment_size // self._hps.data.hop_length,
                n_speakers=self._hps.data.n_speakers,
                **self._hps.model
            ).to(TextToSpeech._DEVICE)
        _ = self._net_g.eval()
        _ = utils.load_checkpoint(self._config.model_path, self._net_g, None)
        TextToSpeech._MODELS_CACHE[cache_key] = self._net_g


    def _split_into_segments(self, text: str) -> list[str]:
        segments = re.split(TextToSpeech._REMAIN_PUNC_REGEX, text.strip())
        return segments


    def _concat_audio_segment(self, concat_audio, seg_audio, pause_duration):
        if (self._config.no_infer):
            return concat_audio
        pause_samples = int(pause_duration * self._hps.data.sampling_rate)
        pause_audio = np.zeros(pause_samples)
        if concat_audio is None:
            concat_audio = np.concatenate((seg_audio, pause_audio))
        else:
            concat_audio = np.concatenate((concat_audio, seg_audio, pause_audio))
        return concat_audio


    @staticmethod
    def save_audio_file(sr, wav, path: str, save_as_mp3: bool = False):
        orig_path = path
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        if save_as_mp3:
            path = "/tmp/temp.wav"
        wavf.write(path, sr, wav.astype(np.int16))
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
    def _query_pause_duration(punctuation):
        if punctuation == '.':
            pause_duration = 0.3
        elif punctuation == '?':
            pause_duration = 0.15
        elif punctuation == '!':
            pause_duration = 0.3
        elif punctuation == ':' or punctuation == ';':
            pause_duration = 0.1
        elif punctuation == '…' or punctuation == '—':
            pause_duration = 0.05
        # elif punctuation == '-' :
        #     pause_duration = 0.001
        # elif punctuation == '[' or punctuation == ']' or punctuation == '(' or punctuation == ')':
        #     pause_duration = 0.001
        else:
            pause_duration = 0        
        return pause_duration


class AbstractSourceTextToSpeech():
    _tts_synthesizer: TextToSpeech = None
    _config = None
    _output_file: str = None

    def __init__(self, config, output_file):
        self._config = config
        self._output_file = output_file
        self._tts_synthesizer = TextToSpeech(config)


    def synthesize(self):
        logger.info(f"Processing: {self._output_file if (not self._config.read_as_corpus) else 'corpus'}")
        file_audio = None
        ipa_text = ""
        with self._open_src_stream_as_iterable() as itr:
            for i, text in enumerate(tqdm(itr)):
                text = text.strip()
                if (text == ''):
                    continue
                if (self._config.read_as_corpus):
                    file_audio = None
                    text = self._parse_corpus_entry(text)
                file_audio, ipa_line = self._tts_synthesizer.synthesize(text, file_audio)
                if (self._config.read_as_corpus):
                    self._write_output_files(file_audio, ipa_line)
                else:
                    ipa_text += ("" if i == 0 else "\n") + ipa_line
        if (self._config.verbose and not self._config.read_as_ipa):
            logger.debug(f"Resultant IPA\n:[{ipa_text}]\n")
        if (not self._config.read_as_corpus):
            self._write_output_files(file_audio, ipa_text)
        return self._output_file
    

    def _write_output_files(self, file_audio, ipa_text):
        if (self._config.prepend_sid_in_filename):
            dir, fn  = os.path.split(self._output_file)
            sid = self._config.sid
            self._output_file = os.path.join(dir, f"s{sid}-{fn}")
        logger.info(f"Writing: {self._output_file}")
        if (self._config.save_ipa_file):
            TextToSpeech.save_ipa_file(ipa_text, self._output_file)
        if ((not self._config.no_infer) and (not self._config.test)):
            TextToSpeech.save_audio_file(self._tts_synthesizer._hps.data.sampling_rate, file_audio, self._output_file, self._config.mp3)


    def _parse_corpus_entry(self, line):
        self._output_file, entry_sid, transcript = line.split("|")
        entry_sid = int(entry_sid)
        if (self._config.sid != entry_sid):
            logger.info(f"Switching to SID: {entry_sid}")
            self._config.sid = int(entry_sid)
            self._sid = LongTensor([self._config.sid]).to(TextToSpeech._DEVICE)
        return transcript
    

    def _open_src_stream_as_iterable(self): 
        pass


    def _repunctuate_src_stream_as_iterable_if_reqd(self, iter):
        ret = iter
        if (self._config.repunctuate_short_seg):
            streamlined = ''
            with iter as itr:
                for i, text in enumerate(tqdm(itr)):
                    text = text.strip()
                    if (text == ''):
                        continue
                    punc = ", " if (text[-1] not in ',.;:!?') else ""
                    streamlined += text + punc
            ret = io.StringIO(streamlined)
        return ret


class TextFileToSpeech(AbstractSourceTextToSpeech):
    _src_file: str = None

    def __init__(self, config, src_file, output_file):
        super().__init__(config, output_file)
        self._src_file = src_file


    def _open_src_stream_as_iterable(self): 
        ret = open(self._src_file, 'r')
        ret = self._repunctuate_src_stream_as_iterable_if_reqd(ret)
        return ret


class StandardInputToSpeech(AbstractSourceTextToSpeech):
    def __init__(self, config, output_file):
        super().__init__(config, output_file)


    def _open_src_stream_as_iterable(self): 
        text = sys.stdin.read().strip()
        ret = io.StringIO(text)
        ret = self._repunctuate_src_stream_as_iterable_if_reqd(ret)
        return ret


class TextDirectoryToSpeech():
    _config = None

    def __init__(self, config, root_src_dir, filter, recurse_dirs, root_output_dir):
        self._config = config
        self.root_src_dir = root_src_dir
        self.filter = filter
        self.recurse_dirs = recurse_dirs
        self.root_output_dir = root_output_dir
        self.skip_existing = config.skip_existing

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
                        synth_file = True
                        if (os.path.exists(output_file) and self.skip_existing):
                            synth_file = False
                            logger.info(f"skipping existing: {output_file}")
                        if (synth_file):
                            last_ret_val = self._synthesize_file(src_file, output_file)
        else:
            files = glob.glob(os.path.join(self.root_src_dir, self.filter))
            for src_file in files:
                output_file = self._get_output_filename(src_file)
                last_ret_val = self._synthesize_file(src_file, output_file)
        return last_ret_val
    

    def _synthesize_file(self, src_file, output_file):
        tts_app = TextFileToSpeech(self._config, src_file, output_file)
        return tts_app.synthesize()


    def _get_output_filename(self, src_file):
        rel_file = os.path.relpath(src_file, self.root_src_dir)
        output_file = os.path.join(self.root_output_dir, rel_file)
        output_file = os.path.splitext(output_file)[0] + ".wav"
        return output_file

def _read_cli_config_into_args(args):
    with open(args.cli_config, "r") as f:
        data = f.read()
        config = json.loads(data)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def tts_cli(args):
    if (args.cli_config is not None):
        args = _read_cli_config_into_args(args)
    app = None
    if (((args.input_file is None) and (args.input_dir is None) and (not args.stdin)) or ((args.input_file is not None) and (args.input_dir is not None) and (args.stdin))):
        raise Exception("Specify one of either input_file or input_dir, stdin")
    if ((not args.no_infer) and (args.config_path is None)):
        raise Exception("Specify one of either no_infer or config_path")

    output_name = None
    if (args.output_file is not None):
        output_name = Path(args.output_file)
        output_name.parent.mkdir(parents=True, exist_ok=True)
    elif (args.output_dir is not None):
        output_name = Path(args.output_dir)
        output_name.mkdir(parents=True, exist_ok=True)

    if (args.verbose):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if (args.stdin):
        app = StandardInputToSpeech(args, output_name)    
    elif (args.input_file):
        app = TextFileToSpeech(args, args.input_file, output_name)
    else:
        app = TextDirectoryToSpeech(args ,args.input_dir, args.filter, args.recurse_dirs, output_name)
    return app.synthesize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vits tts')
    parser.add_argument('--cli_config', type=str, help="JSON formatted config file of the CLI args below")
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-c', '--config_path', type=str)
    parser.add_argument('-s', '--sid', type=int)
    parser.add_argument('-if', '--input_file', type=str, help="Source file to TTS processing.")
    parser.add_argument('-id', '--input_dir', type=str, help="Source root directory to start file TTS processing.")
    parser.add_argument('-f', '--filter', type=str, default="*.txt", help="Filter to use for text file selections in directories")
    parser.add_argument('-r', '--recurse_dirs', action="store_true", help="Process directories recursively when -d is specified.")
    parser.add_argument('-of', '--output_file', type=str, help="Audio save-as filename if -t or stdin is specified.")
    parser.add_argument('-od', '--output_dir', type=str, help="Root output directory if -d is specified.")
    parser.add_argument('-ns', '--noise_scale', type=float,default=.667)
    parser.add_argument('-nsw', '--noise_scale_w', type=float,default=0.6)
    parser.add_argument('-ls', '--length_scale', type=float,default=1)
    parser.add_argument('-pbe', '--ph_backend', type=str, default="espeak", help="The phonemizer backend to use.")
    parser.add_argument('-pla', '--ph_lang', type=str, default="en-us", help="The phonemizer language to use.")
    parser.add_argument('--prepend_sid_in_filename', action="store_true")
    parser.add_argument('--test', action="store_true", help="Test everything except saving audio.")
    parser.add_argument('--skip_existing', action="store_true", help="Skip if the audio file already exists.")
    parser.add_argument('--mp3', action="store_true", help="Save as mp3 rather than wav file.")
    parser.add_argument('--no_infer', action="store_true", help="Dont run the inference; used for phonemization only tasks.")
    parser.add_argument('--save_ipa_file', action="store_true", help="Save IPA text file to stage processing.")
    parser.add_argument('--read_as_ipa', action="store_true", help="Process the source as IPA and by-pass cleaning & phonemization.")
    parser.add_argument('--read_as_corpus', action="store_true", help="Process the source as extended LJSpeech format corpus with path|sid|text columns.")
    parser.add_argument('--repunctuate_short_seg', action="store_true", help="Some models have note been trained with short words and need to be repunctuated.")
    parser.add_argument('--stdin', action="store_true", help="Use standard input instead of file source.")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()

    tts_cli(args)
