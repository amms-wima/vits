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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vits tts')
    parser.add_argument('-m', '--model_path', type=str, default="./build/G_latest.pth")
    parser.add_argument('-c', '--config_path', type=str, default="./build/config.json")
    parser.add_argument('-s', '--sid', type=int)
    parser.add_argument('-t', '--text_file', type=str, required=True)
    parser.add_argument('-o', '--output_name', type=str, default="./test/output.wav")
    parser.add_argument('-ns', '--noise_scale', type=float,default=.667)
    parser.add_argument('-nsw', '--noise_scale_w', type=float,default=0.6)
    parser.add_argument('-ls', '--length_scale', type=float,default=1)
    args = parser.parse_args()

    model_path = args.model_path
    config_path = args.config_path
    output_name = Path(args.output_name)
    output_name.parent.mkdir(parents=True, exist_ok=True)
    sid_val = args.sid
    noise_scale = args.noise_scale
    noise_scale_w = args.noise_scale_w
    length = args.length_scale

    segmenter = pysbd.Segmenter(language="en", clean=False)
    segmenter.language_module.Abbreviation.ABBREVIATIONS.append('ven')
    segmenter.language_module.Abbreviation.PREPOSITIVE_ABBREVIATIONS.append('ven')

    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)


    def _split_using_remaining_puncs(text: str) -> list[str]:
        segments = re.split(r'(?<=[^A-Z].[;:…])\s+', text)
        return segments


    def _split_into_segments(text: str) -> list[str]:
        sentences = segmenter.segment(text)
        segments = []
        for sentence in sentences:
            sen_trimmed = sentence.strip()
            subs = _split_using_remaining_puncs(sen_trimmed)
            segments += subs
        return segments

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
    

    def _concat_audio_segment(file_audio, seg_audio, pause_duration):
        pause_samples = int(pause_duration * hps.data.sampling_rate)
        pause_audio = np.zeros(pause_samples)
        if file_audio is None:
            file_audio = np.concatenate((seg_audio, pause_audio))
        else:
            file_audio = np.concatenate((file_audio, seg_audio, pause_audio))
        return file_audio


    def _synthesize_segment(text):
        audio = None
        stn_tst = get_text(text, hps)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([sid_val]).to(device)
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                length_scale=1.0 / length)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio


    def _clean_segment(text):
        REMOVE_LIST = ["\"", "'", "“", "”"]
        for punc in REMOVE_LIST:
            ret = text.replace(punc, "")
        return ret


    def _synthesize_file():
        file_audio = None
        with open(args.text_file, 'r') as f:
            for i, text in enumerate(f):
                text = text.strip()
                if (text == ''):
                    continue
                text_segments = _split_into_segments(text) 
                for j, text_seg in enumerate(text_segments):
                    cleaned_seg = _clean_segment(text_seg.strip())
                    seg_audio = _synthesize_segment(cleaned_seg)
                    pause_dur = _query_pause_duration(text_seg[-1])
                    print(f"{text_seg} [{pause_dur}]")
                    file_audio = _concat_audio_segment(file_audio, seg_audio, pause_dur)
        wavf.write(output_name, hps.data.sampling_rate, np.array(file_audio))
    
    _synthesize_file()
