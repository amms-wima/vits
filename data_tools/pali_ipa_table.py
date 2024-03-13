import os
import sys
import argparse
import json

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_dir)

from text.pali.pa_si_phonemizer import MAPPING

# remove replacements entry
MAPPING.pop("ṭh", None) 
MAPPING.pop("ya", None)
MAPPING.pop("ṭ", None)

def _dump_csv_format(symbol_dict):
    for i, k in enumerate(symbol_dict):
        print(f"{k}|{symbol_dict[k]}")

def append_to_aggr_letters(new_letters, aggr_letters):
    for letter in new_letters:
        if (letter not in aggr_letters):
            aggr_letters += letter
    return aggr_letters


def main():
    parser = argparse.ArgumentParser(description="Metadata ipa table information for extended LJSpeech dataset.", add_help=False)
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to JSON stats file of cleaned corpus.")
    args = parser.parse_args()
    
    corpus_ipa_table = None
    with open(args.filename, "r") as f:
        data = f.read()
        corpus_ipa_table = json.loads(data)

    pali_aggr_letters = ""
    for i, k in enumerate(MAPPING):
        pali_aggr_letters = append_to_aggr_letters(MAPPING[k], pali_aggr_letters)

    pali_ref_table = {}
    for letter in pali_aggr_letters:
        pali_ref_table[letter] = corpus_ipa_table[letter]

    _dump_csv_format(pali_ref_table)


if __name__ == "__main__":
    main()
