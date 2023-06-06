import os
import sys
import argparse
import json

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_dir)

from text.symbols import symbols 

def _read_metadata_file(file_name) -> str:
    lines = []
    with open(file_name, "r") as f:
        lines = f.readlines()
    return lines


def _init_symbol_counts_table() -> dict:
    symbol_dict = {}
    for symbol in symbols:
        symbol_dict[symbol] = 0    

    return symbol_dict


def _count_occurances_in_line(i, line, symbol_dict):
    for sym in line:
        try:
            symbol_dict[sym] += 1
        except KeyError as e:
            print(f"Error line[{i}] sym[{sym}] not found")


def _dump_csv_format(symbol_dict):
    for i, k in enumerate(symbol_dict):
        print(f"{k}|{symbol_dict[k]}")


def main():
    parser = argparse.ArgumentParser(description="Metadata ipa table information for extended LJSpeech dataset.", add_help=False)
    parser.add_argument("-f", "--filename", type=str, required=True, help="Path to metadata file.")
    parser.add_argument("-s", "--sid", type=int, default=-1, help="Specifies a SID to process on otherwise all speakers are processed.")
    parser.add_argument("--json", action="store_true", help="Dump JSON format to stdout.")
    parser.add_argument("--csv", action="store_true", help="Dump CSV with | delimiter to stdout.")
    parser.add_argument("--verbose", action="store_true", help="Show processing details.")

    args = parser.parse_args()

    symbol_dict = _init_symbol_counts_table()
    lines = _read_metadata_file(args.filename)

    i = 0
    sid = None
    while i < len(lines):
        line = lines[i]
        i += 1
        audio_file_path, speaker, transcript = line.strip().split("|")
        if (args.sid > -1):
            sid = int(speaker)
            if (sid is not args.sid):
                continue
        _count_occurances_in_line(i, transcript, symbol_dict)
        if (args.verbose):
            print(f"Line {i+1} | ")

    if (args.json):
        json.dump(symbol_dict, fp=sys.stdout, indent=2)

    if (args.csv):
        _dump_csv_format(symbol_dict)


if __name__ == "__main__":
    main()
