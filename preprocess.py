import argparse
import text
from tqdm import tqdm
from utils import load_filepaths_and_text

def process_file(filelist, out_extension, text_index, text_cleaners, ph_backend, ph_lang, min_text_len, max_text_len, replace_ext):
    total_lines = 0
    min_pho_len = None
    max_pho_len = 0
    problematic_entries = []

    with open(filelist, "r", encoding="utf-8") as infile:
        for line in infile:
            total_lines += 1

    print(f"preprocess: [{total_lines}] {filelist}")

    if replace_ext:
        new_filelist = filelist[:-3] + out_extension
    else:
        new_filelist = filelist + "." + out_extension

    print(f"Saving to: {new_filelist}")

    with open(filelist, "r", encoding="utf-8") as infile, open(new_filelist, "w", encoding="utf-8") as outfile:
        with tqdm(total=total_lines) as pbar:
            for i, line in enumerate(infile):
                replace_line_entry = line.strip().split("|")
                original_text = replace_line_entry[text_index]
                cleaned_text = text._clean_text(original_text, text_cleaners, ph_backend, ph_lang)
                replace_line_entry[text_index] = cleaned_text
                pho_len = len(cleaned_text)
                if pho_len < min_text_len or pho_len > max_text_len:
                    print(f"Warning: {replace_line_entry[0]}; text length not within required range[{pho_len}]")
                    print(f"phoneme: {cleaned_text}")
                    problematic_entries.append([replace_line_entry[0], pho_len])
                if min_pho_len is None:
                    min_pho_len = pho_len
                min_pho_len = min(min_pho_len, pho_len)
                max_pho_len = max(max_pho_len, pho_len)
                outfile.write("|".join(replace_line_entry) + "\n")
                pbar.update()

    print(f"pho lines> min({min_pho_len}), max({max_pho_len})")
    if problematic_entries:
        print(f"Problematic entries:\n{problematic_entries}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace_ext", action="store_true")
    parser.add_argument("--out_extension", default="cleaned.csv")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
    parser.add_argument("--text_cleaners", nargs="+", default=["en_training_clean_and_phonemize"])
    parser.add_argument('-pbe', '--ph_backend', type=str, default="espeak", help="The phonemizer backend to use.")
    parser.add_argument('-pla', '--ph_lang', type=str, default="en-us", help="The phonemizer language to use.")
    parser.add_argument('-min', '--min_text_len', type=int, default=1, help="Min cleaned text length.")
    parser.add_argument('-max', '--max_text_len', type=int, default=190, help="Max cleaned text length.")

    args = parser.parse_args()

    print(f"Using args: {args}")

    for filelist in args.filelists:
        process_file(filelist, args.out_extension, args.text_index, args.text_cleaners, args.ph_backend,
                     args.ph_lang, args.min_text_len, args.max_text_len, args.replace_ext)
