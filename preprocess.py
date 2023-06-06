import argparse
import text
from tqdm import tqdm

from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--replace_ext", action="store_true")
  parser.add_argument("--out_extension", default="cleaned.csv")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["en_training_clean_and_phonemize"])

  args = parser.parse_args()
    
  print(f"Using args: {args}")
  min_pho_len = None
  max_pho_len = 0
  for filelist in args.filelists:
    filepaths_and_text = load_filepaths_and_text(filelist)
    total_lines = len(filepaths_and_text)
    print(f"preprocess: [{total_lines}] {filelist}")
    if (args.replace_ext):
      new_filelist = filelist[:-3] + args.out_extension
    else:
      new_filelist = filelist + "." + args.out_extension
    print(f"Saving to: {new_filelist}")
    with open(new_filelist, "w", encoding="utf-8") as f:
      with tqdm(total=total_lines) as pbar:
        for i in range(total_lines):
          replace_line_entry = filepaths_and_text[i]
          original_text = filepaths_and_text[i][args.text_index]
          cleaned_text = text._clean_text(original_text, args.text_cleaners)
          replace_line_entry[args.text_index] = cleaned_text
          pho_len = len(cleaned_text)
          if (min_pho_len is None):
            min_pho_len = pho_len
          min_pho_len = min(min_pho_len, pho_len)
          max_pho_len = max(max_pho_len, pho_len)
          f.writelines("|".join(replace_line_entry) + "\n")
          pbar.update()
  print(f"pho lines> min({min_pho_len}), max({max_pho_len})")