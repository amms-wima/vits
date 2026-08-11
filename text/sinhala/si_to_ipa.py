# Referenced sources: 
# https://www.dhammatalks.org/books/ChantingGuide/Section0003.html
# https://www.antvaset.com/c/21hf103jp3 > settings > Google cloud TTS > Eng India > Voice B male
# https://readingfaithfully.org/pali-word-pronunciation-recordings

import sys

MAPPING = {
    # replacements
    "ṭh":    "ṭ",   # silence the 'h'
    "ja":    "ʤɑː",   # retain 'ja' combo but all other a -> ʌ
    "ya":    "yɑː",   # retain 'ya' combo but all other a -> ʌ
    "ll":    "lˌl", 
    "ṭṭ":    "ʈˌʈ", 

    # vowels
    "a":    "ʌ",
    "ʌi":    "aɪə",   # SI only
    "ā":    "ɑːɑː",
    "æ":    "æ",    # SI only
    "ǣ":    "ææ",   # SI only
    "e":    "ɛ",
    "ē":    "ɛɛ",   # SI only
    "i":    "ɪ",
    "ī":    "ɪɪ",
    "o":    "oː",
    "u":    "uː",
    "ū":    "uːuː",

    # consonants
    "b":    "b",
    "c":    "ʧ",
    "d":    "ð",
    "ḍ":    "d",
    "f":    "f",
    "g":    "ɡ",
    # "ɡh":   "gʰ",   # frequencies are too low for g & ʰ
    "h":    "h",
    "ḥ":    "h",
    "j":    "ʤ",
    "k":    "k",
    "l":    "l",
    "ḹ":    "lˈ",
    "ḷ":    "lˌ",
    "m":    "m",
    "ṁ":    "ng",
    "ṃ":    "ng",
    "n":    "n",
    "ññ":    "nˈjj",
    "ñ":    "nˈ",
    "ṅ":    "ŋˈ",
    "ṇ":    "nˌ",
    "ṉ":    "",     # SI only
    "o":    "ɒ",
    "ō":    "oʊ",   # SI only
    "p":    "p",
    "q":    "k",
    "r":    "ɹ",
    "ṛ":    "ɹuː",  # SI only
    "ṝ":    "ɹɹuː", # SI only
    "ṣ":    "ʃ",
    "s":    "s",
    "ś":    "ʃ",    # SI only
    "ş":    "ʃ",    # SI only
    "t":    "θ",
    "ṭ":    "ʈˌ",
    "ʈ":    "t",
    "v":    "w",
    "w":    "v",
    "x":    "ɛk",
    "y":    "j",
    "z":    "z",
    "'s":   "z",
    "-":   " ",
    "(":   " ",
    ")":   " ",
    "[":   " ",
    "]":   " ",
    "<":   " ",
    ">":   " ",
}

MODIFY_ENDINGS = {
    "ˈ": "",
    "ˌ": "",
    "ɑː": "ˈʌ"
}

def si_to_ipa(si_rs_text, debug=False):
    si_rs_text = si_rs_text.lower()
    if debug:
        print(si_rs_text)
    ipa_text = _apply_si_mappings(si_rs_text, debug)
    ipa_text = _finalise_endings(ipa_text, debug)
    return ipa_text


def _apply_si_mappings(si_rs_text, debug):
    for k, v in MAPPING.items():    # Replace special characters first
        si_rs_text = si_rs_text.replace(k, v)
        if debug:
            print(f"[{k}->{v}] \t {si_rs_text}")
    if debug:
        print(f"\n{si_rs_text}")
    return si_rs_text


def _finalise_endings(ipa_text, debug):
    splits = ipa_text.split(" ")
    joins = []
    for split in splits:
        for k, v in MODIFY_ENDINGS.items():
            if (split.endswith(k)):
                split = split[:-len(k)] + v
                if debug:
                    print(f"[{k}->{v}] \t {split}")
                break
        joins.append(split)
    ipa_text = " ".join(joins)
    if debug:
        print(f"\n{ipa_text}")
    return ipa_text

if __name__ == "__main__":
    si_to_ipa(sys.argv[1], True)
