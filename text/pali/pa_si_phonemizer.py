# Referenced sources: 
# https://www.dhammatalks.org/books/ChantingGuide/Section0003.html
# https://www.antvaset.com/c/21hf103jp3 > settings > Google cloud TTS > Eng India > Voice B male
# https://readingfaithfully.org/pali-word-pronunciation-recordings

MAPPING = {
    # replacements
    "ṭh":    "ṭ",   # silence the 'h'
    "ja":    "ʤɑ",   # retain 'ja' combo but all other a -> ə
    "ya":    "yɑ",   # retain 'ya' combo but all other a -> ə

    # vowels
    "a":    "ə",
    "ā":    "ɑɑ",
    "e":    "ɛ",
    "i":    "ɪ",
    "ī":    "ɪɪ",
    "o":    "oː",
    "u":    "uː",
    "ū":    "uːuː",

    # consonants
    # "b":    "b",
    "c":    "ʧ",
    "d":    "ð",
    "ḍ":    "ɖˌ",
    # "f":    "f",
    "g":    "ɡ",
    # "ɡh":   "gʰ",   # frequencies are too low for g & ʰ
    # "h":    "h",
    "ḥ":    "h",
    "j":    "ʤ",
    # "k":    "k",
    # "l":    "l",
    "ḹ":    "lˈ",
    "ḷ":    "lˌ",
    # "m":    "m",
    # "ṁ":    "mgʰ",
    # "ṃ":    "mɡ",
    "ṁ":    "mˈ",
    "ṃ":    "mˌ",
    # "n":    "n",
    "ññ":    "nˈjj",
    "ñ":    "nˈ",
    "ṅ":    "ŋˈ",
    "ṇ":    "nˌ",
    "o":    "ɒ",
    # "p":    "p",
    "q":    "k",
    "r":    "ɹ",
    # "s":    "s",
    "t":    "θ",
    "ṭ":    "ʈˌ",
    # "v":    "v",
    "w":    "v",
    "x":    "ɛk",
    "y":    "j",
    # "z":    "z",
    "'s":   "z",
}

MODIFY_ENDINGS = {
    "ˈ": "",
    "ˌ": "",
    "ɑ": "ˈə"
}

def pali_to_ipa(pali_text, debug=False):
    pali_text = pali_text.lower()
    if debug:
        print(pali_text)
    ipa_text = _apply_pali_mappings(pali_text, debug)
    ipa_text = _finalise_endings(ipa_text, debug)
    # ipa_text = _remove_tailing_stress(ipa_text, debug)
    return ipa_text


def _apply_pali_mappings(pali_text, debug):
    for k, v in MAPPING.items():    # Replace special characters first
        pali_text = pali_text.replace(k, v)
        if debug:
            print(f"[{k}->{v}] \t {pali_text}")
    if debug:
        print(f"\n{pali_text}")
    return pali_text



# def _remove_tailing_stress(ipa_text, debug):
#     if (ipa_text.endswith('ˈ') or (ipa_text.endswith('ˌ'))):
#         if debug:
#             print(f"removed tailing: {ipa_text[-1:]}")
#         ipa_text = ipa_text[:-1]
#     return ipa_text


def _finalise_endings(ipa_text, debug):
    splits = ipa_text.split(" ")
    joins = []
    for split in splits:
        for k, v in MODIFY_ENDINGS.items():
            if (split.endswith(k)):
                # split = split.replace(k, v)
                split = split[:-len(k)] + v
                if debug:
                    print(f"[{k}->{v}] \t {split}")
                break
        joins.append(split)
    ipa_text = " ".join(joins)
    if debug:
        print(f"\n{ipa_text}")
    return ipa_text
