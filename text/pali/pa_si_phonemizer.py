SPECIAL_MAPPING = {
    "'s": "z",
    "’s": "z",
    "bh": "bˈ",
    "ch": "tʃˈ",
    "ḍh": "d",
    "dh": "ddˈ", 
    "gh": "ɡˈ",
    "jh": "dʒˈ",
    "kh": "kˈ",
    "ph": "p",
    "ṭh": "tˈ",
    "th": "θhˈ",

    # promoted from PALI_IPA_MAP in order to support additional rules
    "j": "d͡ʒ",

    # additional rules based on sinhalese pronunciations
    "aya": "əjˈə",
    "bˈa": "bˈɑːɹ", 
    "da": "dθɑːɹ",   
    "ñā": "ŋjjˈɑːɹ", 
    "vi": "vɪˈ",
}

PALI_IPA_MAP = {
    "a": "ɑː",
    "ā": "ɑːɹ", 
    "c": "t͡ʃ",
    "ḍ": "d", 
    "e": "ɛ",
    "g": "ɡ",
    "ḥ": "h",
    "ī": "iː",
    "i": "ɪ",
    "l": "l̩",
    "ḹ": "li",
    "ḷ": "l",
    "ṁ": "m̩",
    "ṃ": "m̩",
    "ṇ": "n̩",
    "n": "nn",
    "ñ": "ŋ",
    "ṅ": "ŋ",
    "o": "ɔː",  
    "q": "k",
    "ṛ": "ɹ",
    "ṝ": "ɹi",
    "r": "ɹɹ",
    "t": "tθ",
    "ṭ": "t",
    "ū": "uː",
    "u": "ˈuː",
    "x": "ɛk",
    "y": "j",
}

SINHALA_ENDINGS = {
    "ɑː": "ˈə"
}

def pali_to_ipa(pali_text, debug=False):
    pali_text = pali_text.lower()
    if debug:
        print(pali_text)
    pali_text = _process_special_combos(pali_text, debug)
    ipa_text = _process_single_characters(pali_text, debug)
    ipa_text = _finalise_endings(ipa_text, debug)
    return ipa_text


def _process_special_combos(pali_text, debug):
    for k, v in SPECIAL_MAPPING.items():    # Replace special characters first
        pali_text = pali_text.replace(k, v)
        if debug:
            print(f"[{k}->{v}] \t {pali_text}")
    if debug:
        print(f"\n{pali_text}")
    return pali_text


def _process_single_characters(pali_text, debug):
    ipa_text = ""
    for i, char in enumerate(pali_text):
        if (char in PALI_IPA_MAP):
            ipa_text += PALI_IPA_MAP[char]
            if debug:
                print(
                    f"{i}: [{char} -> {PALI_IPA_MAP[char]}] \t\t {ipa_text}"
                )
        else:
            ipa_text += char
            if debug:
                print(f"{i}: [{char} == {char}] \t\t {ipa_text}")
    if debug:
        print(f"\n{ipa_text}")
    return ipa_text


def _finalise_endings(ipa_text, debug):
    splits = ipa_text.split(" ")
    joins = []
    for split in splits:
        for k, v in SINHALA_ENDINGS.items():
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
