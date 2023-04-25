# THIS IS FROM COQUI

"""Set of default text cleaners"""

import re

from unidecode import unidecode
from phonemizer import phonemize

from .pali.pa_si_phonemizer import pali_to_ipa

from .english.abbreviations import abbreviations_en
from .english.number_norm import normalize_numbers as en_normalize_numbers
from .english.time_norm import expand_time_english

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def expand_abbreviations(text, lang="en"):
    if lang == "en":
        _abbreviations = abbreviations_en
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def convert_to_ascii(text):
    return unidecode(text)


def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text


def replace_symbols(text, lang="en"):
    """Replace symbols based on the lenguage tag.

    Args:
      text:
       Input text.
      lang:
        Lenguage identifier. ex: "en", "fr", "pt", "ca".

    Returns:
      The modified text
      example:
        input args:
            text: "si l'avi cau, diguem-ho"
            lang: "ca"
        Output:
            text: "si lavi cau, diguemho"
    """
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    return text


def en_pi_si_phonemize(text):
    ret = ''
    sections = re.split(r'[@]', text)
    pali_subsections = re.findall(r'@([^@]+)@', text)
    for i, subsection in enumerate(sections):
        if (subsection == ''):
            continue
        if (subsection in pali_subsections):
            ret += pali_to_ipa(subsection)
        else:
            trimmed_text = subsection.strip()
            if (trimmed_text in [',', '.']):
              ret += subsection
            else:
              en_phonemization = phonemize(trimmed_text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
              ret += en_phonemization
    return ret
    

def en_training_clean_and_phonemize(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_time_english(text)
    text = en_normalize_numbers(text)
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    text = en_pi_si_phonemize(text)
    return text
