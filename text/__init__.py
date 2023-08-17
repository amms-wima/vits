""" from https://github.com/keithito/tacotron """
import logging

from text import cleaners
from text.symbols import symbols

logger = logging


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names, backend=None, lang=None):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  cleaned_text = _clean_text(text, cleaner_names, backend, lang)
  sequence = cleaned_text_to_sequence(cleaned_text)
  return sequence, cleaned_text


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  for symbol in cleaned_text:
    try:
      symbol_id = _symbol_to_id[symbol]
      sequence += [symbol_id]
    except:
      logger.warning(f"symbol[{symbol}] not found in symbol table; ignoring in order to continue...")
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names, backend=None, lang=None):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text, backend, lang)
  return text
