"""Text normalization shared by blocking and feature engineering.

Design (driven by EDA on the training data):
  * lowercase + strip accents (non-ASCII strings only, for speed)
  * split letter/digit boundaries so "7704D" == "7704 d" and "10Th" == "10 th"
  * split on any non-alphanumeric character (handles &, -, (), @, #, commas)
  * strip leading zeros from numbers ("01601" -> "1601")
  * drop placeholder tokens such as <NULL>
No hand-written abbreviation or country-specific dictionaries: the pipeline must work
unchanged on countries not seen in training (France).
"""
import re
import unicodedata

from .translit import skeleton_word, transliterate

_DIGIT_LETTER = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_NULLS = {"null", "nan", "none"}


def clean_tokens(text) -> list:
    """Normalize a raw string and return its list of tokens (order preserved)."""
    if not isinstance(text, str) or not text:
        return []
    s = text.lower()
    if not s.isascii():
        # Indian scripts -> latin first (before accent stripping removes their vowel signs)
        s = transliterate(unicodedata.normalize("NFC", s))
    if not s.isascii():
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _DIGIT_LETTER.sub(" ", s)
    out = []
    for t in _NON_WORD.split(s):
        if not t or t in _NULLS:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(t)
    return out


def name_block_tokens(name) -> list:
    """Blocking keys from a business name.

    Besides the words themselves we add the name squashed without spaces, and squashed
    without its last word. This links website / social-handle style names seen in the data
    ("@piedmonthorizon", "kerrwilliams.com") to "Piedmont Horizon LLC", "Kerr Williams"
    without any hand-written list of legal suffixes.
    """
    toks = clean_tokens(name)
    keys = set(toks)
    if len(toks) >= 2:
        keys.add("".join(toks))
    if len(toks) >= 3:
        keys.add("".join(toks[:-1]))
    return list(keys)


def addr_block_tokens(addr) -> list:
    """Blocking keys from an address: its distinct normalized words/numbers."""
    return list(set(clean_tokens(addr)))


def name_skeleton(tokens) -> list:
    """Phonetic skeletons of the alphabetic name tokens (see translit.skeleton_word)."""
    out = []
    for t in tokens:
        if t.isdigit():
            continue
        sk = skeleton_word(t)
        if sk:
            out.append(sk)
    return out
