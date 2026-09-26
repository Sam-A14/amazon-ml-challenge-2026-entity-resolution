"""Script normalisation for names written phonetically in Indian scripts.

EDA / miss analysis showed that many Source 2/3 records spell the (English) business name in an
Indian script, e.g. "Super Impex Private Limited" vs "सुपर इम्पेक्स प्राइवेट लिमिटेड".
All Indic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
Malayalam) share the same internal layout, so ONE offset table transliterates all of them.
No external data or libraries are used - this is deterministic text processing.

`skeleton()` then maps any Latin text to a phonetic consonant skeleton so that the English
spelling and the transliteration meet:  super impex private limited -> spr mpks prvt lmt
"""
import re

_BLOCKS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
    0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
    0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
    0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
    0x4E: "t",                                                  # Bengali khanda ta
    0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k",   # Malayalam chillu
}
_VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
}
_SIGNS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "rii",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x57: "au",
}
_NASAL = {0x01, 0x02}           # candrabindu, anusvara
_VISARGA = 0x03
_VIRAMA = 0x4D


def _offset(ch):
    o = ord(ch)
    for b in _BLOCKS:
        if b <= o < b + 0x80:
            return o - b
    return None


def transliterate(text: str) -> str:
    """Latin rendering of any Indic-script characters (other characters unchanged).
    The inherent vowel is omitted (the skeleton ignores vowels anyway)."""
    if not text or text.isascii():
        return text
    text = text.replace("\u200c", "").replace("\u200d", "")   # zero-width (non-)joiners
    out = []
    n = len(text)
    for idx, ch in enumerate(text):
        off = _offset(ch)
        if off is None:
            out.append(ch)
        elif off in _CONS:
            out.append(_CONS[off])
        elif off in _VOWELS:
            out.append(_VOWELS[off])
        elif off in _SIGNS:
            out.append(_SIGNS[off])
        elif off in _NASAL:
            nxt = text[idx + 1] if idx + 1 < n else " "
            out.append("n" if _offset(nxt) is not None else "")   # word-final nasal dropped
        elif off == _VISARGA:
            out.append("h")
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))
        # virama, nukta, other marks: nothing
    return "".join(out)


_SUBS = [
    (re.compile(r"tion"), "shn"), (re.compile(r"ck"), "k"), (re.compile(r"x"), "ks"),
    (re.compile(r"qu"), "kv"), (re.compile(r"ph"), "f"),
    (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"g(?=[eiy])"), "j"),
]
_MAP = str.maketrans({"h": "", "c": "k", "q": "k", "g": "k", "b": "p", "d": "t", "f": "p",
                      "w": "v", "z": "s", "a": "", "e": "", "i": "", "o": "", "u": "", "y": ""})
_RUNS = re.compile(r"(.)\1+")


def skeleton_word(w: str) -> str:
    """Phonetic consonant skeleton of one lowercase latin word."""
    for pat, rep in _SUBS:
        w = pat.sub(rep, w)
    return _RUNS.sub(r"\1", w.translate(_MAP))
