"""Robust ECG lead-name normalization (WFDB headers are free-text).

Real-world `.hea`/`sig_name` entries for the same 12 standard leads vary wildly
across datasets: MIT-BIH writes modified-limb names (MLI/MLII/MLIII), older
records use Einthoven (DI/DII/DIII) and historic Goldberger (VR/VL/VF) labels,
chest leads appear as C1-C6 instead of V1-V6, and case / whitespace / unit
suffixes ('II ', 'aVR', '-aVR', 'V1 (mV)') differ everywhere. The previous
12-entry exact-match dict silently dropped anything it didn't recognize — e.g.
a MIT-BIH 'MLII' channel was never mapped to lead II and disappeared.

This module maps names to the 12 canonical forms via (1) a large generated
alias table and (2) a fuzzy fallback for unseen spellings. Anything that still
can't be confidently matched is KEPT under its raw name (never dropped). No Qt
dependency, so the headless batch path can use it too.
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata

_log = logging.getLogger("lead_names")

CANONICAL = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]

# Human-readable variant spellings per canonical lead. Fed through _canon_key()
# so only the normalized key is stored — list the readable forms here.
_VARIANTS: dict[str, list[str]] = {
    "I":   ["i", "1", "d1", "di", "l1", "li", "lead1", "leadi", "limbi",
            "mli", "ml1"],
    "II":  ["ii", "2", "d2", "dii", "l2", "lii", "lead2", "leadii", "limbii",
            "mlii", "ml2"],
    "III": ["iii", "3", "d3", "diii", "l3", "liii", "lead3", "leadiii",
            "limbiii", "mliii", "ml3"],
    "aVR": ["avr", "vr", "goldbergeravr", "augvr"],
    "aVL": ["avl", "vl", "goldbergeravl", "augvl"],
    "aVF": ["avf", "vf", "goldbergeravf", "augvf"],
    "V1":  ["v1", "c1", "precordial1", "chest1", "mv1"],
    "V2":  ["v2", "c2", "precordial2", "chest2"],
    "V3":  ["v3", "c3", "precordial3", "chest3"],
    "V4":  ["v4", "c4", "precordial4", "chest4"],
    "V5":  ["v5", "c5", "precordial5", "chest5"],
    "V6":  ["v6", "c6", "precordial6", "chest6"],
}

_UNIT_RE = re.compile(r"\b(millivolts?|microvolts?|mvolts?|mv|uv|μv)\b")
_BRACKET_RE = re.compile(r"[\(\[\{].*?[\)\]\}]")
_NONALNUM_RE = re.compile(r"[^a-z0-9]")


def _canon_key(name: str) -> str:
    """Normalize a raw lead label to a comparison key.

    NFKD-fold unicode, lowercase, drop bracketed/unit suffixes, strip every
    non-alphanumeric char. 'aVR ' / 'AVR' / '-aVR' -> 'avr'; 'V1 (mV)' -> 'v1'.
    """
    s = unicodedata.normalize("NFKD", str(name)).lower().strip()
    s = _BRACKET_RE.sub("", s)
    s = _UNIT_RE.sub("", s)
    s = _NONALNUM_RE.sub("", s)
    return s


# key -> canonical
_ALIAS: dict[str, str] = {}
for _canon, _vs in _VARIANTS.items():
    for _v in _vs:
        _ALIAS[_canon_key(_v)] = _canon
for _canon in CANONICAL:
    _ALIAS.setdefault(_canon_key(_canon), _canon)

# canonical -> its set of keys (for fuzzy scoring)
_CANON_KEYS: dict[str, list[str]] = {c: [] for c in CANONICAL}
for _k, _c in _ALIAS.items():
    _CANON_KEYS[_c].append(_k)


def _fuzzy_match(key: str, cutoff: float = 0.8) -> tuple[str | None, float]:
    """Closest canonical lead for an unseen key, or (None, best_score).

    Uses difflib ratio against every known key, with a strong bonus (0.85) when
    a canonical key (len >= 2) is a substring of the input or vice-versa — so
    'mlii' favors II via its 'ii' key, 'leadv5x' favors V5. Single-char keys
    ('i') never trigger the substring bonus to avoid greedy mismatches. The
    high pure-ratio cutoff means coincidental overlaps ('ecg1' vs 'c1' = 0.67,
    'avl' vs 'avr' = 0.67) are rejected and kept raw rather than mislabeled;
    only genuine containment ('mlii', 'leadii') clears the bar.
    """
    best_c: str | None = None
    best_score = 0.0
    for canon, keys in _CANON_KEYS.items():
        for k in keys:
            score = difflib.SequenceMatcher(None, key, k).ratio()
            if len(k) >= 2 and (k in key or key in k):
                score = max(score, 0.85)
            if score > best_score:
                best_score, best_c = score, canon
    if best_score >= cutoff:
        return best_c, best_score
    return None, best_score


def normalize_lead_names(names: list[str]) -> list[str]:
    """Map raw WFDB lead names to canonical forms; keep unmatched names as-is.

    Guarantees the output length equals the input length and never silently
    drops a channel. If two raw channels normalize to the same canonical lead
    (a collision), the later one is kept under a distinct (raw-based) name so
    both remain addressable.
    """
    mapped: list[str] = []
    for raw in names:
        key = _canon_key(raw)
        if not key:
            mapped.append(str(raw))
            continue
        if key in _ALIAS:
            canon = _ALIAS[key]
            if canon != raw:
                _log.info("lead '%s' -> '%s' (exact)", raw, canon)
            mapped.append(canon)
            continue
        canon, score = _fuzzy_match(key)
        if canon is not None:
            _log.info("lead '%s' -> '%s' (fuzzy %.2f)", raw, canon, score)
            mapped.append(canon)
        else:
            _log.info("lead '%s' kept raw (best match %.2f < 0.80)", raw, score)
            mapped.append(str(raw))

    # Resolve collisions so every channel stays uniquely addressable.
    seen: set[str] = set()
    result: list[str] = []
    for name, raw in zip(mapped, names):
        if name not in seen:
            result.append(name)
            seen.add(name)
            continue
        cand = str(raw)
        while cand in seen:
            cand += "*"
        _log.info("lead name collision on '%s' -> kept '%s'", name, cand)
        result.append(cand)
        seen.add(cand)
    return result
