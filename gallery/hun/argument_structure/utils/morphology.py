"""Hungarian UniMorph normalization and lexical-form extraction."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional

import requests

from bead.resources.lexical_item import LexicalItem


UNIMORPH_HUN_URL = "https://raw.githubusercontent.com/unimorph/hun/master/hun"


# CASE NORMALIZATION

CASE_MAP = {
    # Core cases
    "NOM": "NOM",
    "ACC": "ACC",
    "DAT": "DAT",

    # Interior local cases
    "IN+ESS": "INE",   # -ban/-ben
    "IN+ALL": "ILL",   # -ba/-be
    "IN+ABL": "ELA",   # -ból/-ből

    # Proximal / "at" local cases
    "AT+ESS": "ADE",   # -nál/-nél
    "AT+ALL": "ALL",   # -hoz/-hez/-höz
    "AT+ABL": "ABL",   # -tól/-től

    # Surface local cases
    "ON+ESS": "SUP",   # -on/-en/-ön
    "ON+ALL": "SUB",   # -ra/-re
    "ON+ABL": "DEL",   # -ról/-ről

    # Other cases
    "INS": "INS",      # -val/-vel
    "CAUS": "CAU_FIN", # -ért
    "TERM": "TER",     # -ig
    "ESS": "ESS_FOR",  # -ként
    "TEMP": "TEMP",    # -kor
    "TRANS": "TRA",    # -vá/-vé
}


# HELPERS

def parse_parenthesized_tag(tag: str) -> tuple[str, list[str]]:
    """Split a UniMorph tag such as ``ACC(DEF)`` into its label and values."""

    match = re.fullmatch(r"([^()]+)\(([^()]*)\)", tag)

    if not match:
        return tag, []

    label = match.group(1)
    values = [value.strip() for value in match.group(2).split(",") if value.strip()]

    return label, values


def is_verb_tag_bundle(tags: str) -> bool:
    """Return whether a UniMorph bundle represents a Hungarian verbal form."""

    first_tag = tags.split(";")[0]
    return first_tag in {"V", "V.PTCP", "V.CVB", "V.MSDR"}


def is_noun_tag_bundle(tags: str) -> bool:
    first_tag = tags.split(";")[0]
    return first_tag == "N"


# VERB FEATURES

def parse_verb_features(tags: str) -> Dict[str, object]:
    """Normalize a Hungarian UniMorph verb analysis for template constraints."""

    raw_tags = tags.split(";")
    tag_set = set(raw_tags)

    features: Dict[str, object] = {
        "pos": "V",
        "unimorph_features": tags,
        "unimorph_tags": raw_tags,
    }

    first_tag = raw_tags[0]

    # NONFINITE FORMS

    if first_tag == "V.PTCP" or "V.PTCP" in tag_set:
        features["finiteness"] = "NFIN"
        features["verb_form"] = "PTCP"

    elif first_tag == "V.CVB" or "V.CVB" in tag_set:
        features["finiteness"] = "NFIN"
        features["verb_form"] = "CVB"

    elif first_tag == "V.MSDR" or "V.MSDR" in tag_set:
        features["finiteness"] = "NFIN"
        features["verb_form"] = "MSDR"

    elif "NFIN" in tag_set:
        features["finiteness"] = "NFIN"
        features["verb_form"] = "INF"

    # POTENTIAL

    # Hungarian -hat/-het.
    #
    # POT is not itself equivalent to NFIN.

    if "POT" in tag_set:
        features["potential"] = True
        features["derivation"] = "POT"

    # MOOD

    for mood in ("IND", "IMP", "COND", "SBJV", "OPT"):
        if mood in tag_set:
            features["mood"] = mood
            break

    # TENSE

    for tense in ("PRS", "PST", "FUT"):
        if tense in tag_set:
            features["tense"] = tense
            break

    # ASPECT

    # Preserve any explicit UniMorph annotation.
    # This does NOT posit an English-style Hungarian progressive paradigm.

    for aspect in ("PFV", "IPFV", "PROG", "HAB"):
        if aspect in tag_set:
            features["aspect"] = aspect
            break

    # VOICE

    for voice in ("ACT", "PASS", "MID", "CAUS"):
        if voice in tag_set:
            features["voice"] = voice
            break

    # SUBJECT AGREEMENT AND OBJECT CONJUGATION

    for tag in raw_tags:
        label, values = parse_parenthesized_tag(tag)


        if label == "NOM" and values:
            if values[0] in {"1", "2", "3"}:
                features["person"] = values[0]

            if len(values) >= 2 and values[1] in {"SG", "PL"}:
                features["number"] = values[1]


        elif label == "ACC" and values:
            if values[0] in {"DEF", "INDF"}:
                features["object_agreement"] = values[0]


            elif values[0] == "2":
                features["object_agreement"] = "SPECIAL"
                features["special_conjugation"] = "1SG_SUBJECT_2ND_PERSON_OBJECT"
                features["object_person"] = "2"

                if len(values) >= 2:
                    features["object_number"] = values[1]

    # FLAT-TAG COMPATIBILITY


    if "DEF" in tag_set:
        features["object_agreement"] = "DEF"

    elif "INDF" in tag_set:
        features["object_agreement"] = "INDF"

    elif "LGSPEC3" in tag_set:
        features["object_agreement"] = "SPECIAL"
        features["special_conjugation"] = "1SG_SUBJECT_2ND_PERSON_OBJECT"
        features["object_person"] = "2"

    if "person" not in features:
        for person in ("1", "2", "3"):
            if person in tag_set:
                features["person"] = person
                break

    if "number" not in features:
        for number in ("SG", "PL"):
            if number in tag_set:
                features["number"] = number
                break

    # FINITENESS


    if "finiteness" not in features:
        finite_evidence = any(
            key in features
            for key in ("mood", "tense", "person", "number", "object_agreement")
        )

        if finite_evidence:
            features["finiteness"] = "FIN"

    # PERSONALIZED INFINITIVES


    if features.get("verb_form") == "INF" and "person" in features:
        features["infinitive_agreement"] = True

    # FINITE FORMS WITHOUT OBJECT-CONJUGATION INFORMATION

    if features.get("finiteness") == "FIN" and "object_agreement" not in features:
        features["object_agreement"] = "NONE"

    return features


# NOUN FEATURES

def parse_noun_features(tags: str) -> Optional[Dict[str, object]]:
    """
    Normalize Hungarian noun morphology.

    Examples:

        N;ACC(SG)
        N;DAT(SG)
        N;IN+ESS(SG)
        N;IN+ALL(SG)
        N;ON+ALL(PL)

    The actual noun surface form is stored by UniMorph.
    We do NOT reconstruct noun + suffix manually.

    Possessive morphology is excluded from Stage 1.
    """

    raw_tags = tags.split(";")

    if not raw_tags or raw_tags[0] != "N":
        return None

    # EXCLUDE POSSESSION FOR STAGE 1

    if any(
        "PSS" in tag or "POSS" in tag or tag.startswith("GEN+")
        for tag in raw_tags
    ):
        return None

    case = None
    number = None

    for tag in raw_tags[1:]:
        label, values = parse_parenthesized_tag(tag)

        if label in {"SG", "PL"} and not values:
            number = label
            continue

        if label in CASE_MAP:
            case = CASE_MAP[label]

            if values and values[0] in {"SG", "PL"}:
                number = values[0]

    if case is None:
        remaining = [tag for tag in raw_tags[1:] if tag not in {"SG", "PL"}]

        if not remaining:
            case = "NOM"

    if case is None:
        return None

    return {
        "pos": "NOUN",
        "person": "3",
        "number": number or "SG",
        "case": case,
        "unimorph_features": tags,
        "unimorph_tags": raw_tags,
    }


# MORPHOLOGY EXTRACTOR

class MorphologyExtractor:
    """
    Hungarian equivalent of the English MorphologyExtractor.

    UniMorph is downloaded once and indexed by lemma.

    generate_lexicons.py can then simply call:

        morph.get_verb_lemmas()
        morph.get_all_verb_forms("olvas")
        morph.get_all_noun_forms("tárgy")
    """

    def __init__(self, url: str = UNIMORPH_HUN_URL):
        self.url = url
        self._loaded = False
        self._verbs = defaultdict(list)
        self._nouns = defaultdict(list)
        self._verb_lemmas = []

    def _load(self):
        if self._loaded:
            return

        print("Downloading Hungarian UniMorph data...")

        response = requests.get(self.url, timeout=60)
        response.raise_for_status()

        seen_verb_lemmas = set()

        for line in response.text.splitlines():
            parts = line.rstrip("\n").split("\t")

            if len(parts) != 3:
                continue

            lemma, form, tags = parts

            # ----------------------------------------------------------------
            # VERBS
            # ----------------------------------------------------------------

            if is_verb_tag_bundle(tags):
                if "-" in lemma:
                    continue

                self._verbs[lemma].append((form, tags))

                if lemma not in seen_verb_lemmas:
                    seen_verb_lemmas.add(lemma)
                    self._verb_lemmas.append(lemma)

            # ----------------------------------------------------------------
            # NOUNS
            # ----------------------------------------------------------------

            elif is_noun_tag_bundle(tags):
                self._nouns[lemma].append((form, tags))

        self._loaded = True

        print(f"Loaded {len(self._verb_lemmas)} Hungarian verb lemmas")
        print(f"Loaded {len(self._nouns)} Hungarian noun lemmas")

    def get_verb_lemmas(self, limit: Optional[int] = None) -> List[str]:
        self._load()

        if limit is None:
            return list(self._verb_lemmas)

        return list(self._verb_lemmas[:limit])

    def get_all_verb_forms(self, lemma: str) -> List[LexicalItem]:
        self._load()

        items = []
        seen = set()

        for form, tags in self._verbs.get(lemma, []):
            key = (form, tags)

            if key in seen:
                continue

            seen.add(key)

            items.append(
                LexicalItem(
                    lemma=lemma,
                    form=form,
                    language_code="hun",
                    features=parse_verb_features(tags),
                    source="UniMorph",
                )
            )

        return items

    def get_all_noun_forms(self, lemma: str) -> List[LexicalItem]:
        self._load()

        items = []
        seen = set()

        for form, tags in self._nouns.get(lemma, []):
            features = parse_noun_features(tags)

            if features is None:
                continue

            key = (form, tags)

            if key in seen:
                continue

            seen.add(key)

            items.append(
                LexicalItem(
                    lemma=lemma,
                    form=form,
                    language_code="hun",
                    features=features,
                    source="UniMorph",
                )
            )

        return items