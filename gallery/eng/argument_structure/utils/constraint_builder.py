"""English-specific constraint builders for argument structure experiment.

This module provides LANGUAGE-SPECIFIC helper functions for English constraints.
For general, language-agnostic constraint builders, see:
    bead/resources/constraints/builders.py - AgreementConstraintBuilder, etc.

This module handles English-specific patterns like:
- English determiner-noun agreement ("a" vs "some" vs "the")
- English bleached lexicon restrictions
- English preposition inventories

NOTE: This is intentionally kept in gallery/ because it's English-specific.
"""

from __future__ import annotations

import csv
from pathlib import Path

from bead.resources.constraints import Constraint


def build_determiner_constraint(det_slot: str, noun_slot: str) -> Constraint:
    """Build determiner-noun agreement constraint.

    Rules:
    - "a" requires singular count nouns
    - "some" requires mass or plural nouns
    - "the" is unrestricted

    Parameters
    ----------
    det_slot : str
        Name of the determiner slot
    noun_slot : str
        Name of the noun slot

    Returns
    -------
    Constraint
        DSL constraint enforcing determiner-noun agreement

    Examples
    --------
    >>> constraint = build_determiner_constraint("det", "noun")
    >>> constraint.expression
    "(det.lemma != 'a' or (noun.features.get('number') == 'singular' and noun.features.get('countability') in ['count', 'count/mass'])) and (det.lemma != 'some' or noun.features.get('number') in ['plural', 'mass'] or noun.features.get('countability') in ['mass', 'count/mass'])"
    """
    expression = (
        f"({det_slot}.lemma != 'a' or "
        f"({noun_slot}.features.get('number') == 'singular' and "
        f"{noun_slot}.features.get('countability') in ['count', 'count/mass'])) "
        f"and "
        f"({det_slot}.lemma != 'some' or "
        f"{noun_slot}.features.get('number') in ['plural', 'mass'] or "
        f"{noun_slot}.features.get('countability') in ['mass', 'count/mass'])"
    )

    return Constraint(
        expression=expression,
        description=f"Determiner-noun agreement for {det_slot} and {noun_slot}",
    )


def load_bleached_nouns(csv_path: str | Path) -> set[str]:
    """Load bleached noun inventory from CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to bleached_nouns.csv

    Returns
    -------
    set[str]
        Set of allowed noun lemmas

    Examples
    --------
    >>> nouns = load_bleached_nouns("resources/bleached_nouns.csv")
    >>> "person" in nouns
    True
    """
    nouns = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nouns.add(row["word"])
    return nouns


def load_bleached_verbs(csv_path: str | Path) -> set[str]:
    """Load bleached verb inventory from CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to bleached_verbs.csv

    Returns
    -------
    set[str]
        Set of allowed verb lemmas

    Examples
    --------
    >>> verbs = load_bleached_verbs("resources/bleached_verbs.csv")
    >>> "do" in verbs
    True
    """
    verbs = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            verbs.add(row["word"])
    return verbs


def load_bleached_adjectives(csv_path: str | Path) -> set[str]:
    """Load bleached adjective inventory from CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to bleached_adjectives.csv

    Returns
    -------
    set[str]
        Set of allowed adjective lemmas

    Examples
    --------
    >>> adjs = load_bleached_adjectives("resources/bleached_adjectives.csv")
    >>> len(adjs) > 0
    True
    """
    adjectives = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            adjectives.add(row["word"])
    return adjectives


def build_bleached_noun_constraint(
    slot: str, csv_path: str | Path = "resources/bleached_nouns.csv"
) -> Constraint:
    """Restrict noun slot to bleached noun inventory.

    Parameters
    ----------
    slot : str
        Name of the noun slot
    csv_path : str | Path
        Path to bleached_nouns.csv (default: "resources/bleached_nouns.csv")

    Returns
    -------
    Constraint
        DSL constraint restricting to bleached nouns

    Examples
    --------
    >>> constraint = build_bleached_noun_constraint("noun")
    >>> "bleached_nouns" in constraint.context
    True
    """
    bleached_nouns = load_bleached_nouns(csv_path)

    expression = f"{slot}.lemma in bleached_nouns"

    return Constraint(
        expression=expression,
        context={"bleached_nouns": bleached_nouns},
        description=f"Restrict {slot} to bleached noun inventory",
    )


def build_bleached_verb_constraint(
    slot: str, csv_path: str | Path = "resources/bleached_verbs.csv"
) -> Constraint:
    """Restrict verb slot to bleached verb inventory.

    Parameters
    ----------
    slot : str
        Name of the verb slot
    csv_path : str | Path
        Path to bleached_verbs.csv (default: "resources/bleached_verbs.csv")

    Returns
    -------
    Constraint
        DSL constraint restricting to bleached verbs

    Examples
    --------
    >>> constraint = build_bleached_verb_constraint("comp_verb")
    >>> "bleached_verbs" in constraint.context
    True
    """
    bleached_verbs = load_bleached_verbs(csv_path)

    expression = f"{slot}.lemma in bleached_verbs"

    return Constraint(
        expression=expression,
        context={"bleached_verbs": bleached_verbs},
        description=f"Restrict {slot} to bleached verb inventory",
    )


def build_bleached_adj_constraint(
    slot: str, csv_path: str | Path = "resources/bleached_adjectives.csv"
) -> Constraint:
    """Restrict adjective slot to bleached adjective inventory.

    Parameters
    ----------
    slot : str
        Name of the adjective slot
    csv_path : str | Path
        Path to bleached_adjectives.csv (default: "resources/bleached_adjectives.csv")

    Returns
    -------
    Constraint
        DSL constraint restricting to bleached adjectives

    Examples
    --------
    >>> constraint = build_bleached_adj_constraint("adjective")
    >>> "bleached_adjectives" in constraint.context
    True
    """
    bleached_adjectives = load_bleached_adjectives(csv_path)

    expression = f"{slot}.lemma in bleached_adjectives"

    return Constraint(
        expression=expression,
        context={"bleached_adjectives": bleached_adjectives},
        description=f"Restrict {slot} to bleached adjective inventory",
    )


def build_preposition_constraint(slot: str) -> Constraint:
    """Allow any English preposition from comprehensive list.

    Parameters
    ----------
    slot : str
        Name of the preposition slot

    Returns
    -------
    Constraint
        DSL constraint allowing comprehensive English prepositions

    Examples
    --------
    >>> constraint = build_preposition_constraint("prep")
    >>> "prepositions" in constraint.context
    True
    """
    # Comprehensive English preposition list
    prepositions = {
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "down",
        "during",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "like",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "out",
        "outside",
        "over",
        "past",
        "since",
        "through",
        "throughout",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "up",
        "upon",
        "with",
        "within",
        "without",
    }

    expression = f"{slot}.lemma in prepositions"

    return Constraint(
        expression=expression,
        context={"prepositions": prepositions},
        description=f"Restrict {slot} to English prepositions",
    )


def build_subject_verb_agreement_constraint(
    det_slot: str, noun_slot: str, verb_slot: str
) -> Constraint:
    """Build subject-verb number agreement constraint for English.

    Rules:
    - Singular noun → verb must be 3SG (3rd person singular)
    - Plural noun → verb must NOT be 3SG

    Note: Lexicons use "singular"/"plural" for noun number, "SG"/"PL" for verb number.

    Parameters
    ----------
    det_slot : str
        Name of the subject determiner slot
    noun_slot : str
        Name of the subject noun slot
    verb_slot : str
        Name of the verb slot

    Returns
    -------
    Constraint
        DSL constraint enforcing subject-verb agreement

    Examples
    --------
    >>> constraint = build_subject_verb_agreement_constraint("det_subj", "noun_subj", "verb")
    >>> "singular" in constraint.expression
    True
    """
    # Agreement logic:
    # If noun is singular → verb should have person=3 and number=SG
    # If noun is plural → verb should NOT have person=3 and number=SG
    expression = (
        f"("
        f"  ({noun_slot}.features.get('number') != 'singular') or "
        f"  ({verb_slot}.features.get('person') == '3' and {verb_slot}.features.get('number') == 'SG')"
        f") and ("
        f"  ({noun_slot}.features.get('number') != 'plural') or "
        f"  (not ({verb_slot}.features.get('person') == '3' and {verb_slot}.features.get('number') == 'SG')))"
    )

    return Constraint(
        expression=expression,
        description=f"Subject-verb number agreement between {noun_slot} and {verb_slot}",
    )


def build_be_subject_agreement_constraint(
    det_slot: str, noun_slot: str, be_slot: str
) -> Constraint:
    """Build subject-be number agreement constraint for English auxiliary 'be'.

    Rules (noun subjects are always 3rd person):
    - Singular noun → be must be 3rd person singular (is/was)
    - Plural noun → be must be plural (are/were)

    This is stricter than build_subject_verb_agreement_constraint because
    'be' has full person/number paradigm, not just 3SG vs non-3SG.

    Parameters
    ----------
    det_slot : str
        Name of the subject determiner slot
    noun_slot : str
        Name of the subject noun slot
    be_slot : str
        Name of the be slot

    Returns
    -------
    Constraint
        DSL constraint enforcing subject-be agreement

    Examples
    --------
    >>> constraint = build_be_subject_agreement_constraint("det_subj", "noun_subj", "be")
    >>> "singular" in constraint.expression
    True
    """
    # Strict agreement for auxiliary be with noun subjects (3rd person):
    # - Singular noun → be must be 3SG (is/was)
    # - Plural noun → be must be PL (are/were)
    expression = (
        f"("
        f"  ({noun_slot}.features.get('number') == 'singular' and "
        f"   {be_slot}.features.get('person') == '3' and "
        f"   {be_slot}.features.get('number') == 'SG')"
        f") or ("
        f"  ({noun_slot}.features.get('number') == 'plural' and "
        f"   {be_slot}.features.get('number') == 'PL')"
        f")"
    )

    return Constraint(
        expression=expression,
        description=f"Subject-verb number agreement between {noun_slot} and {be_slot}",
    )


def build_be_participle_constraint(be_slot: str, verb_slot: str) -> Constraint:
    """Build be + participle agreement constraint for progressive aspect.

    Ensures that progressive templates (e.g., "is running") have:
    1. The verb is a present participle (V.PTCP with PRS tense)
    2. The "be" form agrees with the subject (handled separately)

    Parameters
    ----------
    be_slot : str
        Name of the "be" slot
    verb_slot : str
        Name of the verb slot (must be participle)

    Returns
    -------
    Constraint
        DSL constraint enforcing be + participle structure

    Examples
    --------
    >>> constraint = build_be_participle_constraint("be", "verb")
    >>> "V.PTCP" in constraint.expression
    True
    """
    expression = (
        f"{verb_slot}.features.get('verb_form') == 'V.PTCP' and "
        f"{verb_slot}.features.get('tense') == 'PRS'"
    )

    return Constraint(
        expression=expression,
        description=f"Progressive aspect: {verb_slot} must be present participle when used with {be_slot}",
    )


def build_combined_constraint(*constraints: Constraint) -> Constraint:
    """Combine multiple constraints with AND logic.

    Parameters
    ----------
    *constraints : Constraint
        Variable number of constraints to combine

    Returns
    -------
    Constraint
        Combined constraint with all expressions ANDed together

    Examples
    --------
    >>> c1 = build_bleached_noun_constraint("noun1")
    >>> c2 = build_bleached_noun_constraint("noun2")
    >>> combined = build_combined_constraint(c1, c2)
    >>> "and" in combined.expression
    True
    """
    if not constraints:
        raise ValueError("Must provide at least one constraint")

    if len(constraints) == 1:
        return constraints[0]

    # Combine expressions with AND
    expressions = [f"({c.expression})" for c in constraints]
    combined_expression = " and ".join(expressions)

    # Merge contexts
    combined_context = {}
    for constraint in constraints:
        if constraint.context:
            combined_context.update(constraint.context)

    # Combine descriptions
    descriptions = [c.description for c in constraints if c.description]
    combined_description = "; ".join(descriptions)

    return Constraint(
        expression=combined_expression,
        context=combined_context if combined_context else None,
        description=combined_description,
    )
