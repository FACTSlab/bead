#!/usr/bin/env python3
"""
Generate JSONL lexicon files for argument structure alternation dataset

This script creates all required lexicons:
Output:
    lexicons/verbs.jsonl
    lexicons/bleached_nouns.jsonl
    lexicons/bleached_verbs.jsonl
    lexicons/bleached_adjectives.jsonl
    lexicons/case_markers.jsonl
    lexicons/auxiliary_verbs.jsonl
"""

import argparse
import csv
import requests
from typing import List
import pandas as pd
import re
from pathlib import Path


from bead.resources.lexicon import Lexicon
from bead.resources.lexical_item import LexicalItem

def main(verb_limit: int | None = None, save_csv: bool = True) -> None:
    # Set up paths
    base_dir = Path(__file__).parent
    lexicons_dir = base_dir / "lexicons"
    resources_dir = base_dir / "resources"

    # Ensure directories exist
    lexicons_dir.mkdir(exist_ok=True)
    resources_dir.mkdir(exist_ok=True)

    # Generate verbs lexicon from UniMorph Korean data
    print("=" * 80)
    print("GENERATING VERBS LEXICON FROM UNIMORPH KOR DATA")
    print("=" * 80)

    url = "https://raw.githubusercontent.com/unimorph/kor/master/kor"
    unimorph_kor = requests.get(url)
    data = unimorph_kor.text
    lexicon = Lexicon(name="verbs")
    verbs: List[LexicalItem] = []

    chinese_char_regex = re.compile(r'[\u4e00-\u9fff]')

    if verb_limit is not None:
        print(f"[TEST MODE] Limiting to first {verb_limit} verbs")

    base_verb: set[str] = set()

    for line in data.splitlines():
        parts = line.strip().split('\t')
        if len(parts) == 3:
            base, form, tags = parts

            if verb_limit is not None and len(base_verb) > verb_limit:
                break

            if '-' in base or chinese_char_regex.search(base):
                continue

            base_verb.add(base)

            if 'V;DECL;FIN;PST;FORM' == tags or 'ADJ;DECL;FIN;PST;FORM' == tags: # past tense
                verb = LexicalItem(
                        lemma=base,
                        form=form,
                        language_code="kor",
                        features={"pos": "V","finiteness": "FIN", "tense": "PST", "unimorph_features": tags},
                        source="UniMorph"
                    )
                verbs.append(verb)

            elif 'V;DECL;FIN;PRS;FORM' == tags or 'ADJ;DECL;FIN;PRS;FORM' == tags: # present tense
                verb = LexicalItem(
                        lemma=base,
                        form=form,
                        language_code="kor",
                        features={"pos": "V", "finiteness": "FIN", "tense": "PRS", "unimorph_features": tags},
                        source="UniMorph"
                    )
                verbs.append(verb)

            elif 'V.CVB;NFIN;CONJ' == tags or 'ADJ.CVB;NFIN;CONJ' == tags: # gerund form
                if form.endswith("고"):
                    verb= LexicalItem(
                            lemma=base,
                            form=form,
                            language_code="kor",
                            features={"pos": "V", "finiteness": "NFIN", "verb_form": "V.PTCP", "unimorph_features": tags},
                            source="UniMorph"
                        )
                    verbs.append(verb)

                # Handling UniMorph annotation inconsistencies
                elif '-' in form:
                    verb = LexicalItem(
                            lemma=base,
                            form=form.split('-')[0],
                            language_code="kor",
                            features={"pos": "V", "finiteness": "NFIN", "verb_form": "V.PTCP", "unimorph_features": tags},
                            source="UniMorph"
                        )
                    verbs.append(verb)

                elif '\'' in form:
                    verb = LexicalItem(
                            lemma=base,
                            form=form.split('\'')[0] + " 있다",
                            language_code="kor",
                            features={"pos": "V", "finiteness": "NFIN", "verb_form": "V.PTCP", "unimorph_features": tags},
                            source="UniMorph"
                        )
                    verbs.append(verb)
                
                elif form.endswith("면"):
                    verb = LexicalItem(
                            lemma=base,
                            form=form[:-1] + "고",
                            language_code="kor",
                            features={"pos": "V", "finiteness": "NFIN", "verb_form": "V.PTCP", "unimorph_features": tags},
                            source="UniMorph"
                        )
                    verbs.append(verb)
                    

    lexicon.add_many(verbs)

    print(f"Total base verbs found: {len(base_verb)}")
    print(f"Total verbs found: {len(verbs)}")
    
    lexicon.to_jsonl("./lexicons/verbs.jsonl")

    # 2. Generate bleached nouns, verbs, adjectives, and case markers lexicons
    print("\n" + "=" * 80)
    print("GENERATING BLEACHED NOUNS, VERBS, ADJECTIVES, AND CASE MARKERS LEXICONS")
    print("=" * 80)

    # Generate nominative Case Markers csv and jsonl
    case_markers = pd.DataFrame(columns=['marker', 'case', 'final_consonant'])

    marker = ['이', '가', '을', '를', '으로', '로', '에게', '에서']
    case = ['NOM', 'NOM', 'ACC', 'ACC', 'INST', 'INST', 'DAT', 'LOC']
    final_consonant = ['yes', 'no', 'yes', 'no', 'yes', 'no', None, None]

    case_markers['marker'] = marker
    case_markers['case'] = case
    case_markers['final_consonant'] = final_consonant

    if save_csv:
        case_markers.to_csv('./resources/case_markers.csv', index=False)

    lexicon = Lexicon(name="case_markers")
    with open("resources/case_markers.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["case"] == "NOM":
                item = LexicalItem(
                    lemma=row["marker"],
                    language_code="kor",
                    features={"pos": "PART.NOM", "case": row["case"], "final_consonant": row["final_consonant"]}
                )
                lexicon.add(item)
            if row["case"] == "ACC":
                item = LexicalItem(
                    lemma=row["marker"],
                    language_code="kor",
                    features={"pos": "PART.ACC", "case": row["case"], "final_consonant": row["final_consonant"]}
                )
                lexicon.add(item)
            if row["case"] == "INST":
                item = LexicalItem(
                    lemma=row["marker"],
                    language_code="kor",
                    features={"pos": "PART.INST", "case": row["case"], "final_consonant": row["final_consonant"]}
                )
                lexicon.add(item)
            if row["case"] == "DAT":
                item = LexicalItem(
                    lemma=row["marker"],
                    language_code="kor",
                    features={"pos": "PART.DAT", "case": row["case"], "final_consonant": row["final_consonant"]}
                )
                lexicon.add(item)
            if row["case"] == "LOC":
                item = LexicalItem(
                    lemma=row["marker"],
                    language_code="kor",
                    features={"pos": "PART.LOC", "case": row["case"], "final_consonant": row["final_consonant"]}
                )
                lexicon.add(item)
    lexicon.to_jsonl("./lexicons/case_markers.jsonl")

    print(f"Created {len(case_markers)} case markers.")

    # Generate bleached nouns csv and jsonl
    bleached_nouns = pd.DataFrame(columns=['word', 'semantic_class', 'number', 'countability', 'final_consonant'])

    word = ['사람', '사람들', '단체', '단체들', '물건', '물건들', '장소', '장소들', '사건', '사건들']
    semantic_class = ['animate', 'animate', 'animate', 'animate', 'inanimate_object', 'inanimate_object', 'location', 'location', 'event', 'event']
    number = ['singular', 'plural', 'singular', 'plural', 'singular', 'plural', 'singular', 'plural', 'singular', 'plural']
    countability = ['countable', 'countable', 'countable', 'countable', 'countable', 'countable', 'countable', 'countable', 'countable', 'countable']
    final_consonant = ['yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes']

    bleached_nouns['word'] = word
    bleached_nouns['semantic_class'] = semantic_class
    bleached_nouns['number'] = number
    bleached_nouns['countability'] = countability
    bleached_nouns['final_consonant'] = final_consonant

    if save_csv:
        bleached_nouns.to_csv('./resources/bleached_nouns.csv', index=False)

    lexicon = Lexicon(name="bleached_nouns")

    with open("resources/bleached_nouns.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="kor",
                features={"pos": "NOUN", "number": row["number"], "countability": row["countability"], "final_consonant": row["final_consonant"]},
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/bleached_nouns.jsonl")

    print(f"Created {len(bleached_nouns)} bleached nouns.")


    # Generate bleached verbs csv and jsonl
    bleached_verbs = pd.DataFrame(columns=['word', 'semantic_class', 'aspect', 'valency', 'tenseless_vp', 'gerund', 'tenseless_clause', 'infinitival_clause','tensed_clause'])

    word = ['하다', '가지다', '가다', '갖다', '만들다', '일어나다', '오다'] # be functions differently in Korean
    semantic_class = ['activity', 'state', 'change', 'change', 'causation', 'event', 'change']
    aspect = ['dynamic', 'stative', 'dynamic', 'dynamic', 'dynamic', 'dynamic', 'dynamic']
    valency = ['transitive', 'transitive', 'intransitive', 'transitive', 'transitive', 'intransitive', 'intransitive']
    tenseless_vp = ['{{ object }} 하다', '{{ object }} 가지다', '가다', '{{ object }} 갖다', '{{ object }} 만들다', '일어나다', '오다']
    gerund = ['{{ object }} 하고', '{{ object }} 가지고', '가고', '{{ object }} 갖고', '{{ object }} 만들고', '일어나고', '오고']
    tenseless_clause = ['{{ subject }} {{ object }} 하다', '{{ subject }} {{ object }} 가지다', '{{ subject }} 가다', '{{ subject }} {{ object }} 갖다', '{{ subject }} {{ object }} 만들다', '{{ subject }} 일어나다', '{{ subject }} 오다']
    tensed_clause = ['{{ subject }} {{ object }} 했다', '{{ subject }} {{ object }} 가졌다', '{{ subject }} 갔다', '{{ subject }} {{ object }} 갖았다', '{{ subject }} {{ object }} 만들었다', '{{ subject }} 일어났다', '{{ subject }} 왔다']

    bleached_verbs['word'] = word
    bleached_verbs['semantic_class'] = semantic_class
    bleached_verbs['aspect'] = aspect
    bleached_verbs['valency'] = valency
    bleached_verbs['tenseless_vp'] = tenseless_vp
    bleached_verbs['gerund'] = gerund
    bleached_verbs['tenseless_clause'] = tenseless_clause
    bleached_verbs['infinitival_clause'] = tenseless_clause # same as tenseless_clause in Korean
    bleached_verbs['tensed_clause'] = tensed_clause

    if save_csv:
        bleached_verbs.to_csv("./resources/bleached_verbs.csv", index=False)

    lexicon = Lexicon(name="bleached_verbs")

    with open("resources/bleached_verbs.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="kor",
                features={"pos": "V", "tense": "", "semantic_class": row["semantic_class"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/bleached_verbs.jsonl")

    print(f"Created {len(bleached_verbs)} bleached verbs.")

    # Generate bleached adjectives csv and jsonl
    bleached_adjectives = pd.DataFrame(columns=['word', 'semantic_class', 'gradability', 'stage_vs_individual', 'notes'])

    word = ['좋은', '나쁜', '맞는', '틀린', '괜찮은', '확실한', '준비된', '끝난', '다른', '같은'] # different = other in Korean
    semantic_class = ['evaluation', 'evaluation', 'evaluation', 'evaluation', 'evaluation', 'epistemic', 'aspectual', 'aspectual', 'comparison', 'comparison']
    gradability = ['gradable', 'gradable', 'non-gradable', 'non-gradable', 'gradable', 'non-gradable', 'non-gradable', 'non-gradable', 'gradable', 'gradable']
    stage_vs_individual = ['individual', 'individual', 'stage', 'stage', 'stage', 'stage', 'stage', 'stage', 'individual', 'individual']
    notes = ['', '', '', '', '', '', '', '', '', '']

    bleached_adjectives['word'] = word
    bleached_adjectives['semantic_class'] = semantic_class
    bleached_adjectives['gradability'] = gradability
    bleached_adjectives['stage_vs_individual'] = stage_vs_individual
    bleached_adjectives['notes'] = notes

    if save_csv:
        bleached_adjectives.to_csv("./resources/bleached_adjectives.csv", index=False)

    lexicon = Lexicon(name="bleached_adjectives")

    with open("resources/bleached_adjectives.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["word"],
                language_code="kor",
                features={"pos": "ADJ", "semantic_class": row["semantic_class"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/bleached_adjectives.jsonl")

    print(f"Created {len(bleached_adjectives)} bleached adjectives.")

    # Generate auxiliary verbs (for progressive) csv and jsonl
    auxiliary_verbs = pd.DataFrame(columns=['lemma', 'form', 'pos', 'tense'])

    lemma = ['있다', '있다']
    form = ['있다', '있었다']
    pos = ['AUX', 'AUX']
    tense = ['PRS', 'PST']

    auxiliary_verbs['lemma'] = lemma
    auxiliary_verbs['form'] = form
    auxiliary_verbs['pos'] = pos
    auxiliary_verbs['tense'] = tense

    if save_csv:
        auxiliary_verbs.to_csv("./resources/auxiliary_verbs.csv", index=False)

    lexicon = Lexicon(name="auxiliary_verbs")

    with open("resources/auxiliary_verbs.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = LexicalItem(
                lemma=row["lemma"],
                language_code="kor",
                features={"pos": row["pos"], "tense": row["tense"]}
            )
            lexicon.add(item)

    lexicon.to_jsonl("./lexicons/auxiliary_verbs.jsonl")

    print(f"Created {len(auxiliary_verbs)} auxiliary verbs.")

    # Summary
    print("\n" + "=" * 80)
    print("LEXICON GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {6} lexicon files:")
    print(f"  1. verbs.jsonl:       {len(verbs)} entries")
    print(f"  2. bleached_nouns.jsonl: {len(bleached_nouns)} entries")
    print(f"  3. bleached_verbs.jsonl: {len(bleached_verbs)} entries")
    print(f"  4. bleached_adjectives.jsonl: {len(bleached_adjectives)} entries")
    print(f"  5. case_markers_nom.jsonl:  {len(case_markers)} entries")
    print(f"  6. auxiliary_verbs.jsonl:  {len(auxiliary_verbs)} entries")
    print(f"\nAll files saved to: {lexicons_dir}/")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Generate JSONL lexicon files for argument structure dataset"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of VerbNet verbs to process (for testing)",
    )

    parser.add_argument(
        "--save_csv",
        type=bool,
        default=True,
        help="Whether to save CSV files during processing (default: True, must run if first time)",
    )
    args = parser.parse_args()

    main(verb_limit=args.limit, save_csv=args.save_csv)