#!/usr/bin/env python3
"""
Generate Template objects for argument structure dataset

Output: templates/generic_frames.jsonl
"""

import argparse
from pathlib import Path
from typing import List

from bead.resources import Template, Slot, Constraint


def main(template_limit: int | None = None) -> None:
    """Generate and save Igbo sentence templates.
    
    Parameters
    ----------
    template_limit : int | None
        Limit number of templates to generate (for testing).
    """
    
    # Set up paths
    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    generic_templates: List[Template] = []

 
    # 1: Basic Intransitive (Subject + Verb)
    # Example: "Obi gbafuru." (Obi ran away)
    intransitive_basic = Template(
        name="subj-verb",
        template_string="{noun_subj} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            )
        },
        constraints=[],
        description="Basic intransitive sentence",
        language_code="ibo",
    )
    generic_templates.append(intransitive_basic)


    # 2: Intransitive with Determiner
    # Example: "Nwoke ahụ gbafuru." (The man ran away)
    # Igbo demonstrative comes AFTER noun
    intransitive_det = Template(
        name="subj_det-verb",
        template_string="{noun_subj} {det_subj} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner (post-nominal)",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            )
        },
        constraints=[],
        description="Intransitive sentence with demonstrative determiner",
        language_code="ibo",
    )
    generic_templates.append(intransitive_det)


    # 3: Intransitive with Preposition
    # Example: "Nwoke ahụ nọrọ na ụlọ." (That man stayed at home)
    intransitive_prep = Template(
        name="subj_det-verb-prep_pobj",
        template_string="{noun_subj} {det_subj} {verb} {prep} {noun_pobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "prep": Slot(
                name="prep",
                description="preposition",
                constraints=[Constraint(expression="self.features.get('pos')=='ADP'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="prepositional object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            )
        },
        constraints=[],
        description="Intransitive sentence with preposition",
        language_code="ibo",
    )
    generic_templates.append(intransitive_prep)


    # 4: Intransitive with Preposition and Determiner
    # Example: "Nwata ahụ gara n’ụlọ ahụ." (The child went to that house)
    intransitive_prep_det = Template(
        name="subj_det-verb-prep_pobj_det",
        template_string="{noun_subj} {det_subj} {verb} {prep} {noun_pobj} {det_pobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "prep": Slot(
                name="prep",
                description="preposition",
                constraints=[Constraint(expression="self.features.get('pos')=='ADP'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="prepositional object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_pobj": Slot(
                name="det_pobj",
                description="prepositional object demonstrative",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            )
        },
        constraints=[],
        description="Intransitive sentence with preposition and demonstrative",
        language_code="ibo",
    )
    generic_templates.append(intransitive_prep_det)


    # 5: Basic Transitive (Subject + Verb + Object)
    # Example: "Obi jere ahịa." (Obi went to Market)
    transitive_basic = Template(
        name="subj-verb-obj",
        template_string="{noun_subj} {verb} {noun_dobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            )
        },
        constraints=[],
        description="Basic transitive sentence (SVO)",
        language_code="ibo",
    )
    generic_templates.append(transitive_basic)


    # 6: Transitive with Subject Demonstrative
    # Example: "Nwoke ahụ riri nri." (The man ate food.)
    transitive_det_subj = Template(
        name="subj_det-verb-obj",
        template_string="{noun_subj} {det_subj} {verb} {noun_dobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            )
        },
        constraints=[],
        description="Transitive sentence with subject demonstrative",
        language_code="ibo",
    )
    generic_templates.append(transitive_det_subj)


    # 7: Transitive with Object Demonstrative
    # Example: "Obi riri nri ahụ." (Obi ate that food.)
    transitive_det_obj = Template(
        name="subj-verb-obj_det",
        template_string="{noun_subj} {verb} {noun_dobj} {det_dobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_dobj": Slot(
                name="det_dobj",
                description="direct object demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            )
        },
        constraints=[],
        description="Transitive sentence with object demonstrative",
        language_code="ibo",
    )
    generic_templates.append(transitive_det_obj)


    # 8: Transitive with Both Demonstratives
    # Example: "Nwoke ahụ riri nri ahụ." (The man ate that food.)
    transitive_det_both = Template(
        name="subj_det-verb-obj_det",
        template_string="{noun_subj} {det_subj} {verb} {noun_dobj} {det_dobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_dobj": Slot(
                name="det_dobj",
                description="direct object demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            )
        },
        constraints=[],
        description="Transitive sentence with both demonstratives",
        language_code="ibo",
    )
    generic_templates.append(transitive_det_both)


    #  9: Transitive with Prepositional Phrase
    # Example: "Nwoke ahụ zụrụ nri na ahịa." (The man bought food at market)
    transitive_prep = Template(
        name="subj_det-verb-obj-prep_pobj",
        template_string="{noun_subj} {det_subj} {verb} {noun_dobj} {prep} {noun_pobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "prep": Slot(
                name="prep",
                description="preposition",
                constraints=[Constraint(expression="self.features.get('pos')=='ADP'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="prepositional object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            )
        },
        constraints=[],
        description="Transitive sentence with prepositional phrase",
        language_code="ibo",
    )
    generic_templates.append(transitive_prep)

    # 10: Transitive with Full Demonstratives
    # Example: "Nwoke ahụ zụrụ nri ahụ na ahịa ahụ." (The man bought that food from that market)
    transitive_prep_full_det = Template(
        name="subj_det-verb-obj_det-prep_pobj_det",
        template_string="{noun_subj} {det_subj} {verb} {noun_dobj} {det_dobj} {prep} {noun_pobj} {det_pobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_dobj": Slot(
                name="det_dobj",
                description="direct object demonstrative",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "prep": Slot(
                name="prep",
                description="preposition",
                constraints=[Constraint(expression="self.features.get('pos')=='ADP'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="prepositional object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_pobj": Slot(
                name="det_pobj",
                description="prepositional object demonstrative",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            )
        },
        constraints=[],
        description="Transitive sentence with all demonstratives",
        language_code="ibo",
    )
    generic_templates.append(transitive_prep_full_det)


    # 11: Basic Ditransitive (Subject + Verb + IndirectObj + DirectObj)
    # Example: "Obi nyere Ada ego." (Obi gave Ada money.)
    ditransitive_basic = Template(
        name="subj-verb-iobj-dobj",
        template_string="{noun_subj} {verb} {noun_iobj} {noun_dobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_iobj": Slot(
                name="noun_iobj",
                description="indirect object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            )
        },
        constraints=[],
        description="Basic ditransitive sentence (S V IO DO)",
        language_code="ibo",
    )
    generic_templates.append(ditransitive_basic)


    # 12: Ditransitive with Subject Demonstrative
    # Example: "Nwoke ahụ nyere Ada ego." (The man gave Ada money)
    ditransitive_det_subj = Template(
        name="subj_det-verb-iobj-dobj",
        template_string="{noun_subj} {det_subj} {verb} {noun_iobj} {noun_dobj}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_iobj": Slot(
                name="noun_iobj",
                description="indirect object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            )
        },
        constraints=[],
        description="Ditransitive sentence with subject demonstrative",
        language_code="ibo",
    )
    generic_templates.append(ditransitive_det_subj)


    # 13: Ditransitive with All Demonstratives
    # Example: "Nwoke ahụ nyere nwata ahụ ego ahụ." (The man gave that child the money)
    ditransitive_det_all = Template(
        name="subj_det-verb-iobj_det-dobj_det",
        template_string="{noun_subj} {det_subj} {verb} {noun_iobj} {det_iobj} {noun_dobj} {det_dobj}.",
        slots={
            "noun_subj": Slot( 
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_subj": Slot(
                name="det_subj",
                description="subject demonstrative determiner",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb (simple past)",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('tense')=='PST'")]
            ),
            "noun_iobj": Slot(
                name="noun_iobj",
                description="indirect object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_iobj": Slot(
                name="det_iobj",
                description="indirect object demonstrative",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="direct object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "det_dobj": Slot(
                name="det_dobj",
                description="direct object demonstrative",
                constraints=[Constraint(expression="self.features.get('pos')=='DET'")]
            )
        },
        constraints=[],
        description="Ditransitive sentence with all demonstratives",
        language_code="ibo",
    )
    generic_templates.append(ditransitive_det_all)



    if template_limit:
        generic_templates = generic_templates[:template_limit]

    # Save templates to JSONL
    output_path = templates_dir / "generic_frames.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for template in generic_templates:
            template_json = template.model_dump_json()
            f.write(template_json + "\n")

    print("=" * 80)
    print("TEMPLATE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(generic_templates)} generic templates:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Template objects for Igbo argument structure dataset"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of templates to generate (for testing)",
    )
    args = parser.parse_args()

    main(template_limit=args.limit)