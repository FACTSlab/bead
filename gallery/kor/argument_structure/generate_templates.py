#!/usr/bin/env python3
"""
Generate Template objects for argument structure dataset

Output: templates/generic_frames.jsonl
"""

import argparse

from typing import List
from bead.resources import Template, Slot, Constraint

def main(verb_limit: int | None = None) -> None:
    generic_templates: List[Template] = []

    # intransitive sentence
    intransitive = Template(
        name="subj_nom-verb.",
        template_string="{noun_subj}{nom} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )},
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')")],
        description="Intransitive sentence",
        language_code="kor",
    )
    
    generic_templates.append(intransitive)

    
    # intransitive sentence with dative
    intransitive_dative = Template(
        name="subj_nom-noun_dat-verb.",
        template_string="{noun_subj}{nom} {noun_pobj}{dat} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="dative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "dat": Slot(
                name="dat",
                description="dative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.DAT'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )}, 
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')")],
        description="Intransitive sentence with dative argument",
        language_code="kor",
    )
    
    generic_templates.append(intransitive_dative)

    # intransitive sentence with locative
    intransitive_locative = Template(
        name="subj_nom-noun_loc-verb.",
        template_string="{noun_subj}{nom} {noun_pobj}{loc} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="locative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "loc": Slot(
                name="loc",
                description="locative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.LOC'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )}, 
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')")],
        description="Intransitive sentence with locative argument",
        language_code="kor",
    )
    
    generic_templates.append(intransitive_locative)

    # intransitive sentence with instrumental
    intransitive_instrumental = Template(
        name="subj_nom-noun_inst-verb.",
        template_string="{noun_subj}{nom} {noun_pobj}{inst} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="instrumental noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "inst": Slot(
                name="inst",
                description="instrumental case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.INST")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )},
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_pobj.features.get('final_consonant') == inst.features.get('final_consonant')")],
        description="Intransitive sentence with instrumental arguments",
        language_code="kor",
    )
    
    generic_templates.append(intransitive_instrumental)

    # intransitive sentence with instrumental and locative
    intransitive_inst_loc = Template(
        name="subj_nom-noun_inst-noun_loc-verb.",
        template_string="{noun_subj}{nom} {noun_pobj}{inst} {noun_pobj2}{loc} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos') == 'NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos') == 'PART.NOM'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="instrumental noun phrase",
                constraints=[Constraint(expression="self.features.get('pos') == 'NOUN'")]
            ),
            "inst": Slot(
                name="inst",
                description="instrumental case marker",
                constraints=[Constraint(expression="self.features.get('pos') == 'PART.INST'")]
            ),
            "noun_pobj2": Slot(
                name="noun_pobj2",
                description="locative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos') == 'NOUN'")]
            ),
            "loc": Slot(
                name="loc",
                description="locative case marker",
                constraints=[Constraint(expression="self.features.get('pos') == 'PART.LOC'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos') == 'V' and self.features.get('verb_form') != 'V.PTCP'")]
            )},
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_pobj.features.get('final_consonant') == inst.features.get('final_consonant')")],
        description="Intransitive sentence with instrumental and locative arguments",
        language_code="kor",
    )
    
    generic_templates.append(intransitive_inst_loc)

    # intransitive sentence with instrumental and dative
    intransitive_inst_dat = Template(
        name="subj_nom-noun_inst-noun_dat-verb.",
        template_string="{noun_subj}{nom} {noun_pobj}{inst} {noun_pobj2}{dat} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="instrumental noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "inst": Slot(
                name="inst",
                description="instrumental case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.INST'")]
            ),
            "noun_pobj2": Slot(
                name="noun_pobj2",
                description="dative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "dat": Slot(
                name="dat",
                description="dative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.DAT'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )}, 
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_pobj.features.get('final_consonant') == inst.features.get('final_consonant')")],
        description="Intransitive sentence with instrumental and dative arguments",
        language_code="kor",
    )   
    
    generic_templates.append(intransitive_inst_dat)

    # intransitive sentence with locative and dative
    intransitive_loc_dat = Template(
        name="subj_nom-noun_loc-noun_dat-verb.",
        template_string="{noun_subj}{nom} {noun_pobj}{loc} {noun_pobj2}{dat} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="locative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "loc": Slot(
                name="loc",
                description="locative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.LOC'")]
            ),
            "noun_pobj2": Slot(
                name="noun_pobj2",
                description="dative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "dat": Slot(
                name="dat",
                description="dative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.DAT'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )}, 
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')")],
        description="Intransitive sentence with locative and dative arguments",
        language_code="kor",
    )
    
    generic_templates.append(intransitive_loc_dat)
    

    # transitive sentence
    transitive = Template(
        name="subj_nom-obj_acc-verb.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )}, 
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_dobj.features.get('final_consonant') == acc.features.get('final_consonant')")],
        description="Transitive sentence",
        language_code="kor",
    )
    
    generic_templates.append(transitive)

    # transitive sentence with dative
    transitive_dative = Template(
        name="subj_nom-obj_acc-noun_dat-verb.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{dat} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="dative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "dat": Slot(
                name="dat",
                description="dative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.DAT'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )}, 
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_dobj.features.get('final_consonant') == acc.features.get('final_consonant')")],
        description="Transitive sentence with dative argument",
        language_code="kor",
    )
    
    generic_templates.append(transitive_dative)

    # transitive sentence with locative
    transitive_locative = Template(
        name="subj_nom-obj_acc-noun_loc-verb.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{loc} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="locative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "loc": Slot(
                name="loc",
                description="locative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.LOC'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )}, 
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_dobj.features.get('final_consonant') == acc.features.get('final_consonant')")],
        description="Transitive sentence with locative argument",
        language_code="kor",
    )   
    
    generic_templates.append(transitive_locative)

    # transitive sentence with instrumental
    transitive_instrumental = Template(
        name="subj_nom-obj_acc-noun_inst-verb.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{inst} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="instrumental noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "inst": Slot(
                name="inst",
                description="instrumental case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.INST'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )},
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_dobj.features.get('final_consonant') == acc.features.get('final_consonant')")],
        description="Transitive sentence with instrumental argument",
        language_code="kor",
    )
    
    generic_templates.append(transitive_instrumental)

    # transitive sentence with instrumental and locative
    transitive_inst_loc = Template(
        name="subj_nom-obj_acc-noun_inst-noun_loc-verb.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{inst} {noun_pobj2}{loc} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="instrumental noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "inst": Slot(
                name="inst",
                description="instrumental case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.INST'")]
            ),
            "noun_pobj2": Slot(
                name="noun_pobj2",
                description="locative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "loc": Slot(
                name="loc",
                description="locative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.LOC'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )},
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_dobj.features.get('final_consonant') == acc.features.get('final_consonant')"),
                    Constraint(expression="noun_pobj.features.get('final_consonant') == inst.features.get('final_consonant')")],
        description="Transitive sentence with instrumental and locative arguments",
        language_code="kor",
    )
    
    generic_templates.append(transitive_inst_loc)

    
    # transitive sentence with instrumental and dative
    transitive_inst_dat = Template(
        name="subj_nom-obj_acc-noun_inst-noun_dat-verb.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{inst} {noun_pobj2}{dat} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="instrumental noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "inst": Slot(
                name="inst",
                description="instrumental case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.INST'")]
            ),
            "noun_pobj2": Slot(
                name="noun_pobj2",
                description="dative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "dat": Slot(
                name="dat",
                description="dative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.DAT'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )},
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_dobj.features.get('final_consonant') == acc.features.get('final_consonant')"),
                    Constraint(expression="noun_pobj.features.get('final_consonant') == inst.features.get('final_consonant')")],
        description="Transitive sentence with instrumental and dative arguments",
        language_code="kor",
    )
    
    generic_templates.append(transitive_inst_dat)

    # transitive sentence with locative and dative
    transitive_loc_dat = Template(
        name="subj_nom-obj_acc-noun_loc-noun_dat-verb.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {noun_pobj}{loc} {noun_pobj2}{dat} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "noun_pobj": Slot(
                name="noun_pobj",
                description="locative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "loc": Slot(
                name="loc",
                description="locative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.LOC'")]
            ),
            "noun_pobj2": Slot(
                name="noun_pobj2",
                description="dative noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "dat": Slot(
                name="dat",
                description="dative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.DAT'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )},
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant') == nom.features.get('final_consonant')"),
                    Constraint(expression="noun_dobj.features.get('final_consonant') == acc.features.get('final_consonant')")],
        description="Transitive sentence with locative and dative arguments",
        language_code="kor",
    )
    
    generic_templates.append(transitive_loc_dat)

    #TODO: add templates with whether and that clauses and progressive sentences

    # that-clause complement
    subj_verb_that = Template(
        name="subj-verb-that.",
        template_string="{noun_subj}{nom} {comp_subj}{comp_nom} {comp_verb} {verb}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "comp_subj": Slot(
                name="comp_subj",
                description="complement clause subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "comp_nom": Slot(
                name="comp_nom",
                description="complement clause nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "comp_verb": Slot(
                name="comp_verb",
                description="complement clause verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')=='V.PTCP'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')!='V.PTCP'")]
            )
        },
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant')==nom.features.get('final_consonant')"),
                     Constraint(expression="comp_subj.features.get('final_consonant')==comp_nom.features.get('final_consonant')")],
        description="That-clause complement sentence",
        language_code="kor",
    )
    
    generic_templates.append(subj_verb_that)

    # present progressive intransitive sentence
    prog_intransitive = Template(
        name="subj_nom-verb_prog.",
        template_string="{noun_subj}{nom} {verb}{aux}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb stem",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')=='V.PTCP'")]
            ),
            "aux": Slot(
                name="aux",
                description="progressive auxiliary verb",
                constraints=[Constraint(expression="self.features.get('pos')=='AUX' and self.features.get('tense')=='PRS'")]
            )
        },
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant')==nom.features.get('final_consonant')")],
        description="Present progressive intransitive sentence",
        language_code="kor",
    )
    
    generic_templates.append(prog_intransitive)

    # present progressive transitive sentence
    prog_transitive = Template(
        name="subj_nom-obj_acc-verb_prog.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {verb}{aux}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb stem",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')=='V.PTCP'")]
            ),
            "aux": Slot(
                name="aux",
                description="progressive auxiliary verb",
                constraints=[Constraint(expression="self.features.get('pos')=='AUX' and self.features.get('tense')=='PRS'")]
            )
        },
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant')==nom.features.get('final_consonant')"),
                     Constraint(expression="noun_dobj.features.get('final_consonant')==acc.features.get('final_consonant')")],
        description="Present progressive transitive sentence",
        language_code="kor",
    )
    
    generic_templates.append(prog_transitive)

    # past progressive intransitive sentence
    past_prog_intransitive = Template(
        name="subj_nom-verb_past_prog.",
        template_string="{noun_subj}{nom} {verb}{aux}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb stem",
                constraints=[Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')=='V.PTCP'")]
            ),
            "aux": Slot(
                name="aux",
                description="progressive auxiliary verb",
                constraints=[Constraint(expression="self.features.get('pos')=='AUX' and self.features.get('tense')=='PST'")]
            )
        },
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant')==nom.features.get('final_consonant')")],
        description="Past progressive intransitive sentence",
        language_code="kor",
    )
    
    generic_templates.append(past_prog_intransitive)

    # past progressive transitive sentence
    past_prog_transitive = Template(
        name="subj_nom-obj_acc-verb_past_prog.",
        template_string="{noun_subj}{nom} {noun_dobj}{acc} {verb}{aux}.",
        slots={
            "noun_subj": Slot(
                name="noun_subj",
                description="subject noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "nom": Slot(
                name="nom",
                description="nominative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.NOM'")]
            ),
            "noun_dobj": Slot(
                name="noun_dobj",
                description="object noun phrase",
                constraints=[Constraint(expression="self.features.get('pos')=='NOUN'")]
            ),
            "acc": Slot(
                name="acc",
                description="accusative case marker",
                constraints=[Constraint(expression="self.features.get('pos')=='PART.ACC'")]
            ),
            "verb": Slot(
                name="verb",
                description="main verb stem",
                constraints=
                [Constraint(expression="self.features.get('pos')=='V' and self.features.get('verb_form')=='V.PTCP'")]
            ),
            "aux": Slot(
                name="aux",
                description="progressive auxiliary verb",
                constraints=[Constraint(expression="self.features.get('pos')=='AUX' and self.features.get('tense')=='PST'")]
            )
        },
        constraints=[Constraint(expression="noun_subj.features.get('final_consonant')==nom.features.get('final_consonant')"),
                     Constraint(expression="noun_dobj.features.get('final_consonant')==acc.features.get('final_consonant')")],
        description="Past progressive transitive sentence",
        language_code="kor",
    )
    
    generic_templates.append(past_prog_transitive)

    if verb_limit:
        generic_templates = generic_templates[:verb_limit]

    with open("./templates/generic_frames.jsonl", "w") as f:
        for template in generic_templates:
            template_json = template.model_dump_json()
            f.write(template_json + "\n")
    
    print(f"Generated {len(generic_templates)} generic templates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Template objects from VerbNet frames"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of verb-class pairs to process (for testing)",
    )
    args = parser.parse_args()

    main(verb_limit=args.limit)