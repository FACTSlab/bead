# Data Generation Order

## Korean specific files

1. generate_lexicons.py 
- extract verbs from UniMorph
- generate auxiliary verbs, bleached adjectives, nouns, verbs and case markers

2. generate_templates.py
- create generic frames

3. fill_templates.py
- modified version of eng/argument_structure to handle template rendering for Korean

## Same file as in eng/argument_structure

4. create_2afc_pairs.py

5. generate_lists.py

6. generate_deployment.py