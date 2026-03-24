# import spacy
# import scispacy
# from scispacy.linking import EntityLinker
#
# # 1. Load a large biomedical model
# nlp = spacy.load("en_core_sci_lg")
#
# # 2. Add the UMLS linker to the pipeline
# # This connects text to 4M+ medical concepts
# nlp.add_pipe(
#     "scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}
# )
#
#
# def analyze_sra_metadata(text):
#     doc = nlp(text)
#     linker = nlp.get_pipe("scispacy_linker")
#
#     results = []
#     for ent in doc.ents:
#         # Check if the entity has linked concepts
#         for umls_ent in ent._.kb_ents:
#             concept_id = umls_ent[0]
#             score = umls_ent[1]
#
#             # Fetch the canonical name from the Knowledge Base
#             concept = linker.kb.cui_to_entity[concept_id]
#
#             # Semantic Types for 'Neoplastic Process' (Cancer) are T191, T170, etc.
#             # You can filter results specifically for these types
#             results.append(
#                 {
#                     "term": ent.text,
#                     "canonical": concept.canonical_name,
#                     "cui": concept_id,
#                     "score": score,
#                     "types": concept.types,
#                 }
#             )
#     return results
#
#
# # Example SRA metadata snippet
# metadata_sample = (
#     "Patient with stage III Adenocarcinoma of the lung, negative for KRAS mutation."
# )
# print(analyze_sra_metadata(metadata_sample))

from functions import CANCER_RULE_DEFINITIONS
cancer_rules = []
from medspacy.ner import TargetRule

for literal, category, pattern in CANCER_RULE_DEFINITIONS:
    # print(pattern)
    if pattern:
        cancer_rules.append(TargetRule(literal, category, pattern=pattern))
    else:
        cancer_rules.append(TargetRule(literal, category))
#
# for rule in cancer_rules:
#     print(rule)