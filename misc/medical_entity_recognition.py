from functions import initialize_medspacy_pipeline, get_default_target_rules

cancer_rules, non_cancer_rules = get_default_target_rules()
nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)

# Test
doc = nlp("acute myeloid leukemia")
print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")

doc2 = nlp("glioblastoma multiforme")
print(f"Entities: {[(ent.text, ent.label_) for ent in doc2.ents]}")
