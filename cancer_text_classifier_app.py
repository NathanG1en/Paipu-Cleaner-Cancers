# streamlit_app.py
import streamlit as st
import polars as pl
from pathlib import Path

from functions import (
    initialize_medspacy_pipeline,
    get_default_target_rules,
    generate_disease_rules,
)

st.set_page_config(page_title="Cancer Text Classifier", page_icon="🔬")

DATA_PATH = Path("data/combined_metadata_noncancer_removed.csv")

@st.cache_resource
def load_pipeline():
    """Load and cache the medspacy pipeline with all rules including disease-specific ones."""
    cancer_rules, non_cancer_rules = get_default_target_rules()
    existing_rules = cancer_rules + non_cancer_rules
    
    nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)
    
    if DATA_PATH.exists():
        all_samples = pl.read_csv(
            str(DATA_PATH),
            schema_overrides={"group": pl.Utf8},
            infer_schema_length=0,
        )
        unique_diseases = all_samples.select("disease").unique().to_series().to_list()
        
        auto_rules, skipped = generate_disease_rules(unique_diseases, nlp, existing_rules)
        
        if auto_rules:
            tm = nlp.get_pipe("medspacy_target_matcher")
            tm.add(auto_rules)
        
        return nlp, len(auto_rules), len(skipped)
    
    return nlp, 0, 0

st.title("🔬 Cancer Text Classifier")
st.markdown("Test how text gets classified by the MedSpacy pipeline.")

# Load pipeline (cached)
with st.spinner("Loading MedSpacy pipeline with disease rules..."):
    nlp, auto_rule_count, skipped_count = load_pipeline()

total_rules = len(nlp.get_pipe('medspacy_target_matcher').rules)
st.success(f"Pipeline loaded: {total_rules} total rules ({auto_rule_count} from diseases, {skipped_count} skipped duplicates)")

# Input
text_input = st.text_area(
    "Enter text to classify:",
    placeholder="e.g., breast cancer tissue, normal healthy sample, acute myeloid leukemia...",
    height=100,
)

# Classification logic
if st.button("Classify", type="primary") or text_input.strip():
    if text_input.strip():
        doc = nlp(text_input)
        
        st.subheader("Results")
        
        if doc.ents:
            cancer_terms = []
            non_cancer_terms = []
            negated_terms = []
            
            for ent in doc.ents:
                if ent.label_ == "CANCER":
                    if ent._.is_negated:
                        negated_terms.append(ent.text)
                    else:
                        cancer_terms.append(ent.text)
                elif ent.label_ == "NON_CANCER":
                    if not ent._.is_negated:
                        non_cancer_terms.append(ent.text)
            
            if cancer_terms and not negated_terms:
                label = "🔴 CANCER"
                color = "red"
            elif non_cancer_terms and not cancer_terms:
                label = "🟢 NOT_CANCER"
                color = "green"
            elif cancer_terms and non_cancer_terms:
                label = "🟡 UNCERTAIN"
                color = "orange"
            elif negated_terms:
                label = "🟢 NOT_CANCER (negated)"
                color = "green"
            else:
                label = "⚪ NO_SIGNAL"
                color = "gray"
            
            st.markdown(f"### Classification: :{color}[{label}]")
            
            st.subheader("Detected Entities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**✅ Cancer terms:**")
                if cancer_terms:
                    for term in cancer_terms:
                        st.markdown(f"- `{term}`")
                else:
                    st.markdown("_None_")
            
            with col2:
                st.markdown("**❌ Negated cancer:**")
                if negated_terms:
                    for term in negated_terms:
                        st.markdown(f"- `{term}`")
                else:
                    st.markdown("_None_")
            
            with col3:
                st.markdown("**🟢 Non-cancer terms:**")
                if non_cancer_terms:
                    for term in non_cancer_terms:
                        st.markdown(f"- `{term}`")
                else:
                    st.markdown("_None_")
            
            with st.expander("Raw entity details"):
                for ent in doc.ents:
                    st.write({
                        "text": ent.text,
                        "label": ent.label_,
                        "is_negated": ent._.is_negated,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    })
        else:
            st.markdown("### Classification: :gray[⚪ NO_SIGNAL]")
            st.info("No cancer-related entities detected in the text.")
    else:
        st.warning("Please enter some text to classify.")

# Static examples section
st.divider()
st.subheader("Example texts to try:")
st.markdown("""
Copy and paste any of these into the text box above:

- `breast cancer tissue`
- `normal healthy sample`
- `no evidence of tumor`
- `metastatic carcinoma`
- `acute myeloid leukemia`
- `glioblastoma multiforme`
- `benign lesion`
- `horn cancer`
- `Mus musculus CD8 TILs`
- `type 2 diabetes`
""")

# Sidebar stats
with st.sidebar:
    st.header("📊 Pipeline Stats")
    st.metric("Total Rules", total_rules)
    st.metric("Disease Rules Added", auto_rule_count)
    st.metric("Duplicates Skipped", skipped_count)
    
    with st.expander("View CANCER rules (sample)"):
        cancer_rules_list = [
            r.literal for r in nlp.get_pipe('medspacy_target_matcher').rules 
            if r.category == "CANCER"
        ][:20]
        for r in cancer_rules_list:
            st.code(r)
        if len(cancer_rules_list) == 20:
            st.caption("...and more")
    
    with st.expander("View NON_CANCER rules (sample)"):
        non_cancer_rules_list = [
            r.literal for r in nlp.get_pipe('medspacy_target_matcher').rules 
            if r.category == "NON_CANCER"
        ][:20]
        for r in non_cancer_rules_list:
            st.code(r)
        if len(non_cancer_rules_list) == 20:
            st.caption("...and more")
