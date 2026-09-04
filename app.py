"""
Boiling Point Predictor — Streamlit app
Predicts the normal boiling point of an organic compound from its SMILES
string, using a LightGBM QSPR model trained on DIPPR data.
"""
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from predictor import BoilingPointPredictor, DescriptorError

st.set_page_config(page_title="Boiling Point Predictor", page_icon="🌡️", layout="centered")


@st.cache_resource
def load_predictor():
    return BoilingPointPredictor(
        model_path="best_model_lgbm.pkl",
        scaler_path="scaler.pkl",
        feature_names_path="feature_names.pkl",
        ad_ref_path="ad_reference.pkl",
    )


predictor = load_predictor()

st.title("🌡️ Organic Compound Boiling Point Predictor")
st.markdown(
    "Predicts the **normal boiling point** of an organic compound from its "
    "**SMILES** string, using a LightGBM QSPR model (10 molecular descriptors, "
    "trained on NIST- and DIPPR-sourced data). Companion tool for: Javed et al., "
    "*Interpretable and Parsimonious QSPR Modelling of the Normal Boiling Point "
    "of Structurally Diverse Organic Compounds* (submitted to *J. Chem. Inf. Model.*)."
)

smiles_input = st.text_input(
    "Enter a SMILES string",
    placeholder="e.g. CCO for ethanol, c1ccccc1 for benzene",
)

example_col1, example_col2, example_col3 = st.columns(3)
with example_col1:
    if st.button("Try ethanol"):
        smiles_input = "CCO"
with example_col2:
    if st.button("Try benzene"):
        smiles_input = "c1ccccc1"
with example_col3:
    if st.button("Try aspirin"):
        smiles_input = "CC(=O)OC1=CC=CC=C1C(=O)O"

if smiles_input:
    mol_preview = Chem.MolFromSmiles(smiles_input)

    if mol_preview is None:
        st.error(
            f"Could not parse **'{smiles_input}'** as a valid SMILES string. "
            "Please check the structure and try again."
        )
    else:
        col_img, col_result = st.columns([1, 1.4])

        with col_img:
            img = Draw.MolToImage(mol_preview, size=(280, 280))
            st.image(img, caption="Parsed structure")

        with st.spinner("Computing descriptors and running the model..."):
            try:
                result = predictor.predict(smiles_input)
            except DescriptorError as e:
                result = None
                st.error(f"⚠️ {e}")

        if result is not None:
            with col_result:
                st.metric(
                    "Predicted Boiling Point",
                    f"{result['predicted_bp_K']:.1f} K",
                    delta=f"{result['predicted_bp_C']:.1f} °C",
                    delta_color="off",
                )

                if result["within_applicability_domain"]:
                    st.success(
                        "✅ **Within applicability domain** — this compound is "
                        "structurally similar to the training data, so the "
                        "prediction should be reasonably reliable."
                    )
                else:
                    st.warning(
                        "🚫 **Outside applicability domain** — this compound is "
                        "structurally different from the training data. "
                        "Treat this prediction with caution; it may be an "
                        "unreliable extrapolation."
                    )

                st.caption(
                    f"Typical uncertainty band: ± {result['approx_uncertainty_K']:.0f} K "
                    f"(worst-case, ~3σ of validation error) · "
                    f"Leverage: {result['leverage']:.5f} "
                    f"(threshold h* = {result['h_star']:.5f})"
                )

            with st.expander("Show computed descriptor values"):
                for name, val in result["raw_descriptors"].items():
                    st.write(f"`{name}` = {val:.6g}")

st.divider()
st.caption(
    "Model: LightGBM, 10 descriptors (1 RDKit-2D + 9 Mordred). "
    "Trained on NIST- and DIPPR-sourced boiling point data. "
    "Predictions outside the applicability domain, or for compounds that "
    "decompose/sublime rather than boil cleanly, may not reflect physical reality. "
    "Full model code and reproducibility archive: "
    "[BP-boiling-point-model](https://github.com/iqbaljaved54568/BP-boiling-point-model)."
)
    "Model: LightGBM, 10 descriptors (1 RDKit-2D + 9 Mordred). "
    "Trained on DIPPR normal boiling point data. "
    "Predictions outside the applicability domain, or for compounds that "
    "decompose/sublime rather than boil cleanly, may not reflect physical reality."
)
