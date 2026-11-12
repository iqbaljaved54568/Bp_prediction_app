# Boiling Point Predictor (web application)

Predicts the normal boiling point (K) of an organic compound from its SMILES string, with an applicability-domain check.

- **Live app:** https://bppredictionapp-eghwsz9hbquq3kcrcgjfk5.streamlit.app/
- **Archive:** https://doi.org/10.5281/zenodo.22311228
- **Model code and data:** https://github.com/iqbaljaved54568/QSPR-model-to-predict-the-normal-boiling-point-of-organic-compounds (https://doi.org/10.5281/zenodo.22311330)

Paper: Javed, M. I.; Waters, L.; Belton, D.; Cooke, D. *A Ten-Descriptor LightGBM Model for Normal Boiling Point Prediction of Structurally Diverse Organic Compounds: Validation, Applicability Domain and Online Deployment.* Submitted to *ACS Omega*, 2026.

## What it does

1. Reads a SMILES string and adds explicit hydrogens (RDKit).
2. Calculates the ten descriptors of the final model: RDKit `Ipc` and nine Mordred descriptors (MATS1se, GATS1m, ATSC1c, BCUTi-1h, ATS3m, BCUTi-1l, ATSC1i, AATSC1d, AATS1i). All are two-dimensional; no 3D conformer is generated.
3. Scales them with the training-set scaler and predicts Tb with the final LightGBM model (test set: R² = 0.9596, MAE = 12.32 K, RMSE = 18.39 K).
4. Calculates the leverage of the compound relative to the 4,056 training compounds and flags it as **outside the applicability domain** if h > h* = 0.00814.

A prediction takes about 9 ms after start-up.

## Files

| File | Content |
|---|---|
| `app.py` | Streamlit interface |
| `predictor.py` | SMILES → descriptors → scaling → prediction and leverage |
| `best_model_lgbm.pkl` | Final LightGBM model |
| `scaler.pkl` | Training-set StandardScaler |
| `feature_names.pkl` | Descriptor order |
| `ad_reference.pkl` | Standardised training descriptors and h* for the applicability domain (no boiling-point values) |
| `requirements.txt`, `packages.txt` | Python and system packages |

## Run locally

```bash
git clone https://github.com/iqbaljaved54568/Bp_prediction_app.git
cd Bp_prediction_app
pip install -r requirements.txt
streamlit run app.py
```

## Scope and limitations

- Trained on compounds containing only C, H, N, O, P, S, Cl, Br and I, with 2–28 heavy atoms and Tb of 238–743 K. Predictions outside this scope (e.g. organosilicon compounds) are extrapolations and are flagged by the applicability-domain check.
- The model cannot know whether a compound decomposes or sublimes before boiling; use chemical judgement for such compounds.

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0) – see `LICENSE`.

## Contact

Muhammad Iqbal Javed, University of Huddersfield – muhammadiqbal.javed@hud.ac.uk
