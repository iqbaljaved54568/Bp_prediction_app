"""
Core prediction logic for the Boiling Point Predictor app.
Reproduces the exact descriptor-computation, scaling, and LightGBM
inference pipeline from the original training script.
"""
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import descList
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from mordred import Calculator, descriptors as mord_desc

RANDOM_SEED = 42
MORDRED_NAMES = ["MATS1se", "GATS1m", "ATSC1c", "BCUTi-1h", "ATS3m",
                  "AATS1i", "ATSC1i", "BCUTi-1l", "AATSC1d"]

_mordred_calc = Calculator(mord_desc, ignore_3D=False)
_rdkit_2d_map = dict(descList)


class DescriptorError(Exception):
    """Raised when a SMILES string can't be turned into descriptors."""
    pass


def smiles_to_descriptors(smiles: str, feature_order: list) -> np.ndarray:
    """
    Reproduces the training pipeline exactly:
      - rdkit2d_Ipc  <- computed on the plain 2D mol (Chem.MolFromSmiles)
      - mordred_*    <- computed on the H-explicit, UFF-optimized 3D mol
    Returns a (1, n_features) array in `feature_order`, or raises DescriptorError.
    """
    smiles = smiles.strip()
    if not smiles:
        raise DescriptorError("Empty SMILES string.")

    mol2d = Chem.MolFromSmiles(smiles)
    if mol2d is None:
        raise DescriptorError(f"Could not parse SMILES: '{smiles}'. Check it is valid.")

    # ---- rdkit 2D: Ipc ----
    try:
        ipc_val = _rdkit_2d_map["Ipc"](mol2d)
    except Exception as e:
        raise DescriptorError(f"Failed computing rdkit2d_Ipc: {e}")

    # ---- 3D embed + UFF optimize (same settings as training) ----
    mol3d = Chem.AddHs(mol2d)
    embed_status = AllChem.EmbedMolecule(
        mol3d, randomSeed=RANDOM_SEED,
        useExpTorsionAnglePrefs=True, useBasicKnowledge=True,
    )
    if embed_status != 0:
        raise DescriptorError(
            "3D conformer embedding failed for this molecule. "
            "This compound cannot currently be scored by the model."
        )
    opt_status = AllChem.UFFOptimizeMolecule(mol3d, maxIters=500)
    if opt_status != 0:
        raise DescriptorError(
            "3D geometry optimization (UFF) failed to converge for this molecule."
        )

    try:
        mordred_res = _mordred_calc(mol3d).asdict()
    except Exception as e:
        raise DescriptorError(f"Mordred descriptor calculation failed: {e}")

    values = {"rdkit2d_Ipc": ipc_val}
    for name in MORDRED_NAMES:
        val = mordred_res.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise DescriptorError(f"Descriptor mordred_{name} could not be computed (missing/NaN).")
        values[f"mordred_{name}"] = float(val)

    try:
        ordered = np.array([[values[f] for f in feature_order]], dtype=float)
    except KeyError as e:
        raise DescriptorError(f"Missing expected descriptor: {e}")

    return ordered


def compute_leverage(X_ref: np.ndarray, x_query: np.ndarray) -> float:
    """Williams-plot leverage of a query point against the training reference set."""
    U, S, Vt = np.linalg.svd(X_ref, full_matrices=False)
    tol = S.max() * max(X_ref.shape) * np.finfo(float).eps * 100
    S_inv = np.where(S > tol, 1.0 / S**2, 0.0)
    h = (x_query @ Vt.T) ** 2 @ S_inv
    return float(h[0])


class BoilingPointPredictor:
    def __init__(self, model_path, scaler_path, feature_names_path, ad_ref_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(feature_names_path)
        ad_ref = joblib.load(ad_ref_path)
        self.h_star = ad_ref["h_star"]
        self.sigma = ad_ref["sigma"]
        self.X_train_sc = ad_ref["X_train_sc"]

    def predict(self, smiles: str) -> dict:
        x_raw = smiles_to_descriptors(smiles, self.feature_names)   # (1, 10)
        x_scaled = self.scaler.transform(x_raw)                      # (1, 10)
        y_pred = float(self.model.predict(x_scaled)[0])              # Kelvin

        h = compute_leverage(self.X_train_sc, x_scaled)
        in_ad = h <= self.h_star

        return {
            "smiles": smiles,
            "predicted_bp_K": y_pred,
            "predicted_bp_C": y_pred - 273.15,
            "leverage": h,
            "h_star": self.h_star,
            "within_applicability_domain": in_ad,
            "approx_uncertainty_K": 3 * self.sigma,  # +/- band, ~99% of in-domain errors
            "raw_descriptors": dict(zip(self.feature_names, x_raw[0].tolist())),
        }
