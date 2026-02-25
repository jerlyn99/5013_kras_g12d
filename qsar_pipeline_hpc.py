import numpy as np
import pandas as pd
import sys
import joblib
import os
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, RDKFingerprint, SDWriter
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.feature_selection import mutual_info_regression
from concurrent.futures import ProcessPoolExecutor
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

# Inside your main() or init_worker
# This ensures each worker only uses 1 thread for math, 
# letting the ProcessPoolExecutor handle the parallelism.
threadpool_limits(limits=1, user_api='blas')

# --- 1. GLOBAL CONSTANTS (Structural Rules) ---
# Defined here so EVERY function can see them immediately
ECFP_LEN = 2048
MACCS_LEN = 167
RDKIT_LEN = 2048
TOTAL_LEN = ECFP_LEN + MACCS_LEN + RDKIT_LEN

# --- 2. GLOBAL VARIABLES (Heavy Data) ---
# Initialized as None; populated by init_worker later
MODEL = None
TRAIN_REF = None

class clean_features(BaseEstimator, TransformerMixin):
    """
    Remove features with more than 10% missing values

    Returns: list of features to keep
    """
    def __init__(self):
        self.to_keep_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Remove descriptors with >10% missing values
        missing_counts = X.isna().sum()
        valid_features = missing_counts[missing_counts <= len(X) * 0.1].index.tolist()
        self.to_keep_ = valid_features

        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_filtered = X[self.to_keep_].copy()
        return X_filtered

class remove_correlated_features(BaseEstimator, TransformerMixin):
    """
    Remove highly correlated features from a DataFrame.

    For each pair with |correlation| > threshold, removes the
    feature with higher mean absolute correlation to all others.

    Parameters:
    - threshold: correlation threshold (default 0.95)

    Returns: list of features to keep
    """
    def __init__(self, threshold = 0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        target_corr = X.corrwith(y).abs()

        # Get upper triangle (avoid duplicate pairs)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        for col in upper_tri.columns:
            # Find features correlated above threshold with this column
            correlated = upper_tri.index[upper_tri[col] > self.threshold].tolist()
            for corr_feat in correlated:
                # Keep feature with lower mean correlation
                if target_corr[col] > target_corr[corr_feat]:
                    self.to_drop_.append(corr_feat)
                else:
                    self.to_drop_.append(col)
        self.to_drop_ = list(set(self.to_drop_))
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        kept_columns = [c for c in X.columns if c not in self.to_drop_]
        X_filtered = X[kept_columns].copy()
        return X_filtered

def mi_reproducible(X, y):
    return mutual_info_regression(X, y, random_state=42)

def standardize_smiles(smiles):
    """
    Standardize a SMILES string for consistent fingerprint generation.
    
    Steps:
    1. Parse SMILES
    2. Remove salts and counterions
    3. Neutralize charges where possible
    4. Generate canonical SMILES
    
    Returns: (standardized_smiles, error_message)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        
        # Remove salts
        remover = rdMolStandardize.FragmentRemover()
        mol = remover.remove(mol)
        
        # Neutralize
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        # Generate canonical SMILES
        std_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        return std_smiles, None
        
    except Exception as e:
        return None, str(e)

def generate_ecfp4(smiles, radius=2, n_bits=2048):
    """
    Generate ECFP4 fingerprint from SMILES.
    
    Parameters:
    - smiles: Input SMILES string
    - radius: Morgan fingerprint radius (2 for ECFP4)
    - n_bits: Fingerprint length (2048 standard)
    
    Returns: (fingerprint_array, error_message)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        fp_array = morgan_gen.GetFingerprintAsNumPy(mol)
        
        return fp_array, None
        
    except Exception as e:
        return None, str(e)
    

def generate_maccs_fingerprint(smiles):
    """
    Generate MACCS keys fingerprint (166 bits).

    MACCS keys are predefined structural patterns.
    Each bit has a known chemical interpretation.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        
        fp = MACCSkeys.GenMACCSKeys(mol)

        return np.array(fp, dtype=float), None
    
    except Exception as e:
        return None, str(e)

def generate_rdkit_fingerprint(smiles, n_bits=2048):
    """
    Generate RDKit topological fingerprint.

    Encodes hashed paths through the molecular graph.
    Complementary to circular fingerprints like ECFP.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        
        # shape is 1D (n_bits,)
        fp = RDKFingerprint(mol, fpSize=n_bits)

        return np.array(fp), None
    
    except Exception as e:
        return None, str(e)

def calculate_tanimoto_similarity(fp1, fp_matrix):
    """
    Calculate Tanimoto similarity between a fingerprint and a matrix of fingerprints.
    """
    intersection = fp1 @ fp_matrix.T
    sum_fp1 = fp1.sum()
    sum_fp_matrix = fp_matrix.sum(axis=1)
    union = sum_fp1 + sum_fp_matrix - intersection
    
    similarity = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0
    )
    
    return similarity
 
def assess_applicability_domain(combined_fp, X_train_ref, k=5, threshold=0.3):
    """
    Assess if a compound is within the applicability domain.
    
    Returns: (inside_ad, knn_similarity, nearest_neighbor_sim)
    """

    if X_train_ref is None:
        return None, None, None

    fp_dict = {
        "ecfp4": combined_fp[:, :ECFP_LEN],
        "maccs": combined_fp[:, ECFP_LEN:ECFP_LEN+MACCS_LEN],
        "rdkit": combined_fp[:, ECFP_LEN+MACCS_LEN:]
    }

    X_train_dict = {
        "ecfp4": X_train_ref[:, :ECFP_LEN],
        "maccs": X_train_ref[:, ECFP_LEN:ECFP_LEN+MACCS_LEN],
        "rdkit": X_train_ref[:, ECFP_LEN+MACCS_LEN:]
    }

    ad_results = {}
    similarities_all = []

    for name, fp in fp_dict.items():
        #print(fp.shape)
        #print(X_train_dict[name].shape)
        sims = calculate_tanimoto_similarity(fp, X_train_dict[name])

        top_k = np.sort(sims)[-k:]
        knn_sim = top_k.mean()
        nn_sim = sims.max()

        ad_results[name] = {
            "inside_ad": knn_sim >= threshold,
            "knn_similarity": knn_sim,
            "nearest_neighbor_sim": nn_sim
        }

        similarities_all.append(knn_sim)

    #sims = calculate_tanimoto_similarity(combined_fp, X_train_ref)
    #top_k = np.sort(sims)[-k:]
    #knn_sim = top_k.mean()
    #nn_sim = sims.max()

    # Aggregation strategy
    combined_knn = np.mean(similarities_all)
    inside_ad_combined = combined_knn >= threshold

    #ad_results = {
    #        "inside_ad": inside_ad_combined,
    #        "knn_similarity": knn_sim,
    #        "nearest_neighbor_sim": nn_sim
    #    }

    return inside_ad_combined, combined_knn, ad_results

    #return knn_sim >= threshold, knn_sim, ad_results


def init_worker(model_path, train_ref_path):
    """
    Loads the model and reference data into the worker's global memory.
    On Linux, this often uses 'Copy-on-Write', making it memory efficient.
    """
    global MODEL, TRAIN_REF
    MODEL = joblib.load(model_path)
    # Load reference data; ensure this matches the feature count used in training
    TRAIN_REF = pd.read_csv(train_ref_path)
    # Remove metadata if present to leave only numeric features
    metadata_cols = ['molecule_chembl_id', 'smiles', 'pIC50']
    TRAIN_REF = TRAIN_REF.drop(columns=[c for c in metadata_cols if c in TRAIN_REF.columns]).values

def process_molecule(args):
    """
    The function executed by each CPU core.
    """
    smiles, name, ad_k, ad_threshold = args

    # Use your standardization function
    std_smiles, error = standardize_smiles(smiles)
    if error:
        return {'compound_name': name, 'valid': False, 'error': error}
    
    # Use your fingerprint functions
    ecfp4_fp, err1 = generate_ecfp4(std_smiles)
    maccs_fp, err2 = generate_maccs_fingerprint(std_smiles)
    rdkit_fp, err3 = generate_rdkit_fingerprint(std_smiles)

    # Collect all errors that are not None
    errors = [e for e in [err1, err2, err3] if e is not None]

    if errors:
        # Join errors into a single string to store in the CSV
        combined_error_msg = " | ".join(errors)
        return {
            'compound_name': name, 
            'input_smiles': smiles,
            'valid': False, 
            'error': f"Fingerprinting failed: {combined_error_msg}"
        }
    
    
    # 3. Predict
    combined_fp = np.concatenate([ecfp4_fp, maccs_fp, rdkit_fp])
    # Column names must match training (ECFP4_0... MACCS_0... RDKit_0...)
    ecfp4_cols = [f'ECFP4_{i}' for i in range(ECFP_LEN)]
    maccs_cols = [f'MACCS_{i}' for i in range(MACCS_LEN)]
    rdkit_cols = [f'RDKit_{i}' for i in range(RDKIT_LEN)]
    
    combined_df = pd.DataFrame([combined_fp], columns=ecfp4_cols + maccs_cols + rdkit_cols)
    pIC50_pred = MODEL.predict(combined_df)[0]

    # 4. AD Assessment
    inside_ad, knn_sim, _ = assess_applicability_domain(
        combined_df.values, TRAIN_REF, ad_k, ad_threshold
    )
    # 5. Return dict with flattened fingerprint for easy DF conversion
    res = {
        'compound_name': name,
        'input_smiles': smiles,
        'standardized_smiles': std_smiles,
        'valid': True,
        'pIC50_pred': pIC50_pred,
        'inside_ad': inside_ad,
        'knn_similarity': knn_sim,
        'reliability': 'High' if inside_ad else 'Low'
    }
    
    # Append fingerprint bits directly to dict
    for i, val in enumerate(ecfp4_fp): res[f'ECFP4_{i}'] = val
    for i, val in enumerate(maccs_fp): res[f'MACCS_{i}'] = val
    for i, val in enumerate(rdkit_fp): res[f'RDKit_{i}'] = val
    
    return res

def export_to_sdf(self, results_df, output_file='predictions.sdf'):
        """Export predictions to SDF format with properties."""
        writer = SDWriter(output_file)
        n_written = 0
        
        for _, row in results_df.iterrows():
            if not row['valid']:
                continue
            
            mol = Chem.MolFromSmiles(row['standardized_smiles'])
            if mol is None:
                continue
            
            # Add properties
            mol.SetProp('compound_name', str(row['compound_name']))
            mol.SetProp('pIC50_pred', f"{row['pIC50_pred']:.3f}")
            mol.SetProp('reliability', str(row['reliability']))
            
            if row['inside_ad'] is not None:
                mol.SetProp('inside_AD', str(row['inside_ad']))
                mol.SetProp('kNN_similarity', f"{row['knn_similarity']:.3f}")
            
            writer.write(mol)
            n_written += 1
        
        writer.close()
        print(f"Exported {n_written} compounds to {output_file}")

def main():
    # File paths from CLI
    model_file = sys.argv[1]
    train_file = sys.argv[2]
    input_file = sys.argv[3]
    n_workers = int(sys.argv[4])
    
    # Setup Scratch Storage
    # PBS usually defines $TMPDIR as the local node SSD
    scratch_dir = os.environ.get('TMPDIR', './scratch_local')
    os.makedirs(scratch_dir, exist_ok=True)
    print(f"Using scratch directory: {scratch_dir}")

    lib = pd.read_csv(input_file)
    chunk_size = 10000
    chunk_files = []

    # Initialize Pool with the 'initializer'
    with ProcessPoolExecutor(max_workers=n_workers, 
                             initializer=init_worker, 
                             initargs=(model_file, train_file)) as executor:
        
        for i in range(0, len(lib), chunk_size):
            chunk_df = lib.iloc[i : i + chunk_size]
            tasks = [(row['SMILES'], row['ID'], 5, 0.3) for _, row in chunk_df.iterrows()]
            
            # Map the tasks to workers
            results = list(tqdm(executor.map(process_molecule, tasks), 
                                total=len(tasks), desc=f"Batch {i//chunk_size}"))
            
            # Save to LOCAL SCRATCH
            chunk_path = os.path.join(scratch_dir, f"chunk_{i}.csv")
            pd.DataFrame(results).to_csv(chunk_path, index=False)
            chunk_files.append(chunk_path)

    # Final Consolidation: Read from scratch and save to final destination
    print("Consolidating results to permanent storage...")
    final_df = pd.concat([pd.read_csv(f) for f in chunk_files])
    final_df.to_csv("predictions_final.csv", index=False)
    print("Done!")

    # Export minimal results (no fingerprints)
    minimal_cols = [
        'compound_name', 'input_smiles', 'standardized_smiles', 'valid',
        'pIC50_pred',
        'inside_ad', 'knn_similarity', 'reliability'
    ]

    results_minimal = results[minimal_cols]
    results_minimal.to_csv('predictions/predictions_minimal.csv', index=False)
    print("Minimal results saved to: predictions_minimal.csv")

    # Filter for high-confidence predictions
    high_confidence = results[
        (results['valid'] == True) &
        (results['reliability'] == 'High')
    ].copy()

    # Sort by predicted potency
    high_confidence = high_confidence.sort_values('pIC50_pred', ascending=False)

    print(f"High-confidence predictions: {len(high_confidence)} compounds")
    print()
    print("Top predictions (sorted by pIC50):")
    print(high_confidence[['compound_name', 'pIC50_pred', 'reliability']].head(10).to_string(index=False))

    export_to_sdf(results, 'predictions/prediction.sdf')

if __name__ == "__main__":
    main()