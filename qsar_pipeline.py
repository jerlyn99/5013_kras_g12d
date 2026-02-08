import numpy as np
import pandas as pd
import sys
import joblib
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator, MACCSkeys, RDKFingerprint, SDWriter
from rdkit.Chem.MolStandardize import rdMolStandardize

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

        return np.array(fp), None
    
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
    
    ECFP_LEN = 2048
    MACCS_LEN = 167

    fp_dict = {
        "ecfp4": combined_fp[:ECFP_LEN],
        "maccs": combined_fp[ECFP_LEN:ECFP_LEN+MACCS_LEN],
        "rdkit": combined_fp[ECFP_LEN+MACCS_LEN:]
    }

    X_train_dict = {
        "ecfp4": X_train_ref[:ECFP_LEN],
        "maccs": X_train_ref[ECFP_LEN:ECFP_LEN+MACCS_LEN],
        "rdkit": X_train_ref[ECFP_LEN+MACCS_LEN:]
    }

    ad_results = {}
    similarities_all = []

    for name, fp in fp_dict.items():
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

    # Aggregation strategy
    combined_knn = np.mean(similarities_all)
    inside_ad_combined = combined_knn >= threshold

    return inside_ad_combined, combined_knn, ad_results

class KRASInhibitorPredictor:
    """
    Complete prediction pipeline for KRAS G12D pIC50 prediction.
    """
    
    def __init__(self, model, X_train_ref=None, ad_threshold=0.3, ad_k=5):
        self.model = model
        self.X_train_ref = X_train_ref
        self.ad_threshold = ad_threshold
        self.ad_k = ad_k
        
        print("KRASInhibitorPredictor initialized:")
        print(f"  AD enabled: {X_train_ref is not None}")
        if X_train_ref is not None:
            print(f"  AD threshold: {ad_threshold}")
            print(f"  AD k-neighbors: {ad_k}")
    
    def predict_single(self, smiles, compound_name='Unknown'):
        """Predict for a single compound."""
        result = {
            'compound_name': compound_name,
            'input_smiles': smiles,
            'standardized_smiles': None,
            'valid': False,
            'error': None,
            'pIC50_pred': None,
            'inside_ad': None,
            'knn_similarity': None,
            'nn_similarity': None,
            'reliability': None,
            'fingerprint': None
        }
        
        # Standardize
        std_smiles, error = standardize_smiles(smiles)
        if error:
            result['error'] = f'Standardization failed: {error}'
            return result
        result['standardized_smiles'] = std_smiles
        
        # Generate fingerprint
        ecfp4_fp, error_ecfp4 = generate_ecfp4(std_smiles)
        maccs_fp, error_maccs = generate_maccs_fingerprint(std_smiles)
        rdkit_fp, error_rdkit = generate_rdkit_fingerprint(std_smiles)
        if error_ecfp4 or error_maccs or error_rdkit:
            result['valid'] = False
            result['error'] = {
                'ecfp4': error_ecfp4,
                'maccs': error_maccs,
                'rdkit': error_rdkit
            }
            return result

        result['ecfp4_fp'] = np.asarray(ecfp4_fp, dtype=float)
        result['maccs_fp'] = np.asarray(maccs_fp, dtype=float)
        result['rdkit_fp'] = np.asarray(rdkit_fp, dtype=float)
        result['valid'] = True
        
        # Predict
        ecfp4_fp = np.asarray(ecfp4_fp, dtype=float)
        maccs_fp = np.asarray(maccs_fp, dtype=float)
        rdkit_fp = np.asarray(rdkit_fp, dtype=float)

        # Concatenate in TRAINING ORDER (CRITICAL!)
        combined_fp = np.concatenate([ecfp4_fp, maccs_fp, rdkit_fp])

        # Reshape for sklearn
        combined_fp_2d = combined_fp.reshape(1, -1)
        result['pIC50_pred'] = self.model.predict(combined_fp_2d)[0]
        
        # AD assessment
        inside_ad, knn_sim, nn_sim = assess_applicability_domain(
            {'ecfp4': ecfp4_fp, 'maccs': maccs_fp, 'rdkit': rdkit_fp}, self.X_train_ref, self.ad_k, self.ad_threshold
        )
        result['inside_ad'] = inside_ad
        result['knn_similarity'] = knn_sim
        result['nn_similarity'] = nn_sim
        
        # Reliability classification
        if inside_ad is None:
            result['reliability'] = 'Unknown (no AD)'
        elif inside_ad:
            result['reliability'] = 'High'
        else:
            result['reliability'] = 'Low'
        
        return result
    
    def predict_batch(self, df, smiles_col='SMILES', name_col='compound_name'):
        """
        Predict for a batch of compounds from a DataFrame.
        """
        results = []
        n_total = len(df)
        
        print(f"Processing {n_total} compounds...")
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row[smiles_col]
            name = row.get(name_col, f'Compound_{i}')
            
            result = self.predict_single(smiles, name)
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_total}...")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add ecfp4 fingerprint columns
        ecfp4_cols = [f'ECFP4_{i}' for i in range(2048)]
        ecfp4_matrix = np.vstack([
            r['ecfp4_fp'] if r['ecfp4_fp'] is not None else np.zeros(2048)
            for r in results
        ])
        ecfp4_df = pd.DataFrame(ecfp4_matrix, columns=ecfp4_cols)

        # Add maccs fingerprint columns
        maccs_cols = [f'MACCS_{i}' for i in range(167)]
        maccs_matrix = np.vstack([
            r['maccs_fp'] if r['maccs_fp'] is not None else np.zeros(167)
            for r in results
        ])
        maccs_df = pd.DataFrame(maccs_matrix, columns=maccs_cols)

        # Add rdkit fingerprint columns
        rdkit_cols = [f'RDKit_{i}' for i in range(2048)]
        rdkit_matrix = np.vstack([
            r['rdkit_fp'] if r['rdkit_fp'] is not None else np.zeros(2048)
            for r in results
        ])
        rdkit_df = pd.DataFrame(rdkit_matrix, columns=rdkit_cols)
        
        # Combine (without duplicate fingerprint column)
        results_df = results_df.drop(columns=['ecfp4_fp', 'maccs_fp', 'rdkit_fp'])
        results_df = pd.concat([results_df, ecfp4_df, maccs_df, rdkit_df], axis=1)
        
        # Summary
        n_valid = results_df['valid'].sum()
        n_inside_ad = results_df['inside_ad'].sum() if self.X_train_ref is not None else 'N/A'
        
        print(f"\nPrediction complete:")
        print(f"  Valid compounds: {n_valid}/{n_total}")
        print(f"  Inside AD: {n_inside_ad}")
        
        return results_df
 
    def export_to_sdf(results_df, output_file='predictions.sdf'):
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


if __name__ == "__main__":
    model_file = sys.argv[1]
    training_data_file = sys.argv[2]
    input_file = sys.argv[3]

    training_data = pd.read_csv(training_data_file)
    model = joblib.load(model_file)

    metadata_cols=['molecule_chembl_id', 'smiles', 'pIC50']
    feature_cols = [c for c in training_data.columns if c not in metadata_cols]

    predictor = KRASInhibitorPredictor(model,  training_data[feature_cols].values)
    results = predictor.predict_batch(input_file, smiles_col='SMILES', name_col='zinc_id')
 
    # Display key results
    display_cols = [
        'compound_name', 'valid', 'pIC50_pred',
        'inside_ad', 'knn_similarity', 'reliability'
    ]
    print("\nPrediction Results:")
    print(results[display_cols].to_string(index=False))

    # Save full results with fingerprints
    results.to_csv('predictions_full.csv', index=False)
    print("Full results saved to: predictions_full.csv")
    print(f"Columns: {len(results.columns)}")

    # Export minimal results (no fingerprints)
    minimal_cols = [
        'compound_name', 'input_smiles', 'standardized_smiles', 'valid', 'error',
        'pIC50_pred',
        'inside_ad', 'knn_similarity', 'nn_similarity', 'reliability'
    ]

    results_minimal = results[minimal_cols]
    results_minimal.to_csv('predictions_minimal.csv', index=False)
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

    predictor.export_to_sdf(results, 'predictions.sdf')


