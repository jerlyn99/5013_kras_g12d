import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

def sanitize_smiles(smiles):
    if not isinstance(smiles, str) or not smiles: return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        
        # Standardization Pipeline
        mol = rdMolStandardize.Cleanup(mol)
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        if not mol: return None
        
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        
        te = rdMolStandardize.TautomerEnumerator()
        mol = te.Canonicalize(mol)
        
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        return None

def process_library(df, smiles_column='canonical_smiles'):
    # Determine number of workers (leave 1-2 cores free for your OS)
    n_cores = max(1, cpu_count() - 2)
    smiles_list = df[smiles_column].tolist()
    
    print(f"Starting sanitization on {n_cores} cores...")
    
    results = []
    with Pool(n_cores) as pool:
        # pool.imap allows tqdm to track progress as items are completed
        # we use a chunksize to reduce the overhead of passing data between cores
        for result in tqdm(pool.imap(sanitize_smiles, smiles_list, chunksize=1000), 
                           total=len(smiles_list), 
                           desc="Sanitizing ChEMBL"):
            results.append(result)
            
    df['clean_smiles'] = results
    
    # Post-processing: remove failed molecules and duplicates
    initial_count = len(df)
    df = df.dropna(subset=['clean_smiles'])
    df = df.drop_duplicates(subset=['clean_smiles'])
    df = df.rename(columns={"clean_smiles": "SMILES", "chembl_id": "ID"})
    
    print(f"Finished! Kept {len(df)} of {initial_count} molecules.")
    return df

if __name__ == "__main__":
    library_file = sys.argv[1]
    name = sys.argv[2]
    df = process_library(pd.read_csv(library_file))
    df.to_csv(f"data/{name}", index=False)
    