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

def process_library(df, n_workers, smiles_column='canonical_smiles', id_column='ID'):
    smiles_list = df[smiles_column].tolist()
    
    print(f"Starting sanitization on {n_workers} cores...")
    
    results = []
    with Pool(n_workers) as pool:
        # pool.imap allows tqdm to track progress as items are completed
        # we use a chunksize to reduce the overhead of passing data between cores
        for result in tqdm(pool.imap(sanitize_smiles, smiles_list, chunksize=1000), 
                           total=len(smiles_list), 
                           desc="Sanitizing library"):
            results.append(result)
            
    df['clean_smiles'] = results
    
    # Post-processing: remove failed molecules and duplicates
    initial_count = len(df)
    df = df.dropna(subset=['clean_smiles'])
    df = df.drop_duplicates(subset=['clean_smiles'])
    df = df[["clean_smiles", id_column]].rename(columns={"clean_smiles": "SMILES", id_column: "ID"})
    
    print(f"Finished! Kept {len(df)} of {initial_count} molecules.")
    return df

if __name__ == "__main__":
    library_file = sys.argv[1]
    name = sys.argv[2]
    id_column = sys.argv[3]
    smiles_column = sys.argv[4]
    n_workers = int(sys.argv[5])
    df = process_library(pd.read_csv(library_file), n_workers, smiles_column, id_column)
    df.to_csv(f"{name}", index=False)
    
