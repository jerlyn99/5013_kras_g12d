#sanity check for PAINS, aggregators

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

def init_pains():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)

PAINS = init_pains()


REACTIVE_SMARTS = {
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "acyl_halide": "C(=O)[Cl,Br,I]",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "isocyanate": "N=C=O",
    "aldehyde": "[CX3H1](=O)[#6]",
    "michael_acceptor": "C=CC=O",
    "thiol": "[SH]",
    "hydrazine": "NN",
}

REACTIVE = {
    name: Chem.MolFromSmarts(sma)
    for name, sma in REACTIVE_SMARTS.items()
}

def aggregator_flags(mol):
    flags = []

    mw = rdMolDescriptors.CalcExactMolWt(mol)
    logp = Crippen.MolLogP(mol)
    arom = Lipinski.NumAromaticRings(mol)
    psa = rdMolDescriptors.CalcTPSA(mol)

    #poor aqueous solubility and high off target risk
    if mw > 500 and logp > 4.5:
        flags.append("high_MW_high_logP")

    #flat and promiscuous, high toxicity risk and poorly solvated
    if arom >= 3 and psa < 60:
        flags.append("aromatic_hydrophobic")

    return flags

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import csv
import os
import sys

BATCH_SIZE = 100_000

# ---------------------------
# Top-level function (must be picklable)
# ---------------------------
def process_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        smi, zid = line.split()
    except ValueError:
        return [line, "BAD_FORMAT", "", "", ""]

    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return [smi, zid, "INVALID_SMILES", "", ""]
        Chem.SanitizeMol(mol)

        pains = "YES" if PAINS.HasMatch(mol) else "NO"

        reactive_hits = [
            name for name, patt in REACTIVE.items()
            if mol.HasSubstructMatch(patt)
        ]
        reactive_str = ",".join(reactive_hits) if reactive_hits else "NONE"

        agg_flags = aggregator_flags(mol)
        agg_str = ",".join(agg_flags) if agg_flags else "NONE"

        return [smi, zid, pains, reactive_str, agg_str]

    except Exception as e:
        return [smi, zid, f"INVALID_SMILES:{str(e)}", "", ""]

# ---------------------------
# Batch processing
# ---------------------------
def process_batch(lines, outfile, num_cores, write_header):
    with Pool(num_cores) as pool, open(outfile, "a", newline='') as fout:
        writer = csv.writer(fout, delimiter="\t")
        if write_header:
            writer.writerow(["SMILES", "ZINC_ID", "PAINS", "REACTIVE_GROUPS", "AGGREGATOR_FLAGS"])

        for row in tqdm(pool.imap_unordered(process_line, lines), total=len(lines), desc="Batch processing"):
            if row:
                writer.writerow(row)

# ---------------------------
# Main sanity pass
# ---------------------------
def sanity_pass_batched(infile, outfile):
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores.")

    file_exists = os.path.exists(outfile)
    mode = "a" if file_exists else "w"

    with open(infile) as f:
        batch = []
        batch_num = 1
        for line in f:
            batch.append(line)
            if len(batch) >= BATCH_SIZE:
                print(f"Processing batch {batch_num} ({len(batch)} molecules)")
                process_batch(batch, outfile, num_cores, not file_exists)
                batch = []
                batch_num += 1
                file_exists = True

        if batch:
            print(f"Processing batch {batch_num} ({len(batch)} molecules)")
            process_batch(batch, outfile, num_cores, not file_exists)

# ---------------------------
# Important: only run multiprocessing under this guard
# ---------------------------
if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    sanity_pass_batched(infile, outfile)