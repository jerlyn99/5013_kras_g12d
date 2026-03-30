import sys, time, tempfile, warnings
from rdkit import Chem, RDLogger
from openbabel import openbabel as ob
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdDistGeom
from vina import Vina
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit.Chem.MolStandardize import rdMolStandardize
from copy import deepcopy

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


np.int = int  # Manually re-add the missing alias

#Global Variables
SCORING_FUNCTION = "vinardo"
EXHAUSTIVENESS   = 8
N_POSES          = 1
RANDOM_SEED      = 42
MMFF_MAX_ITERS = 2000
N_CONFORMERS = 1
TARGET_PH= 7.4
PROTONATION_METHOD = "openbabel"


def standardise_mol(mol):
    """Largest fragment selection followed by canonical tautomer."""
    lfc = rdMolStandardize.LargestFragmentChooser()
    te  = rdMolStandardize.TautomerEnumerator()
    mol = lfc.choose(mol)
    mol = te.Canonicalize(mol)
    return mol

def protonate_rdkit(mol):
    """Neutralise all formal charges (pH-unaware)."""
    uc = rdMolStandardize.Uncharger()
    return uc.uncharge(mol)

def protonate_openbabel(mol, ph=7.4):
    """pH-aware protonation via OpenBabel (SMILES round-trip)."""
    try:
        smi    = Chem.MolToSmiles(mol)
        obconv = ob.OBConversion()
        obconv.SetInAndOutFormats("smi", "smi")
        obmol  = ob.OBMol()
        obconv.ReadString(obmol, smi)
        obmol.AddHydrogens(False, True, ph)
        out_smi = obconv.WriteString(obmol).strip()
        out_mol = Chem.MolFromSmiles(out_smi)
        if out_mol is None:
            return mol
        out_mol.SetProp("_Name", mol.GetPropsAsDict().get("_Name", ""))
        return out_mol
    except Exception:
        return mol

def embed_and_minimise(mol, n_confs, max_iters):
    """
    Generate multiple 3D conformers with ETKDGv3, minimise each with
    MMFF94s, and return only the lowest-energy conformer.
    """
    params = rdDistGeom.ETKDGv3()
    params.randomSeed            = 42
    params.numThreads            = 0
    params.useSmallRingTorsions  = True
    params.useMacrocycleTorsions = True
    params.enforceChirality      = True

    mol3d    = Chem.AddHs(mol)
    conf_ids = AllChem.EmbedMultipleConfs(mol3d, numConfs=n_confs, params=params)
    if not conf_ids:
        # Fallback to single-conformer ETKDG
        AllChem.EmbedMolecule(mol3d, AllChem.ETKDG())

    results = AllChem.MMFFOptimizeMoleculeConfs(
        mol3d, mmffVariant="MMFF94s", maxIters=max_iters
    )
    if not results:
        return mol3d, False

    # Select the lowest-energy converged conformer
    energies = [(r[1], i) for i, r in enumerate(results) if r[0] == 0]
    if not energies:
        converged, best_conf = False, 0
    else:
        converged = True
        _, best_conf = min(energies)

    # Remove all conformers except the best one
    mol_out = deepcopy(mol3d)
    for cid in reversed(range(mol_out.GetNumConformers())):
        if cid != best_conf:
            mol_out.RemoveConformer(cid)
    return mol_out, converged

def smiles_to_3d_mol(smiles, name):
    # 1. SMILES to 2D Mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    try:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            raise ValueError(f"Sanitisation failed: {e}")

        # [b+c] Standardise (largest fragment + canonical tautomer)
        mol = standardise_mol(mol)
        mol.SetProp("_Name", name)

        # [d] Protonation
        if PROTONATION_METHOD == "openbabel":
            mol = protonate_openbabel(mol, ph=TARGET_PH)
        else:
            mol = protonate_rdkit(mol)
        mol.SetProp("_Name", name)

        # [e] Explicit hydrogens
        mol = Chem.AddHs(mol, addCoords=False)

        # [g+h] 3D embedding + MMFF94s minimisation
        mol3d, converged = embed_and_minimise(mol, n_confs=N_CONFORMERS,
                                              max_iters=MMFF_MAX_ITERS)

        if mol3d is None or mol3d.GetNumConformers() == 0:
            return None, "3D embedding failed"

        mol3d.SetProp("_Name",     name)
        mol3d.SetProp("Converged", str(converged))
        return mol3d, "SUCCESS"

    except Exception as err:
        return None, str(err)

def _ob_sdf_to_pdbqt(sdf_path, pdbqt_path):
    """Convert a single-molecule SDF to PDBQT via OpenBabel (flexible ligand)."""
    obc = ob.OBConversion()
    obc.SetInAndOutFormats("sdf", "pdbqt")
    mol = ob.OBMol()
    if not obc.ReadFile(mol, str(sdf_path)):
        raise RuntimeError(f"OpenBabel could not read {sdf_path.name}")
    cm2 = ob.OBChargeModel.FindType("gasteiger")
    if cm2:
        cm2.ComputeCharges(mol)
    obc.WriteFile(mol, str(pdbqt_path))


def _dock_one(lig_name, smiles, label, global_idx, total,
                  prot_pdbqt_path, center, size, scoring_func, exhaustiveness,
                  n_poses, seed_base, cores_per_dock):
    """
    HPC-optimized docking: Uses /tmp for I/O and captures results.
    """
    # 1. Create a temporary directory on the local compute node
    with tempfile.TemporaryDirectory(prefix=f"vina_{lig_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            # 1. Convert SMILES to 3D RDKit Mol
            mol_3d, status = smiles_to_3d_mol(smiles, lig_name)
            if mol_3d is None:
                return {
                    "Ligand_name": lig_name,
                    "Status": f"FAILED: {status}",
                    "Is_active": 1 if label == "active" else 0,
                    "Best_affinity_kcal_mol": None
                }

            # -- Molecular descriptors (CPU intensive, done in parallel) --
            mw   = Descriptors.ExactMolWt(mol_3d)
            logp = Descriptors.MolLogP(mol_3d)
            hbd  = rdMolDescriptors.CalcNumHBD(mol_3d)
            hba  = rdMolDescriptors.CalcNumHBA(mol_3d)
            rotb = rdMolDescriptors.CalcNumRotatableBonds(mol_3d)
            tpsa = Descriptors.TPSA(mol_3d)
            formula = rdMolDescriptors.CalcMolFormula(mol_3d)

            # -- File Prep in /tmp --
            lig_sdf   = tmp_path / "ligand.sdf"
            lig_pdbqt = tmp_path / "ligand.pdbqt"

            w = Chem.SDWriter(str(lig_sdf))
            w.write(mol_3d); w.close()

            # Convert to PDBQT in /tmp
            _ob_sdf_to_pdbqt(lig_sdf, lig_pdbqt)

            # -- Vina docking --
            # Note: We give each Vina instance a subif mol3d.GetNumConformers() == 0:
            v = Vina(sf_name=scoring_func, cpu=cores_per_dock, 
                     seed=seed_base + global_idx, verbosity=0)
            v.set_receptor(str(prot_pdbqt_path))
            v.set_ligand_from_file(str(lig_pdbqt))
            v.compute_vina_maps(center=center, box_size=size)

            t0 = time.time()
            v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            elapsed = time.time() - t0

            energies = v.energies(n_poses=n_poses)
            best_aff = energies[0][0]

            # (Optional) If you want to keep the poses, move them out of /tmp
            # final_output_dir = Path("results/poses")
            # final_output_dir.mkdir(parents=True, exist_ok=True)
            # shutil.copy(poses_pdbqt, final_output_dir / f"{lig_name}_docked.pdbqt")

            return {
                "Ligand_name": lig_name, "Label": label, "Is_active": 1 if label == "active" else 0,
                "Formula": formula, "MW_Da": round(mw, 3), "LogP": round(logp, 3),
                "HBD": hbd, "HBA": hba, "RotB": rotb, "TPSA_A2": round(tpsa, 2),
                "Best_affinity_kcal_mol": round(best_aff, 4), "Runtime_s": round(elapsed, 2),
                "Status": "SUCCESS"
            }

        except Exception as err:
            return {"Ligand_name": lig_name, "Status": f"FAILED: {err}", "Is_active": 1 if label == "active" else 0}

if __name__ == "__main__":
    protein_pdb = sys.argv[1]
    ref_ligand = sys.argv[2]
    actives = sys.argv[3]
    decoys = sys.argv[4]
    CPU_CORES = int(sys.argv[5])
    ref_mol = Chem.MolFromPDBFile(ref_ligand, removeHs=False)

    #prep binding box
    REF_LIGAND_PADDING = 6.0
    CORES_PER_VINA = 1
    WORKERS = CPU_CORES // CORES_PER_VINA

    if ref_mol is None:
        print(f"    WARNING: RDKit parsing failed -- trying OpenBabel ...")
        try:
            _oc = ob.OBConversion(); _oc.SetInAndOutFormats("pdb", "sdf")
            _om = ob.OBMol(); _oc.ReadFile(_om, ref_ligand)
            _tmp = "data/ref_conv.sdf"; _oc.WriteFile(_om, str(_tmp))
            ref_mol = next((m for m in Chem.SDMolSupplier(str(_tmp), removeHs=False)
                            if m is not None), None)
            if ref_mol: print("    [OK]  OpenBabel conversion OK")
        except Exception as _e:
            print(f"    WARNING: OpenBabel also failed: {_e}")
    else:
        if ref_mol.GetNumConformers() == 0:
            raise RuntimeError("Reference ligand has no 3D coordinates.")
        ref_pos  = np.array([ref_mol.GetConformer().GetAtomPosition(i)
                             for i in range(ref_mol.GetNumAtoms())])
        center   = ref_pos.mean(axis=0).tolist()
        size     = (ref_pos.max(axis=0) - ref_pos.min(axis=0) + 2*REF_LIGAND_PADDING).clip(min=15).tolist()
        print(f"    [OK]  Reference centroid ({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) A")

    vol = size[0]*size[1]*size[2]
    print(f"    [OK]  Centre ({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) A")
    print(f"    [OK]  Size   ({size[0]:.2f},{size[1]:.2f},{size[2]:.2f}) A  "
        f"vol={vol:,.0f} A^3" + ("  WARNING: Large box" if vol > 500_000 else ""))

    #prep protein format
    obC = ob.OBConversion()
    obC.SetInAndOutFormats("pdb", "pdbqt")
    obC.AddOption("r", ob.OBConversion.OUTOPTIONS)   # rigid receptor

    prot_mol = ob.OBMol()
    if not obC.ReadFile(prot_mol, protein_pdb):
        raise RuntimeError("OpenBabel could not read protein PDB.")

    cm = ob.OBChargeModel.FindType("gasteiger")
    if cm:
        cm.ComputeCharges(prot_mol)

    prot_pdbqt = "data/protein.pdbqt"
    obC.WriteFile(prot_mol, str(prot_pdbqt))
    n_rec = sum(1 for ln in open(prot_pdbqt) if ln.startswith(("ATOM","HETATM")))
    print(f"    [OK]  {n_rec:,} PDBQT records \n")

    #prep actives and ligands
    # 1. Load your data
    act_mols = pd.read_csv(actives) # Assumes columns 'molecule_chembl_id' and 'smiles'
    dec_mols  = pd.read_csv(decoys)

    # 2. Build your list for the Parallel Executor
    all_ligands = []

    # Add actives
    for _, row in act_mols.iterrows():
        all_ligands.append((row['molecule_chembl_id'], row['smiles'], "active"))

    # Add decoys
    for _, row in dec_mols.iterrows():
        all_ligands.append((row['molecule_chembl_id'], row['smiles'], "decoy"))

    n_actives = len(act_mols)
    n_decoys  = len(dec_mols)
    n_total   = n_actives + n_decoys
    print(f"\n  Total: {n_actives} actives + {n_decoys} decoys = {n_total} ligands to dock\n")

    #docking
    dock_results  = []   # list of dicts: name, label, best_affinity, ...
    t0_total      = time.time()
    _W = 56

    print(f"  [HPC] Starting Parallel Pool: {WORKERS} workers, {CORES_PER_VINA} core(s) per ligand")

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # 1. Map the futures to their names (or indices) for easier tracking
        future_to_ligand = {
            executor.submit(
                _dock_one, name, mol, label, idx, n_total,
                prot_pdbqt, center, size, SCORING_FUNCTION,
                EXHAUSTIVENESS, N_POSES, RANDOM_SEED, CORES_PER_VINA
            ): name for idx, (name, mol, label) in enumerate(all_ligands)
        }
        # 2. Process them as they finish
        for future in as_completed(future_to_ligand):
            ligand_name = future_to_ligand[future]
            try:
                res = future.result()
                dock_results.append(res)
            except Exception as e:
                print(f"    Error docking {ligand_name}: {e}", flush=True)

            # 3. Print progress immediately
            if len(dock_results) % 10 == 0:
                print(f"    Progress: {len(dock_results)}/{n_total} finished...", flush=True)

    total_elapsed = time.time() - t0_total
    n_success = sum(1 for r in dock_results if r["Status"] == "SUCCESS")
    n_failed  = len(dock_results) - n_success

    print(f"\n  {'='*_W}")
    print(f"  Docking complete | {n_success} succeeded | {n_failed} failed | {total_elapsed:.1f}s\n")

    # -- Collect successful results ---------------------------------------------
    valid = [r for r in dock_results
            if r["Status"] == "SUCCESS" and r["Best_affinity_kcal_mol"] is not None]

    if len(valid) < 2:
        raise RuntimeError("Not enough successful docking results for ROC analysis.")

    n_act_success = sum(1 for r in valid if r["Is_active"] == 1)
    n_dec_success = sum(1 for r in valid if r["Is_active"] == 0)
    print(f"  Valid results: {n_act_success} actives + {n_dec_success} decoys = {len(valid)} total")

    if n_act_success == 0 or n_dec_success == 0:
        raise RuntimeError("Need at least 1 active AND 1 decoy with successful docking for ROC.")

    # -- Sort by affinity (more negative = better, rank ascending) --------------
    valid.sort(key=lambda r: r["Best_affinity_kcal_mol"])

    labels   = np.array([r["Is_active"] for r in valid])
    scores   = np.array([-r["Best_affinity_kcal_mol"] for r in valid])  # negate so higher = better

    # Combine arrays into a DataFrame
    df = pd.DataFrame({'Label': labels, 'Score': scores})

    # Export to CSV
    df.to_csv('docking_results.csv', index=False)
