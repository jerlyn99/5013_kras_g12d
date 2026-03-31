import sys, time, tempfile, warnings, os, multiprocessing
from rdkit import Chem, RDLogger
from openbabel import openbabel as ob
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdDistGeom
from vina import Vina
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit.Chem.MolStandardize import rdMolStandardize
import signal
import oddt
from oddt.scoring.functions import rfscore
import joblib

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

#Global Variables
SCORING_FUNCTION = "vinardo"
EXHAUSTIVENESS   = 16
N_POSES          = 1
RANDOM_SEED      = 42
MMFF_MAX_ITERS = 2000
N_CONFORMERS = 1
TARGET_PH= 7.4
PROTONATION_METHOD = "openbabel"

_rf_model = None
_worker_vina_obj = None
_oddt_receptor = None

class DockingTimeoutError(Exception):
    pass

def handler(signum, frame):
    raise DockingTimeoutError("Molecules took too long!")

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

    mol3d = Chem.AddHs(mol)
    
    conf_ids = list(AllChem.EmbedMultipleConfs(mol3d, numConfs=n_confs, params=params))
    
    if not conf_ids:
        # Emergency fallback for difficult structures
        if AllChem.EmbedMolecule(mol3d, AllChem.ETKDG()) == -1:
            return mol3d, False
        conf_ids = [0]

    results = AllChem.MMFFOptimizeMoleculeConfs(mol3d, mmffVariant="MMFF94s", maxIters=max_iters)
    if not results:
        return mol3d, False

    # Selection: Find absolute minimum energy among all generated poses
    # r[0] is status, r[1] is energy
    best_conf = 0
    min_energy = float('inf')
    for i, r in enumerate(results):
        if r[1] < min_energy:
            min_energy = r[1]
            best_conf = i
    
    converged = (results[best_conf][0] == 0)

    # Efficiently keep only the best conformer
    new_mol = Chem.Mol(mol3d)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(mol3d.GetConformer(best_conf), assignId=True)
    
    return new_mol, converged

def smiles_to_3d_mol(smiles, name):
    # 1. SMILES to 2D Mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    try:
        Chem.SanitizeMol(mol)
        mol = standardise_mol(mol)

        # [d] Protonation (Choose pH-aware OB for better validation results)
        if PROTONATION_METHOD == "openbabel":
            mol = protonate_openbabel(mol, ph=TARGET_PH)
        else:
            mol = protonate_rdkit(mol)

        # [g+h] 3D embedding + MMFF94s minimisation
        mol3d, converged = embed_and_minimise(mol, n_confs=N_CONFORMERS, 
                                              max_iters=MMFF_MAX_ITERS)

        if mol3d is None or mol3d.GetNumConformers() == 0:
            return None, "3D embedding failed"

        mol3d.SetProp("_Name", name)
        mol3d.SetProp("Converged", str(converged))
        return mol3d, "SUCCESS"

    except Exception as err:
        return None, str(err)
    
def worker_init(prot_path, center, size, scoring_func, cpu_count, base_seed, counter, lock):
    global _worker_vina_obj, _rf_model, _oddt_receptor

    # 1. Get a stable Worker ID (0, 1, 2...) instead of using PIDs
    with lock:
        worker_id = counter.value
        counter.value += 1
    
    time.sleep(worker_id * 0.1) 
    
    # Each "slot" in your pool gets a unique but repeatable seed
    stable_seed = base_seed + worker_id
    
    try:
        _oddt_receptor = next(oddt.toolkit.readfile('pdbqt', prot_path))
        _oddt_receptor.protein = True
        _ = _oddt_receptor.atom_dict
        _rf_model = rfscore(1, _oddt_receptor)
        local_weights = os.path.join(os.getcwd(), 'RFScore_v1_pdbbind2016.pickle')
        if os.path.exists(local_weights):
            try:
                # 1. Manually load the raw data from the pickle
                loaded_data = joblib.load(local_weights)
                
                # 2. Check what was actually inside that pickle
                # Sometimes ODDT saves the whole Scorer, sometimes just the Regressor
                if hasattr(loaded_data, 'model'):
                    # It's a full RFScore object
                    _rf_model.model = loaded_data.model
                else:
                    # It's just the Scikit-Learn Regressor
                    _rf_model.model = loaded_data                
            except Exception as e:
                print(f"Worker {os.getpid()}: Surgery failed! Error: {e}")

        _worker_vina_obj = Vina(sf_name=scoring_func, cpu=cpu_count, seed=stable_seed, verbosity=0)
        _worker_vina_obj.set_receptor(str(prot_path))
        _worker_vina_obj.compute_vina_maps(center=center, box_size=size)
    except Exception as e:
        print(f"Worker init failed: {e}")

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

def _pdbqt_to_sdf(pdbqt_path, sdf_path):
    """Convert docked PDBQT back to SDF via OpenBabel."""
    obc = ob.OBConversion()
    obc.SetInAndOutFormats("pdbqt", "sdf")
    mol = ob.OBMol()
    if obc.ReadFile(mol, str(pdbqt_path)):
        obc.WriteFile(mol, str(sdf_path))
    # Explicitly clear the molecule from memory
    mol.Clear()
    del mol

def _dock_one(lig_name, smiles, label, exhaustiveness, n_poses):
    """
    HPC-optimized docking: Uses /tmp for I/O and captures results.
    """
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(300) # seconds

    #username = os.environ.get('USER', 'default_user')
    #base_scratch = Path(f"/scratch/{username}")

    #job_tmp_dir = base_scratch / "vina_temp_jobs"
    #job_tmp_dir.mkdir(parents=True, exist_ok=True)

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
            v = _worker_vina_obj
            v.set_ligand_from_file(str(lig_pdbqt))        

            t0 = time.time()
            v.dock(exhaustiveness=exhaustiveness, 
                    n_poses=n_poses)
            
            elapsed = time.time() - t0

            energies     = v.energies(n_poses=n_poses)
            best_aff     = energies[0][0]
            best_rmsd_lb = energies[0][1]
            best_rmsd_ub = energies[0][2]

            out_pdbqt = tmp_path / "docked_poses.pdbqt"
            out_sdf   = tmp_path / "docked_poses.sdf"
            v.write_poses(str(out_pdbqt), n_poses=n_poses, overwrite=True)

            try:
                _pdbqt_to_sdf(out_pdbqt, out_sdf)
                if os.path.exists(out_sdf):
                    mols_out = [m for m in Chem.SDMolSupplier(str(out_sdf), removeHs=False, sanitize=False)
                                if m is not None]
                    top_pose_mol = None
                    if mols_out:
                        # We only take the best pose (index 0) for the ROC analysis
                        top_pose_mol = mols_out[0]

                        # Perform "High-Tolerance" Sanitization
                        # This ensures RF-Score sees aromatic rings and donors/acceptors correctly
                        Chem.SanitizeMol(top_pose_mol, 
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)

                        # 3. Add all your Vina and Molecular descriptors
                        top_pose_mol.SetProp("_Name", f"{lig_name}_docked")
                        top_pose_mol.SetProp("Vina_Affinity_kcal_mol", f"{best_aff:.4f}")
                        top_pose_mol.SetProp("Is_Active", "1" if label == "active" else "0")
                        top_pose_mol.SetProp("RMSD_lower_bound_A",     f"{best_rmsd_lb:.4f}")
                        top_pose_mol.SetProp("RMSD_upper_bound_A",     f"{best_rmsd_ub:.4f}")
                        top_pose_mol.SetProp("Formula",                formula)
                        top_pose_mol.SetProp("MW",                     f"{mw:.3f}")
                        top_pose_mol.SetProp("LogP",                   f"{logp:.3f}")
                        top_pose_mol.SetProp("RotatableBonds",         str(rotb))
                    else:
                        sys.stderr.write(f"\n[ERROR] RDKit could not parse the converted SDF for {lig_name}\n")
                else:
                    sys.stderr.write(f"\n[ERROR] _pdbqt_to_sdf failed to create file for {lig_name}\n")
            except Exception as e_sdf:
                sys.stderr.write(f"\n[CRITICAL] SDF logic failed: {str(e_sdf)}\n")

            # 6. RF-Score Rescoring (Only for Top Pose)
            if top_pose_mol is not None:
                try:
                    # Convert RDKit mol to ODDT mol for scoring
                    # We use the SDF string to ensure coordinates/bonds transfer correctly
                    mol_block = Chem.MolToMolBlock(top_pose_mol)
                    oddt_ligand = oddt.toolkit.readstring('sdf', mol_block)
                    _ = oddt_ligand.atom_dict

                    _rf_model.receptor = _oddt_receptor
                    # 2. BRUTE FORCE SEARCH for the descriptor engine
                    # We look for any attribute inside _rf_model that knows how to 'build' features
                    engine = None
                    for attr_name in dir(_rf_model):
                        attr = getattr(_rf_model, attr_name, None)
                        # A descriptor engine usually has 'build', 'make_full_v1', or 'calculate'
                        if hasattr(attr, 'build') or hasattr(attr, 'make_full_v1'):
                            engine = attr
                            break
                    
                    if engine is None:
                        # Last ditch effort: Is the model itself the descriptor?
                        if hasattr(_rf_model, 'make_features'):
                            features = _rf_model.make_features(oddt_ligand)
                        else:
                            raise AttributeError("Internal ODDT structure unrecognized.")
                    else:
                        # 3. Calculate features using the found engine
                        # We pass the protein explicitly to fix the 'int' pointer error
                        try:
                            features = engine.build(oddt_ligand, protein=_oddt_receptor)
                        except:
                            features = engine.make_full_v1(oddt_ligand, protein=_oddt_receptor)          
                                            
                    # Predict pKd (-log Ki/Kd)
                    rf_val = _rf_model.model.predict(features.reshape(1, -1))[0]
                    
                    # Add to RDKit molecule properties
                    top_pose_mol.SetProp("RFScore_pKd", f"{rf_val:.3f}")
                except Exception as e_rf:
                    sys.stderr.write(f"\n[WARNING] RF-Score failed for {lig_name}: {e_rf}\n")
            
            signal.alarm(0) # Disable alarm
            
            return {
                "Ligand_name": lig_name, "Label": label, "Is_active": 1 if label == "active" else 0,
                "Formula": formula, "MW_Da": round(mw, 3), "LogP": round(logp, 3),
                "HBD": hbd, "HBA": hba, "RotB": rotb, "TPSA_A2": round(tpsa, 2),
                "Best_affinity_kcal_mol": round(best_aff, 4), 
                "RF_Score_pKd": round(rf_val, 3) if rf_val is not None else None,
                "Runtime_s": round(elapsed, 2),
                "Status": "SUCCESS",
                "cleaned_mol": mol_3d,      # Returned for master file writing
                "top_pose_mol": top_pose_mol # Returned for master file writing
            }
            
        except DockingTimeoutError:
            return {"Ligand_name": lig_name, "Status": "FAILED: Timeout"}

        except Exception as err:
            signal.alarm(0)
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
        size     = [15,15,15]
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

    all_ligands.sort(key=lambda x: x[0]) 

    n_actives = len(act_mols)
    n_decoys  = len(dec_mols)
    n_total   = n_actives + n_decoys
    print(f"\n  Total: {n_actives} actives + {n_decoys} decoys = {n_total} ligands to dock\n")

    #docking
    dock_results  = []   # list of dicts: name, label, best_affinity, ...
    t0_total      = time.time()
    _W = 56

    print(f"  [HPC] Starting Parallel Pool: {WORKERS} workers, {CORES_PER_VINA} core(s) per ligand")

    cleaned_writer = Chem.SDWriter("all_cleaned_ligands.sdf")
    poses_writer   = Chem.SDWriter("all_top_poses.sdf")

    manager = multiprocessing.Manager()
    worker_counter = manager.Value('i', 0)
    worker_lock = manager.Lock()

    #create pickle file for rfscore
    m = rfscore(1, None)
    m.load()

    with ProcessPoolExecutor(
        max_workers=WORKERS, 
        initializer=worker_init,
        initargs=(prot_pdbqt,center, size, SCORING_FUNCTION, CORES_PER_VINA, RANDOM_SEED, worker_counter, worker_lock)) as executor:
        # 1. Map the futures to their names (or indices) for easier tracking
        future_to_ligand = {
            executor.submit(
                _dock_one, name, mol, label, EXHAUSTIVENESS, N_POSES
            ): name for idx, (name, mol, label) in enumerate(all_ligands)
        }
        # 2. Process them as they finish
        for future in as_completed(future_to_ligand):
            try:
                res = future.result()
                # 1. Handle Successful Docking
                if res.get("Status") == "SUCCESS":
                    if res.get("cleaned_mol"):
                        cleaned_writer.write(res["cleaned_mol"])
                    if res.get("top_pose_mol"):
                        poses_writer.write(res["top_pose_mol"])
                
                # 2. Extract numeric data (removing heavy mol objects)
                final_data = {k: v for k, v in res.items() if k not in ['cleaned_mol', 'top_pose_mol']}
                dock_results.append(final_data)
                
                # 3. Explicitly help Python Garbage Collector
                res = None 
                
            except Exception as e:
                print(f"    Critical Error in future: {e}", flush=True)

            # 3. Print progress immediately
            if len(dock_results) % 10 == 0:
                pd.DataFrame(dock_results).to_csv("docking_checkpoint.csv", index=False)
                print(f"    Progress: {len(dock_results)}/{n_total} finished...", flush=True)

    cleaned_writer.close()
    poses_writer.close()
    manager.shutdown()

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

    else:
        # 2. Convert to DataFrame
        df = pd.DataFrame(valid)

        # 3. Handle RF-Score edge cases
        # If ODDT failed for a ligand, we fill with a neutral/low score (e.g., 0.0)
        if 'RF_Score_pKd' in df.columns:
            df['RF_Score_pKd'] = df['RF_Score_pKd'].fillna(0.0)
        else:
            # If the entire run didn't use RF-Score, initialize column to avoid errors
            df['RF_Score_pKd'] = 0.0

        # 4. Calculate Ranks for Consensus Scoring
        # rank_vina: Lower kcal/mol is better (1 is most negative)
        df['rank_vina'] = df['Best_affinity_kcal_mol'].rank(ascending=True)
        
        # rank_rf: Higher pKd is better (1 is highest value)
        df['rank_rf'] = df['RF_Score_pKd'].rank(ascending=False)

        # 5. Consensus Score: Rank Product (Lower is better)
        # This combines the 'Physics' (Vina) and 'Knowledge' (RF) metrics
        df['Consensus_Score'] = np.sqrt(df['rank_vina'] * df['rank_rf'])

        # 6. Final Sort: Show the best Consensus hits at the top
        df = df.sort_values("Consensus_Score", ascending=True)

        # 7. Save to CSV
        output_file = "docking_final_results.csv"
        df.to_csv(output_file, index=False)
