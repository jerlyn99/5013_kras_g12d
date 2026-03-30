import sys, re, time, shutil, tempfile
import traceback
from rdkit import Chem
from openbabel import openbabel as ob
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, EnumerateStereoisomers, Descriptors, rdMolDescriptors
from vina import Vina
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def smiles_to_3d_mol(smiles, name, random_seed):
    # 1. SMILES to 2D Mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 2. Add Hydrogens
    mol = Chem.AddHs(mol)
    
    # 3. Generate 3D Coordinates (ETKDGv3 is the modern standard)
    # This creates a "conformer" (the 3D shape)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    embed_status = AllChem.EmbedMolecule(mol, params)
    
    if embed_status == -1:
        # Retry with random coordinates if standard embedding fails
        embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(useRandomCoords=True))

    # 4. Energy Minimization (Clean up bond lengths/angles)
    if embed_status != -1:
        AllChem.MMFFOptimizeMolecule(mol)
        mol.SetProp("_Name", name)
        return mol
    
    return None

def _collect_mols(sdf_path):
    """Read SDF with RDKit and return list of (name, mol)."""
    out = []
    for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf_path),
                                                  removeHs=False, sanitize=True)):
        if mol is None:
            print(f"    WARNING: Molecule #{idx+1} unparseable -- skipped")
            continue
        raw = mol.GetPropsAsDict().get("_Name", "").strip()
        name = re.sub(r"[^\w\-]", "_", raw) if raw else f"LIG_{idx+1:04d}"
        out.append((name, mol))
    return out
    
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
            mol_3d = smiles_to_3d_mol(smiles, lig_name, seed_base)
            if mol_3d is None:
                return {"Status": "FAILED: 3D embedding failed"}
            
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
            # Note: We give each Vina instance a subset of the total cores
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
            print(f"DEBUG: {lig_name} failed with error: {err}")
            traceback.print_exc() # This will print the line number and specific error
            return {"Ligand_name": lig_name, "Status": f"FAILED: {err}", "Is_active": 1 if label == "active" else 0}

if __name__ == "__main__":
    protein_pdb = sys.argv[1]
    ref_ligand = sys.argv[2]
    actives = sys.argv[3]
    decoys = sys.argv[4]
    ref_mol = Chem.MolFromPDBFile(ref_ligand, removeHs=False)

    #prep binding box
    REF_LIGAND_PADDING = 6.0

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
    SCORING_FUNCTION = "vinardo"
    EXHAUSTIVENESS   = 8
    N_POSES          = 1
    CPU_CORES        = int(sys.argv[5])
    RANDOM_SEED      = 42

    dock_results  = []   # list of dicts: name, label, best_affinity, ...
    t0_total      = time.time()
    _W = 56

    CORES_PER_VINA = 1 
    WORKERS = CPU_CORES // CORES_PER_VINA

    print(f"  [HPC] Starting Parallel Pool: {WORKERS} workers, {CORES_PER_VINA} core(s) per ligand")

    dock_results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = []
        for idx, (name, mol, label) in enumerate(all_ligands):
            # Pass all necessary parameters to the function
            futures.append(executor.submit(
                _dock_one, name, mol, label, idx, n_total,
                prot_pdbqt, center, size, SCORING_FUNCTION, 
                EXHAUSTIVENESS, N_POSES, RANDOM_SEED, CORES_PER_VINA
            ))
        
        for f in futures:
            res = f.result()
            dock_results.append(res)
            # Optional: Print progress every 10 ligands
            if len(dock_results) % 10 == 0:
                print(f"    Progress: {len(dock_results)}/{n_total} finished...")

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

    # -- ROC curve computation (manual, no sklearn dependency) ------------------
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    # Sort by descending score (best binders first)
    sorted_idx = np.argsort(-scores)
    sorted_lab = labels[sorted_idx]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for lab in sorted_lab:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    # -- AUC (trapezoidal rule) ------------------------------------------------
    auc = np.trapz(tpr, fpr)
    print(f"\n  ROC AUC = {auc:.4f}")

    # -- Enrichment Factors ----------------------------------------------------
    print(f"\n  Enrichment Factors\n")
    _Ra = n_pos / len(valid)   # ratio of actives in the total set

    ef_results = {}
    print(f"  {'Fraction':>10}  {'Top-N':>6}  {'Actives found':>14}  {'EF':>8}  {'Max EF':>8}")
    print(f"  {'-'*54}")

    for frac in [0.01, 0.05]:
        top_n = max(1, int(round(frac * len(valid))))
        top_labels = sorted_lab[:top_n]
        n_act_in_top = int(top_labels.sum())
        hits_rate = n_act_in_top / top_n
        ef = hits_rate / _Ra if _Ra > 0 else 0.0
        max_ef = min(n_pos, top_n) / (top_n * _Ra) if _Ra > 0 else 0.0
        ef_results[frac] = {"top_n": top_n, "actives_found": n_act_in_top,
                            "ef": ef, "max_ef": max_ef}
        print(f"  {frac:>10.1%}  {top_n:>6}  {n_act_in_top:>14}  {ef:>8.2f}  {max_ef:>8.2f}")

    # -- BEDROC (Boltzmann-Enhanced Discrimination of ROC, alpha=20) ------------
    alpha = 20.0
    n = len(sorted_lab)
    sum_exp = 0.0
    for i, lab in enumerate(sorted_lab):
        if lab == 1:
            ri = (i + 1) / n
            sum_exp += np.exp(-alpha * ri)

    ra = n_pos / n
    ri_ideal = np.sum(np.exp(-alpha * (np.arange(1, n_pos + 1) / n)))
    ri_rand  = (ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))

    bedroc = (sum_exp - ri_rand) / (ri_ideal - ri_rand) if (ri_ideal - ri_rand) != 0 else 0.0
    print(f"\n  BEDROC (alpha=20) = {bedroc:.4f}")

    # -- Plot ROC Curve and Enrichment Curve ------------------------------------
    print("\n  Generating plots ...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Virtual Screening Performance", fontsize=14, fontweight="bold")

    # Panel 1: ROC Curve
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color="#2563eb", linewidth=2.2, label=f"ROC (AUC = {auc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Random (AUC = 0.500)")
    ax1.fill_between(fpr, tpr, alpha=0.12, color="#2563eb")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax1.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal")
    ax1.xaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.grid(True, alpha=0.3)

    # Panel 2: Enrichment Curve
    ax2 = axes[1]
    frac_screened = np.arange(0, len(sorted_lab) + 1) / len(sorted_lab)
    frac_actives  = np.concatenate([[0], np.cumsum(sorted_lab) / n_pos])

    ax2.plot(frac_screened, frac_actives, color="#dc2626", linewidth=2.2,
            label=f"Docking ({SCORING_FUNCTION})")
    ax2.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Random")

    # Ideal enrichment curve
    ideal_x = [0, min(n_pos/len(sorted_lab), 1.0), 1.0]
    ideal_y = [0, 1.0, 1.0]
    ax2.plot(ideal_x, ideal_y, ":", color="#16a34a", linewidth=1.5, label="Ideal")

    # Mark EF fractions on the curve
    for frac, ef_data in ef_results.items():
        idx = ef_data["top_n"]
        x_pt = idx / len(sorted_lab)
        y_pt = ef_data["actives_found"] / n_pos
        ax2.plot(x_pt, y_pt, "o", color="#dc2626", markersize=7, zorder=5)
        ax2.annotate(f"EF{frac:.0%}={ef_data['ef']:.1f}",
                    xy=(x_pt, y_pt), xytext=(x_pt+0.04, y_pt-0.06),
                    fontsize=8, color="#dc2626",
                    arrowprops=dict(arrowstyle="-", color="#dc2626", lw=0.8))

    ax2.set_xlabel("Fraction of Database Screened", fontsize=11)
    ax2.set_ylabel("Fraction of Actives Found", fontsize=11)
    ax2.set_title("Enrichment Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_aspect("equal")
    ax2.xaxis.set_major_locator(MultipleLocator(0.2))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "data/roc_enrichment_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  [OK]  Plot saved -> {plot_path.name}")

    # -- Affinity distribution plot ---------------------------------------------
    fig2, ax3 = plt.subplots(figsize=(8, 5))

    act_affs = [r["Best_affinity_kcal_mol"] for r in valid if r["Is_active"] == 1]
    dec_affs = [r["Best_affinity_kcal_mol"] for r in valid if r["Is_active"] == 0]

    bins = np.linspace(min(act_affs + dec_affs) - 0.5, max(act_affs + dec_affs) + 0.5, 30)
    ax3.hist(act_affs, bins=bins, alpha=0.65, color="#2563eb", edgecolor="white",
            linewidth=0.8, label=f"Actives (n={len(act_affs)})")
    ax3.hist(dec_affs, bins=bins, alpha=0.65, color="#dc2626", edgecolor="white",
            linewidth=0.8, label=f"Decoys (n={len(dec_affs)})")
    ax3.set_xlabel("Best Vina Affinity (kcal/mol)", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Score Distribution -- Actives vs Decoys", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    dist_path = "data/score_distribution.png"
    fig2.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  [OK]  Distribution plot saved -> {dist_path.name}")
