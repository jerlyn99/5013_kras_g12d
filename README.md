**Ligand Prep Steps**

Files: actives_prep.ipynb

1. Download known actives from chemBL
2. Clean data and filter for relevant actives
3. Output list of cleaned compounds in "data" folder (DATASET_ic50_cleaned_compounds.csv)
4. Generate fingerprints and molecular descriptors for actives
5. Output (fingerprints, molecular descriptors) in "data" folder (e.g. DATASET_ecfp4.csv)

**Library Prep Steps**

Files: library_prep.py

1. Download database from any virtual library website
2. Run HPC pipeline for library_prep to generate standardized SMILES for screening

**Virtual Screening Steps**

Files: qsar_model.ipynb, qsar_model_valdiation.ipynb, qsar_pipeline.py

1. Create QSAR model from known actives
2. Conduct scaffold splitting and feature selection using *qsar_model.ipynb*
3. Validate chosen models using *qsar_model_validation.ipynb*
4. Using qsar_pipeline.py, use model to generate hits from reference library
5. Filter for top candidate hits from reference library

**Molecular Docking Steps**
1. Create molecular docking protocol
2. Obtain cleaned target and ligand structures
3. Perform redocking of cleaned ligand onto target
4. Use known actives from chemBL to evaluate docking protocol
5. Dock top candidates from previous step and obtain rankings

**Structural Based Screening Steps**

1. Using PharmacoNet, automatically generate pharmacophore models based on protein structure
2. Based on the output of PharmacoNet, generate and optimise 3D ligand conformations to fit the predefined pharmacophore using DiffPhore
3. Validate results using molecular docking such as UniDock
