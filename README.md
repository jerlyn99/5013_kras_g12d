This repository is for a drug discovery project
"In Silico Prediction of Potential Inhibitors of Kirsten Rat Sarcoma G12D Using Machine Learning-Based Virtual Screening, Molecular Docking, and Molecular Dynamic Simulation Approaches"

**Ligand Prep Steps**

Files: actives_prep.ipynb

1. Download known actives from chemBL
2. Clean data and filter for relevant actives
3. Output list of cleaned compounds in "data" folder (DATASET_ic50_cleaned_compounds.csv)
4. Generate fingerprints and molecular descriptors for actives
5. Output (fingerprints, molecular descriptors) in "data" folder (e.g. DATASET_ecfp4.csv)
6. Results and code are found in ligand_prep folder

**Library Prep Steps**

Files: library_prep.ipynb, library_prep.py

1. Download database from any virtual library website
2. Explore library using library_prep.ipynb
3. Run HPC pipeline for library_prep to generate standardized SMILES for screening
4. Results and code are found in library_prep folder

**QSAR Model Building Steps**

Files: qsar_model.ipynb, qsar_model_valdiation.ipynb, qsar_pipeline.py, hits_analysis.ipynb

1. Create QSAR model from known actives
2. Conduct scaffold splitting and feature selection using *qsar_model.ipynb*
3. Validate chosen models using *qsar_model_validation.ipynb*
4. Using qsar_pipeline_hpc.py, use chosen model to generate hits from virtual library
5. Filter for top candidate hits from virtual library and explore using hits_analysis.ipynb
6. Results and code are found in qsar folder

**Molecular Docking Steps**

Files: redocking_validation.ipynb, mol_docking_results_analysis.ipynb, molecular_docking_pipeline.py

1. Create molecular docking protocol
2. Obtain cleaned target and ligand structures
3. Perform redocking of cleaned ligand onto target using redocking_validation.ipynb
4. Use known actives from chemBL to evaluate docking protocol in HPC via molecular_docking_pipeline.py
5. Analyse results of docking validation using mol_docking_results_analysis.ipynb
6. Dock top candidates from previous step and obtain rankings
7. Results and code are found in mol_docking folder

**Molecular Dynamics Steps**

1. Perform MD Simulation for top 3 molecules from molecular docking step
2. Results and code are found in md_sim folder
