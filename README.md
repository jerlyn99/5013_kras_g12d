**Ligand Prep Steps**

1. Download known actives from ChemBL
2. Use ligand_prep.ipynb to prepare the ligands

**Library Prep Steps**

1. Download database from ZINC20 website
2. Filter through SMILES using sanity_check.py script
3. Run library_prep.ipynb

**Virtual Screening Steps**

1. Create QSAR model from known actives -> pending
2. Create Pharmacophore model from known actives -> pending
3. Validate that models are able to differentiate actives from decoys
4. Put library through models to generate rankings and hits
