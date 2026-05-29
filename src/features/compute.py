# Imports
from modlamp.descriptors import *
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from .calc_hydrophob import calculate_hydrophobicity_features
import pandas as pd
import numpy as np
import os
from colorama import Fore, Style
import peptides
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

def get_modlamp_descriptors(seq, pH=7.4):
    """Calculate descriptors using modlAMP for a given sequence and pH"""
    features = {}
    # ----- Calculate global descriptors with modlAMP -----
    global_desc = GlobalDescriptor(seq)
    global_desc.calculate_all(ph=pH, amide=True)

    # Collect features into dictionary
    for feat, value in zip(global_desc.featurenames, global_desc.descriptor[0]):
        features[feat] = value

    # ----- Calculate peptide descriptors from modlAMP -----
        # -- Flexibility --
    peptide_desc = PeptideDescriptor(seq, 'flexibility')
    peptide_desc.calculate_global()
    features['Flexibility'] = peptide_desc.descriptor[0][0]
        # -- Bulkiness --
    peptide_desc = PeptideDescriptor(seq, 'bulkiness')
    peptide_desc.calculate_global()
    features['Bulkiness'] = peptide_desc.descriptor[0][0]
        # -- Polarity --
    peptide_desc = PeptideDescriptor(seq, 'polarity')
    peptide_desc.calculate_global()
    features['Polarity'] = peptide_desc.descriptor[0][0]

    return features

def get_heliquest_descriptors(seq, charge):
    """Calculate hydrophobicity features using custom calc_hydrophob.py derived from heliquest"""
    features = {}
    hydro_features = calculate_hydrophobicity_features(seq, charge)
    for feat, value in hydro_features.items():
        features[f'{feat}'] = value

    return features

def get_peptide_factors(seq):
    """Get peptide factors using peptides.py library"""
    # Create Peptide object
    peptide_obj = peptides.Peptide(seq)

    # Get Atchley factors, T-scales, VHSE-scales, Z-scales
    atchley_factors = peptide_obj.atchley_factors()
    t_scales = peptide_obj.t_scales()
    VHSE_scales = peptide_obj.vhse_scales()
    z_scales = peptide_obj.z_scales()

    # Additional features from peptides.py
    features = {}
    features['Entropy'] = peptide_obj.entropy() # Shannon entropy
    features['mz'] = peptide_obj.mz()

    for i, val in enumerate(atchley_factors):
        features[f'Atchley_Factor_{i+1}'] = val
    for i, val in enumerate(t_scales):
        features[f'T_Scale_{i+1}'] = val
    for i, val in enumerate(VHSE_scales):
        features[f'VHSE_Scale_{i+1}'] = val
    for i, val in enumerate(z_scales):
        features[f'Z_Scale_{i+1}'] = val

    return features

def get_rdkit_descriptors(seq):
    """Get all rdkit descriptors for a sequence"""
    m = Chem.MolFromSequence(seq)

    desc_list = Descriptors._descList[:132] # Exclude fraction descriptors

    # Prune features that are just zeros or constants 
    #remove_desc = ['NumRadicalElectrons', 'PEOE_VSA13', 'PEOE_VSA5', 'SlogP_VSA10', 'SMR_VSA8', 'SlogP_VSA7', 'SlogP_VSA9', 'VSA_EState1', 'VSA_EState9', 
                   #'NumAliphaticCarbocycles', 'NumBridgeheadAtoms', 'NumSaturatedCarbocycles', 'NumSpiroAtoms', 'NumUnspecifiedAtomStereoCenters']    

    #desc_list = [item for item in desc_list if item[0] not in remove_desc]

    values = {
        name: func(m)
        for name, func in desc_list 
    }

    return values

def get_mordred_descriptors(seq):
    calc = Calculator(descriptors, ignore_3D=True)
    mol = Chem.MolFromSequence(seq)
    
    desc = calc(mol)
    desc_dict = desc.asdict()
    
    return desc_dict

def getMD_sidechain_descriptors(pdb_path):
    """
    Calculate sidechain descriptors using MDAnalysis for a given PDB file
    
    Returns three types of sidechain length descriptors
    -----
    - cb: distance from CA to CB (or COM if no CB)
    - far: distance from CA to farthest sidechain atom
    - com: distance from CA to center of mass of sidechain

    mean and max values across all residues are returned for each type
    """
    BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}
    u = mda.Universe(pdb_path)
    protein = u.select_atoms("protein")

    per_res = []
    # Iterate residues
    for res in protein.residues:
        # CA atom
        ca = res.atoms.select_atoms("name CA")
        if len(ca) != 1:
            continue
        ca_pos = ca.positions[0]

        # Sidechain atom group = residue atoms excluding backbone
        side = res.atoms[[a.name not in BACKBONE_NAMES for a in res.atoms]]

        if len(side) == 0:
            # e.g., glycine may end up empty if you exclude CB-based definitions;
            # or unusual/trimmed residues.
            per_res.append({
                "resid": int(res.resid),
                "resname": str(res.resname),
                "length_cb": np.nan,
                "length_far": np.nan,
                "length_com": np.nan,
            })
            continue

        # Beta carbon
        cb = res.atoms.select_atoms("name CB")
        if len(cb) == 1:
            sc_pos = cb.positions[0]
        else: # fallback: com if no CB (glycine, etc.)
            sc_pos = side.center_of_mass()

        v = sc_pos - ca_pos
        length_cb = float(np.linalg.norm(v))

        # Farthest sidechain atom from CA
        d = np.linalg.norm(side.positions - ca_pos, axis=1)
        sc_pos = side.positions[int(np.argmax(d))]
        v = sc_pos - ca_pos
        length_far = float(np.linalg.norm(v))

        # Center of mass of sidechain
        sc_pos = side.center_of_mass()
        v = sc_pos - ca_pos
        length_com = float(np.linalg.norm(v))

        per_res.append({
            "resid": int(res.resid),
            "resname": str(res.resname),
            "length_cb": length_cb,
            "length_far": length_far,
            "length_com": length_com,
        })

    lengths_cb = [r["length_cb"] for r in per_res if not np.isnan(r["length_cb"])]
    lengths_far = [r["length_far"] for r in per_res if not np.isnan(r["length_far"])]
    lengths_com = [r["length_com"] for r in per_res if not np.isnan(r["length_com"])]

    # Remove the std features
    summary = {
        "sidechain_len_mean_cb": float(np.mean(lengths_cb)) if len(lengths_cb) else np.nan,
        #"sidechain_len_std_cb": float(np.std(lengths_cb)) if len(lengths_cb) else np.nan,
        "sidechain_len_max_cb": float(np.max(lengths_cb)) if len(lengths_cb) else np.nan,
        "sidechain_len_mean_far": float(np.mean(lengths_far)) if len(lengths_far) else np.nan,
        #"sidechain_len_std_far": float(np.std(lengths_far)) if len(lengths_far) else np.nan,
        "sidechain_len_max_far": float(np.max(lengths_far)) if len(lengths_far) else np.nan,
        "sidechain_len_mean_com": float(np.mean(lengths_com)) if len(lengths_com) else np.nan,
        #"sidechain_len_std_com": float(np.std(lengths_com)) if len(lengths_com) else np.nan,
        "sidechain_len_max_com": float(np.max(lengths_com)) if len(lengths_com) else np.nan,
    }

    return summary

def getMD_analysis_descriptors(ompp_id):
    """ Get descriptors from MDAnalysis library for a given sequence """
    # Get the pdb path
    ompp_id = ompp_id.replace("-", "_") # Replace - with _ for file naming
    pdb_path = os.path.join('data', 'ompp-pdb', f'{ompp_id}.pdb')
    u = mda.Universe(pdb_path)
    ca = u.select_atoms("name CA")

    # Some distance metrics
    # Calculate Euclidean distance between first and last CA atom
    ca_coords = ca.positions
    end_to_end = float(np.linalg.norm(ca_coords[0] - ca_coords[-1]))

    # Calculate distance between first and last atom in the entire structure
    all_coords = u.atoms.positions # Same as protein.positions
    end_to_end_all = float(np.linalg.norm(all_coords[0] - all_coords[-1]))

    # Calculate average Euclidean distance between all pairs of CA atoms
    dist_array = []
    for i in range(len(ca_coords)):
        for j in range(i + 1, len(ca_coords)):
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            dist_array.append(dist)
    avg_ca_distance = np.mean(dist_array)

    outputs = {
        "end_to_end_CA": end_to_end,
        "end_to_end_all": end_to_end_all,
        "avg_CA_distance": avg_ca_distance
    }

    # Get sidechain descriptors
    sidechain_descriptors = getMD_sidechain_descriptors(pdb_path)

    outputs.update(sidechain_descriptors)

    return outputs


def get_descriptors(seq, pH=7.4, include_mordred=False):
    """The main function to calculate all descriptors for a given sequence using modlAMP and other libraries"""

    out_features = {"Sequence": seq}
    # ----- Calculate modlAMP descriptors -----
    modlamp_features = get_modlamp_descriptors(seq, pH=pH)
    out_features.update(modlamp_features)

    # ----- Calculate heliquest-derived hydrophobicity features -----
    heliq_features = get_heliquest_descriptors(seq, out_features['Charge'])
    out_features.update(heliq_features)

    # ----- Calculate peptide descriptors from peptides.py -----
    peptides_features = get_peptide_factors(seq)
    out_features.update(peptides_features)

    # ----- Calculate descriptors using rdkit -----
    rdkit_desc = get_rdkit_descriptors(seq)
    out_features.update(rdkit_desc)

    # ----- Calculate mordred descriptors -----
    if include_mordred:
        mordred_desc = get_mordred_descriptors(seq)
        out_features.update(mordred_desc)

    return out_features


def validate_input_feature_file(file_name: str) -> pd.DataFrame:
    """Validate and load the input sequence CSV used for feature computation."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Input file not found: {file_name}")

    df = pd.read_csv(file_name)
    if 'Sequence' not in df.columns:
        raise ValueError(f"Missing required column 'Sequence' in {file_name}")
    return df

def compute_features_from_file(file_name: str = "data/OMPP_master_long.csv", output_path: str | None = None, pH: float = 7.4, include_mordred: bool = False):
    """
    Compute all descriptors for sequences in a CSV file and save the resulting feature table.

    Expects an input column named 'Sequence'.
    """
    df_seqs = validate_input_feature_file(file_name)
    
    df = compute_features_for_sequences(df_seqs, pH=pH, include_mordred=include_mordred)

    print(Fore.GREEN + f"\nFeature calculation completed! {len(df.columns)} features calculated." + Style.RESET_ALL)

    save_path = output_path or os.path.join(file_name.replace('.csv', '_features.csv'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Sequences with features saved to: {save_path}")

    return df, save_path

def compute_features_for_sequences(df: pd.DataFrame, pH: float = 7.4, include_mordred: bool = False, include_raw_charge: bool = False) -> dict:
    """
    Compute features from a DataFrame containing sequences, returning a new DataFrame with features.
    This function is used for predictions so we want to always compute as many features as possible, thus include_mordred is True by default here.
    """

    seqs = df['Sequence'].tolist()
    if pd.Series(seqs).duplicated().any():
        duplicate_count = int(pd.Series(seqs).duplicated().sum())
        raise ValueError(f"Duplicate sequences detected in input ({duplicate_count} duplicates). Sequences must be unique before feature computation.")
    
    all_features = []
    has_ompp_col = 'OMPP nr' in df.columns
    
    for seq in tqdm(seqs, desc="Sequences"):
        features = get_descriptors(seq, pH=pH, include_mordred=include_mordred)

        if has_ompp_col:
            ompp_ids = df.loc[df['Sequence'] == seq, 'OMPP nr']
            if len(ompp_ids) > 0 and pd.notna(ompp_ids.values[0]):
                try:
                    md_features = getMD_analysis_descriptors(str(ompp_ids.values[0]))
                    features.update(md_features)
                except FileNotFoundError:
                    pass

        if 'Charge' in features and not include_raw_charge:
            del features['Charge']

        all_features.append(features)

    features_df = pd.DataFrame(all_features)
    df = pd.concat([df, features_df.drop(columns=['Sequence'])], axis=1)

    # Remove columns where any value is a string (e.g., due to failed descriptor calculations)
    df.drop(columns=[col for col in df.columns if df[col].dtype == 'object' and col not in {"OMPP nr", "Peptide", "Sequence"}], inplace=True)

    return df

