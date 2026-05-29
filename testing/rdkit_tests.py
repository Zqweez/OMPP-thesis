from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import pandas as pd

def get_descriptors(seq):
    """Calculate rdkit descriptors for a given sequence"""
    m = Chem.MolFromSequence(seq)

    img = Draw.MolToFile(m, "outputs/test/peptide.svg", size=(600, 600))

    features_to_include = [
        "MolLogP", "NumHDonors", "NumHAcceptors", "NumValenceElectrons",
        "NumRotatableBonds", "NHOHCount", "NOCount", "NumAromaticRings", "TPSA", 
    ]

    print(f"MolWt: {Descriptors.MolWt(m)}")
    print(f"MolLogP: {Descriptors.MolLogP(m)}")
    print(f"NumHDonors: {Descriptors.NumHDonors(m)}")
    print(f"NumHAcceptors: {Descriptors.NumHAcceptors(m)}")
    print(f"NumValenceElectrons: {Descriptors.NumValenceElectrons(m)}")
    print(f"NumRotatableBonds: {Descriptors.NumRotatableBonds(m)}")
    print(f"NHOHCount: {Descriptors.NHOHCount(m)}")
    print(f"NOCount: {Descriptors.NOCount(m)}")
    print(f"NumAromaticRings: {Descriptors.NumAromaticRings(m)}")

    print(f"TPSA: {Descriptors.TPSA(m)}")

    values = {
        name: func(m)
        for name, func in Descriptors._descList
    }
    
    return values
    # Create a DataFrame from the values dictionary and save to CSV
    #df = pd.DataFrame(list(values.items())[:132], columns=['key', 'value'])
    #df.to_csv("outputs/test/peptide_descriptors.csv", index=False)
    
    return values

def main():
    # Get descriptors and then load correlation matrix from csv to prune descriptors
    descriptors = pd.read_csv("data/feature-selection/rd_descriptors.csv")

    correlation_matrix = pd.read_csv("data/APD6/correlated_features_1000_all.csv")    

    threshold = 0.85
    prune_list = []
    for descriptor in descriptors["key"]:
        # rows where descriptor appears in either column
        mask = (
            (correlation_matrix["feature_1"] == descriptor) |
            (correlation_matrix["feature_2"] == descriptor)
        )

        if not mask.any():
            continue  # descriptor not present at all

        max_corr = correlation_matrix.loc[mask, "correlation"].abs().max()

        if max_corr > threshold:
            prune_list.append([descriptor, max_corr])

    print(f"Descriptors to prune ({len(prune_list)}):")
    print(prune_list)

    pd.DataFrame(prune_list).to_csv("data/feature-selection/rdkit_descriptors_to_prune.csv", index=False)

if __name__ == "__main__":
    main()