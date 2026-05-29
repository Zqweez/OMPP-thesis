import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import py3Dmol
import pathlib

def plot_plddt_and_distance_map(pdb_path: str):
    u = mda.Universe(pdb_path)
    ca = u.select_atoms("protein and name CA")
    if len(ca) == 0:
        raise ValueError("No CA atoms found. Is this a protein PDB?")

    plddt = ca.tempfactors  # AlphaFold pLDDT in B-factor field
    coords = ca.positions
    dmat = distance_array(coords, coords)

    plt.figure()
    plt.plot(np.arange(1, len(plddt) + 1), plddt)
    plt.xlabel("Residue index (CA order)")
    plt.ylabel("pLDDT (B-factor)")
    plt.title("AlphaFold confidence along sequence")
    plt.show()

    plt.figure()
    plt.imshow(dmat, aspect="auto")
    plt.colorbar(label="Distance (Å)")
    plt.title("Cα–Cα distance map")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.show()

def view_pdb(path):
    with open(path) as ifile:
        system = "".join([x for x in ifile])
    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(system)
    view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
    view.zoomTo()
    view.show()
    html = view._make_html()
    out = "viewer.html"
    pathlib.Path(out).write_text(html)



def main():
    pdb_path = "data/ompp-pdb/ompp_1.pdb"
    # view_pdb(pdb_path)
    #plot_plddt_and_distance_map(pdb_path)

    

if __name__ == "__main__":
    main()