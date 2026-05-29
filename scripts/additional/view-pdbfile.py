from pymol import cmd
from pathlib import Path
from PIL import Image

# ===== SETTINGS =====
pdb_dir = Path("data/ompp-pdb")
output_dir = Path("outputs/pdb-view/front-page")
output_dir.mkdir(parents=True, exist_ok=True)

pdb_files = sorted(pdb_dir.glob("*.pdb"))
for pdb_file in pdb_files: 
    png_file = output_dir / f"{pdb_file.stem}.png"
    pdf_file = output_dir / f"{pdb_file.stem}.pdf"

    # ===== RESET =====
    cmd.reinitialize()

    # ===== LOAD =====
    cmd.load(str(pdb_file), "peptide")
    cmd.remove("solvent")
    cmd.hide("everything", "all")

    # ===== MAIN REPRESENTATION =====
    # Backbone as cartoon
    cmd.show("cartoon", "peptide")

    # Sidechains only surface
    cmd.select("sidechains", "peptide and polymer.protein and not name N+C+O")

    # ===== STYLING =====
    # Cartoon style
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)

    # Colors by secondary structure
    cmd.color("tv_red", "ss h")         # helices
    cmd.color("wheat", "ss s")          # sheets
    cmd.color("grey70", "ss l+")        # loops/other

    cmd.show("sticks", "sidechains")
    cmd.set("stick_radius", 0.18, "sidechains")

    # Surface style
    cmd.show("surface", "peptide")
    cmd.color("br7", "peptide")
    cmd.set("transparency", 0.65, "peptide")

    cmd.show("surface", "sidechains")
    cmd.color("lightblue", "sidechains")
    cmd.set("transparency", 0.75, "sidechains")
    cmd.set("surface_quality", 1)
    
    # Positively charged residues
    cmd.color("palegreen", "resn LYS+ARG+HIS")

    # Negatively charged residues
    cmd.color("paleyellow", "resn ASP+GLU")

    # Background
    cmd.bg_color("white")

    # ===== RENDER SETTINGS =====
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)          # cleaner / less harsh
    cmd.set("specular", 0.2)
    cmd.set("shininess", 20)
    cmd.set("ambient", 0.35)
    cmd.set("direct", 0.45)
    cmd.set("orthoscopic", 1)          # cleaner projection for figures

    # Optional outline effect
    cmd.set("ray_trace_gain", 0.1)
    cmd.set("ray_trace_disco_factor", 1)

    # ===== VIEW =====
    cmd.orient("peptide")
    cmd.zoom("peptide", 4)

    # Optional slight rotation for nicer perspective
    cmd.turn("y", 20)
    cmd.turn("x", -15)

    # ===== EXPORT PNG =====
    cmd.ray(2400, 1800) # (4800, 3600) for higher res
    cmd.png(str(png_file), dpi=600)

    print(f"Saved PNG: {png_file}")
