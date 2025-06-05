import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from diffpy.structure import loadStructure
# from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.pdffit2 import PdfFit



# === User parameters ===
# data_dir  = "/Users/matthewgreenberg/Desktop/BNL in-situ CsPbBr3 Manuscript/2025 Beamtime"  # directory with your CIFs
data_dir = '/nsls2/users/clin1/Documents/Git_BNL/xpd-profile-collection-ldrd20-31/scripts/Matt_multi_phase'
raw_cifs  = ["Cs4PbBr6.cif", "CsBr.cif", "CsPbBr3.cif"]    # original CIF filenames
qmin, qmax = 1.0, 20.0    # Q-range for PDF (Å⁻¹)
rmin, rmax, dr = 2.0, 20.0, 0.01  # r-range and step size (Å)



def pmg_to_diffpy_str(raw_cif, cifwriter_kwargs={"symprec": 0.1}):
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "structure_clean.cif"
        pm_str = Structure.from_file(raw_cif)
        pm_str.remove_oxidation_states()
        w = CifWriter(pm_str, **cifwriter_kwargs)
        w.write_file(str(path))   
        diffpy_structure = loadStructure(str(path))
    return diffpy_structure




def _calculate_pdf(
    diffpy_structure,
    diffpy_structure_attributes={"Uisoequiv": 0.01},
    pdf_calculator_kwargs={
        "qmin": 1, 
        "qmax": 20,
        "rmin": 1.0,
        "rmax": 20.0,
        "qdamp": 0.06,
        "qbroad": 0.06
    }
):
    """Computes the PDF of the given structure.
    
    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Materials structure.
    diffpy_structure_attributes : dict, optional
        Attributes to set on the diffpy structure object.
    pdf_calculator_kwargs : dict, optional
        Keyword arguments to pass to the diffpy PDF calculator.

    Returns
    -------
    numpy.ndarray
    """

    for key, value in diffpy_structure_attributes.items():
        setattr(diffpy_structure, key, value)

    # ## PDFCalculator is for diffpy.srreal and will be replaced by diffpy.pdffit2
    # dpc = PDFCalculator(**pdf_calculator_kwargs)
    # r1, g1 = dpc(diffpy_structure)

    pf = PdfFit()
    pf.alloc('X', 
             pdf_calculator_kwargs['qmax'], 
             pdf_calculator_kwargs['qdamp'], 
             pdf_calculator_kwargs['rmin'], 
             pdf_calculator_kwargs['rmax'], 
             1800
             )
    pf.setvar(pf.qbroad, pdf_calculator_kwargs['qbroad'])
    pf.add_structure(diffpy_structure)
    pf.calc()

    r1 = np.asarray(pf.getR())
    g1 = np.asarray(pf.getpdf_fit())

    # return np.array([r1, g1]).T
    return r1, g1




def calculate_pdf_save(raw_cifs, data_dir, is_save = True, is_plot = True):
    for cif in raw_cifs:
        raw_path = os.path.join(data_dir, cif)
        diffpy_structure = pmg_to_diffpy_str(raw_path)
        r, G = _calculate_pdf(diffpy_structure)

        # 4) Save to .gr (space-delimited)
        if is_save:
            out_name = os.path.splitext(cif)[0] + ".gr"
            out_path = os.path.join(data_dir, out_name)
            np.savetxt(out_path, np.column_stack((r, G)), fmt="%.6f %.6f")
            print(f"Saved {out_name} ({len(r)} points)")
        
        # 3) Plot
        if is_plot:
            ## TODO: Add a plug-in to plot figure from _plot_helper.py
            # plt.plot(r, G, label=os.path.splitext(cif)[0])
            pass

            

            
            
            
# Directory containing both experimental and simulated .gr files
data_dir = "/Users/matthewgreenberg/Desktop/BNL in-situ CsPbBr3 Manuscript/2025 Beamtime"

# Names of your simulated files (adjust if needed)
simulated_files = ["CsBr.gr", "CsPbBr3.gr", "Cs4PbBr6.gr"]

# Find all .gr files in the directory
all_gr = [f for f in os.listdir(data_dir) if f.endswith(".gr")]

# Split into experimental vs. simulated
exp_files = [f for f in all_gr if f not in simulated_files]

# Load simulated data into a dict
sim_data = {}
for fname in simulated_files:
    path = os.path.join(data_dir, fname)
    r_sim, G_sim = np.loadtxt(path).T
    sim_data[fname] = (r_sim, G_sim)

# Prepare results container
results = []

# Loop over experimental files
for fname in exp_files:
    path = os.path.join(data_dir, fname)
    r_exp, G_exp = np.loadtxt(path,skiprows=27).T
    
    # Slice experimental data between 2.0 and 20.0 Å
    mask = (r_exp >= 2.0) & (r_exp <= 20.0)
    r_slice = r_exp[mask]
    G_slice = G_exp[mask]
    
    # Compute Pearson r against each simulated PDF
    row = {"exp_file": fname}
    for sim_fname, (r_sim, G_sim) in sim_data.items():
        # interpolate simulated G onto experimental r-grid
        G_sim_i = np.interp(r_slice, r_sim, G_sim)
        # pearson correlation
        pearson_r = np.corrcoef(G_slice, G_sim_i)[0,1]
        row[sim_fname] = pearson_r
    
    results.append(row)

# Build DataFrame and save
df = pd.DataFrame(results).set_index("exp_file")
print(df.to_string(float_format="%.4f"))

out_csv = os.path.join(data_dir, "pearson_correlations.csv")
df.to_csv(out_csv)
print(f"\nSaved correlations to {out_csv}")