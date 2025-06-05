import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
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




def calculate_pdf(
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


# Configure PDFCalculator
# dpc = PDFCalculator()
# dpc.qmin, dpc.qmax = qmin, qmax
# dpc.rmin, dpc.rmax, dpc.dr = rmin, rmax, dr

plt.figure(figsize=(6,4))
for cif in raw_cifs:
    raw_path = os.path.join(data_dir, cif)
    
    # 1) Clean with pymatgen
    # pm_struct = Structure.from_file(raw_path)
    # pm_struct.remove_oxidation_states()
    # clean_name = os.path.splitext(cif)[0] + "_pym.cif"
    # clean_path = os.path.join(data_dir, clean_name)
    # CifWriter(pm_struct, symprec=0.1).write_file(clean_path)
    
    diffpy_structure = pmg_to_diffpy_str(raw_path)
    
    # 2) Load cleaned CIF and compute PDF
    # struct = loadStructure(clean_path)
    # struct.Uisoequiv = 0.01
    # r, G = dpc(struct)
    
    r, G = calculate_pdf(diffpy_structure)

    
    
    # 3) Plot
    plt.plot(r, G, label=os.path.splitext(cif)[0])
    
    # 4) Save to .gr (space-delimited)
    out_name = os.path.splitext(cif)[0] + ".gr"
    out_path = os.path.join(data_dir, out_name)
    np.savetxt(out_path, np.column_stack((r, G)), fmt="%.6f %.6f")
    print(f"Saved {out_name} ({len(r)} points)")

# Finalize
plt.xlabel("r (Å)")
plt.ylabel("G(r)")
plt.legend()
plt.tight_layout()
plt.show()