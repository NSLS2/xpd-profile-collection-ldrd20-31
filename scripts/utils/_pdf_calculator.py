import pymatgen as pm
from diffpy.pdffit2 import PdfFit
from diffpy.structure import loadStructure
from pymatgen.io.cif import CifWriter
from pymatgen.io.cif import CifParser
import os



def _no_oxidation_cif(cif_file):
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0] 
    structure.remove_oxidation_states()
    print(structure, '\n\n')
    cif_pym = os.path.dirname(cif_file) + '/' + os.path.basename(cif_file)[:-4] + '_pym.cif'
    w = CifWriter(structure,symprec=0.1)
    w.write_file(cif_pym) 
    
    return cif_pym
    


def _pdffit2_CsPbX3(gr_data, cif_list, qmax=18, rmax=120, qdamp=0.031, qbroad=0.032, fix_APD=True, toler=0.000001, return_pf=False):

    # Initialize the CifParser with the path to your .cif file
    # Parse the .cif file
    pym_cif = []
    for cif in cif_list:
        cif_new = _no_oxidation_cif(cif)
        pym_cif.append(cif_new)

    # Create new PDF calculator object.
    pf = PdfFit()

    # Load experimental x-ray PDF data
    # qmax = 20.0  # Q-cutoff used in PDF calculation in 1/A
    # qdamp = 0.03 # instrument Q-resolution factor, responsible for PDF decay
    pf.read_data(gr_data, 'X', qmax, qdamp)

    # Load and add structure ------------------------------------------------------------------
    for pym in pym_cif:
        stru = loadStructure(pym)
        stru.Uisoequiv = 0.04
        stru.title = os.path.basename(pym)[:-4]
        #Add loaded .cif 
        pf.add_structure(stru)
    
    # set contrains for lattice parameter, ADP
    if (len(cif_list) == 1) and ('CsPbBr' in os.path.basename(cif_list[0])):
        _set_CsPbBr3_constrain(pf, phase_idx=1, fix_APD=fix_APD)
    elif (len(cif_list) == 2) and ('CsPbCl' in os.path.basename(cif_list[-1])):
        _set_CsPbBr3_constrain(pf, phase_idx=1, fix_APD=fix_APD)
        _set_CsPbCl3_constrain(pf, phase_idx=2, fix_APD=fix_APD)

    # set constrain for data scale
    pf.constrain(pf.dscale, '@902')
    pf.setpar(902, 1.0)

    # Set value for Qdamp, Qbraod
    pf.setvar(pf.qdamp, qdamp)
    pf.setvar(pf.qbroad, qbroad)

    # Refine 
    pf.pdfrange(1, 2.5, rmax)
    pf.refine(toler=toler)

    if return_pf:
        return pf
    else:
        phase_fraction = pf.phase_fractions()['mass']
        particel_size = []
        for i in range(pf.num_phases()):
            pf.setphase(i+1)
            particel_size.append(pf.getvar(pf.spdiameter))
        return phase_fraction, particel_size



'''
lat_dic = {'a':11, 'b':12, 'c':13}
'''
def _set_CsPbBr3_constrain(PDF_calculator_object, phase_idx=1, fix_APD=True):
    pf = PDF_calculator_object
    pf.setphase(phase_idx)

    # Refine lattice parameters a, b, c.
    pf.constrain(pf.lat(1), "@11")
    pf.constrain(pf.lat(2), "@12")
    pf.constrain(pf.lat(3), "@13")
    # set initial value of parameter @1, @2, @3
    pf.setpar(11, pf.lat(1))
    pf.setpar(12, pf.lat(2))
    pf.setpar(13, pf.lat(3))

    # Refine phase scale factor.  Right side can have formulas.
    pf.constrain('pscale', '@111')
    pf.setpar(111, 1.0)
    # pf.setpar(20, pf.getvar(pf.pscale) / 2.0)

    # Refine sharpening factor for correlated motion of close atoms.
    pf.constrain(pf.delta2, '@122')
    pf.setpar(122, 6.87)
    pf.fixpar(122)

    # Refine diameter for the spherical particle
    pf.constrain(pf.spdiameter, '@133')
    pf.setpar(133, 80)
    # pf.fixpar(133)

    # Set temperature factors isotropic to each atom
    # idx starts from 1 not 0
    # 1-4: Cs
    for idx in range(1, 5):
        pf.constrain(pf.u11(idx), '@101')
        pf.constrain(pf.u22(idx), '@101')
        pf.constrain(pf.u33(idx), '@101')
    pf.setpar(101, 0.029385)

    # 5-8: Pb
    for idx in range(5, 9):
        pf.constrain(pf.u11(idx), '@102')
        pf.constrain(pf.u22(idx), '@102')
        pf.constrain(pf.u33(idx), '@102')
    pf.setpar(102, 0.027296)

    # 9-16: Br
    for idx in range(9, 17):
        pf.constrain(pf.u11(idx), '@103')
        pf.constrain(pf.u22(idx), '@103')
        pf.constrain(pf.u33(idx), '@103')
    pf.setpar(103, 0.041577)

    # 16-20: Br
    for idx in range(17, 21):
        pf.constrain(pf.u11(idx), '@104')
        pf.constrain(pf.u22(idx), '@104')
        pf.constrain(pf.u33(idx), '@104')
    pf.setpar(104, 0.028164)


    if fix_APD:
        for par in [101, 102, 103, 104]:
            pf.fixpar(par)



def _set_CsPbCl3_constrain(PDF_calculator_object, phase_idx=1, fix_APD=True):
    pf = PDF_calculator_object
    pf.setphase(phase_idx)

    # Refine lattice parameters a, b, c.
    pf.constrain(pf.lat(1), "@21")
    pf.constrain(pf.lat(2), "@21")
    pf.constrain(pf.lat(3), "@21")
    # set initial value of parameter @1, @2, @3
    pf.setpar(21, pf.lat(1))
    pf.setpar(21, pf.lat(2))
    pf.setpar(21, pf.lat(3))

    # Refine phase scale factor.  Right side can have formulas.
    pf.constrain('pscale', '@111')
    pf.setpar(111, 0.85)
    # pf.setpar(20, pf.getvar(pf.pscale) / 2.0)

    # Refine sharpening factor for correlated motion of close atoms.
    pf.constrain('pscale', '1-@111')
    # pf.constrain(pf.delta2, '@122')
    # pf.setpar(122, 6.87)
    # pf.fixpar(122)

    # Refine diameter for the spherical particle
    pf.constrain(pf.spdiameter, '@233')
    pf.setpar(233, 20)
    # pf.fixpar(133)

    # Set temperature factors isotropic to each atom with an initial value 0.04
    # idx starts from 1 not 0
    # 1: Cs  -->  same temperature factor as Cs in phase 1
    for idx in range(1, 2):
        pf.constrain(pf.u11(idx), '@101')
        pf.constrain(pf.u22(idx), '@101')
        pf.constrain(pf.u33(idx), '@101')
    pf.setpar(101, 0.029385)


    # 2: Pb  -->  same temperature factor as Pb in phase 1
    for idx in range(2, 3):
        pf.constrain(pf.u11(idx), '@102')
        pf.constrain(pf.u22(idx), '@102')
        pf.constrain(pf.u33(idx), '@102')
    pf.setpar(102, 0.027296)

    # 3-5: Cl
    for idx in range(3, 6):
        pf.constrain(pf.u11(idx), '@203')
        pf.constrain(pf.u22(idx), '@203')
        pf.constrain(pf.u33(idx), '@203')
    pf.setpar(203, 0.064436)


    if fix_APD:
        for par in [101, 102, 203]:
            pf.fixpar(par)


