import os
from pymatgen.io.cif import CifParser
from os import path
from pymatgen.core.structure import Structure
from contextlib import redirect_stdout
from copy import deepcopy
from functools import lru_cache
from io import StringIO
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from warnings import warn

try:
    from diffpy.Structure import loadStructure
except ImportError:
    warn("diffpy.Structure import failed")
try:
    from diffpy.srreal.pdfcalculator import PDFCalculator
except ImportError:
    warn("diffpy.srreal import failed")
try:
    from diffpy.srfit.pdf import PDFContribution
    from diffpy.srfit.fitbase import FitRecipe, FitResults
    from diffpy.srfit.structure import constrainAsSpaceGroup
except ImportError:
    warn("diffpy.srfit import failed")

try:
    from pyobjcryst import loadCrystal
except ImportError:
    warn("pyobjcryst import failed")

try:
    from diffpy.pdffit2 import PdfFit
except ImportError:
    warn("diffpy.pdffit2 import failed")
    
from monty.json import MSONable
import numpy as np
from pymatgen.io.cif import CifWriter
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize.minpack import leastsq
from scipy.stats import pearsonr
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
# import numpy as np

#These functions are being called in the class Refinery
def pmg_structure_to_diffpy_structure(
    structure,
    cifwriter_kwargs={"symprec": 0.1}
):
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "structure.cif"
        structure.remove_oxidation_states()
        w = CifWriter(structure, **cifwriter_kwargs)
        w.write_file(str(path))   
        diffpy_structure = loadStructure(str(path))
        crystal = loadCrystal(str(path))
    return diffpy_structure, crystal



def calculate_pdf(
    structure,
    cifwriter_kwargs={"symprec": 0.1},
    diffpy_structure_attributes={"Uisoequiv": 0.04},
    pdf_calculator_kwargs={
        "qmin": 1, 
        "qmax": 18,
        "rmin": 1.5,
        "rmax": 22,
        "qdamp": 0.06,
        "qbroad": 0.06
    }
):
    """Computes the PDF of the given structure.
    
    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        Materials structure.
    cifwriter_kwargs : dict, optional
        Keyword arguments to pass to the pymatgen.io.cif.CifWriter class.
    diffpy_structure_attributes : dict, optional
        Attributes to set on the diffpy structure object.
    pdf_calculator_kwargs : dict, optional
        Keyword arguments to pass to the diffpy PDF calculator.

    Returns
    -------
    numpy.ndarray
    """
    diffpy_structure, _ = pmg_structure_to_diffpy_structure(
        structure, cifwriter_kwargs
    )

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
             1000
             )
    pf.setvar(pf.qbroad, pdf_calculator_kwargs['qbroad'])
    pf.add_structure(diffpy_structure)
    pf.calc()

    r1 = pf.getR()
    g1 = pf.getpdf_fit()

    return np.array([r1, g1]).T
    
class Structure2(MSONable):

    @property
    def pmg_structure(self):
        return self._pmg_structure 

    @property
    def material_id(self):
        return self._material_id

    @property
    def unique_elements(self):
        return set([xx.specie.symbol for xx in self._pmg_structure.sites])

    @property
    def pdf(self):
        return self._pdf

    @property
    def refined_pdf(self):
        return self._refined_pdf

    @property
    def metrics(self):
        return self._metrics

    @property
    def diffpy_fit_report(self):
        return self._diffpy_fit_report

    @property
    def status(self):
        return self._status    

    def __repr__(self):
        return self._pmg_structure.__repr__()

    def __eq__(self, s):
        if s.pdf is not None and self.pdf is None:
            return False
        if s.pdf is None and self.pdf is not None:
            return False
        c1 = self.pmg_structure == s.pmg_structure
        if c1 and s.pdf is None and self.pdf is None:
            return True
        return c1 and np.allclose(self.pdf, s.pdf)

    def __init__(
        self,
        pmg_structure,
        pdf=None,
        metrics=None,
        status="original",
        material_id=None,
        diffpy_fit_report=None,
    ):
        self._pmg_structure = pmg_structure
        self._pdf = pdf
        self._metrics = metrics
        self._status = status
        self._material_id = material_id
        self._diffpy_fit_report = diffpy_fit_report

    def calculate_pdf_(
        self,
        cifwriter_kwargs,
        diffpy_structure_attributes,
        pdf_calculator_kwargs,
        interpolation_grid,
    ):
        """Summary
        
        Parameters
        ----------
        cifwriter_kwargs : dict
        diffpy_structure_attributes : dict
        pdf_calculator_kwargs : dict
        interpolation_grid : array_like
            The grid to interpolate all of the pdfs onto. This should be the
            same x-axis grid as the mystery compound.
        """
        ## Add material id as title for diffpy.Structure.loadStructure
        diffpy_structure_attributes.update({'title': self._material_id})

        pdf = calculate_pdf(
            self._pmg_structure,
            cifwriter_kwargs,
            diffpy_structure_attributes,
            pdf_calculator_kwargs,
        )
        ius = InterpolatedUnivariateSpline(pdf[:, 0], pdf[:, 1], k=3)
        self._pdf = ius(interpolation_grid)

    def calculate_metrics_(self, mystery):
        """Summary
        
        Parameters
        ----------
        mystery : array_like
            Description
        """

        if self._pdf is not None:
            pcc = pearsonr(self._pdf, mystery)[0]
            mse = ((self._pdf - mystery)**2).sum()
            self._metrics = {"pearsonr": pcc, "mse": mse}
        else:
            warn("pdf not set")    
    
    def refine_pdf(
        self,
        mystery_path,
        interpolation_grid,
        cifwriter_kwargs={"symprec": 0.1},
        diffpy_structure_attributes={
            "Uisoequiv": 0.04,
        },
        setCalculationRange_kwargs={
            "xmin": 10.0,
            "xmax": 20.0,
            "dx": 0.01
        },
        fitVars={
            "pdfObj.scale": {"args": [1.0], "kwargs": {}},
            "pdfObj.qdamp": {"args": [0.06], "kwargs": {"fixed": True}},
            "pdfObj._NAME_.delta2": {"args": [5.0], "kwargs": {}}
        },
        qmin=1.0,
        qmax=18.0,
        Biso_default=1.0,
        Biso_lower_bound=0.1,
        Biso_upper_bound=10.0,
        refine_lattice_parameters=True,
    ):

        f = StringIO()
        with redirect_stdout(f):

            name = "tmp_name"
            pdfObj = PDFContribution(name)

            # Load the data and set the r-range over which we'll fit
            pdfObj.loadData(mystery_path)
            pdfObj.setCalculationRange(**setCalculationRange_kwargs)

            # Convert the pymatgen structure to the diffpy structure
            diffpy_structure, crystal = pmg_structure_to_diffpy_structure( #crystal
                self._pmg_structure, cifwriter_kwargs
            )
            for key, value in diffpy_structure_attributes.items():
                setattr(diffpy_structure, key, value)
            pdfObj.addStructure(name, diffpy_structure, periodic=True)

            # The FitRecipe does the work of managing one or more contributions
            # that are optimized together.  In addition, FitRecipe configures
            # fit variables that are tied to the model parameters and thus
            # controls the calculated profiles.
            fitObj = FitRecipe()

            # give the PDFContribution to the FitRecipe
            fitObj.addContribution(pdfObj)

            # Apply some attributes
            for key, value in fitVars.items():
                key = key.replace("_NAME_", name)
                fitObj.addVar(eval(key), *value["args"], **value["kwargs"])

            if refine_lattice_parameters:
                try:
                    spaceGroup = crystal.GetSpaceGroup()
                    sgpars = constrainAsSpaceGroup(
                        eval(f"pdfObj.{name}.phase"),
                        spaceGroup.GetName()
                    )
                except ValueError:
                    # The space group identifier was not recognized. Manually set to 'P1'.
                    sgpars = constrainAsSpaceGroup(
                        eval(f"pdfObj.{name}.phase"),
                        'P1'
                    )

                    # Assigns a seperate variable for each independent lattice
                    # paramater #Changes
                for par in sgpars.latpars:
                    fitObj.addVar(par)
                    
                    # Add a lower bound to the lattice parameters

            # Set qmin and qmax
            # The evals here are terrible but what can we do
            eval(f"pdfObj.{name}.setQmin")(qmin)
            eval(f"pdfObj.{name}.setQmax")(qmax)

            # For every unique element type in the structure we apply some
            # values for the thermal noise factor
            # We create the variables of ADP and assign the initial value to
            # them. In this example, we use isotropic ADP for all atoms
            # Most should be around 1, it's a fine default value

            el_iso_map = {}
            for el in self.unique_elements:
                var = f"{el}_Biso"
                v = fitObj.newVar(var, value=Biso_default)
                el_iso_map[el] = v
                fitObj.restrain(var, lb=Biso_lower_bound, ub=Biso_upper_bound)

            atoms = pdfObj.tmp_name.phase.getScatterers()
            for atom in atoms:
                fitObj.constrain(atom.Biso, el_iso_map[atom.element])

            fitObj.clearFitHooks()

            # We can now execute the fit using scipy's least square optimizer.
            print("Refine PDF using scipy's least-squares optimizer:")
            print("  variables:", fitObj.names)
            print("  initial values:", fitObj.values)
            leastsq(fitObj.residual, fitObj.values)
            print("  final values:", fitObj.values)
            print()

            # Obtain and display the fit results.
            pbsResults = FitResults(fitObj)
            print("FIT RESULTS\n")
            print(pbsResults)

            # Get the experimental data from the recipe
            r = eval(f"fitObj.{name}.profile.x")
            gcalc = eval(f"fitObj.{name}.evaluate()")

        ius = InterpolatedUnivariateSpline(r, gcalc, k=3)
        refined_pdf = ius(interpolation_grid)
        fit_report = f.getvalue()
        print(fit_report)

        return Structure2(
            pmg_structure=deepcopy(self._pmg_structure),
            pdf=refined_pdf,
            status="diffpy_refined",
            material_id=self._material_id,
            diffpy_fit_report=fit_report
        )
    
# I am not adding the material project function cuz I already generated cif files
#Now the class Refinery and the calling function above :

def calculate_all_pdfs(
    structures,
    cifwriter_kwargs,
    diffpy_structure_attributes,
    pdf_calculator_kwargs,
    interpolation_grid,
):
    """Runs PDF calculations for a list of provided structures.
    
    Parameters
    ----------
    structures : list
        List of dictionaries as computed in previous steps.
    cifwriter_kwargs : dict
    diffpy_structure_attributes : dict
    pdf_calculator_kwargs : dict
    interpolation_grid : array_like
        The grid to interpolate all of the pdfs onto. This should be the same
        x-axis grid as the mystery compound.
    
    Returns
    -------
    list
    """

    for struct in tqdm(structures):
        struct.calculate_pdf_(
            cifwriter_kwargs,
            diffpy_structure_attributes,
            pdf_calculator_kwargs,
            interpolation_grid
        )


class Refinery(MSONable):
    """Summary
    """

    @property
    def structures(self):
        return self._structures
    
    #changes Added a setter
    @structures.setter
    def structures(self, value):
        self._structures = value
    

    @property
    @lru_cache(None)
    def mystery(self):
        # Prepare the mystery; this will do nothing and is pretty cheap if
        # loading from file

        mystery = np.loadtxt(self._mystery_path)

        ius = InterpolatedUnivariateSpline(
            mystery[:, 0], mystery[:, 1], k=3
        )
        return ius(self.R)

    @property
    @lru_cache(None)
    def R(self):
        rmin = self._pdf_calculator_kwargs["rmin"]
        rmax = self._pdf_calculator_kwargs["rmax"]
        return np.linspace(rmin, rmax, self._N)

    @property
    def cifwriter_kwargs(self):
        return self._cifwriter_kwargs

    @cifwriter_kwargs.setter
    def cifwriter_kwargs(self, x):
        self._cifwriter_kwargs = x

    @property
    def diffpy_structure_attributes(self):
        return self._diffpy_structure_attributes

    @diffpy_structure_attributes.setter
    def diffpy_structure_attributes(self, x):
        self._diffpy_structure_attributes = x

    @property
    def pdf_calculator_kwargs(self):
        return self._pdf_calculator_kwargs

    @pdf_calculator_kwargs.setter
    def pdf_calculator_kwargs(self, x):
        self._pdf_calculator_kwargs = x

    @property
    def setCalculationRange_kwargs(self):
        return self._setCalculationRange_kwargs

    @setCalculationRange_kwargs.setter
    def setCalculationRange_kwargs(self, x):
        self._setCalculationRange_kwargs = x

    @property
    def fitVars(self):
        return self._fitVars

    @fitVars.setter
    def fitVars(self, x):
        self._fitVars = x

    @property
    def other_refine_pdf_kwargs(self):
        return self._other_refine_pdf_kwargs

    @other_refine_pdf_kwargs.setter
    def other_refine_pdf_kwargs(self, x):
        self._other_refine_pdf_kwargs = x

    def __init__(
        self,
        mystery_path,
        results_path, #added this line changes
        criteria={
            "elements": {
                "$in": ["Pb", "S"],
                "$all": ["Pb", "S"]
            }
        },
        strict=["Pb", "S"],
        cifwriter_kwargs={"symprec": 0.1},
        diffpy_structure_attributes={"Uisoequiv": 0.04},
        pdf_calculator_kwargs={
            "qmin": 1.0, 
            "qmax": 18.0,
            "rmin": 1.5,
            "rmax": 22.0,
            "qdamp": 0.06,
            "qbroad": 0.06
        },
        fitVars={
            "pdfObj.scale": {"args": [1.0], "kwargs": {}},
            "pdfObj.qdamp": {"args": [0.06], "kwargs": {"fixed": True}},
            "pdfObj._NAME_.delta2": {"args": [5.0], "kwargs": {}}
        },
        other_refine_pdf_kwargs={
            "Biso_default": 1.0,
            "Biso_lower_bound": 0.1,
            "Biso_upper_bound": 10.0,
        },
        structures=[],
        N=1000,
        refine_lattice_parameters=True,
    ):
        for key, value in locals().items():
            setattr(self, f"_{key}", value)
            
    def populate_structures_(self):
        """
        This method populates the 'structures' attribute of the Refinery object with structures from the CIF files 
        located in the directory specified by 'results_path' during object initialization.

        For each CIF file found in the directory, the structure is loaded using Pymatgen's Structure.from_file method 
        and then wrapped into a Structure2 object, also adding the material_id which is derived from the filename. 
        The Structure2 objects are stored in a list, which is finally assigned to 'self.structures'.

        Note: This method is intended to be used after initialization and requires that the 'results_path' attribute 
        points to a valid directory containing CIF files. If 'results_path' does not point to a valid directory or 
        contains no CIF files, 'self.structures' will be an empty list.
        """
        structures = []
        for filename in os.listdir(self._results_path):
            if filename.endswith(".cif"):
                structure_path = os.path.join(self._results_path, filename)
                pmg_struct = Structure.from_file(structure_path)
                material_id = os.path.splitext(filename)[0]
                structures.append(Structure2(pmg_structure=pmg_struct, material_id=material_id))
        self.structures = structures



    def populate_pdfs_(self):
        """Computes the PDFs for every structure in the current attribute list.
        """
        
        calculate_all_pdfs(
            self._structures,
            self._cifwriter_kwargs,
            self._diffpy_structure_attributes,
            self._pdf_calculator_kwargs,
            self.R,
        )

    def apply_metrics_(self):
        """Applies the specified metrics to the data. For example, if
        metrics=["pearsonr", "mse"] and sort_by="pearsonr", then the pearson
        correlation coefficient and L2 differences will be applied, and the
        results sorted by the pearson correlation coefficient, with the best
        results at the beginning of the list.
        
        Parameters
        ----------
        metrics : list, optional
            A list of strings with options {"pearsonr", "mse"}.
        sort_by : str, optional
            The metric to sort by.
        """

        for struct in self._structures:
            struct.calculate_metrics_(self.mystery)

    def get_sorted_structures(self, metric, status="original"):
        """Returns the structures sorted by the provided metric. This requires
        apply_metrics_ to have been run on all stored structures matching the
        provided status.
        
        Parameters
        ----------
        metric : str
        status : {"original", "diffpy_refined", None}
        
        Returns
        -------
        list
        """

        subset_structures = [
            s for s in self._structures if s.status == status
        ] if status is not None else self._structures

        r = False
        if metric == "pearsonr":
            r = True
        return sorted(
            subset_structures, key=lambda x: x.metrics[metric], reverse=r
        )

    def refine_selected_(self, top_n=5, metric="pearsonr"):
        """Takes the specified top structures and provided metric, and refines
        the structure parameters using diffpy.
        
        Parameters
        ----------
        top_n : int, optional
        metric : str, optional
        """

        setCalculationRange_kwargs = {
            "xmin": self.R[0],
            "xmax": self.R[-1],
            "dx": self.R[1] - self.R[0]
        }

        sorted_structures = self.get_sorted_structures(metric)
        new_structures = []
        for struct in tqdm(sorted_structures[:top_n]):
            new_struct = struct.refine_pdf(
                mystery_path=self._mystery_path,
                interpolation_grid=self.R,
                cifwriter_kwargs=self._cifwriter_kwargs,
                diffpy_structure_attributes=self._diffpy_structure_attributes,
                setCalculationRange_kwargs=setCalculationRange_kwargs,
                fitVars=self._fitVars,
                qmin=self._pdf_calculator_kwargs["qmin"],
                qmax=self._pdf_calculator_kwargs["qmax"],
                refine_lattice_parameters=self._refine_lattice_parameters,
                **self._other_refine_pdf_kwargs,
            )
            new_structures.append(new_struct)

        # Concatenate the new lists!
        self._structures = self._structures + new_structures
        

        
if __name__ == "__main__":
    #Old Code
    refinery = Refinery(
        mystery_path="/home/xf28id2/Documents/ChengHung/pdffit2_example/CsPbBr3/CsPbBr3.gr", 
        results_path="/home/xf28id2/Documents/ChengHung/pdffit2_example/results_CsPbBr_chemsys_search",  # add this line
        criteria={"elements":
                  {#["Pb","Se"],
                   #"$in": ["Cs"], 
                   "$all": ["Pb"],

        }},
        strict=[],
        # strict=["Pb", "S"],
        pdf_calculator_kwargs={
            "qmin": 1.0, 
            "qmax": 18.0,
            "rmin": 2.0,
            "rmax": 60.0,
            "qdamp": 0.031,
            "qbroad": 0.032
        },
    )
