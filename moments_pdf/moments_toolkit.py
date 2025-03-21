######################################################
## moments_toolkit.py                               ##
## created by Emilio Taggi - 2025/01/15             ##
######################################################

#########################################################################
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 3 of the License, or     #
# (at your option) any later version.                                   #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
# GNU General Public License for more details.                          #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program.  If not, see <http://www.gnu.org/licenses/>. #
#########################################################################

########################## Program Usage ################################
#
# ...  ...
# ... ...
# ...
#
#########################################################################



######################## Library Imports ################################

## general libraries
import numpy as np #to handle matrices
from pathlib import Path #to check whether directories exist or not
from pylatex import Document, Command, Section, Subsection, Alignat #to produce a pdf documents with catalogue of operators
from pylatex.utils import NoEscape #also to produce a pdf document
import subprocess #to open pdf files
import time #to use sleep and pause the code
import matplotlib.pyplot as plt #to plot stuff
from scipy.optimize import curve_fit #to extract the mass using a fit of the two point correlator
import gvar as gv #to handle gaussian variables (library by Lepage: https://gvar.readthedocs.io/en/latest/)
import itertools as it #for fancy iterations (product:to loop over indices; cycle: to cycle over markers)
from typing import Any #to use annotations for fig and ax
from matplotlib.figure import Figure #to use annotations for fig and ax
from functools import partial #to specify arguments of functions




## custom made libraries
from building_blocks_reader import bulding_block #to read the 3p and 2p correlators
from moments_operator import Operator, Operator_from_file, make_operator_database #to handle lattice operators
from utilities import all_equal #auxiliary function used to check if all elements in an iterable are equal (credit: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-equal)
from utilities import jackknife, jackknife_resamples #to perform a statistical analysis using the jackknife resampling technique
from utilities import bootstrap, bootstrap_resamples #to perform a statistical analysis using the bootstrap resampling technique
from utilities import plateau_search #function used to search for the plateau region of a 1D array
import correlatoranalyser as CA #to perform proper fits (Marcel's library: https://github.com/Marcel-Rodekamp/CorrelatorAnalyser)



######################## Global Variables ###############################




######################## Main Class ####################################

#each dataset corresponds to an instance of the class, its methods provide useful analysis tools
class moments_toolkit(bulding_block):
    """
    Create an instance of the class to setup the analysis framework associated to the given dataset
    """

    ### Global variable shared by all class instances

    #values of hbar times c
    hbarc = gv.gvar(197.3269631, 0.0000049)

    #useful mass values
    #from PDG
    m_p   : gv._gvarcore.GVar  = gv.gvar(938.27208816, 0.00000029)
    m_pi  : gv._gvarcore.GVar  = gv.gvar(139.5701, 0.0003) #pi+-
    m_pi0 : gv._gvarcore.GVar  = gv.gvar(134.9770, 0.0005) #pi0
    #from reference paper
    m_pi_coarse : gv._gvarcore.GVar  = gv.gvar(136, 2)
    m_pi_fine   : gv._gvarcore.GVar  = gv.gvar(133, 1)

    #values of the lattice spacing for the two lattices of interest
    a_coarse : gv._gvarcore.GVar = gv.gvar(0.1163, 0.0004) 
    a_fine   : gv._gvarcore.GVar = gv.gvar(0.0926, 0.0006)

    #list with the available energy units that can be used
    energy_units = ["lattice", "MeV"]

    #list with the available resampling techniques
    resampling_list = ["jackknife", "bootstrap"]

    #value that is used as threshold for 0
    eps: float = 10**(-20)

    ## default values of some class attributes

    #isospin
    default_isospin: str = 'U-D'

    #resampling technique specifics
    default_central_value_fit:            bool = True
    default_central_value_fit_correlated: bool = True
    default_resample_fit:                 bool = False
    default_resample_fit_correlated:      bool = False
    default_resample_fit_resample_prior:  bool = False
    default_svdcut: float|None  = None
    default_maxiter: int = 10_000

                        




    ### Methods of the class

    ## Initialization Method

    #Initialization function #TO DO; add properly the skip3p option and skipop option
    def __init__(self, p3_folder:str, p2_folder:str,
                 max_n:int=3, plot_folder:str="plots", skipop:bool=False, verbose:bool=False,
                 operator_folder:str="operator_database", **kwargs) -> None:
        
        """
        Initializaiton of the class containing data analysis routines related to moments of nucleon parton distribution functions

        Input:
            - p3_folder: folder having as sub folders all the folders with the 3-point correlators at different time separations T
            - p2_folder: folder with the 2-point correlators (related to the 3 point ones)
            - max_n: maximum number of indices  of the lattice operators we want to have to deal with (changing this parameter will change the number of available operators)
            - plot_folder: str, the file path to the folder where the plot will be saved
            - skipop, bool, if True the creation of the operator list will be avoided
            - operator_folder: str, the file path to the folder where the operator database is
            - verbose: bool, if True info print are provided while the class instance is being constructed
            - **kwargs: all the optional argument take as input by the part class building_block

        Output:
            - None (an instance of the moments_toolkit class is created)
        """


        #Info Print
        if verbose:
            print("\nInitializing the moments_toolkit class instance...\n")

        #we call the initialization of the parent class
        super().__init__(p3_folder=p3_folder, p2_folder=p2_folder, verbose=verbose, **kwargs)


        #we store the variable where we want the plots to be saved
        self.plots_folder: str = plot_folder
        #we create such folder if it does not exist
        Path(plot_folder).mkdir(parents=True, exist_ok=True)

        #input check on the maximum number of indices
        if (type(max_n) is not int) or max_n<=1:
            print("\nAchtung: the maximum number of indices should be at least 2 - Switching from the input value to the defualt max_n=2\n")
            self.max_n = 2
        #if the input is ok we just store the maximum number of indices
        else:
            self.max_n = max_n

        #we initialize a list with all the available number of indices
        self.n_list: list[int] = [i for i in range(2,self.max_n+1)] 

        #we initialize as empty the list with operators selected for the analysis
        self.selected_op: list[Operator] = []

        #and we also put the number of selected operators to 0
        self.Nop: int = 0

        #we initialize the values of the times chosen in the analysis
        self.chosen_T_list = self.T_list[:]
        self.nT = len(self.chosen_T_list)

        #we initialize the default value of the isospin
        self.isospin = self.default_isospin

        #we initialize the resampling technique used in the analysis to be the jackknife
        self.resampling = jackknife
        self.resamples_array = jackknife_resamples
        self.resampling_type = "jackknife"


        ## We initialize to None some variables that will be later accessed using getter methods

        #we initialize the value of the ground state energy
        self.E0: gv._gvarcore.GVar | None = None

        #we initialize the value of the energy difference between first excited and ground state
        self.dE: gv._gvarcore.GVar | None = None

        #we initialize the value of the mass
        self.m: gv._gvarcore.GVar | None = None

        #we initialize the list with the kinematic factors of all the selected operators
        self.Klist: list[gv._gvarcore.GVar] | None = None

        #we initialize the matrix element(M) and the moments(x) array (from the S extraction) with the various methods (fit and finite difference)
        self.M_from_S_fit:  np.ndarray[gv._gvarcore.GVar] | None = None #shape = (Nop,NT) --> with zero as padding for the values of T not allowed
        self.x_from_S_fit:  np.ndarray[gv._gvarcore.GVar] | None = None
        self.M_from_S_diff: np.ndarray[gv._gvarcore.GVar] | None = None
        self.x_from_S_diff: np.ndarray[gv._gvarcore.GVar] | None = None

        #we initialize the parameters we have to specify in the fit
        self.central_value_fit:            bool = self.default_central_value_fit
        self.central_value_fit_correlated: bool = self.default_central_value_fit_correlated
        self.resample_fit:                 bool = self.default_resample_fit
        self.resample_fit_correlated:      bool = self.default_resample_fit_correlated
        self.resample_fit_resample_prior:  bool = self.default_resample_fit_resample_prior
        self.svdcut: float|None  = self.default_svdcut
        self.maxiter: int = self.default_maxiter

        #we instantiate the fit function we are going to use to perform the various analysis (with some of the paramters already specified)
        self.fit = partial(CA.fit,
                           central_value_fit=self.central_value_fit, central_value_fit_correlated=self.central_value_fit_correlated, # <- args for fit strategy
                           resample_fit=self.resample_fit, resample_fit_correlated=self.resample_fit_correlated,
                           resample_fit_resample_prior=self.resample_fit_resample_prior,
                           svdcut=self.svdcut, maxiter=self.maxiter)                                                                 # <- args for lsqfit

        ## We build the list of all the available operators

        #To Do: remove this option
        if skipop==False:

            ## First we check that everything is ok with the database passed as input

            #we compute the number elements that should be in the database
            n_operators = np.sum([6*4**n for n in self.n_list]) # n V, and n A have 4**n operators each, n T has 4**(n+1)

            #we check that the database exists and that it has all the elements it should have, if that is not the case we regenerate the database
            if Path(operator_folder).is_dir()==False or len([x for x in Path(operator_folder).glob('**/*') if x.is_file()])<n_operators:

                #info print
                if verbose:
                    print("\nBuilding the operator database ...\n")

                #we regenerate the opeator database before going on
                make_operator_database(operator_folder=operator_folder, max_n=max_n, verbose=True)


            ## At this point we can go on and read all the operators from the database

            #info print
            if verbose:
                print("\nReading the list of all the available operators from the database...\n")

            #we initialize the list as empty
            self.operator_list: list[Operator] = []

            #we also store all the operators in a dict, as to access them using their specifics (the keys of the dict)
            self.operators_dict: dict[Operator] = {}
                
            #we instantiate the path object related to the folder
            path = Path(operator_folder).glob('**/*')

            #we list the operator files
            operator_files = [x for x in path if x.is_file()]

            #we sort the files according to the operator number
            operator_files.sort(key=lambda x: int(x.name.split("_")[1]))

            #to construct the the operators we loop over the related files
            for file in operator_files[:n_operators]:

                #we obtain the operator from the file (from its name)
                op = Operator_from_file(file.as_posix())

                #we append the operator to the list
                self.operator_list.append(op)

                #we append the operator to the dict
                
                #we first handle the creation of the keys
                if (op.n,op.X) not in self.operators_dict.keys():
                    self.operators_dict[(op.n,op.X)] = {}
                if ( op.irrep, op.block) not in self.operators_dict[(op.n,op.X)].keys():
                    self.operators_dict[(op.n,op.X)][op.irrep, op.block] = []

                #then we apped the operator to the dict
                self.operators_dict[(op.n,op.X)][op.irrep, op.block].append(op)

        #info print
        if verbose:
            print("\nClass Initialization Complete!\n")



    ## Info Printing Methods (methods used to show informative output to the user)

    #Function used to print to pdf and show to the user all the available operators that can be chosen
    def operator_show(self, title:str|None=None, author:str="E.T.", doc_name:str|None=None, verbose:bool=False, show:bool=True, remove_pdf:bool=False, clean_tex:bool=True) -> None:
        """
        Function used to produce and show the .pdf file containing the list of all the available operators.

        Input:
            - title: str, the title of the pdf document (i.e. what is shown on the first page of the pdf)
            - auhor: the name of the author shown in the pdf
            - doc_name: the name of the .pdf file
            - show: bool, if True the .pdf file is opened and shown to the user 
            - remove_pdf: bool, if True the .pdf file is removed right after being created (useful it the user just wants to see it once)
            - clean_tex: bool, if True the tex files used to create the .pdf file are removed after the pdf creation

        Output:
            - None (the function just shows and/or save the pdf with the catalouge of lattice operators)
        """

        #we set the title param to the default value
        if title is None:
            title = f"Available Operators (with up to n={self.max_n} indices)"

        #we set the document name to the default value
        if doc_name is None:
            doc_name = "operator_catalogue"
        
        #the name of the pdf file is then
        file_name = f"{doc_name}.pdf"
        

        #we instantiate the .tex file
        doc = Document(default_filepath=f'{doc_name}.tex', documentclass='article',page_numbers=False)#, font_size='' )

        #create document preamble
        doc.preamble.append(Command("title", title))
        doc.preamble.append(Command("author", author))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        doc.append(NoEscape(r"\newpage"))

        doc.append(Command('fontsize', arguments = ['8', '12']))

        #we loop over the keys of the operators dict to add the operators to the document
        for n,X in self.operators_dict.keys():

            #we make a section, putting in the title the specifics common to all operators (X and n)
            section = Section(f"X={X}, n={n}",numbering=False)

            for irrep, imul in self.operators_dict[(n,X)].keys():

                #we asses the C parity, trace condition and index symmetry of the whole block of operators

                #first we construct a list with all the properties for all the operators in the block
                C_list = [op.C for op in self.operators_dict[(n,X)][(irrep,imul)]]
                tr_list = [op.tr for op in self.operators_dict[(n,X)][(irrep,imul)]]
                symm_list = [op.symm for op in self.operators_dict[(n,X)][(irrep,imul)]]

                #then to asses the global properties we check if all the elements in the lists are the same
                C = C_list[0] if all_equal(C_list) else 'mixed'
                tr = tr_list[0] if all_equal(tr_list) else 'mixed'
                symm = symm_list[0] if all_equal(symm_list) else 'Mixed Symmetry'

                #we make a subsection, putting in the title the specifics common to all operators (irrep, block and the shared symmetries)
                subsection = Subsection(f"{irrep} Block {imul}:  Trace {tr}, {symm}, C = {C}",numbering=False)

                #we instantiate the math environment where we print the operators
                agn = Alignat(numbering=False, escape=False)

                #we loop over the operators in the dict
                for op in self.operators_dict[(n,X)][(irrep,imul)]:

                    #we append first the operator number (its id)
                    agn.append(r"\text{Operator "+str(op.id)+r"}&\\")
                
                    #we append the output to the mathematical latex environment
                    agn.append(r"\!"*20 + r" O_{}^{} &= {} \\".format(op.index_block,'{'+f"{X}{irrep},{imul}"+'}',op))

                    #we append the kinematic factor to the math environment
                    agn.append(r"\!"*20 + r" K_{}^{} &= {} \\\\\\".format(op.index_block,'{'+f"{X}{irrep},{imul}"+'}',op.latex_K))

                #we append the math expression to the subsection
                subsection.append(agn)

                #and we append the subsection to the section
                section.append(subsection) 
            
            #then we append the section to the document
            doc.append(section)
            doc.append(NoEscape(r"\newpage"))

        #then we generate the pdf
        
        #info print
        if verbose:
            print("\nGenerating the operators catalogue ...\n")

        #pdf generation
        doc.generate_pdf(doc_name, clean_tex=clean_tex)

        #info print
        if verbose:
            print("\nOperators catalogue generated\n")

  

        #we show to screen the generated pdf
        if show:
            #we xdg-open the pdf
            subprocess.call(["xdg-open", file_name])
            #we wait 1.5 seconds so that the pdf can be seen before its deletion
            time.sleep(1.5)
            #info print
            if verbose:
                print("\nOperators catalogue shown\n")

        #we remove the pdf
        if remove_pdf:

                #file elimination
                Path(file_name).unlink()

                #info print
                if verbose:
                    print("\nOperators catalogue removed from the device\n")

        elif verbose:
            print(f"\nOperators catalogue available in {file_name}\n")

    #overwrite of the bulting repr method to show to the user all the relavant paramters of the class
    def __repr__(self) -> str:
        """
        Function used to show to the user all the relevant paramters of the class

        Input:
            - None: all the information is already stored inside the class
        
        Output:
            - out_string: a string containing all the relevant information about the class instance
        """

        #we create the string to be returned
        string = f"Instance of the moments_toolkit class\n\n"

        string += "Dataset Specifics:\n"
        string += f"Number of configurations: {self.nconf}\n"
        string += f"P: {self.n_P_vec}\n"
        string += f"q: {self.n_q_vec}\n\n"

        string += "Current Selection of Parameters:\n"
        string += f"Number of Selected Operators: {self.Nop}\n"
        string += f"Selected Isospin: {self.isospin}\n"
        string += f"Selected T values: {self.chosen_T_list}\n\n"

        string += "Fit parameters:\n"
        string += f" - Central Value Fit:            {self.central_value_fit}\n"
        string += f" - Central Value Fit Correlated: {self.central_value_fit_correlated}\n"
        string += f" - Resample Fit:                 {self.resample_fit}\n"
        string += f" - Resample Fit Correlated:      {self.resample_fit_correlated}\n"
        string += f" - Resample Fit Resample Prior:  {self.resample_fit_resample_prior}\n"
        string += f" - SVD Cut: {self.svdcut}\n"
        string += f" - Max Iterations: {self.maxiter}\n\n"

        string += f"Resampling Technique: {self.resampling_type}\n"

        #we return the string
        return string
        


    ##  Setter Methods (methods used to set the values of important parameters used in the analysis)

    #method used to the set the resempling technique used in the analysis (either jackknife or bootstrap)
    def set_resampling_type(self, resampling:str,
                            binsize:int=1, first_conf:int=0, last_conf:int=None,
                            Nres:int|None=None, sample_per_resamples:int|None=None) -> None:
        """
        Function used to set the resampling technique used in the analysis
        
        Input:
            - resampling: str, either "jackknife" or "bootstrap", the resampling technique used in the analysis
            - binsize, first_conf, last_conf: int, the parameters used to set the jackknife resampling technique
            - Nres, sample_per_resamples: int, the parameters used to set the bootstrap resampling technique
        
        Output:
            - None (the resampling technique used in the analysis is updated)
        """

        #input control
        if resampling not in self.resampling_list:
            raise ValueError(f"The resampling technique must be one in the list {self.resampling_list}, but instead {resampling} was chosen.")
        
        #if the input is ok we change the resampling technique

        #we set the resampling technique to be either the jackknife ...
        elif resampling == "jackknife":

            #the number of bins and the first and last configuration are set
            self.resampling = partial( jackknife, binsize=binsize, first_conf=first_conf, last_conf=last_conf )
            self.resamples_array = partial( jackknife_resamples, binsize=binsize, first_conf=first_conf, last_conf=last_conf )
            self.resampling_type = "jackknife" 

        #... or the bootstrap
        elif resampling == "bootstrap":

            #we adjust the input to the standard values if we have to
            Nres = Nres if Nres is not None else self.nconf * 2
            sample_per_resamples = sample_per_resamples if sample_per_resamples is not None else self.nconf

            #the number of resamples and the samples per resamples are set 
            self.resampling = partial( bootstrap, Nres=Nres, sample_per_resamples=sample_per_resamples, new_resamples=False )
            self.resamples_array = partial( bootstrap_resamples, Nres=Nres, sample_per_resamples=sample_per_resamples, new_resamples=False ) #new_resamples=False so that the configurations are drawn randomly only once
            self.resampling_type = "bootstrap" 

        #the variable relying on some estimation through the jackknife or bootstrap resampling technique are re-initialized
        self.E0 = None
        self.dE = None
        self.m = None
        self.Klist= None
        self.M_from_S_fit = None 
        self.x_from_S_fit = None
        self.M_from_S_diff= None
        self.x_from_S_diff= None

    #method used to select the default value of the isospin used by default by the other methods #TO DO: add isospin variables re-initialization
    def set_isospin(self, isospin:str) -> None:
        """
        Function used to change the default value of the isospin (by default it is set to "U-D" by the init method)
        
        Input:
            - isospin: str, either 'U', 'D', 'U-D' or 'U+D', the default value of isospin that will be used by all the method calls if no other isospin value is specified
        
        Output:
            - None (the default value of the isospin stored in the class gets updated)
        """
        
        #input control
        if isospin not in self.isospin_list:
            raise ValueError(f"The isospin value must be one in the list {self.isospin_list}, but instead {isospin} was chosen.")
        
        #if the input is ok we change the default value of the isospin (the one that will be used by the other methods if the isospin parameter is not specified when calling them)
        else:
            self.isospin = isospin

    #function used to deselect some source-sink separation values T from the analysis
    def remove_T(self, *args: int, verbose:bool=False) -> None:
        """
        Function used to deselect some source-sink separation values T from the analyses performed by the class
        
        Input:
            - args: list[int], list containing the values T of the source-sink separation that won't be used by the analysy (if this is empty then all the available T will be used)
            - verbose: bool, if True an info print will be given along the function call
            
        Output:
            - None: the list with the chosen T value for the analysis gets updated
        """

        #we read the list with the values of T to remove from the analysis from the input
        T_to_remove_list = args

        #we reset the chosen T to be all the availables ones
        self.chosen_T_list = self.T_list[:]

        #for each T specified by the user we remove it from the analysis
        for T_to_remove in T_to_remove_list:
            if T_to_remove in self.chosen_T_list:
                self.chosen_T_list.remove(T_to_remove)

        #we also update the total number of T in the analysis
        self.nT = len(self.chosen_T_list)

        #we re-initialize all the variables depending on T_list
        self.re_initialize_T_variables()

        #info print
        if verbose:
                print(f"\nAvailable source-skink separation values: {self.T_list}\nChosen source-sink separation values: {self.chosen_T_list}")

    #function used to select which operators we want to study
    def select_operator(self, *kwarg: int) -> None:
        """
        Function used to select the operators on which perform the analysis.

        Input:
            - an int for each operator that the user wants to select: the correspondence between the int and the operator is given in the operator catalogue (accessible via the method operator_show)

        Output:
            - None (the selected_op list, i.e. the list with the selected operators gets updated)
        """

        #the chosen ids are
        chosen_ids = kwarg

        #first we reset the list of selected operators
        self.selected_op = []

        #then for every id we append the corresponding operator to the list of chosen ones
        for id in chosen_ids:

            #we perform an input check
            if type(id) is not int or id<1 or id>len(self.operator_list):
                raise ValueError(f"\nAchtung: the operator id {id} is not valid, please select an id between 1 and {len(self.operator_list)} (included)\n")

            #we append the operator to the list
            self.selected_op.append(self.operator_list[id-1]) #-1 because in the pdf the numbering starts from 1

        #we update the number of selected operators
        self.Nop = len(self.selected_op)

        #we re-initialize all the variables depending on the list selected_op
        self.re_initialize_operator_variables()

    #function used to append an operator (not necessarily one in the catalogue) to the list of selected operators
    def append_operator(self, new_operator: Operator) -> None:
        """
        Input:
            - new_operator: an instance of the Operator class
            
        Output:
            - None, but as a result of the function call the input operator is added to the list of selected operators
        """

        #input check
        if type(new_operator) is not Operator:
            raise ValueError("\nAchtung: the input must be an instance of the Operator class!\n")
        
        #if the input is ok we append the input operator to the list of selected operators
        self.selected_op.append(new_operator)

        #we update the number of selected operators
        self.Nop +=1

        #we re-initialize all the variables depending on the list selected_op
        self.re_initialize_operator_variables()
        
        #we return None
        return None

    #function used to remove operators from the list of selected operators
    def deselect_operator(self, old_operator: Operator|None=None) -> None:
        """
        Function used to remove an Operator from the list of selected operator (if no argument is passed the list is emptied)
        
        Input:
            - old_operator: the operator already in the list of selected operators that want to be removed (if None all the operators are removed from the list)
            
        Output:
            - None (the list of selected operators gets updated)
        """

        #we empty the list no input is specified
        if old_operator is None:
            self.selected_op = []
            self.Nop = 0
            self.re_initialize_operator_variables()
            return None

        #raise an error if the input is specified but it's not of the correct type
        if type(old_operator) is not Operator:
            raise ValueError("\nAchtung: the input must be an instance of the Operator class!\n")
        
        #if the operator is not in the list we raise an error
        elif old_operator not in self.selected_op:
            raise ValueError("\nThe specified operator is not in the list of selected operators, hence it cannot be removed\n")
        
        #if an operator is correctly specified we just remove it from the list
        else:
            self.selected_op.remove(old_operator)
            self.Nop -= 1
            self.re_initialize_operator_variables()
            return None

    #function used to select the fit parameters used for all the fits throughout the analysis
    def set_fit_parms(self, central_value_fit:bool|None=None, central_value_fit_correlated:bool|None=None,
                      resample_fit:bool|None=None, resample_fit_correlated:bool|None=None,
                      resample_fit_resample_prior:bool|None=None,
                      svdcut:float|None=None, maxiter:int|None=None) -> None:
        """
        Function used to specify the paramters that will be used in all the fits performed in the analysis (see  https://github.com/Marcel-Rodekamp/CorrelatorAnalyser).
        The specified paramters will be changed according to the user choice, the others will be given their default value
        Consequently all the quantites that have been computed using the previous fit parameters will be re-initialized.
        
        Input:
            - central_value_fit: bool, if True a central value fit is performed
            - central_value_fit_correlated: bool, if True the correlation is taken into account in the central value fit
            - resample_fit: bool, if True a resample fit is performed
            - resample_fit_correlated: bool, if True the correlation is taken into account in the resample fit
            - resample_fit_resample_prior: bool, if True the mean value of the prior is resampled in the resample fit
            - sdvcut: float, paramter of the lsqfit by Lepage
            - maxiter: int, the maximum number of iterations in the lsqfit by Lepage
            
        Output:
            - None (the fit parameters used in the analysis are updated)
        """

        #we update the parmaters if they have been specified by the user
        self.central_value_fit            = central_value_fit            if central_value_fit            is not None else self.default_central_value_fit
        self.central_value_fit_correlated = central_value_fit_correlated if central_value_fit_correlated is not None else self.default_central_value_fit_correlated
        self.resample_fit                 = resample_fit                 if resample_fit                 is not None else self.default_resample_fit
        self.resample_fit_correlated      = resample_fit_correlated      if resample_fit_correlated      is not None else self.default_resample_fit_correlated
        self.resample_fit_resample_prior  = resample_fit_resample_prior  if resample_fit_resample_prior  is not None else self.default_resample_fit_resample_prior
        self.svdcut                       = svdcut                       if svdcut                       is not None else self.default_svdcut
        self.maxiter                      = maxiter                      if maxiter                      is not None else self.default_maxiter

        #we update the fit function used in the analysis
        self.fit = partial(CA.fit,
                           central_value_fit=self.central_value_fit, central_value_fit_correlated=self.central_value_fit_correlated, # <- args for fit strategy
                           resample_fit=self.resample_fit, resample_fit_correlated=self.resample_fit_correlated, 
                           resample_fit_resample_prior=self.resample_fit_resample_prior,
                           svdcut=self.svdcut, maxiter=self.maxiter)                                                                 # <- args for lsqfit
        
        #we re-initialize all the variables depending on the fit results
        self.E0 = None
        self.dE = None
        self.m = None
        self.Klist= None
        self.M_from_S_fit = None 
        self.x_from_S_fit = None
        self.M_from_S_diff= None
        self.x_from_S_diff= None



    ## Getter Methods (methods used to access properly the data stored in the attributes of the class)

    #function that returns the operator according to the label given in the catalogue
    def get_operator(self, op_number: int) -> Operator:
        """
        Input:
            - op_number: the number of the operator one wants to get as given in the operator catalogue

        Output:
            - an instance of the Operator class with all the specifics of the selected operator
        """
        
        #we perform an input check
        if type(op_number) is not int or op_number>len(self.operator_list) or op_number<1:
            raise ValueError(f"\nAchtung: the operator id {op_number} is not valid, please select an id between 1 and {len(self.operator_list)}\n")

        #we just select the right operator and send it back
        return self.operator_list[op_number-1] #-1 because the numbering starts from 1 in the catalogue, not from 0

    #function used to get the 2 point correlators (with the correct cast)
    def get_p2corr(self) -> np.ndarray:
        """
        Function used to get the 2 point correlators (one for each configuration)
        
        Input:
            - None: all the information is already stored inside the class
        
        Output:
            - p2_corr: two point correlator, shape = (nconf, latticeT), dtype=float (i.e. they are casted to real numbers)
        """

        #we just return what we have already stored, just casting it to real
        return self.p2_corr.real

    #function used to get the 3 point correlation functions related to the selected operators
    def get_p3corr(self, isospin:str|None=None) -> np.ndarray:
        """
        Function used to get the 3 point correlators (the building block, one for each configuration) of the selected operators

        Input:
            - isospin: str, either 'U', 'D', 'U-D' or 'U+D'

        Output:
            - p3_corr: array with the 3 point correlators of the selected operators, shape = (nop, nconf, nT, maxT+1), dtype=float
        """

        #if the isospin parameter is not specified we use the default value (the input control is performed in the last method being called in the tree)
        isospin = isospin if isospin is not None else self.isospin
 

        #we compute the dimensionality of the tau axis (so for T<maxT there is a padding with zeros from taus bigger than their max value)
        maxT = np.max(self.chosen_T_list)

        #we initialize the output array with zeros
        p3_corr = np.zeros(shape=(self.Nop, self.nconf, self.nT, maxT+1), dtype=float) #+1 because tau goes from 0 to T included

        #we now fill the array using the method of the building block class to extract the combination corresponding to the selected operator

        #loop over the selected operators
        for iop,op in enumerate(self.selected_op):


            #loop over the available times T
            for iT,T in enumerate(self.chosen_T_list):

                #we compute the relevant 3 point correlator (that isthe building block related to the operator under study)

                #we have to take the real or imaginary part depending on the kinematic factor (according to the chosen convention, this 3p corr has to be real or imaginary depending if i*Kinematic_factor is)
                if op.p3corr_is_real:
                    p3_corr[iop,:,iT,:T+1] = self.operatorBB(T,isospin, op).real          
                else:  
                    p3_corr[iop,:,iT,:T+1] = self.operatorBB(T,isospin, op).imag          #the last axis of R is padded with zeros

        #we return the 3 point correlators
        return p3_corr

    #function used to get the momentum P at the sink
    def get_P(self, units:str="lattice") -> tuple[gv._gvarcore.GVar,gv._gvarcore.GVar,gv._gvarcore.GVar]:
        """
        Function returning the3 gvar variables with corresponding to Px, Py and Pz, the 3 components of the momentum P at the sink
        
        Input:
            - units: str, either "lattice" or "MeV", the chosen energy units in which the result will be returned
            
        Output:
            - Px, Py Pz: gv.gvar, variables with mean value and std of the 3 components of the momentum 3-vector P
        """

        #input control on the chosen units
        if units not in self.energy_units:
            raise ValueError(f"Error: the value of units must be one in the list {self.energy_units}, but instead units={units} was chosen.")
        
        #we read the components of P from the values stored (and read by) the building block class
        Px = gv.gvar(self.P_vec[0], 0) #(the values have no uncertainty) #-> TO DO: check that statement
        Py = gv.gvar(self.P_vec[1], 0)
        Pz = gv.gvar(self.P_vec[2], 0)

        #we convert the values to MeV if the user asks for it
        if units=="MeV":
            Px = self.lattice_to_MeV(Px)
            Py = self.lattice_to_MeV(Py)
            Pz = self.lattice_to_MeV(Pz)

        #we return the three components of the momentum
        return Px, Py, Pz



    ## Advanced Getter Methods (methods that return parameters stored in the class, but that require a computation being made each time the method is called)

    #function used to obtain the ground state energy from the fit to the two point function
    def get_E(self, force_fit:bool=False,  units:str="lattice") -> gv._gvarcore.GVar:
        """
        Function returning the gvar variable with the ground state energy obtained from the fit to the two point correlator
        
        Input:
            - force_fit: bool, if True the fit is performed again even though the value of the energy could have been fetched from the class
            - units: str, either "lattice" or "MeV", the chosen energy units in which the result will be returneds
            
        Output:
            - E0: gv.gvar, mean value and std of the ground state energy
        """

        #input control on the chosen units
        if units not in self.energy_units:
            raise ValueError(f"Error: the value of units must be one in the list {self.energy_units}, but instead units={units} was chosen.")

        #we check whether we can avoid doing the fit
        if self.E0 is None or force_fit==True:

            #first we perform the fit with the main method
            fit2p = self.fit_2pcorr(show=False, save=False, fit_doubt_factor=3) #TO DO: adjust resamples, i.e. check whether resample fit can be True

            #then we grep the parameters from the models average
            results = fit2p.model_average()

            #we update the value of the ground state energy stored in the class
            self.E0 = gv.gvar( results["est"]["E0"], results["err"]["E0"] )

        #we select a value of the energy according to the units specified by the user
        E0 = self.E0 if units == "lattice" else self.lattice_to_MeV(self.E0)

        #then we return a gv variable containing mean and std of the best estimate of the ground state energy
        return E0
    
    #function used to obtain the energy difference between the the first excited state and the ground state, from the fit to the two point function
    def get_dE(self, force_fit:bool=False,  units:str="lattice") -> gv._gvarcore.GVar:
        """
        Function returning the gvar variable with the delta E (first excited state - ground state) obtained from the fit to the two point correlator
        
        Input:
            - force_fit: bool, if True the fit is performed again even though the value of the energy could have been fetched from the class
            - units: str, either "lattice" or "MeV", the chosen energy units in which the result will be returneds
            
        Output:
            - dE: gv.gvar, mean value and std of the energy difference between the first excited state and the ground state
        """

        #input control on the chosen units
        if units not in self.energy_units:
            raise ValueError(f"Error: the value of units must be one in the list {self.energy_units}, but instead units={units} was chosen.")

        #we check whether we can avoid doing the fit
        if self.dE is None or force_fit==True:

            #first we perform the fit with the main method
            fit2p = self.fit_2pcorr(show=False, save=False,fit_doubt_factor=3) #TO DO: adjust resamples, i.e. check whether resample fit can be True

            #then we grep the parameters from the models average
            results = fit2p.model_average()

            #we update the value of the ground state energy stored in the class
            self.dE = gv.gvar( results["est"]["dE1"], results["err"]["dE1"] )

        #we select a value of the energy according to the units specified by the user
        dE = self.dE if units == "lattice" else self.lattice_to_MeV(self.dE)

        #then we return a gv variable containing mean and std of the best estimate of the ground state energy
        return dE

    #function used to obtain the mass from the ground state energy obtained from the fit to the two point function
    def get_m(self, force_fit:bool=False, units:str="lattice") -> gv._gvarcore.GVar:
        """
        Function returning the gvar variable with the mass, obtained from the ground state energy, obtained from the fit to the two point correlator
        
        Input:
            - force_fit: bool, if True the fit is performed again even though the value of the energy could have been fetched from the class
            - units: str, either "lattice" or "MeV", the chosen energy units in which the result will be returned
            
        Output:
            - m: gv.gvar, mean value and std of the mass extracted from the fit
        """

        #input control on the chosen units
        if units not in self.energy_units:
            raise ValueError(f"Error: the value of units must be one in the list {self.energy_units}, but instead units={units} was chosen.")

        #we check whether we can avoid doing the computation again
        if self.m is None or force_fit==True:
            
            #we update the value of the mass
            self.m = np.sqrt( self.get_E(force_fit=force_fit)**2 - self.P_vec @ self.P_vec ) # m = sqrt( E^2 - p^2 )

        #we select a value of the mass according to the units specified by the user
        m = self.m if units == "lattice" else self.lattice_to_MeV(self.m)

        #then we return a gv variable containing mean and std of the best estimate for the mass
        return m

    #function used to get the list with the kinenatic factors associated to each of the selected operator
    def get_Klist(self, force_computation: bool=False, force_fit: bool=False) -> list[gv._gvarcore.GVar]:
        """
        Function used to obtain the list with all the numerical values of the kinematic factors associated to the selected operators
        
        Input
            - force_computation: bool, if True the K factorsa are computed again even though they could have been fetched from a class variable
            - force_fit: bool, if True the fit of the 2 point correlator is performed again even though the value of the energy  and the mass could have been fetched from the class
            
        Output:
            - Klist: the list with the kinematic factors (as gvar variables), shape = (Nop,)
        """

        #we check whether we have to do the computation
        if self.Klist is None or force_computation==True:

            #we first fetch the quantities we need to evaluate the kinematic factors (energy, mass and momentum)
            m = self.get_m(force_fit=force_fit)
            E0 = self.get_E() #(force fit not required here because the value of E0 gets updated by the call to get_m )
            p1, p2, p3 = self.get_P()

            #we create the list containing the kinematic factor for each operator
            self.Klist = [op.evaluate_K_gvar(m_value=m, E_value=E0, p1_value=p1, p2_value=p2, p3_value=p3) for op in self.selected_op]

        #we return a the list with the kinematic factors
        return self.Klist 

    #function used to compute the ratio R(T,tau)
    def get_R(self, isospin:str|None=None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Input:
            - isospin: str, either 'U', 'D', 'U-D' or 'U+D'

        Output:
            - Rmean(iop,T,tau): the mean resulting from the jackknife analysis performed using as observable the ratio R, shape = (nop, nT, maxT+1)
            - Rstd(iop,T,tau): the std reasulting from the jackknife analysis performed using as observable the ratio R, shape = (nop, nT, maxT+1)
            - Rcovmat(iop,T,tau1, tau2): the whole covariance matrix reasulting from the jackknife analysis performed using as observable the ratio R, shape = (nop, nT, maxT+1, maxT+1)
        """

        #We first take the 3 point and 2 point correlators needed to compute the ratio
        p3_corr = self.get_p3corr(isospin=isospin) #shape = (nop, nconf, nT, maxT+1)
        p2_corr = self.get_p2corr() #shape = (nconf, latticeT)

        #the shape of the ratio is given by (nop, nT, maxT+1), i.e.
        R_shape = p3_corr[:,0,:,:].shape

        #we instantiate the output ratio
        Rmean = np.zeros(shape=R_shape, dtype=float) 
        Rstd = np.zeros(shape=R_shape, dtype=float)
        Rcovmat = np.zeros(shape=R_shape + (R_shape[-1],), dtype=float)

        #we loop over all the T values we have
        for iT,T in enumerate(self.chosen_T_list):

            #we perform the jackknife analysis (the observable being the ratio we want to compute)
            Rmean[:,iT,:], Rstd[:,iT,:], Rcovmat[:,iT,:,:] = self.resampling([p3_corr[:,:,iT,:], p2_corr], lambda x,y: ratio_formula(x,y, T=T, gauge_axis=1), res_axis_list=[1,0], time_axis=-1)

        #we return the ratios just computed and the results of the jackknife analysis
        return Rmean, Rstd, Rcovmat

    #function used to to compute the sum of ratios S
    def get_S(self, tskip: int, isospin:str|None=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Method used to obtain, using a jackknife analysis, the sum of ratios given by S(T,tskip) = sum_(t=tskip)^(T-tskip) R(T,t)

        Input:
            - tskip = tau_skip = gap in time when performing the sum of ratios
            - isospin: either 'U', 'D', 'U-D' or 'U+D'

        Output:
            - Smean: the mean resulting from the jackknife analysis performed using S as observable, shape = (nop, nT)
            - Sstd: the std resulting from the jackknife analysis performed using S as observable, shape = (nop, nT)
        """

        #We first take the 3 point and 2 point correlators needed to compute the ratio and consequently the Summed ratios S
        p3_corr = self.get_p3corr(isospin=isospin) #shape = (nop, nconf, nT, maxT+1)
        p2_corr = self.get_p2corr() #shape = (nconf, latticeT)

        #the shape of the ratio is given by (nop, nT), i.e.
        S_shape =  (self.Nop, self.nT)

        #we instantiate the output ratio
        Smean = np.zeros(shape=S_shape, dtype=float) 
        Sstd = np.zeros(shape=S_shape, dtype=float)

        #we loop over all the T values we have
        for iT,T in enumerate(self.chosen_T_list):
            
            #we compute S using the jackknife algorithm
            Smean[:,iT], Sstd[:,iT], _ = self.resampling( [p3_corr[:,:,iT,:], p2_corr], lambda x,y: sum_ratios_formula( ratio_formula(x,y, T=T, gauge_axis=1), T, tskip, time_axis=-1), res_axis_list=[1,0], time_axis=None )

        #we return S
        return Smean, Sstd

    #function used to extract the matrix elements from the summed ratios
    def get_M_from_S(self, method:str="fit", tskip_list:list[int] = [1,2,3], delta_list:list[int]=[1,2,3], isospin:str|None=None, moments:bool=False, force_computation:bool=False) -> np.ndarray[gv._gvarcore.GVar]:
        """
        Function performing the extraction of the matrix element from the summed ratios using one of the two possible methods (finite differences or fit)

        Input:
            - method: str, either "fit or "finite differences", is the method that will be used to extract the matrix element from the summed ratios
            - tskip_list: list of tau skip we want to use in the analysis
            - delta_list: list of delta that we want to use in the analysis (only used if method == "finite differences")
            - isospin: str, either 'U', 'D', 'U-D' or 'U+D'
            - moments: bool, if True the moments are returned (i.e. matrix elements normalized to the kinematic factors)
            - force_computation: bool, if True the matrix element is computed again even though it could have been fetched from a class variable
        
        Output:
            - matrix_elemets: np.ndarray of gvar variables, with shape (Nop, NT) containing the values of the matrix elements (or moments) - Achtung: some values are padded with zeros to match the shape of self.chosen_T_list
        """

        ## We perform a different calculation depending on the chosen method

        #fit method
        if method=="fit": #TO DO: move computation to another sub-routine
            
            #we check if we have to do the computation
            if self.M_from_S_fit is None or force_computation==True:
                
                #We first compute all the summed ratios we need

                #we initialize the array  summed ratios
                Smean_array = np.zeros(shape=(len(tskip_list),self.Nop,self.nT))
                Sstd_array = np.zeros(shape=(len(tskip_list),self.Nop,self.nT))

                #for each tskip we compute S
                for itskip,tskip in enumerate(tskip_list): #TO DO: pythonize this for loop (after changing the output format of the get_S function)
                    Smean, Sstd = self.get_S(tskip=tskip, isospin=isospin)
                    Smean_array[itskip] = Smean
                    Sstd_array[itskip] = Sstd

                #we also take the list of kinematic factors
                Klist = self.get_Klist()

                #We now acutally do the fit

                #we fill the output arrays with zeros
                self.M_from_S_fit = np.zeros(shape=(self.Nop, self.nT), dtype=object ) #shape = (Nop, nT)
                self.x_from_S_fit = np.zeros(shape=(self.Nop, self.nT), dtype=object ) #shape = (Nop, nT)


                #loop over different operators
                for iop,op in enumerate(self.selected_op):

                    #loop over starting times
                    for iTstart, Tstart in enumerate(self.chosen_T_list):

                        #for each tuple (operator,starting time) we instantiate a fit state
                        fit_state = CA.FitState()

                        ## we want to average over various fit changing the value of tskip and that of the last time considered 

                        for itskip,tskip in enumerate(tskip_list):

                            #we skip the value of tskip that are not allowed
                            if tskip >= (Tstart-1)/2: continue

                            #we loop over the maximum time considered
                            for iTend,Tend in enumerate(self.chosen_T_list):

                                #we skip all the endtimes that are smaller (by 2) w.r.t. the start time
                                if iTend-iTstart < 3: continue

                                #we determine the fit abscissa and ordinate
                                abscissa = np.asarray( self.chosen_T_list[iTstart:iTend+1] )
                                ordinate = Smean_array[itskip,iop,iTstart:iTend+1]
                                ordinate_err = Sstd_array[itskip,iop,iTstart:iTend+1]

                                #we determine the prior

                                #we instantiate the prior dict
                                prior = gv.BufferDict()

                                #we get an estimate on slope and intercept
                                slope = (ordinate[-1] - ordinate[0]) / (abscissa[-1] - abscissa[0])
                                intercept = ordinate[0] - slope * abscissa[0]

                                #we give a 100% error on the value of the slope and of the intercept
                                prior["m"] = gv.gvar(slope,slope)
                                prior["q"] = gv.gvar(intercept,intercept)


                                #we do the fit
                                fit_result = CA.fit(

                                    abscissa                = abscissa,
                                    
                                    ordinate_est            = ordinate, 
                                    ordinate_std            =  ordinate_err, 
                                    ordinate_cov            =   None, 
                                    
                                    resample_ordinate_est   = None,
                                    resample_ordinate_std   = None, 
                                    resample_ordinate_cov   = None,

                                    # fit strategy, default: only uncorrelated central value fit:
                                    central_value_fit            = True,
                                    central_value_fit_correlated = False,

                                    resample_fit                 = False,
                                    resample_fit_correlated      = False,
                                    
                                    resample_fit_resample_prior  = False,
                                    resample_type               = "bst" if self.resampling_type=="bootstrap" else "jkn", #"jkn",                  #TO DO: update this param as soon as the class gets corrected

                                    # args for lsqfit:
                                    model   = lambda x,p: p["m"]*x+p["q"],
                                    prior   = prior,
                                    p0      = None,

                                    svdcut  = None,
                                    maxiter = 10_000,
                                )

                                #we append the fit result to the fit state
                                fit_state.append(fit_result)


                        #we compute the matrix elements using the fit model average (we pad with 0 +- 0 if the fit was not possible)
                        self.M_from_S_fit[iop,iTstart] = gv.gvar(fit_state.model_average()["est"]["m"],fit_state.model_average()["err"]["m"]) if len(fit_state.model_average())>0 else gv.gvar(0,0)
                        self.x_from_S_fit[iop,iTstart] = self.M_from_S_fit[iop,iTstart] / Klist[iop]

            #after computing it we return the matrix element (or moment) array
            return self.x_from_S_fit if moments==True else self.M_from_S_fit

        #finite differences calculation
        elif method=="finite differences":

             #we check if we have to do the computation
            if self.M_from_S_diff is None or force_computation==True:
                
                #we first take the correlators we need to compute everything
                p2corr = self.get_p2corr() #shape = (Nconf, latticeT)
                p3corr = self.get_p3corr(isospin=isospin) #shape = (Nop, Nconf, NT, maxT+1)

                #we also take the list of kinematic factors
                Klist = self.get_Klist()

                #we fill the output arrays with zeros
                self.M_from_S_diff = np.zeros(shape=(self.Nop, self.nT), dtype=object ) #shape = (Nop, nT)
                self.x_from_S_diff = np.zeros(shape=(self.Nop, self.nT), dtype=object ) #shape = (Nop, nT)

                #we fill the output array using the formula for the matrix element from S
                for iop in range(self.Nop):

                    #we compute mean and std of the matrix element using the jackknife #TO DO: check whether the resampling function can be called once and not for each operator
                    mat_ele, mat_ele_std, _ = self.resampling([p3corr[iop],p2corr], observable = lambda x,y: MatEle_from_slope_formula(p3_corr=x, p2_corr=y, T_list=self.chosen_T_list, delta_list=delta_list, tskip_list=tskip_list), res_axis_list=[0,0], time_axis=None)

                    #we put them into a gvar variable and store it into the array
                    self.M_from_S_diff[iop] = gv.gvar(mat_ele,mat_ele_std)
                    self.x_from_S_diff[iop] = gv.gvar(mat_ele,mat_ele_std) / Klist[iop]

            #after computing it we return the matrix element (or moment) array
            return self.x_from_S_diff if moments==True else self.M_from_S_diff

        #raise an error if something else is specified
        else:
            raise ValueError(f"The variable method can only assume values in the list ['fit', 'finite differences'], however method={method} was specified.")



    ## Plotter Methods (methods used to make the relevant plots of the data stored in the class)

    #function used to plot the ratio R for all the selected operators
    def plot_R(self, isospin:str|None=None, show:bool=True, save:bool=False, figname:str='plotR',
               figsize:tuple[int,int]=(20,8), fontsize_title:int=24, fontsize_x:int=18, fontsize_y:int=18, markersize:int=8,
               rescale=False) -> list[tuple[Figure, Any]]:
        """
        Input:
            - isospin: either 'U', 'D', 'U-D' or 'U+D
            - show: bool, if True the plot with R is shown
            - save: bool, if True the plot is saved to .png
            - figname: the name prefix of the .png files (one for each of the selected operators)
            - figsize: size of the matplotlib figure
            - fontsize_title: font size for the title of the plot
            - fontsize_x: font size for the x label of the plot
            - fontsize_y: font size for the y label of the plot
            - markersize: size of the markers on the plot

        Output:
            - fig_ax_list: list with each element being a tuple of the kind (fig, ax), which are the output of the plt.subplots() call, so that the user can modify the figures if he wants to
        """

        #check on the number of selected operators
        if self.Nop==0:
            raise ValueError("\nAchtung: no operator has been selected so no plot will be shown (operators can be selected using the select_operator method)\n")

        
        #we first fetch R using the dedicate method
        Rmean, Rstd, Rcovmat = self.get_R(isospin=isospin)

        #wewe instantiate the output list where we will store all the figure and axes
        fig_ax_list:list[tuple[Figure, Any]]  = []


        #loop over selected operators (for each we make a plot)
        for iop,op in enumerate(self.selected_op):
            
            #instantiate figure
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=figsize)

            #we add figure and axes to the output list
            fig_ax_list.append((fig,ax))

            #we cycle on the markers
            marker = it.cycle(('>', 'D', '<', '+', 'o', 'v', 's', '*', '.', ',')) 

            #we loop over T and each time we add a graph to the plot
            for iT, T in enumerate(self.chosen_T_list):

                times = np.arange(-T/2+1,T/2)

                #we grep the interesting part of the array and we ignore the padding along the last axis
                ratio = Rmean[iop,iT,:T+1]
                ratio_err = Rstd[iop,iT,:T+1]

                #we discard the endpoints
                r = ratio[1:-1]
                r_err = ratio_err[1:-1]

                #we put R into a gvar variable
                r_gvar = gv.gvar(r,r_err)

                #we rescale to the kfactor #TO DO: check the kinematics factors
                if rescale:

                    #we take mass and energy of g.s.
                    E0 = self.get_E()
                    mass = self.get_m()
                    

                    #we set the momentum to its value
                    p1 = gv.gvar(self.P_vec[0], 0)
                    p2 = gv.gvar(self.P_vec[1] ,0)
                    p3 = gv.gvar(self.P_vec[2] ,0)

                    #we compute the kinematic factor
                    kin = op.evaluate_K_gvar(m_value=mass, E_value=E0, p1_value=p1, p2_value=p2, p3_value=p3)

                    #ratio /= kin if kin!=0 else 1
                    
                    r_gvar /= kin if kin!=0 else 1

                #we recast everything into np array
                r = np.asarray( [c.mean for c in r_gvar])
                r_err = np.asarray( [c.sdev for c in r_gvar])

                #_=plt.plot(times,r,marker = 'o', linewidth = 0.3, linestyle='dashed',label=i)
                #ax.errorbar(times, r,yerr=ratio_err, marker = 'o', linewidth = 0.3, linestyle='dashed',label=f"T{T}")
                ax.errorbar(times, r,yerr=r_err, marker = next(marker), markersize = markersize, elinewidth=1, capsize=2, linestyle='',label=f"T{T}")
                ax.legend()

                ax.set_title(r"R(T,$\tau$) - Operator = ${}$".format(op),fontsize=fontsize_title)
                ax.set_xlabel(r"$\tau$", fontsize=fontsize_x)
                ax.set_ylabel('R', fontsize=fontsize_y)

                #ax.grid()

            #we save the plot if the user asks for it
            if save:
                plt.savefig(f"{self.plots_folder}/{figname}_operator{op.id}.png")
            
            #we show the plot if the user asks for it
            if show:
                plt.show()

        #we return fig and ax
        return fig_ax_list
    
    #function used to plot S
    def plot_S(self, tskip:int, isospin:str|None=None, show:bool=True, save:bool=True, figname:str='plotS',
               figsize:tuple[int,int]=(20,8), fontsize_title:int=24, fontsize_x:int=18, fontsize_y:int=18, markersize:int=8,) -> tuple[Figure, Any]:
        """
        Input:
            - tskip = tau_skip = gap in time when performing the sum of ratios
            - isospin: str, either 'U', 'D', 'U-D' or 'U+D'
            - show: bool, if True plots are shown to screen
            - save: bool, if True plots are saved to .png files

        Output:
            - fig, ax: the output of the plt.subplots() call, so that the user can modify the figure if he wants to
        """

        #first thing first we compute S with the fiven t skip 
        #S, Smean, Sstd = self.get_S(tskip=tskip) #shapes = (Nop, Nconf, NT), (Nop, NT), (Nop, NT)
        Smean, Sstd = self.get_S(tskip=tskip, isospin=isospin)  #shapes =  (Nop, NT), (Nop, NT)

        ## we then remove from the plot all the value of S that are 0 (because the given T is too small compared ti tau skip)

        #the treshold value that should be removed is
        T_treshold = 1 + 2*tskip #because we want to have (T+1) -2 -2tau_skip > 0

        #we instantiate the times T to plot to the full list
        T_plot = self.chosen_T_list[:]

        #we loop over all the smaller values of T to find the one from which we should start cutting the data
        for T in range(T_treshold, self.chosen_T_list[0]-1,-1):

            #when (and if) we find the biggest value that can be removed, we remove from it onward and stop the loop
            if T in self.chosen_T_list:

                #the index from where we will cut is
                iT_cut = self.chosen_T_list.index(T) + 1

                #we cut the relevant arrays (the x and y of the plots)
                T_plot = self.chosen_T_list[iT_cut:]
                Smean = Smean[:,iT_cut:]
                Sstd = Sstd[:,iT_cut:]

                #we stop the loop if we find one of such values
                break

        #we instantiate the figure
        fig, ax = plt.subplots(nrows=1,ncols=3,figsize=figsize,sharex=False,sharey=False)

        #we cycle on the markers
        #marker = it.cycle(('>', 'D', '<', '+', 'o', 'v', 's', '*', '.', ',')) 

        #we loop over the operators
        for iop, op in enumerate(self.selected_op):

            #depending on the X structure of the operator we decide in which of the three plots to put it
            plot_index = self.X_list.index(op.X)


            #we take mass and energy of g.s.
            E0 = self.get_E()
            mass = self.get_m()

            #we set the momentum to its value
            p1,p2,p3 = self.get_P()

            #we compute the kinematic factor
            kin = op.evaluate_K_gvar(m_value=mass, E_value=E0, p1_value=p1, p2_value=p2, p3_value=p3)

            #we only plot if the kin factor is not 0
            if kin!=0:

                #we put S into a gvar var
                S_gvar = gv.gvar(Smean[iop],Sstd[iop])

                #we normalize to the kin factor
                S_gvar /= kin

                #we recast into np array
                Smean[iop] = np.asarray( [c.mean for c in S_gvar] )
                Sstd[iop] = np.asarray( [c.sdev for c in S_gvar] )

                #then we plot it
                ax[plot_index].errorbar(T_plot, Smean[iop],yerr=Sstd[iop], marker = 'o', markersize = markersize, linewidth = 0.3, linestyle='dashed',label=r"${}$".format(op.latex_O))


        #we set the title of the plot
        for i,X in enumerate(self.X_list):
            ax[i].set_title(X)
            ax[i].set_xlabel('T/a')
            ax[i].legend()

        ax[0].set_ylabel(r'$\bar{S}(T, t_{skip}=$' +str(tskip) +r'$)$')


        #we save the plot if the user asks for it
        if save:
            plt.savefig(f"{self.plots_folder}/{figname}_tskip{tskip}.png")

        #we show the plot if the user asks for it
        if show:
            plt.show()

        #we return fig and ax
        return fig, ax



    ## Backbone Methods (core methods implementing the underlying logic of the computations carried out during the data analysis) 

    #function returning the building block of the specified operator
    def operatorBB(self, T:int, isospin: str, operator: Operator) -> np.ndarray:
        """
        Input:
            - T: int, the time separation of the 3-point correlator
            - operator: the instance of the Operator class representing the operator under study
            - normalize: if True then the output is the ratio to the two point function

        Output:
            - the building block (a np array) of the one operator specified by cgmat (and with the other features specified by the other inputs) (shape= (nconf,T+1))
        """

        #first thing first we fetch the building block of the basic operators
        bb = self.get_bb(T, operator.X, isospin, operator.nder) #shape = (nconf, T, 4,4) (an extra 4 if X==T)

        #we can then instantiate the ouput array with the right dimensionality
        opBB = np.zeros(shape=(self.nconf,T+1), dtype=complex)

        #we now loop over all the possible indices combinations (n is the number of indices of the operartor)
        for indices in it.product(range(4),repeat=operator.n):

            #using the matrix with the cg coefficients related to the operator we construct its building block
            opBB[:,:] += operator.cgmat[indices] * bb[:,:,*indices]

        #we return the building block of the operator identified by the cgmat passed as input
        return opBB    

    #function used to perform the fit of the two point correlator and to extract from it the ground state energy
    def fit_2pcorr(self, 
                   chi2_treshold:float=1.0, fit_doubt_factor:float=3, cut_treshold:float=-0.2, #statistical analysis params
                   zoom:int=0, show=True, save=True, verbose=False) -> CA.FitState:            #output printing params
        """
        Input:
            - chi2_treshold: float, treshold value of the chi2 used for the plateau determination
            - fit_doubt_factor: enhancement to the std used as prior that is obtained from a simple scipy fit
            - cut_treshold: float, treshold value used to cut the effective mass array (cut is performed when the ratio between the effective mass and its std is below this value)
            - zoom: int, number of points around the plateau to be shown in the plot (i.e. how much "zoom out" should be used in the plot)

        Output:
            - fit_state: instance of the FitState class containing all the information regarding the fits performed
        """

        #info print
        if verbose:
            print("\nPreparing the fit for the two point correlator...\n")

        #we first take the 2point corr
        p2corr = self.get_p2corr() #shape = (nconf, latticeT)

        #first we determine the gauge avg of the 2p corr using the jackknife and we store it for a later use
        p2corr_jack, p2corr_jack_std, p2corr_jack_cov = self.resampling(p2corr, lambda x: np.mean(x, axis=0), res_axis_list=0, time_axis=-1)

        #then we use the jackknife to compute a value of the effective mass (mean, std and cov)
        meff_raw, meff_std_raw, meff_covmat_raw = self.resampling(p2corr, effective_mass_formula, res_axis_list=0, time_axis=-1) #these values of the effective mass are "raw" because they still contain <=0 values (and also padding from the effective mass function)

        #we look at the point where the mass starts to be negative (i.e. the first point that is always set to 0, and hence as a std equal to 0)
        cut=np.where(meff_raw <= cut_treshold * meff_std_raw)[0][0]

        #and we cut the meff arrays there
        meff = meff_raw[:cut]
        meff_std = meff_std_raw[:cut]
        meff_covmat = meff_covmat_raw[:cut,:cut]

        #we can now identify the boundaries of the plateau region
        start_plateau, end_plateau = plateau_search(meff, meff_covmat, only_sig=True, chi2_treshold=chi2_treshold) #we search the plateau not using the whole correlation matrix (the covariance determination is not reliable -> too many params to determine with too little input)

        #we define the gaussian variables corresponding to the valeus of the effective mass in the plateau region
        gv_meff = gv.gvar(meff[start_plateau:end_plateau], meff_std[start_plateau:end_plateau])

        #we then obtain the plateau value by taking the weighted average of these values (weighted by the inverse of the variance)
        gv_meff_plateau = np.average(gv_meff,weights=[1/e.sdev**2 for e in gv_meff])


        #we make an intermediate plot regarding the effective mass and the plateau determination if the user asks either to see it or to save it to file
        if show or save:

            #we instantiate the figure
            fig, axlist = plt.subplots(nrows=2,ncols=1,figsize=(32, 14))

            #we adjust the beginning of the plost according to the zoom out parameter given by the user
            start_plot = start_plateau-zoom if start_plateau-zoom > 0 else 0

            ## plot m_eff vs t

            #we use the first axis for this plot
            ax1 = axlist[0]

            #we determine the time values to be displayed on the plot (x axis) #TO DO: check if a centered effective mass can be used, and then adjust the m_times accordingly --> and check how the start plateau change in that case
            m_times = np.arange(np.shape(meff_raw)[0]) #+ 0.5

            #we plot all the value of the effective mass in the chosen plot range
            _ = ax1.errorbar(m_times[start_plot:end_plateau+zoom], meff_raw[start_plot:end_plateau+zoom], yerr=meff_std_raw[start_plot:end_plateau+zoom], linewidth=0.7, marker='o', markersize=6, elinewidth=1.0, label="Effective Mass")

            #we plot again in a different color the value of the effective mass actually used to determine the plateau, still in the chosen plot range
            _ = ax1.errorbar( np.arange(np.shape(meff)[0])[start_plot:end_plateau+zoom], meff[start_plot:end_plateau+zoom], yerr=meff_std[start_plot:end_plateau+zoom], linewidth=0.7, marker='o', markersize=6, elinewidth=1.0, label="Effective Mass (used for plateau search)")

            #we plot an orizontal line denoting the plateau value
            _ = ax1.hlines(gv_meff_plateau.mean, start_plateau, end_plateau-1, color='red', label="Plateau Region")
            
            #we plot the +- 1 sigma region around the plateau
            _ = ax1.fill_between(m_times[start_plateau:end_plateau], gv_meff_plateau.mean - gv_meff_plateau.sdev, gv_meff_plateau.mean + gv_meff_plateau.sdev, alpha=0.2, color="red")

            #we adjust the plot styling
            _ = ax1.set_title("Effective Mass Plateau")
            _ = ax1.set_ylabel(r"$m_{eff}(t)$")
            _ = ax1.set_xlabel(r"$t$")
            _ = ax1.grid()
            _ = ax1.legend()

            ## plot  C_2pt vs t

            #we use the second axis for this plot  
            ax2 = axlist[1]

            #we determine the time values to be displayed on the plot (x axis)
            times = np.arange(np.shape(p2corr_jack)[0]) #+ 0.5

            #we plot the two point correlator around the plateau region
            _ = ax2.errorbar(times[start_plot:end_plateau+zoom], p2corr_jack[start_plot:end_plateau+zoom], yerr=p2corr_jack_std[start_plot:end_plateau+zoom], linewidth=0.7, marker='o', markersize=6, elinewidth=1.0, label="Two Point Correlator")
            
            #we highlight the plateau region by means of a transparent layer 
            _ = ax2.fill_between(times[start_plateau:end_plateau], np.min(p2corr_jack[start_plot:end_plateau+zoom]), np.max(p2corr_jack[start_plot:end_plateau+zoom]), alpha=0.2, color="red", label="Plateau Region")

            #we adjust the plot styling
            _ = ax2.set_title("Two Point Correlator")
            _ = ax2.set_ylabel(r"$C_{2pt}(t)$")
            _ = ax2.set_xlabel(r"$t$")
            _ = ax2.set_yscale("log")
            _ = ax2.grid()
            _ = ax2.legend()


            #we save the figure if the user ask for it
            if save:
                plt.savefig(f"{self.plots_folder}/plateau_fit_search.png")

            #we show the figure if the user ask for it
            if show:
                plt.show()
            

        ## we now proceed with the proper fit of the two point correlator

        #we declare that we are going to perform only one state and two state fits
        nstates_list = [1,2]

        #we also declare that we are going to perform fits only with a minimum time up to half of the plateau left boundary value
        t_start_list = np.arange(start_plateau, int(start_plateau/2), -1)
        t_end = end_plateau

        #if the user asks for a plot we instantiate the figure we're going to use
        if show or save:
            fig, axs = plt.subplots(nrows=len(t_start_list), ncols=len(nstates_list), figsize=(32, 14), sharex=True, sharey=True)
            fig.text(0.5, 0.04, r"$t$", ha='center', fontsize=16)
            fig.text(0.04, 0.5, r"$C_{2pt}(t)$", va='center', rotation='vertical', fontsize=16)
            fig.suptitle("Two Point Correlator Fits", fontsize=16)
            fig.tight_layout()
            plt.subplots_adjust(left=0.085,bottom=0.1)


        #we instanatiate the states of the fits we want to do
        fit_state = CA.FitState()


        #we now loop over the free parameters of the fit (the number of states and the starting time of the fit)
        for i_state, nstates in enumerate(nstates_list):

            #we then loop over the starting times
            for i_tstart, t_start in enumerate(t_start_list):

                #first we determine the gauge avg of the 2p corr using the jackknife and we store it for a later use
                p2corr_jack_plat, p2corr_jack_plat_std, p2corr_jack_plat_cov = self.resampling(p2corr[:,t_start:t_end], lambda x: np.mean(x, axis=0), res_axis_list=0, time_axis=-1)
                p2corr_resamples_plat = self.resamples_array(p2corr[:,t_start:t_end], lambda x: np.mean( x, axis=0), res_axis_list=0)

                #we handle the prior determination of the parameters

                #we instantiate a prior dict
                prior = gv.BufferDict()

                #we get a first estimate for the mass and the amplitude from  the scipy fit + jackknife analysis
                m_A_fit, m_A_fit_std , _= self.resampling(p2corr[:,t_start:t_end], lambda x: fit_mass(x, t0=t_start, guess_mass=gv_meff_plateau.mean,), res_axis_list=0, time_axis=None) #first entry of m_A_fit is the amplitude, the second is the mass

                #we store the values in the prior dict and we rescale their uncertainy by a factor accounting for the fact that we don't fully trust the simple scipy fit
                prior["A0"] = gv.gvar(m_A_fit[1], m_A_fit_std[1] * fit_doubt_factor)
                E0 = gv.gvar(m_A_fit[0], m_A_fit_std[0] * fit_doubt_factor)
                prior["log(E0)"] = np.log(E0)

                #we then model a prior for the exponential corresponding to the efirst excited state, if needed
                if nstates==2:

                    #for the energy we don't know, so we just give a wide prior assuming the energy doubles from ground to first excited state
                    dE1 = gv.gvar( self.MeV_to_lattice(self.m_pi).mean,  self.MeV_to_lattice(self.m_pi).mean * E0.sdev/E0.mean )
                    prior[f"log(dE1)"]= np.log(dE1)

                    #the amplitude of the term corresponding to the first excited state we extract by all the other information we have using the functional form of the correlator
                    #t_probe = t_start
                    #A1 = (p2corr_jack[t_probe] - prior["A0"] * np.exp(-t_probe*E0) ) * np.exp( t_probe * ( dE1 + E0) )
                    A1_list = []
                    for t_probe in np.arange(t_start,t_end):
                        A1_list.append((p2corr_jack[t_probe] - prior["A0"] * np.exp(-t_probe*E0) ) * np.exp( t_probe * ( dE1 + E0) ))
                    A1 = np.mean(A1_list)
                    prior[f"A1"] = gv.gvar( A1.mean, prior["A0"].sdev/prior["A0"].mean * A1.mean)

                #we actually do the fit using the correlator analyser library

                #we do the fit
                fit_result = self.fit(

                    abscissa                = np.arange(t_start,t_end),
                    
                    ordinate_est            = p2corr_jack_plat, 
                    ordinate_std            =  p2corr_jack_plat_std, 
                    ordinate_cov            =   p2corr_jack_plat_cov, 
                    
                    resample_ordinate_est   = p2corr_resamples_plat,
                    resample_ordinate_std   = p2corr_jack_plat_std, 
                    resample_ordinate_cov   = p2corr_jack_plat_cov,

                    # fit strategy, default: only uncorrelated central value fit:
                    resample_type               = "bst" if self.resampling_type=="bootstrap" else "jkn", #"jkn",                  #TO DO: update this param as soon as the class gets corrected

                    # args for lsqfit:
                    model   = SumOrderedExponentials(nstates),
                    prior   = prior,
                    p0      = None,
                )

                #we append the fit result to the fit_state
                fit_state.append(fit_result)

                #if the user wants a plot we do it
                if show or save:

                    #first we compute the 2p correlator in the region of interest
                    Corr = gv.gvar(p2corr_jack_plat, p2corr_jack_plat_std )

                    #we plot the fit
                    make_fitplot_2pcorr(fit_result=fit_result,correlator=Corr, ax=axs[i_tstart,i_state], nstates=nstates, Ngrad=15)

        #we save the figure if the user asks for it
        if save:
            plt.savefig(f"{self.plots_folder}/2p_corr_fit.png")

        #we sho the figure in the user asks for it
        if show:
            plt.show()

        #we return the fit state
        return fit_state



    ## Auxiliary Methods (useful methods implementing computations and auxiliary routines that are not at the core of the data analysis)

    #function used to convert an energy value from lattice units to MeV
    def lattice_to_MeV(self, input_value:float|np.ndarray|gv._gvarcore.GVar) -> np.ndarray|gv._gvarcore.GVar:
        """
        Function used to convert an energy value from lattice units to MeV
        
        Input:
            - input_value: the value (float, array, gvar variable) that represents energy(ies) in lattice units
        
        Output:
            - output_value: the input converted in MeV
        """

        #we multiply the input by hbar c
        output_value = input_value * self.hbarc

        #then we divide by the lattice spacing #TO DO: add proper determination of coarse and fine lattice
        output_value /= self.a_coarse if self.latticeT==48 else self.a_fine

        #we return the value in MeV
        return output_value
    
    #function used to convert an energy value from lattice units to MeV
    def MeV_to_lattice(self, input_value:float|np.ndarray|gv._gvarcore.GVar) -> np.ndarray|gv._gvarcore.GVar:
        """
        Function used to convert an energy value from MeV to lattice units
        
        Input:
            - input_value: the value (float, array, gvar variable) that represents energy(ies) in MeV
        
        Output:
            - output_value: the input converted in lattice units
        """

        #we divide the input by hbar c
        output_value = input_value / self.hbarc

        #then we multiply by the lattice spacing #TO DO: add proper determination of coarse and fine lattice
        output_value *= self.a_coarse if self.latticeT==48 else self.a_fine

        #we return the value in lattice units
        return output_value

    #function used to re-initialize all the operator dependant class variables
    def re_initialize_T_variables(self) -> None:
        """
        Function that needs to be called after a change in the list of the source-sink separations T to be sure that all the T_list dependant variables are correctly re-initialized
        
        Input:
            - None
        
        Output:
            - None (all the relevant class variables get re-initialized)
        """

        #we reset the array with the matrix elements and moments (shape = (Nop, NT))
        self.M_from_S_fit  = None
        self.x_from_S_fit  = None
        self.M_from_S_diff = None
        self.x_from_S_diff = None

    #function used to re-initialize all the operator dependant class variables
    def re_initialize_operator_variables(self) -> None:
        """
        Function that needs to be called after a change in the list of the selected operators to be sure that all the operator dependant variables are correctly re-initialized
        
        Input:
            - None
        
        Output:
            - None (all the relevant class variables get re-initialized)
        """

        #we reset the list of the kinematic factors (shape = (Nop,))
        self.Klist = None

        #we reset the array with the matrix elements and moments (shape = (Nop, NT))
        self.M_from_S_fit  = None
        self.x_from_S_fit  = None
        self.M_from_S_diff = None
        self.x_from_S_diff = None

    #function used to display the selected operators inside a jupyter notebook
    def display_operators(self) -> None:
        """
        Function used to display into a notebook each of the selected operators.
        
        Input:
            - None
        
        Output:
            - None (the operators in the list of selected operators are shown in the notebook)
        """

        for op in self.selected_op:
            op.display()



    ## Work in Progress Methods (stuff still in development)

    def fit_ratio(self, chi2_treshold=1.0,  fit_doubt_factor=3, tskip_list=[1,2], show=True, save=True, verbose=False, rescale=True,
                        figsize:tuple[int,int]=(20,8), fontsize_title:int=24, fontsize_x:int=18, fontsize_y:int=18, markersize:int=8):
        """
        Input:
            - 

        Output:
            - 
        """

        #info print
        if verbose:
            print("\nPreparing the fit for the ratio of the correlators...\n")

        ## We construct the abscissa for the fit as the list of tuples (T,tau) of values that have a plateau

        #we instantiate the abscissa as an empty list
        #abscissa = np.empty(shape=(self.Nop,), dtype=list)
        abscissa_list = []

        #we take the values of R
        Rmean, Rstd, Rcovmat = self.get_R()

        #we instantiate the dict where we will store the range of the plateaux used to define the abscissa
        plateau_dict = {}

        for iop,op in enumerate(self.selected_op):

            #abscissa[iop] = []
            tmp_a = []

            for iT,T in enumerate(self.chosen_T_list):

                start_plateau, end_plateau = plateau_search(Rmean[iop,iT,:T+1 ], Rcovmat[iop,iT,:T+1 , :T+1 ], only_sig=False, chi2_treshold=chi2_treshold)

                plateau_dict[(iop,iT)] = start_plateau, end_plateau

                for tau in range(start_plateau,end_plateau):
                    #abscissa[iop].append( (T,tau) )
                    tmp_a.append( (T,tau) )

            abscissa_list.append( np.asarray(tmp_a) )

        #the number of resamples is given by
        #nres = abscissa_list[0].shape[0]


        ## Next we construct the ordinate

        #we take the correlators
        p3corr = self.get_p3corr() #shape = (Nop, nconf, nT, maxT+1)
        p2corr = self.get_p2corr()

        #we resample the ratios, len = Nop,, each element with shape = (Nres, Nallowed(T,tau) )
        ratio_resamples_list = [ self.resamples_array([p3corr,p2corr], lambda x,y: np.asarray( [e for l in [ ratio_formula(x, y, T, gauge_axis=1)[iop,iT, plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1] ] for iT,T in enumerate(self.chosen_T_list) ] for e in l] ), res_axis_list=[1,0] ) for iop in range(self.Nop) ]

        #the number of resamples is given by
        nres = ratio_resamples_list[0].shape[0]

        #ratio_list = [ np.asarray( [e for l in [ ratio_formula(p3corr, p2corr, T, gauge_axis=1)[iop,iT, plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1] ] for iT,T in enumerate(self.chosen_T_list) ] for e in l] ) for iop in range(self.Nop) ]
        ratio_list = [ self.resampling([p3corr,p2corr], lambda x,y: np.asarray( [e for l in [ ratio_formula(x, y, T, gauge_axis=1)[iop,iT, plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1] ] for iT,T in enumerate(self.chosen_T_list) ] for e in l] ), res_axis_list=[1,0] ) for iop in range(self.Nop) ]
        ratio_mean_list = [ np.asarray([e for l in  [Rmean[iop,iT, plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1]] for iT,T in enumerate(self.chosen_T_list) ] for e in l] ) for iop in range(self.Nop)]
        ratio_std_list = [ np.asarray([e for l in  [Rstd[iop,iT, plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1]] for iT,T in enumerate(self.chosen_T_list) ] for e in l] ) for iop in range(self.Nop)]
        #ratio_cov_list = [ np.asarray([e for l in  [Rcovmat[iop,iT, plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1], plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1]] for iT,T in enumerate(self.chosen_T_list) ] for e in l] ) for iop in range(self.Nop)]

        ## Then the prior construction

        #we use the fit of the 2 point function
        fit2p_parms = self.fit_2pcorr(show=False,save=False).model_average()

        #from the result of the fit we take the energy
        #dE = gv.gvar(fit2p_parms['est']['dE1'], fit2p_parms['err']['dE1'])

        #we will use as prior the value extracted from S
        matele_fromS = self.get_M_from_S(method="fit",moments=False)

        

        #we instantiate a prior dict for each operator
        priordict_list = []

        for iop in range(self.Nop):

            #we instantiate the dict
            prior = gv.BufferDict()

            #we fill it
            prior["dE"] = gv.gvar(fit2p_parms['est']['dE1'], fit2p_parms['err']['dE1'])

            prior["M"] = np.average(np.asarray([e for e in matele_fromS[iop] if np.abs(e.mean)>self.eps]), weights= [ele.sdev**(-2) for ele in matele_fromS[iop] if np.abs(ele.mean)>self.eps] )

            prior["R1"] = gv.gvar(0.5,1)
            prior["R2"] = gv.gvar(0,1)
            prior["R3"] = gv.gvar(0,1)

            priordict_list.append(prior)
        

        ## Now we are ready to do the fit

        #we instanatiate the states of the fits we want to do (one fit state for each operator)
        fit_state_list = [ CA.FitState() for iop in range(self.Nop) ]


        #for each operator we do a series of fits

        for iop in range(self.Nop):

            #we take the right resampling
            ratio_res = ratio_resamples_list[iop]

            #we take the ratio
            ratio, ratio_std, ratio_cov = ratio_list[iop]

            #we loop over the possible parameters
            for r2 in [False, True]:
                for r3 in [False]:#, True]:


                    #we do the fit
                    fit_result = self.fit(

                        abscissa                = abscissa_list[iop],
                        
                        ordinate_est            = ratio, #np.mean(ratio_res, axis = 0), # ratio_mean_list[iop], #ratio, #np.mean(ratio_res, axis = 0),
                        ordinate_std            = ratio_std, #np.sqrt(nres-1) *np.std (ratio_res, axis = 0), # ratio_std_list[iop], #ratio_std, ##np.sqrt((nres-1)/nres) * np.std (ratio_res, axis = 0),
                        ordinate_cov            = ratio_cov, #(nres-1) * np.cov (ratio_res, rowvar=False), #(nres-1)/nres * np.cov (ratio_res, rowvar=False),
                        
                        resample_ordinate_est   = ratio_res,
                        resample_ordinate_std   = np.sqrt((nres-1)) * np.std (ratio_res, axis = 0),
                        resample_ordinate_cov   = (nres-1) * np.cov (ratio_res, rowvar=False),

                        # fit strategy, default: only uncorrelated central value fit:
                        resample_type               = "bst" if self.resampling_type=="bootstrap" else "jkn",#"jkn",

                        # args for lsqfit:
                        model   = ratio_func_form(r1=True,r2=r2,r3=r3),
                        prior   = priordict_list[iop],
                        p0      = None,
                    )

                    #we append the fit to the list
                    fit_state_list[iop].append(fit_result)

        return fit_state_list

        if show or save:

            #we first fetch R using the dedicate method
            Rmean, Rstd, Rcovmat = self.get_R()

            #wewe instantiate the output list where we will store all the figure and axes
            #fig_ax_list:list[tuple[Figure, Any]]  = []
            #fig_ax_list = self.plot_R(show=False,save=True) #TO DO: adjust save - show condition in plot_R



            #loop over selected operators (for each we make a plot)
            for iop,op in enumerate(self.selected_op):
                

                
                
                
                #fit avg results
                avg_result = fit_state_list[iop].model_average()
                dE = gv.gvar(avg_result["est"]['dE'], avg_result["err"]['dE'])
                matele = gv.gvar(avg_result["est"]['M'], avg_result["err"]['M'])
                R1 = gv.gvar(avg_result["est"]['R1'], avg_result["err"]['R1'])
                R2 = gv.gvar(avg_result["est"]['R2'], avg_result["err"]['R2'])
                R3 = gv.gvar(avg_result["est"]['R3'], avg_result["err"]['R3'])

                post_dict = {key:gv.gvar(avg_result["est"][key], avg_result["err"][key]) for key in avg_result["est"].keys()}

                
                #instantiate figure
                fig,ax = plt.subplots(nrows=1,ncols=1,figsize=figsize)

                #we add figure and axes to the output list
                #fig_ax_list.append((fig,ax))

                #we cycle on the markers
                marker = it.cycle(('>', 'D', '<', '+', 'o', 'v', 's', '*', '.', ',')) 

                #we loop over T and each time we add a graph to the plot
                for iT, T in enumerate(self.chosen_T_list):

                    #times = np.arange(-T/2+1,T/2)

                    start_plateau, end_plateau = plateau_dict[(iop,iT)]

                    times = np.arange(start_plateau, end_plateau)

                    #we grep the interesting part of the array and we ignore the padding along the last axis
                    ratio = Rmean[iop,iT,start_plateau:end_plateau]
                    ratio_err = Rstd[iop,iT,start_plateau:end_plateau]

                    #we discard the endpoints
                    r = ratio#[1:-1]
                    r_err = ratio_err#[1:-1]

                    #we rescale to the kfactor #TO DO: check the kinematics factors
                    if rescale:
                        #mass = self.get_meff()[0]
                        #mass = self.fit_2pcorr(save=False,show=False).model_average()['est']['E0']
                        a = 0.1163 #we cheat
                        hc = 197.327
                        mp_mev = 1000
                        mass = mp_mev/hc * a
                        #kin = 1j * op.evaluate_K(m_value=mass,E_value=mass,p1_value=0,p2_value=0,p3_value=0) #this 1j in front comes from the fact that mat_ele = <x> * i K
                        #if np.iscomplex(kin):
                        #    kin *= -1j
                        #kin = kin.real
                        kin = op.evaluate_K_real(m_value=mass,E_value=mass,p1_value=0,p2_value=0,p3_value=0)
                        ratio /= kin if kin!=0 else 1
                        #ratio /= np.abs( op.evaluate_K(m_value=mass,E_value=mass,p1_value=0,p2_value=0,p3_value=0) )


                    #_=plt.plot(times,r,marker = 'o', linewidth = 0.3, linestyle='dashed',label=i)
                    #ax.errorbar(times, r,yerr=ratio_err, marker = 'o', linewidth = 0.3, linestyle='dashed',label=f"T{T}")
                    ax.errorbar(times, r,yerr=r_err, marker = next(marker), markersize = markersize, linewidth = 0.3, linestyle='dashed',label=f"T{T}")
                    ax.legend()

                    ax.set_title(r"R(T,$\tau$) - Operator = ${}$".format(op),fontsize=fontsize_title)
                    ax.set_xlabel(r"$\tau$", fontsize=fontsize_x)
                    ax.set_ylabel('R', fontsize=fontsize_y)

                    model = ratio_func_form(r1=True,r2=True,r3=True)

                    ax.plot(times,model( np.asarray( [(T,tau) for tau in times] ).mean ,post_dict))


                plt.show()










    

            




######################## Auxiliary Functions ##########################


#function used to compute the ratio of the 3 point correlator to the two point correlator
def ratio_formula(p3_corr:np.ndarray, p2_corr:np.ndarray, T:int, gauge_axis:int=0) -> np.ndarray:
    """
    Function implementing the formula for the ratio of the three point correlator to the two point correalator
    
    Input:
        - p3_corr: the 3 point correlator, i.e the numerator of the ration, with shape (..., nconf, ..., T+1, ...), related to the indices (iconf, tau)
        - p2_corr: the 2 point correlator, i.e. the denominator of the ratio, with shape (nconf, latticeT)
        - T: int, the source sink separation related to the p3_corr
        - gauge_axis: int, the axis of the 3 point corr over which there are the configurations
        
    Output:
        - R(tau): the ratio of the 3 point to the 2 point correlator, with shape (..., ..., T+1, ...)
    """
    
    #First thing first we compute the correlators, and to do so we have to perform the gauge averages
    C_3pt = np.mean( p3_corr, axis=gauge_axis) 
    C_2pt = np.mean( p2_corr, axis=0) #the gauge axis is the first one for the 2 point function

    #then we return the ratio according to the formula
    return C_3pt / C_2pt[T]



#function translating R to S (i.e. the array with ratios to the array where the tau dimension has been summed appropiately)
def sum_ratios_formula(ratio: np.ndarray, T:int, tskip: int, time_axis:int=-1) -> np.ndarray:
    """
    Input:
        - ratio: the arrays R, with a generic shape, as obtained from the ratio formula
        - T: int, the source sink separation related to the p3_corr
        - tskip: int, the tau skip used while summing the ratios
        - time_axis: int, the axis over which the sum should be performed (the tau axis)

    Output:
        - S: the sum of ratios, with only one dimension (it's a number)
    """

    #we implement the formula for the sum of rations in a fancy way (so that we can index the right dimension without knowing how many other dimensions there are)
    return np.sum( np.take(ratio, range(1 + tskip, T+1 -1 -tskip), axis=time_axis) , axis=time_axis) #the extra +1 and -1 are there to discard the endpoints



#function used to extract the matrix element as the slop of the summed ratio function
def MatEle_from_slope_formula(p3_corr:np.ndarray, p2_corr:np.ndarray, T_list:list[int],  tskip_list:list[int] = [1,2], delta_list:list[int] = [1,2,3]) -> float:
    """
    Function implementing the extraction of the matrix element as the slope of the summed ratios
    
    Input:
        - p3_corr: the 3 point correlator, i.e the numerator of the ration, with shape (Nconf, NT, maxT+1 )
        - p2_corr: the 2 point correlator, i.e. the denominator of the ratio, with shape (nconf, latticeT)
        - T_list: int, the lsit with the source sink separation related to the p3_corr
        - tskip_list: list with the taus to be used in the analysis
        - delta_list: list with the deltas to be used in the analysis (for the meaning of tau and delta see the reference paper) 
    
    Output:
        - mat_ele_array: array, with len nT, containing the values of the matrix element obteined as the average over the possible deltas and tau skip #To DO: implement something more than a plain unweighted average
    """

    ## First the calculation of the the summed ratios using the formula

    #we start by instantiatinh the list with the summed ratios we are going to use, shape = (NT, Ntskip)
    S_list = np.zeros(shape=(len(T_list), len(tskip_list)),  dtype=float)

    #loop over selected tau skip
    for i_tskip, tskip in enumerate(tskip_list):

        #loop over available source sink seprations T
        for iT , T in enumerate(T_list):

            #we compute the summed ratio with the formula
            S_list[iT,i_tskip] = sum_ratios_formula( ratio_formula(p3_corr[:,iT,:], p2_corr, T=T, gauge_axis=0), T, tskip, time_axis=-1)

    ## Then the computation of all the matrix elements (one for each available compination of delta+T, and one for each tau skip)

    #we instantiate the list with the allowed matrix elements as empty
    mat_ele_array = np.zeros(shape=(len(T_list),), dtype=float) #shape = (nT,)


    #we loop over the source-sink separations T
    for iT, T in enumerate(T_list):

        #we instantiate a tmp list where we store all the matrix elements related to the given T
        tmp_mat_ele_list = []

        #we loop over the values of tau skip
        for itskip,tskip in enumerate(tskip_list):

            #we skip the not allowed values
            if np.abs( S_list[iT,itskip] ) < 10**(-18): continue

            #we loop over the delta we want to use in the analysis (delta is the separation we use to look at the slope)
            for delta in delta_list:
                
                #a combination T,delta is allowed only if their sum is in the available Ts
                if T + delta not in T_list: continue

                #we check what is the index of the T we have to consider
                iT_plus_delta = T_list.index(T + delta)

                #we compute the matrix element as the slope of the summed ratio function
                tmp_mat_ele_list.append( (S_list[iT_plus_delta,itskip] - S_list[iT,itskip])/delta )

        #for the given T we extract a value of the matrix element, and we just take a simple unnweighted average over all the values of tskip and the allowed values of T+delta
        mat_ele_array[iT] = np.mean(tmp_mat_ele_list) if len(tmp_mat_ele_list)!=0 else 0 #TO DO: check if something better can be done rather than the plain unweighted average

    #we return the array with the matrix element just computed
    return mat_ele_array



#function used to extract the effective mass for the two-point correlators
def effective_mass_formula(corr_2p: np.ndarray, conf_axis:int=0) -> np.ndarray:
    """
    Input:
        - corr_2p: two point correlators, with shape (nconf, Tlat) (with Tlat being the time extent of the lattice)
        - conf_axis: int, the axis with the configurations

    Output:
        - m_eff(t): the effective mass, with shape (Tlat,) (the configuration axis gets removed) (the shape is Tlat and not Tlat-1, so that the jackknife function can be used)
    """

    #we first take the gauge average of the two point correlator
    corr_gavg = np.mean(corr_2p, axis=conf_axis)

    #we get the lattice time T
    Tlat = np.shape(corr_gavg)[0]

    #we instantiate the eff mass array with first dimension of size Tlat (even tough the effective mass should have Tlat-1 time values --> the last will be a 0 of padding)
    meff = np.zeros((Tlat))

    #we compute the effective mass (the loop as it should goes up to Tlat-1, so that the last entry of meff is a 0 of padding)
    for t in range(Tlat-1): 
        
        #to compute the mass we need the ratio of the 2-points correlator
        ratio_2p = corr_gavg[t]/corr_gavg[t+1]

        #then we just need to take the log, but for that the ratio has to be bigger than 1 (??)   (if that does not happen then the mass is set to 0 -> as done in the initialization above)
        if ratio_2p > 1.0:
            meff[t] = np.log( ratio_2p )

    #we send back the effective mass
    return meff



#function used to extract the fit mass from the two-point correlators
def fit_mass(corr_2p: np.ndarray, t0:int, conf_axis:int=0, guess_mass:float|None=None, guess_amp:float|None=None) -> np.ndarray:
    """
    Input:
        - corr_2p: two point correlators, with shape (nconf, Tlat) (with Tlat being the time extent of the lattice)
        - t0: int, the starting lattice time of the input correlator
        - conf_axis: the axis with the configurations
        - guess_mass: the first guess for the mass we want to extract from the fit
        - guess_amp: the first guess for the amplitude we want to extract from the fit

    Output:
        - (mift, mfit_std): the mean value and the std of the mass extracted from the fit
    """



    #we first take the gauge average of the two point correlator
    corr_gavg = np.mean(corr_2p, axis=conf_axis)

    #then we define the first guess for the parameters of the fit
    if guess_amp is None:
        guess_amp = corr_gavg[0]
    if guess_mass is None:
        guess_mass = np.log( corr_gavg[0]/corr_gavg[1] )
    guess = [guess_amp,guess_mass]

    #we define the x and y arrays used for the fit (which are respectively times and corr_gavg)
    times = np.arange(t0, t0+np.shape(corr_gavg)[0])

    #we perform the fit #TO DO: look at maxfev
    popt,pcov = curve_fit(lambda t,amp,mass: amp*np.exp(-t*mass), times, corr_gavg, p0=guess)#,maxfev = 1300) #popt,pcov being mean and covariance matrix of the parameters extracted from the fit
    #perr = np.sqrt(np.diag(pcov)) #perr being the std of the parameters extracted from the fit

    #we read the mass (that's the only thing we're interested about, the amplitude we discard)
    fit_mass = np.array( popt[1] )
    #fit_mass_std = np.array( perr[1] )

    #same thing for amplitude
    fit_amp = np.array( popt[0] )
    #fit_amp_std = np.array( perr[0] )


    #we return the fit mass and the fit amp
    return np.asarray([fit_mass,fit_amp])

#auxiliary function used to plot the fit of the two point correlators
def make_fitplot_2pcorr(fit_result:CA.FitResult, correlator:gv._gvarcore.GVar ,ax:Any, nstates:int=2, Nsigma:float=2, Ngrad:int=30) -> None:
    """
    Function that given a fit result can be used to make the plot of the linear fit of the two point correlator, and append it to the given axis
    
    Input:
        - fit_result: instance of the FitResult class containing the result of the performed fit to the two point function
        - corrrelator: gvar variable with the array of values of the 2 point function to be shown on the plot
        - ax: matplotlib axis where the fit result will be plotted on
        - nstates: int, the number of states used in the given fit
        - Nsigma: float, the number of standard deviations that will be shown around the mean value
        - Ngrad: int, the number of lines that will be plotted (with an inversionally proportional alpha) to highlight the Nsigma region
    
    Output:
        - None (the plot gets appended to the axis passed as input)
    """

    #we take the x and y of the fit result function
    abscissa = np.arange(fit_result.ts, fit_result.te+1)
    ordinate = fit_result.eval(abscissa)

    #we plot the fit input data (the correlator) with its std
    _ = ax.errorbar(x=abscissa, y=gv.mean(correlator), yerr=gv.sdev(correlator), marker='.', linewidth=0, elinewidth=1, label="Correlator data", color="red")

    #we plot the fit result function
    (line,) = ax.plot( abscissa, ordinate["est"],
                      label=rf"$N_\text{{states}}={nstates}, ({abscissa[0]},{abscissa[-1]}) "
                      rf"\chi^2/_\mathrm{{dof}}~[\mathrm{{dof}}] = {fit_result.chi2/fit_result.dof:g}~[{fit_result.dof:g}], "
                      rf"\text{{AIC}} = {fit_result.AIC:g} $",
                      linestyle="-",)

    #we plot the Nsigma region using a fill_between Ngrad times (we plot a "gradient" of color)
    for igrad in range(Ngrad):
        _ = ax.fill_between(abscissa, ordinate["est"] + Nsigma*ordinate["err"] * igrad/Ngrad, ordinate["est"] - Nsigma*ordinate["err"] * igrad/Ngrad,
                            color=line.get_color(), alpha= 1.0/(Nsigma*np.sqrt(Ngrad)) * (1.0-igrad/Ngrad),)

    #we prepare the text with all the parameters' information to be shown inside the plot
    str_list = ['Estimated parameters:']
    for key in fit_result.result_params()["est"].keys():
        str_list.append( f"{key} =" + str( gv.gvar(fit_result.result_params()["est"][key], fit_result.result_params()["err"][key]) ) )
    textstr = '\n'.join(str_list)    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    #we display the text box with the relevant parameters inside the plot (in a region where we know by inspection that there will not be overlap with the correlator's plot)
    _ = ax.text(0.11, 0.55, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    #we adjust the plot styling
    _ = ax.set_yscale("log")
    _ = ax.grid()
    _ = ax.legend()


#auxiliary function used to average values of the moments over various values of T
def average_moments_over_T(in_array:np.ndarray[gv._gvarcore.GVar], chi2:float=1.0) -> gv._gvarcore.GVar:
    """
    Function used to average the moments over different source sink separations, as in eq 14 of the reference paper
    
    Input:
        - in_array: 1D np array of gvar variables, shape = (Ntimes,), either moments or matrix elements that needs to be averaged
        - chi2: value of the chi2 used as treshold for the average determination
    
    Output:
        - average: gvar variable with mean and std of the average moment (or matrix element)
    """

    #first we remove the padding
    in_array = np.asarray( [ e for e in in_array if np.abs(e.mean)>0 ] )

    #then we grep the mean and std from the gvar variables
    mean_array = np.asarray( [ e.mean for e in in_array ] )
    std_array = np.asarray( [ e.sdev for e in in_array ] )

    #we compute the weights (inverse sigma squared)
    weights = std_array**(-2)

    #we now loop over the values of T, starting from the lowest, and we cut from there
    for iTmin in range(int(len(mean_array)/2)):

        #we compute the average value
        avg = np.average(in_array[iTmin:], weights=weights[iTmin:])

        #if the chi2 is smaller than the treshold then we return the array
        if np.sum( ((mean_array[iTmin:] - avg.mean)/std_array[iTmin:])**2 ) / len(in_array[iTmin:]) < chi2: 
            return avg, iTmin

    #if the chi2 is never smaller than the treshold we just return the last value
    return avg, iTmin





#auxiliary class used to fit the two point correlators
class SumOrderedExponentials:
    def __init__(self, number_states):
        self.number_states = number_states

    def __call__(self,t,p):
        E = np.exp(p["log(E0)"]) if "log(E0)" in p.keys() else p["E0"]
        out = p[f"A{0}"] * np.exp( -t*E )
    
        for n in range(1,self.number_states):
            #    ΔE_n = E_n - E_{n-1}
            # =>  E_n = E_{n-1} + ΔE_n
            E += np.exp(p[f"log(dE{n})"]) if f"log(dE{n})" in p.keys() else p[f"dE{n}"]
    
            out += p[f"A{n}"] * np.exp( -t*E )
    
        return out

#auxiliary class used to fit the ratios
class ratio_func_form:

    def __init__(self,r1:bool=True,r2:bool=True,r3:bool=False):
        self.r1:bool=r1
        self.r2:bool=r2
        self.r3:bool=r3
        
    def __call__(self, t:tuple[int,int], parms:dict):

        #we grep the input
        T = np.asarray(t[:,0],dtype=float)
        tau = np.asarray(t[:,1],dtype=float)
        MatEle = parms["M"]
        R1 = parms["R1"]
        R2 = parms["R2"]
        R3 = parms["R3"]
        dE = parms["dE"]

        out = MatEle

        if self.r1:
            #out += np.sqrt(R1*R1) * np.exp(-T/2*dE)*np.cosh( (T/2 - tau) * dE)
            out += R1 * np.exp(-T/2*dE)*np.cosh( (T/2 - tau) * dE)
        if self.r2:
            out += R2 * np.exp(-T*dE)

        if self.r3:
            out /= (1 + R3 * np.exp(-T*dE))

        return out