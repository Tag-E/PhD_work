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
from typing import Any, Callable #to use annotations for functions
from scipy.optimize import curve_fit #to extract the mass using a fit of the two point correlator
import gvar as gv #to handle gaussian variables (library by Lepage: https://gvar.readthedocs.io/en/latest/)
import itertools as it #for fancy iterations (product:to loop over indices; cycle: to cycle over markers; groupby: used to asses the equality of elements in a list)
from typing import Any #to use annotations for fig and ax
from matplotlib.figure import Figure #to use annotations for fig and ax




## custom made libraries
from building_blocks_reader import bulding_block #to read the 3p and 2p correlators
from moments_operator import Operator, Operator_from_file, make_operator_database #to handle lattice operators
import correlatoranalyser as CA #to perform proper fits (Marcel's library: https://github.com/Marcel-Rodekamp/CorrelatorAnalyser)



######################## Global Variables ###############################




######################## Main Class ####################################

#each dataset corresponds to an instance of the class, its methods provide useful analysis tools
class moments_toolkit(bulding_block):
    """
    Create an instance of the class to setup the analysis framework associated to the given dataset
    """

    ## Global variable shared by all class instances

    #list of available structures = vector,axial,tensor
    X_list = ['V', 'A', 'T']




    ## Methods of the class

    #Initialization function #TO DO; add properly the skip3p option and skipop option
    def __init__(self, p3_folder:str, p2_folder:str,
                 max_n:int=3, plot_folder:str="plots", skipop:bool=False, verbose:bool=False,
                 operator_folder="operator_database", **kwargs) -> None:
        
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


    #function used to get the 3 point correlation functions related to the selected operators
    def get_p3corr(self, isospin:str='U-D') -> np.ndarray:
        """
        Function used to get the 3 point correlators (the building block, one for each configuration) of the selected operators

        Input:
            - isospin: either 'U', 'D', 'U-D' or 'U+D'

        Output:
            - p3_corr: array with the 3 point correlators of the selected operators, shape = (nop, nconf, nT, maxT+1), dtype=float
        """

        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

        #we compute the dimensionality of the tau axis (so for T<maxT there is a padding with zeros from taus bigger than their max value)
        maxT = np.max(self.T_list)

        #we initialize the output array with zeros
        p3_corr = np.zeros(shape=(self.Nop, self.nconf, self.nT, maxT+1), dtype=float) #+1 because tau goes from 0 to T included

        #we now fill the array using the method of the building block class to extract the combination corresponding to the selected operator

        #loop over the selected operators
        for iop,op in enumerate(self.selected_op):


            #loop over the available times T
            for iT,T in enumerate(self.T_list):

                #we compute the relevant 3 point correlator (that isthe building block related to the operator under study)

                #we have to take the real or imaginary part depending on the kinematic factor (according to the chosen convention, this 3p corr has to be real or imaginary depending if i*Kinematic_factor is)
                if op.p3corr_is_real:
                    p3_corr[iop,:,iT,:T+1] = self.operatorBB(T,isospin, op).real          
                else:  
                    p3_corr[iop,:,iT,:T+1] = self.operatorBB(T,isospin, op).imag          #the last axis of R is padded with zeros

        #we return the 3 point correlators
        return p3_corr
    

    #function used to get the 2 point correlators (with the correct cast)
    def get_p2corr(self) -> np.ndarray:
        """
        Function used to get the 2 point correlators (one for each configuration)
        
        Input:
            - None: every information is stored inside the class
        
        Output:
            - p2_corr: two point correlator, shape = (nconf, latticeT), dtype=float (i.e. they are casted to real numbers)
        """

        #we just return what we have already stored, just casting it to real
        return self.p2_corr.real


    #function used to compute the ratio R(T,tau)
    def get_R(self, isospin:str='U-D') -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Input:
            - isospin: either 'U', 'D', 'U-D' or 'U+D'

        Output:
            - Rmean(iop,T,tau): the mean resulting from the jackknife analysis performed using as observable the ratio R, shape = (nop, nT, maxT+1)
            - Rstd(iop,T,tau): the std reasulting from the jackknife analysis performed using as observable the ratio R, shape = (nop, nT, maxT+1)
            - Rcovmat(iop,T,tau1, tau2): the whole covariance matrix reasulting from the jackknife analysis performed using as observable the ratio R, shape = (nop, nT, maxT+1)
        """
        
        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

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
        for iT,T in enumerate(self.T_list):

            #we perform the jackknife analysis (the observable being the ratio we want to compute)
            Rmean[:,iT,:], Rstd[:,iT,:], Rcovmat[:,iT,:,:] = jackknife([p3_corr[:,:,iT,:], p2_corr], lambda x,y: ratio_formula(x,y, T=T, gauge_axis=1), jack_axis_list=[1,0], time_axis=-1)

        #we return the ratios just computed and the results of the jackknife analysis
        return Rmean, Rstd, Rcovmat
    

    #function used to plot the ratio R for all the selected operators
    def plot_R(self, isospin:str='U-D', show:bool=True, save:bool=False, figname:str='plotR',
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
        
        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

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
            for iT, T in enumerate(self.T_list):

                times = np.arange(-T/2+1,T/2)

                #we grep the interesting part of the array and we ignore the padding along the last axis
                ratio = Rmean[iop,iT,:T+1]
                ratio_err = Rstd[iop,iT,:T+1]

                #we discard the endpoints
                r = ratio[1:-1]
                r_err = ratio_err[1:-1]

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

                #ax.grid()

            #we save the plot if the user asks for it
            if save:
                plt.savefig(f"{self.plots_folder}/{figname}_operator{op.id}.png")
            
            #we show the plot if the user asks for it
            if show:
                plt.show()

        #we return fig and ax
        return fig_ax_list

    #function used to to compute the sum of ratios S
    def get_S(self, tskip: int, isospin:str='U-D') -> tuple[np.ndarray, float, float]:
        """
        Method used to obtain, using a jackknife analysis, the sum of ratios given by S(T,tskip) = sum_(t=tskip)^(T-tskip) R(T,t)

        Input:
            - tskip = tau_skip = gap in time when performing the sum of ratios
            - isospin: either 'U', 'D', 'U-D' or 'U+D'

        Output:
            - Smean: the mean resulting from the jackknife analysis performed using S as observable, shape = (nop, nT)
            - Sstd: the std resulting from the jackknife analysis performed using S as observable, shape = (nop, nT)
        """
        
        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

        #We first take the 3 point and 2 point correlators needed to compute the ratio and consequently the Summed ratios S
        p3_corr = self.get_p3corr(isospin=isospin) #shape = (nop, nconf, nT, maxT+1)
        p2_corr = self.get_p2corr() #shape = (nconf, latticeT)

        #the shape of the ratio is given by (nop, nT), i.e.
        S_shape =  (self.Nop, self.nT)

        #we instantiate the output ratio
        Smean = np.zeros(shape=S_shape, dtype=float) 
        Sstd = np.zeros(shape=S_shape, dtype=float)

        #we loop over all the T values we have
        for iT,T in enumerate(self.T_list):
            
            #we compute S using the jackknife algorithm
            Smean[:,iT], Sstd[:,iT], _ = jackknife( [p3_corr[:,:,iT,:], p2_corr], lambda x,y: sum_ratios_formula( ratio_formula(x,y, T=T, gauge_axis=1), T, tskip, time_axis=-1), jack_axis_list=[1,0], time_axis=None )

        #we return S
        return Smean, Sstd


    #function used to plot S
    def plot_S(self, tskip:int, isospin:str='U-D', show:bool=True, save:bool=True, figname:str='plotS',
               figsize:tuple[int,int]=(20,8), fontsize_title:int=24, fontsize_x:int=18, fontsize_y:int=18, markersize:int=8,
               abs=False, rescale=False) -> tuple[Figure, Any]:
        """
        Input:
            - tskip = tau_skip = gap in time when performing the sum of ratios
            - isospin: either 'U', 'D', 'U-D' or 'U+D'
            - show: bool, if True plots are shown to screen
            - save: bool, if True plots are saved to .png files

        Output:
            - fig, ax: the output of the plt.subplots() call, so that the user can modify the figure if he wants to
        """

        #first thing first we compute S with the fiven t skip 
        #S, Smean, Sstd = self.get_S(tskip=tskip) #shapes = (Nop, Nconf, NT), (Nop, NT), (Nop, NT)
        Smean, Sstd = self.get_S(tskip=tskip)  #shapes =  (Nop, NT), (Nop, NT)

        #we instantiate the figure
        fig, ax = plt.subplots(nrows=1,ncols=3,figsize=figsize,sharex=False,sharey=False)

        #we cycle on the markers
        #marker = it.cycle(('>', 'D', '<', '+', 'o', 'v', 's', '*', '.', ',')) 

        #we loop over the operators
        for iop, op in enumerate(self.selected_op):

            #depending on the X structure of the operator we decide in which of the three plots to put it
            plot_index = self.X_list.index(op.X)


            #To Do: adjust kin factor 
            a = 0.1163 #we cheat
            hc = 197.327
            mp_mev = 1000
            mass = mp_mev/hc * a
            kin = 1j * op.evaluate_K(m_value=mass,E_value=mass,p1_value=0,p2_value=0,p3_value=0) #this 1j in front comes from the fact that mat_ele = <x> * i K
            if np.iscomplex(kin):
                kin *= -1j
            kin = kin.real

            #we only plot if the kin factor is not 0
            if kin!=0:

                #then we plot it
                ax[plot_index].errorbar(self.T_list, Smean[iop]/kin,yerr=Sstd[iop]/np.abs(kin), marker = 'o', markersize = markersize, linewidth = 0.3, linestyle='dashed',label=r"${}$".format(op.latex_O))


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


    #method to extract the matrix element from the summed ratios
    def MatEle_from_S(self, tskip_list:list[int] = [1,2,3], delta_list:list[int] = [1,2,3], isospin:str='U-D') -> np.ndarray:
        """
        Function returning a value of the (unrenormalized) matrix element for each operator, extracting them from the summed ratios S
        
        Input:
            - tskip_list: list of tau skip we want to use in the analysis
            - delta_list: list of delta that we want to use in the analysis (see reference paper for their meaning)
            - isospin: either 'U', 'D', 'U-D' or 'U+D'
        
        Output:
            - mat_ele: np array with shape (Nop, nT-1), with one value of the matrix element for each operator and for each allowed time value
        """

        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

        #we first take the correlators we need to compute everything
        p2corr = self.get_p2corr() #shape = (Nconf, latticeT)
        p3corr = self.get_p3corr(isospin=isospin) #shape = (Nop, Nconf, NT, maxT+1)

        #we instantiate the output array
        mat_ele_array = np.zeros(shape=(self.Nop, self.nT), dtype=object ) #shape = (Nop, nT)

        #we fill the output array using the formula for the matrix element from S
        for iop in range(self.Nop):

            #we compute mean and std of the matrix element using the jackknife
            mat_ele, mat_ele_std, _ = jackknife([p3corr[iop],p2corr], observable = lambda x,y: MatEle_from_slope_formula(p3_corr=x, p2_corr=y, T_list=self.T_list, delta_list=delta_list, tskip_list=tskip_list), jack_axis_list=[0,0], time_axis=None)

            #we put them into a gvar variable and store it into the array
            mat_ele_array[iop] = gv.gvar(mat_ele,mat_ele_std)

        #if delta=0 is not one of the values considered then the last value of T is removed (since it has no allowed value of T+delta)
        if 0 not in delta_list:
            mat_ele_array = np.delete(mat_ele_array, self.nT-1, axis=1)
        
        #we return the matrix element array
        return mat_ele_array



    #function to get a value of the mass from the two point function
    def get_meff(self, show:bool=False, save:bool=False, chi2_treshold:float=1.0, zoom:int=0, figname:str='mass_plateau') -> tuple[float, float]:
        """
        Input:
            - show: bool, if True the effective mass vs time plot is shown to screen
            - save: bool, if True the effective mass vs time plot is saved to file
            - chi2_treshold:  treshold for the plateau determination using a chi2 analysis
            - zoom: int, number of extra points plotted on the sides of the plateau
            - figname: name of the .png file containing the plot
        
        Output:
            - meff_plat, meff_plat_std: value of the mass extracted from the plateau of the effective mass, with related std
        """

        #first we recall the two point correlators
        corr_2p = self. get_p2corr() #the cast to real is done here

        #we use the jackknife to compute the effective mass (mean and std)
        meff, meff_std, meff_covmat = jackknife(corr_2p, effective_mass, jack_axis_list=0, time_axis=-1)

        #to remove this problematic part we look for the first time the std is 0
        cut=np.where(meff_std<=0)[0][0]

        #and we cut the meff arrays there
        meff = meff[:cut]
        meff_std = meff_std[:cut]
        meff_covmat = meff_covmat[:cut,:cut]


        #we compute a mean value and a std from the plateau

        #first we identify the boundaries of the plateau region
        start_plateau, end_plateau = plateau_search(meff, meff_covmat, chi2_treshold=chi2_treshold)

        #then we get the mass from the plateau value
        meff_plat = np.mean(meff[start_plateau:end_plateau])
        #its std is given by sqrt of sum variance/ N
        meff_plat_std = np.sqrt( np.mean( meff_std[start_plateau:end_plateau]**2 ) )

        #we make a plot 

        #the plot is made if the user ask either to see it or to save it to file
        if show or save:
            
            #we instantiate the figure
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(32, 14))

            #we determine the time values to be displayed on the plot (x axis)
            m_times = np.arange(np.shape(meff)[0]) + 0.5

            #we plot the plateau and the neighbouring effective mass values
            plt.errorbar(m_times[start_plateau-zoom:end_plateau+zoom], meff[start_plateau-zoom:end_plateau+zoom], yerr=meff_std[start_plateau-zoom:end_plateau+zoom],linewidth=0.5,marker='o',markersize=4,elinewidth=1.0)
            plt.hlines(meff_plat ,start_plateau+0.5, end_plateau+0.5, color='red')
            plt.hlines(meff_plat + meff_plat_std, start_plateau+0.5, end_plateau+0.5, color='red', linestyles='dashed')
            plt.hlines(meff_plat - meff_plat_std, start_plateau+0.5, end_plateau+0.5, color='red', linestyles='dashed')

            #plot styling
            plt.title("Effective Mass Plateau")
            plt.ylabel(r"$m_{eff}(t)$")
            plt.xlabel(r"$t$")


            #we save the figure if asked
            if save:
                plt.savefig(f"{self.plots_folder}/{figname}.png")

            #we show the figure if asked
            if show:
                plt.show()


        #we return the plateau values
        return meff_plat, meff_plat_std
    

    #function used to extract a value of the fit mass from the two point correlators
    def get_mfit(self) -> tuple[float,float]:
        """
        Input:
            - None

        Output:
            - mfit, mfit_std: mean value and std of the mass extracted from the fit, using the jackknife analysis
        """

        #first we recall the two point correlators
        #corr_2p = self.bb_list[0].p2_corr.real
        corr_2p = self.get_p2corr()

        #then we use the jackknife to compute the fit mass and its std
        mfit, mfit_std , _= jackknife(corr_2p, fit_mass, jack_axis_list=0, time_axis=None)

        #we return the fit mass and its std
        return mfit, mfit_std


    def fit_2pcorr(self, chi2_treshold=1.0, zoom=0, fit_doubt_factor=10, show=True, save=True, verbose=False):
        """
        Input:
            - 

        Output:
            - 
        """

        #info print
        if verbose:
            print("\nPreparing the fit for the two point correlator...\n")

        #we first take the 2point corr
        p2corr = self.get_p2corr()

        #first we determine the gauge avg of the 2p corr using the jackknife and we store it for a later use
        p2corr_jack, p2corr_jack_std, p2corr_jack_cov = jackknife(p2corr, lambda x: np.mean( x, axis=0), jack_axis_list=0, time_axis=-1)

        #then we se the jackknife to compute a value of the effective mass (mean, std and cov)
        meff_raw, meff_std_raw, meff_covmat_raw = jackknife(p2corr, effective_mass, jack_axis_list=0, time_axis=-1) #raw becaus there still are values of the mass that are 0

        #these values of the effective mass are "raw" because they still contain <=0 values (and also padding from the effective mass function)

        #we look at the point where the mass starts to be negative
        cut=np.where(meff_std_raw<=0)[0][0]

        #and we cut the meff arrays there
        meff = meff_raw[:cut]
        meff_std = meff_std_raw[:cut]
        meff_covmat = meff_covmat_raw[:cut,:cut]

        #we can now identify the boundaries of the plateau region
        start_plateau, end_plateau = plateau_search(meff, meff_covmat, only_sig=True, chi2_treshold=1.0) #we search the plateau not using the whole correlation matrix

        #we define the gaussian variables corresponding to the valeus of the effective mass in the plateau region
        gv_meff = gv.gvar(meff[start_plateau:end_plateau], meff_std[start_plateau:end_plateau])

        #we then obtain the plateau value by taking the weighted average of these values (weighted by the inverse of the variance)
        gv_meff_plateau = np.average(gv_meff,weights=[1/e.sdev**2 for e in gv_meff])


        #we make a plot if the user asks either to see it or to save it to file
        if show or save:

            #we instantiate the figure
            fig, axlist = plt.subplots(nrows=2,ncols=1,figsize=(32, 14))

            ax1 = axlist[0]


            #we determine the time values to be displayed on the plot (x axis)
            m_times = np.arange(np.shape(meff_raw)[0]) #+ 0.5

            #we adjust the beginning of the plot according to the zoom out parameter given by the user
            start_plot = start_plateau-zoom if start_plateau-zoom > 0 else 0

            #we plot the plateau and the neighbouring effective mass values
            _ = ax1.errorbar(m_times[start_plot:end_plateau+zoom], meff_raw[start_plot:end_plateau+zoom], yerr=meff_std_raw[start_plot:end_plateau+zoom], linewidth=0.7, marker='o', markersize=6, elinewidth=1.0, label="Effective Mass")
            _ = ax1.errorbar( np.arange(np.shape(meff)[0])[start_plot:end_plateau+zoom], meff[start_plot:end_plateau+zoom], yerr=meff_std[start_plot:end_plateau+zoom], linewidth=0.7, marker='o', markersize=6, elinewidth=1.0, label="Effective Mass (used for plateau search)")
            _ = ax1.hlines(gv_meff_plateau.mean, start_plateau, end_plateau-1, color='red', label="Plateau Region")
            #plt.hlines(meff_plat + meff_plat_std, start_plateau+0.5, end_plateau+0.5, color='red', linestyles='dashed')
            #plt.hlines(meff_plat - meff_plat_std, start_plateau+0.5, end_plateau+0.5, color='red', linestyles='dashed')
            _ = ax1.fill_between(m_times[start_plateau:end_plateau], gv_meff_plateau.mean - gv_meff_plateau.sdev, gv_meff_plateau.mean + gv_meff_plateau.sdev, alpha=0.2, color="red")

            #plot styling
            _ = ax1.set_title("Effective Mass Plateau")
            _ = ax1.set_ylabel(r"$m_{eff}(t)$")
            _ = ax1.set_xlabel(r"$t$")
            _ = ax1.grid()
            _ = ax1.legend()





            ax2 = axlist[1]


            #we determine the time values to be displayed on the plot (x axis)
            times = np.arange(np.shape(p2corr_jack)[0]) #+ 0.5

            #we decide how much we want to zoom out of the plateau
            #zoom = 7

            #we plot the plateau and the neighbouring effective mass values
            _ = ax2.errorbar(times[start_plot:end_plateau+zoom], p2corr_jack[start_plot:end_plateau+zoom], yerr=p2corr_jack_std[start_plot:end_plateau+zoom], linewidth=0.7, marker='o', markersize=6, elinewidth=1.0, label="Two Point Correlator")
            #ax1.errorbar( np.arange(np.shape(meff)[0]), meff, yerr=meff_std, linewidth=0.7, marker='o', markersize=6, elinewidth=1.0, label="Effective Mass (used for plateau search)")
            #ax1.hlines(gv_meff_mean.mean, start_plateau, end_plateau-1, color='red', label="Plateau Region")
            #plt.hlines(meff_plat + meff_plat_std, start_plateau+0.5, end_plateau+0.5, color='red', linestyles='dashed')
            #plt.hlines(meff_plat - meff_plat_std, start_plateau+0.5, end_plateau+0.5, color='red', linestyles='dashed')
            _ = ax2.fill_between(times[start_plateau:end_plateau], np.min(p2corr_jack[start_plot:end_plateau+zoom]), np.max(p2corr_jack[start_plot:end_plateau+zoom]), alpha=0.2, color="red", label="Plateau Region")

            #plot styling
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

        #we construct the resamples for the 2 point correlator
        p2corr_resamples = jackknife_resamples(p2corr, lambda x: np.mean( x, axis=0), jack_axis_list=0)
        nres = p2corr_resamples.shape[0]

        #we define the parameters of interest
        nstates_list = [1,2] #only 1 or two state fits
        t_start_list = np.arange(start_plateau, int(start_plateau/2), -1)
        t_end = end_plateau

        #we instanatiate the states of the fits we want to do
        fit_state = CA.FitState()


        #if the user asks for a plot we instantiate the figure we're going to use
        if show or save:
            fig, axs = plt.subplots(nrows=len(t_start_list), ncols=len(nstates_list), figsize=(32, 14), sharex=True, sharey=True)
            fig.text(0.5, 0.04, r"$t$", ha='center', fontsize=16)
            fig.text(0.04, 0.5, r"$C_{2pt}(t)$", va='center', rotation='vertical', fontsize=16)
            fig.suptitle("Two Point Correlator Fits", fontsize=16)
            fig.tight_layout()
            plt.subplots_adjust(left=0.085,bottom=0.1)

        #we now loop over the free parameters of the fit (the number of states and the starting time of the fit)
        for i_state, nstates in enumerate(nstates_list):

            #we handle the prior determination of the parameters

            #we instantiate a prior dict
            prior = gv.BufferDict()

            #we get a first estimate for the mass and the amplitude from  the scipy fit + jackknife analysis
            mfit, mfit_std , _= jackknife(p2corr, lambda x: fit_mass(x, guess_mass=gv_meff_plateau.mean, par="mass"), jack_axis_list=0, time_axis=None) #TO DO: have a look at this fit to be sure that it is ok to use it as prior
            Afit, Afit_std , _= jackknife(p2corr, lambda x: fit_mass(x, guess_mass=gv_meff_plateau.mean, par="amp"), jack_axis_list=0, time_axis=None)

            #we store the values in the prior dict and we rescale their uncertainy by a factor accounting for the fact that we don't trust the simple scipy fit
            prior["A0"] = gv.gvar(Afit,Afit_std * fit_doubt_factor)
            prior["E0"] = gv.gvar(mfit, mfit_std * fit_doubt_factor)

            #we then model a prior for the exponential corresponding to the efirst excited state, if needed
            if nstates==2:

                #for the energy we don't know, so we just give a wide prior assuming the energy doubles from ground to first excited state
                prior[f"dE1"]= gv.gvar(
                                        gv.mean(mfit),
                                        gv.mean(mfit),
                                    )

                #the amplitude of the term corresponding to the first excited state we extract by all the other information we have using the functional form of the correlator
                t_probe = 1
                tmp = (np.mean(self.p2_corr.real,axis=0)[t_probe] - prior["A0"] * np.exp(-t_probe*prior["E0"]) ) * np.exp( t_probe * ( prior["dE1"] + prior["E0"]) )
                prior[f"A1"] = gv.gvar( 
                    gv.mean(tmp),gv.mean(tmp)
                )


            #we then loop over the starting times
            for i_tstart, t_start in enumerate(t_start_list):

                #we do the fit
                fit_result = CA.fit(

                    abscissa                = np.arange(t_start,t_end),
                    
                    ordinate_est            = np.mean(p2corr_resamples[:,t_start:t_end], axis = 0), #np.mean(p2corr[:,t_start:t_end], axis = 0),
                    ordinate_std            = np.sqrt((nres-1)/nres) * np.std(p2corr_resamples[:,t_start:t_end], axis = 0), #np.std (p2corr[:,t_start:t_end], axis = 0),
                    ordinate_cov            =  (nres-1)/nres * np.cov(p2corr_resamples[:,t_start:t_end], rowvar=False), #np.cov (p2corr[:,t_start:t_end], rowvar=False),
                    
                    resample_ordinate_est   = p2corr_resamples[:,t_start:t_end],
                    resample_ordinate_std   = np.sqrt((nres-1)/nres) * np.std (p2corr_resamples[:,t_start:t_end], axis = 0),
                    resample_ordinate_cov   = (nres-1)/nres * np.cov (p2corr_resamples[:,t_start:t_end], rowvar=False),

                    # fit strategy, default: only uncorrelated central value fit:
                    central_value_fit            = True,
                    central_value_fit_correlated = True,

                    resample_fit                 = True,
                    resample_fit_correlated      = True,
                    
                    resample_fit_resample_prior  = False,
                    resample_type               = "bst", #"jkn",

                    # args for lsqfit:
                    model   = SumOrderedExponentials(nstates),
                    prior   = prior,
                    p0      = None,

                    svdcut  = None,
                    maxiter = 10_000,
                )

                #we append the fit result to the fit_state
                fit_state.append(fit_result)

                #if the user wants a plot we do it
                if show or save:

                    #first we compute the 2p correlator in the region of interest
                    C = gv.gvar( np.mean(self.p2_corr.real[:,fit_result.ts:fit_result.te], axis = 0), np.std(self.p2_corr.real[:,fit_result.ts:fit_result.te], axis = 0))

                    #we plot the fit
                    plot_fit_2pcorr(fit_result=fit_result,correlator=C, axs=axs[i_tstart,i_state], nstates=nstates, Ngrad=15)



        #we save the figure if the user asks for it
        if save:
            plt.savefig(f"{self.plots_folder}/2p_corr_fit.png")

        #we sho the figure in the user asks for it
        if show:
            plt.show()

        

        #we return the fit state
        return fit_state




    def fit_ratio(self, chi2_treshold=1.0,  fit_doubt_factor=10, tskip_list=[1], show=True, save=True, verbose=False, rescale=True,
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

            for iT,T in enumerate(self.T_list):

                start_plateau, end_plateau = plateau_search(Rmean[iop,iT,:T+1 ], Rcovmat[iop,iT,:T+1 , :T+1 ], only_sig=False, chi2_treshold=1.0)

                plateau_dict[(iop,iT)] = start_plateau, end_plateau

                for tau in range(start_plateau,end_plateau):
                    #abscissa[iop].append( (T,tau) )
                    tmp_a.append( (T,tau) )

            abscissa_list.append( np.asarray(tmp_a) )

        #the number of resamples is given by
        nres = abscissa_list[0].shape[0]


        ## Next we construct the ordinate

        #we take the correlators
        p3corr = self.get_p3corr() #shape = (Nop, nconf, nT, maxT+1)
        p2corr = self.get_p2corr()

        #we resample the ratios, len = Nop,, each element with shape = (Nres, Nallowed(T,tau) )
        ratio_resamples_list = [ jackknife_resamples([p3corr,p2corr], lambda x,y: np.asarray( [e for l in [ ratio_formula(x, y, T, gauge_axis=1)[iop,iT, plateau_dict[(iop,iT)][0] : plateau_dict[(iop,iT)][1] ] for iT,T in enumerate(self.T_list) ] for e in l] ), jack_axis_list=[1,0] ) for iop in range(self.Nop) ]



        ## Then the prior construction

        #we use the fit of the 2 point function
        fit2p_parms = self.fit_2pcorr(show=False,save=False).model_average()

        #from the result of the fit we take the energy
        #dE = gv.gvar(fit2p_parms['est']['dE1'], fit2p_parms['err']['dE1'])

        #we will use as prior the value extracted from S
        matele_fromS = self.MatEle_from_S(tskip_list=tskip_list)

        #we instantiate a prior dict for each operator
        priordict_list = []

        for iop in range(self.Nop):

            #we instantiate the dict
            prior = gv.BufferDict()

            #we fill it
            prior["dE"] = gv.gvar(fit2p_parms['est']['dE1'], fit2p_parms['err']['dE1'])

            prior["M"] = np.average(matele_fromS[iop], weights= [ele.sdev**(-2) for ele in matele_fromS[iop]] )

            prior["R1"] = gv.gvar(0,10)
            prior["R2"] = gv.gvar(0,10)
            prior["R3"] = gv.gvar(0,10)
        

        ## Now we are ready to do the fit

        #we instanatiate the states of the fits we want to do (one fit state for each operator)
        fit_state_list = [ CA.FitState() for iop in range(self.Nop) ]


        #for each operator we do a series of fits

        for iop in range(self.Nop):

            #we take the right resampling
            ratio_res = ratio_resamples_list[iop]

            #we loop over the possible parameters
            for r2 in [False, True]:
                for r3 in [False, True]:


                    #we do the fit
                    fit_result = CA.fit(

                        abscissa                = abscissa_list[iop],
                        
                        ordinate_est            = np.mean(ratio_res, axis = 0),
                        ordinate_std            = np.sqrt((nres-1)/nres) * np.std (ratio_res, axis = 0),
                        ordinate_cov            = (nres-1)/nres * np.cov (ratio_res, rowvar=False),
                        
                        resample_ordinate_est   = ratio_res,
                        resample_ordinate_std   = np.sqrt((nres-1)/nres) * np.std (ratio_res, axis = 0),
                        resample_ordinate_cov   = (nres-1)/nres * np.cov (ratio_res, rowvar=False),

                        # fit strategy, default: only uncorrelated central value fit:
                        central_value_fit            = True,
                        central_value_fit_correlated = True,

                        resample_fit                 = True,
                        resample_fit_correlated      = True,
                        
                        resample_fit_resample_prior  = False,
                        resample_type               = "bst",#"jkn",

                        # args for lsqfit:
                        model   = ratio_func_form(r1=True,r2=r2,r3=r3),
                        prior   = prior,
                        p0      = None,

                        svdcut  = None,
                        maxiter = 10_000,
                    )

                    #we append the fit to the list
                    fit_state_list[iop].append(fit_result)


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
                for iT, T in enumerate(self.T_list):

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






    #function that returns the operator according to the label given in the catalogue #TO DO: add input control
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
            return None
            




######################## Auxiliary Functions ##########################


#function implementing the jackknife analysis
def jackknife(in_array_list: np.ndarray|list[np.ndarray], observable: Callable[[], Any], jack_axis_list:int|list[int|None]=0, time_axis:int|None=-1, binsize:int=1,first_conf:int=0,last_conf:int|None=None) -> list[np.ndarray]:
    """
    Function implemeneting the Jackknife mean and std estimation. The input array(s) has to match the input required by the observable function. If a list of array is given then also a list of
    jackknife axis and time axis has to be given.

    Input:
        - in_array_list: input array to be jackknifed, or a list containing such arrays
        - observable: function taking as input an array of the same shape of in_array (i.e. an observable that should be computed over it), and giving as output an array with the jackknife axis (i.e. conf axis) removed
        - jack_axis_list: the axis over which perform the jacknife analysis (from a physics p.o.v. the axis with the configurations) (or a list with such axis for every input array)
        - time_axis: axis on the output array (i.e. after observable is applyied!!) over to which look for the autocorrelation, (if None the covariance matrix is not computed)
        - binsize: binning of the jackknife procedure
        - first_conf: index of the first configuration taken into account while performing the jackknife procedure
        - last_conf: index of the last configuration taken into account while performing the jackknife procedure (if not specified then the last available configuration is used)

    Output:
        - list with [mean, std, cov] where mean and std are np array with same the same shape as the input one minus the jackknife dimension, and the cov has one extra time dimension (the new time dimension is now the last one)
    """

    #we make a check on the input to asses that the number of input_array, jackknife axes and time_axes is consistend
    if type(in_array_list) is list and (type(jack_axis_list) is not list or len(in_array_list)!=len(jack_axis_list) ):
        raise ValueError("The input array is a list, hence also the jackknife axis should be a list and have the same lenght, but that is not the case")
    
    #if the given input is just one array and not a list of arrays, then we put it in a list
    if type(in_array_list) is not list:
        in_array_list = [in_array_list]
        jack_axis_list = [jack_axis_list]

    #we set last conf to its default value
    if last_conf is None:
        last_conf = np.shape(in_array_list[0])[jack_axis_list[0]]

    #step 1: creation of the jackknife resamples (we create a jack resample for input array in the list)
    jack_resamples_list = [ np.asarray( [np.delete(in_array, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=jack_axis_list[i]) for iconf in range(first_conf,last_conf,binsize)] ) for i,in_array in enumerate(in_array_list)]#shape = (nresamp,) + shape(in_array) (with nconf -> nconf-binsize)
    #print("jack resamples")
    #for e in jack_resamples_list:
    #    print(np.shape(e))

    #the number of resamples is len(jack_resmaples[0]) or also
    #nresamp = int((last_conf-first_conf)/binsize)
    nresamp = np.shape(jack_resamples_list[0])[0] #the 0th axis now is the resample axis, (and axis has nconf-1 conf in the standard case (binsize=1 ecc.) )

    #step 2; for each resample we compute the observable of interest
    #we use the resampled input array to compute the observable we want, and we have nresamp of them
    obs_resamp = np.asarray( [observable( *[jack_resamples[i] for jack_resamples in jack_resamples_list] ) for i in range(nresamp) ] )                                                                          #shape = (nresamp,) + output_shape
    #print("obs resamples")
    #print(np.shape(obs_resamp))

    #step 3: we compute the observable also on the whole dataset
    obs_total = observable(*in_array_list)                                                                                                                                   #shape = output_shape
    #print("obs")
    #print(np.shape(obs_total))

    #step4: compute estimate, bias and std according to the jackknife method
    
    #the estimate is the mean of the resamples
    jack_mean = np.mean(obs_resamp,axis=0) #axis 0 is the resamples one                                                                                         #shape = (nresamp,) + output_shape - (nresamp,) = output_shape
    #print("jack mean")
    #print(np.shape(jack_mean))

    #the jackknife bias is given by the following formula 
    bias = (nresamp-1) * (jack_mean - obs_total)                                                                                                                     #shape = output_shape
    #print("bias")
    #print(np.shape(bias))

    #TO DO: add proper cast to real

    #the jack std is given by the following formula
    obs_std = np.sqrt( (nresamp-1)/nresamp * np.sum( (obs_resamp - jack_mean)**2, axis=0 ) ) #the axis is the resamples one                                        #shape = (nresamp,) + output_shape - (nresamp,) = output_shape
    #print("obs std")
    #print(np.shape(obs_std))

    #to obtain the final estimate we correct the jack mean by the bias
    #obs_mean = jack_mean - bias 
    obs_mean = obs_total - bias                                                                                                                                  #shape = output_shape


    #step 5: covariance matrix computation

    #we perform such a computation only if there actually see a time axis over which to look for a correlation
    if time_axis is not None:

        #to account for the fact that we have removed the jackknife dimension we change the time dimension

        #first we compute the lenght in the time dimension (by looking at the output array)
        lenT = np.shape(obs_total)[time_axis]

        #we the instantiate the covariance matrix (we add an extra time dimension so that we can compute the correlation)
        covmat = np.zeros(shape = np.shape(obs_mean) + (lenT,), dtype=float )

        #the time axis is translated to a positive value
        if time_axis<0:
            #time_axis = lenT+time_axis
            time_axis = len(np.shape(obs_total))+time_axis

        #TO DO: add cast to real values before computing covariance matrix

        #we then loop over the times and fill the covariance matrix
        for t1 in range(lenT):
            for t2 in range(lenT):

                #we do a little bit of black magic to addres the right indices combinations (credit https://stackoverflow.com/questions/68437726/numpy-array-assignment-along-axis-and-index)
                s = [slice(None)] * len(np.shape(covmat))
                axe1 = time_axis #position of the first time axis
                s[axe1] = slice(t1,t1+1)
                axe2 =  len(np.shape(covmat))-1 #because the new time axis is at the end of the array
                s[axe2] = slice(t2,t2+1)

                #we update the covariance matrix
                #covmat[tuple(s)] = np.expand_dims( (nresamp-1)/nresamp * np.sum( (  np.take(obs_resamp,t1,axis=time_axis) - np.take(obs,t1,axis=new_time_axis) ) * (  np.take(obs_resamp,t2,axis=time_axis) - np.take(obs,t2,axis=new_time_axis) ), axis=0 ),
                 #                                  [axe1,axe2])
                covmat[tuple(s)] = np.expand_dims( (nresamp-1)/nresamp * np.sum( (  np.take(obs_resamp,t1,axis=time_axis+1) - np.take(jack_mean,t1,axis=time_axis) ) * (  np.take(obs_resamp,t2,axis=time_axis+1) - np.take(jack_mean,t2,axis=time_axis) ), axis=0 ),
                                                   [axe1,axe2]) #--> obs_resamp has a +1 because it has the resample dimension at the beginning!
    #if instead there is not a time axis we just send back the std in place of the covmat
    else:
        covmat = obs_std


    #we return mean and std 
    return [obs_mean, obs_std, covmat]


#function used to obtain the jackknife resamplings for the given observable with the given input
def jackknife_resamples(in_array_list: np.ndarray|list[np.ndarray], observable: Callable[[], Any], jack_axis_list:int|list[int|None]=0, binsize:int=1,first_conf:int=0, last_conf:int|None=None) -> list[np.ndarray]:
    """
    Function returning the jackknife resamples of the given observablem that can be computed with the given inputs.

    Input:
        - in_array_list: input array to be jackknifed, or a list containing such arrays
        - observable: function taking as input an array of the same shape of in_array (i.e. an observable that should be computed over it), and giving as output an array with the jackknife axis (i.e. conf axis) removed
        - jack_axis_list: the axis over which perform the jacknife analysis (from a physics p.o.v. the axis with the configurations) (or a list with such axis for every input array)
        - binsize: binning of the jackknife procedure
        - first_conf: index of the first configuration taken into account while performing the jackknife procedure
        - last_conf: index of the last configuration taken into account while performing the jackknife procedure (if not specified then the last available configuration is used)

    Output:
        - list with [mean, std, cov] where mean and std are np array with same the same shape as the input one minus the jackknife dimension, and the cov has one extra time dimension (the new time dimension is now the last one)
    """

    #we make a check on the input to asses that the number of input_array, jackknife axes and time_axes is consistend
    if type(in_array_list) is list and (type(jack_axis_list) is not list or len(in_array_list)!=len(jack_axis_list) ):
        raise ValueError("The input array is a list, hence also the jackknife axis should be a list and have the same lenght, but that is not the case")
    
    #if the given input is just one array and not a list of arrays, then we put it in a list
    if type(in_array_list) is not list:
        in_array_list = [in_array_list]
        jack_axis_list = [jack_axis_list]

    #we set last conf to its default value
    if last_conf is None:
        last_conf = np.shape(in_array_list[0])[jack_axis_list[0]]

    #step 1: creation of the jackknife resamples (we create a jack resample for input array in the list)
    jack_resamples_list = [ np.asarray( [np.delete(in_array, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=jack_axis_list[i]) for iconf in range(first_conf,last_conf,binsize)] ) for i,in_array in enumerate(in_array_list)]#shape = (nresamp,) + shape(in_array) (with nconf -> nconf-binsize)

    #the number of resamples is len(jack_resmaples[0]) or also
    #nresamp = int((last_conf-first_conf)/binsize)
    nresamp = np.shape(jack_resamples_list[0])[0] #the 0th axis now is the resample axis, (and axis has nconf-1 conf in the standard case (binsize=1 ecc.) )

    #step 2; for each resample we compute the observable of interest
    #we use the resampled input array to compute the observable we want, and we have nresamp of them
    obs_resamp = np.asarray( [observable( *[jack_resamples[i] for jack_resamples in jack_resamples_list] ) for i in range(nresamp) ] )                                                                          #shape = (nresamp,) + output_shape

    return obs_resamp


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

    #tskip=tskip+1 #TO DO: check whether this should be done, i.e. if the range should be (1+tskip, T-tskip)

    #we implement the formula for the sum of rations in a fancy way (so that we can index the right dimension without knowing how many other dimensions there are)
    return np.sum( np.take(ratio, range(tskip, T+1-tskip), axis=time_axis) , axis=time_axis) #TO DO: check if also the pace of the sum is tskip (instead of 1 as it is used here)



#function used to extract the matrix element as the slop of the summed ratio function
def MatEle_from_slope_formula(p3_corr:np.ndarray, p2_corr:np.ndarray, T_list:list[int],  tskip_list:list[int] = [1,2,3], delta_list:list[int] = [1,2,3]) -> float:
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
            S_list[iT,i_tskip] = sum_ratios_formula( ratio_formula(p3_corr[:,iT,:], p2_corr, T=T, gauge_axis=1), T, tskip, time_axis=-1)

    ## Then the computation of all the matrix elements (one for each available compination of delta+T, and one for each tau skip)

    #we instantiate the list with the allowed matrix elements as empty
    mat_ele_array = np.zeros(shape=(len(T_list),), dtype=float) #shape = (nT,)

    #we loop over the source-sink separations T
    for iT, T in enumerate(T_list):

        #we instantiate a tmp list where we store all the matrix elements related to the given T
        tmp_mat_ele_list = []

        #we loopv over the delta we want to use in the analysis (delta is the separation we use to look at the slope)
        for delta in delta_list:
            
            #a combination T,delta is allowed only if their sum is in the available Ts
            if T + delta in T_list:

                #we check what is the index of the T we have to consider
                iT_plus_delta = T_list.index(T + delta)

                #we compute the matrix element as the slope of the summed ratio function
                tmp_mat_ele_list.append( (S_list[iT_plus_delta,:] - S_list[iT,:])/delta )

        #for the given T we extract a value of the matrix element, and we just take a simple unnweighted average over all the values of tskip and the allowed values of T+delta
        mat_ele_array[iT] = np.mean(tmp_mat_ele_list) if len(tmp_mat_ele_list)!=0 else 0 #TO DO: check if something better can be done than the plain unweighted average

    #we return the array with the matrix element just computed
    return mat_ele_array



#function used to convert the ratio of correlators to a value of the effective mass (iterative procedure - takes into account the periodicity of the lattice) TO DO: double check this function with computations (!!!!) #TO DO: remove this function if not used
def ratio2p_to_mass(ratio2p: float, t: int, T: int, case:int=2, max_it:int=1000) -> float:
    """
    Input:
        - ratio2p: the ratio of the two point correlators at two consecutive times
        - t: the time of the correlator at the numerator
        - T: time extent of the lattice
        - case: case considered (cosh, sinh or exp)
        - max_it: maximum number of iteration of the iterative algorithm for the mass determination
    
    Output:
        - m: the mass value extracted from the ratio of the correlators
    """

    sign = [1.0, -1.0, 0.0]
    

    
    # If the ratio is less than or equal to 1.0 we return a value of the mass that is 0
    if ratio2p <= 1.0:
        return 0.0
    
    #if the ratio is bigger than 1 we proceed with the iterative determination of the effective mass

    #0th value of the mass in the iterative procedure
    m0 = np.log(ratio2p)

    #we also instantiate the value of the previous iteration to be the 0th value
    old_m = m0
    m_new = m0

    #then we loop over the iteration of the iterative algorithm
    for it in range(max_it):
        
        # Specific conditions for early exit
        if ((T - 2 * (t - 1) + 2) == 0 and sign[case] == -1) or ((T - 2 * (t - 1) - 2) == 0 and sign[case] == -1):
            break
        
        if t <= (T / 2):
            d = 1.0 + sign[case] * np.exp(-old_m * (T - 2.0 * (t - 1)))
            u = 1.0 + sign[case] * np.exp(-old_m * (T - 2.0 * (t - 1) - 2.0))
        else:
            d = 1.0 + sign[case] * np.exp(old_m * (T - 2.0 * (t - 1)))
            u = 1.0 + sign[case] * np.exp(old_m * (T - 2.0 * (t - 1) + 2.0))
        
        rud = u / d
        m_new = np.log(ratio2p * rud)
        

        #if the new mass exceeds a max value or it is smaller than 0 we stop the algortihm
        if abs(m_new) > 4.0 or m_new <= 0.0:
            return 10.0
        
        #if the change in the mass due to the iteration process is small we stop the iterative procedure
        if abs(old_m - m_new) <= 3.0e-7:
            break
        
        old_m = m_new

    #in the end we return the new mass
    return m_new



#function used to extract the effective mass for the two-point correlators
def effective_mass(corr_2p: np.ndarray, conf_axis:int=0) -> np.ndarray:
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



#function used to compute the reduced chi2 of a 1D array using the covariance matrix
def redchi2_cov(in_array: np.ndarray, fit_array: np.ndarray, covmat: np.ndarray, only_sig:bool=False) -> float:
    """
    Input:
        - in_array: a 1D array, with len T
        - fit_array: a 1D array, also with len T, representing the function we want the in_arrya fitted to
        - covmat: a 2D array with shape (T,T), representing the covariance matrix of in_array
        - only_sig: bool, if True the diag of the covmat is used (the covariance) instead of the whole covmat

    Output:
        - chi2: the reduced chi2 of the fit
    """

    #then we compute the differences between the fit and input array
    deltas = in_array - fit_array

    #then we compute the number of d.o.f. (the len of the plateau)
    ndof = np.shape(in_array)[0]

    #TO DO: fix the issue with the covmat
    if only_sig==False:
        #first we invert the covariance array
        cov_inv = np.linalg.inv(covmat)
        #then we compute the reduced chi2 according to its formula and we return it
        return np.einsum( 'j,j->' , deltas, np.einsum('jk,k->j',cov_inv,deltas) ) / ndof
    else:
        sig = np.sqrt(np.diag(covmat))
        return np.sum( (deltas/sig)**2 ) / ndof

#function that given a 1D array returns the values of the indices identifying its plateau (the first and last index)
def plateau_search(in_array: np.ndarray, covmat: np.ndarray, chi2_treshold:float=1.0, only_sig:bool=True) -> tuple[int,int]:
    """
    Input:
        - in_array: the 1D array we want to search the plateau of
        - covmat: a 2D array, representing the covariance matrix of in_array
        - chi2_treshold: the treshold for the plateau determination
        - only_sig: bool, if True only the standard deviation, and not the whole cavariance matrix, is used for the plateau determination
    
    Output:
        - (start_plateau,end_plateau): indices such that in_array[start_plateau,end_plateau] is the region with the plateau
    """

    #first we compute the len of the array
    len_array = np.shape(in_array)[0]

    #we loop over all the possible plateau lenghts, starting from the biggest possible one and then diminishing it up to a plataeau of len 2
    for len_plat in range(len_array,1,-1):

        #we instantiate a tmp dictionary where we are gonna store the values of the chi2 corresponding to the different starting value of the plateau (for a fixed lenght)
        tmp_chi2_dict = {}

        #then we loop over the possible initial points of the plateau
        for start_plateau in range(0,len_array-len_plat+1,1):

                #the suggested plateau region in this case is
                plat = in_array[start_plateau:start_plateau+len_plat]

                #we also have to reshape the covariance matrix
                covmat_plat = covmat[start_plateau:start_plateau+len_plat, start_plateau:start_plateau+len_plat]

                #the value of the plateau is
                plat_value = np.average(plat, weights = np.diag(np.linalg.inv(covmat_plat)), axis=0, keepdims=True) #the weights are the inverse of the sigma squared

                #we compute the chi2 of the current plateau
                chi2 = redchi2_cov(plat, plat_value, covmat_plat,only_sig=only_sig)

                #we see if the chi2 meets the condition
                if chi2 < chi2_treshold: #TO DO: in this case put the value in a list and then at the end of the inner loop search for the better one

                    #in that case we add the values of the starting and ending point fo the plateau to a dictionary, along with the associated chi2
                    tmp_chi2_dict[(start_plateau, start_plateau+len_plat)] = chi2
        
        #after looking at all the possible starting values, for a fixed plateau len, if at least one chi2 was <1, we return the smallest
        if len(tmp_chi2_dict) > 0:
            return min(tmp_chi2_dict, key=tmp_chi2_dict.get)
                
    #if by the end of the loop the chi2 condition is never met (i.e. if len_plat is 1) we return the point corresponding to the middle of the dataset
    return int(len_array/2), int(len_array/2)+1



#exponential function used in the fit for the mass extraction
def exp_fit_func(t: np.ndarray, amp: float, mass: float) -> np.ndarray:
    """
    This function is only used to fit the two point correlators to an exponential to extract the mass using scipy curve fit
    
    Input:
        - t: numpy array with the times (the x array of the fit)
        - amp: the amplitude of the exponential
        - mass: the mass at the exponent (the parameter we actually want to extract from the fit)
        
    Output:
        - corr_2p(t) = amp * exp(-t * m): the values of the correlator in the purely exponential form (the y array of the fit)
    """
    
    #we just return the exponential
    return amp * np.exp(-t * mass)



#function used to extract the fit mass from the two-point correlators
def fit_mass(corr_2p: np.ndarray, conf_axis:int=0, guess_mass:float|None=None, guess_amp:float|None=None, par:str="mass") -> np.ndarray:
    """
    Input:
        - corr_2p: two point correlators, with shape (nconf, Tlat) (with Tlat being the time extent of the lattice)
        - conf_axis: the axis with the configurations
        - guess_mass: the first guess for the mass we want to extract from the fit

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
    times = np.arange(np.shape(corr_gavg)[conf_axis])

    #we perform the fit
    popt,pcov = curve_fit(exp_fit_func, times, corr_gavg, p0=guess)#,maxfev = 1300) #popt,pcov being mean and covariance matrix of the parameters extracted from the fit
    perr = np.sqrt(np.diag(pcov)) #perr being the std of the parameters extracted from the fit

    #we read the mass (that's the only thing we're interested about, the amplitude we discard)
    fit_mass = np.array( popt[1] )
    fit_mass_std = np.array( perr[1] )

    #same thing for amplitude
    fit_amp = np.array( popt[0] )
    fit_amp_std = np.array( perr[0] )


    #we return the fit mass or the amp depending on the user request
    if par=="mass":
        #we return the fit mass and its std
        return fit_mass#, fit_mass_std, fit_amp, fit_amp_std 
    elif par=="amp":
        return fit_amp
    

#auxiliary function used to plot the fit of the two point correlators
def plot_fit_2pcorr(fit_result,correlator,axs,nstates=2, Ngrad = 30, Nsigma = 2):

    abscissa = np.arange(fit_result.ts, fit_result.te)
    ordinate = fit_result.eval(abscissa)
    #nstates = fit_result.fcn.Nstates

    axs.errorbar(
            x=abscissa, y=gv.mean(correlator), yerr=gv.sdev(correlator), marker='.', linewidth=0, elinewidth=1, label="Correlator data", color="red"
        )


    (line,) = axs.plot(
        abscissa,
        ordinate["est"],
        #"-",
        label=rf"$N_\text{{states}}={nstates}, ({abscissa[0]},{abscissa[-1]}) "
        rf"\chi^2/_\mathrm{{dof}}~[\mathrm{{dof}}] = {fit_result.chi2/fit_result.dof:g}~[{fit_result.dof:g}], "
        rf"\text{{AIC}} = {fit_result.AIC:g} $",
        linestyle="-",
    )




    
    for igrad in range(Ngrad):
        _ = axs.fill_between(
                abscissa,
                ordinate["est"] + Nsigma*ordinate["err"] * igrad/Ngrad,
                ordinate["est"] - Nsigma*ordinate["err"] * igrad/Ngrad,
                color=line.get_color(),
                alpha= 1.0/(Nsigma*np.sqrt(Ngrad)) * (1.0-igrad/Ngrad),
            )



    axs.legend()





    str_list = ['Estimated parameters:']
    for key in fit_result.result_params()["est"].keys():
        str_list.append( f"{key} =" + str( gv.gvar(fit_result.result_params()["est"][key], fit_result.result_params()["err"][key]) ) )


    #Display text box with frelevant parameters outside the plot

    textstr = '\n'.join(str_list)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place the text box in upper left in axes coords
    _ = axs.text(0.11, 0.55, textstr, transform=axs.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #_ = axs.set_title("Two Point Correlator")
    #_ = axs.set_ylabel(r"$C_{2pt}(t)$")
    #_ = axs.set_ylabel("")
    #_ = axs.set_xlabel("")
    #_ = axs.set_xlabel(r"$t$")
    _ = axs.set_yscale("log")
    _ = axs.grid()
    _ = axs.legend()


#auxiliary function used to check if all elements in an iterable are equal (credit: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-equal)
def all_equal(iterable):
    """
    Function used to check if all elements in a list are equal (credit: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-equal)
    
    Input:
        - iterable: the list (or iterable in general) to be checked
        
    Output:
        - True if all elements are equal, False otherwise
    """
    g = it.groupby(iterable)
    return next(g, True) and not next(g, False)

#auxiliary class used to fit the two point correlators
class SumOrderedExponentials:
    def __init__(self, number_states):
        self.number_states = number_states

    def __call__(self,t,p):
        E = p["E0"]
        out = p[f"A{0}"] * np.exp( -t*E )
    
        for n in range(1,self.number_states):
            #    ΔE_n = E_n - E_{n-1}
            # =>  E_n = E_{n-1} + ΔE_n
            E += p[f"dE{n}"]
    
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
        T = t[:,0]
        tau = t[:,1]
        MatEle = parms["M"]
        R1 = parms["R1"]
        R2 = parms["R2"]
        R3 = parms["R3"]
        dE = parms["dE"]

        out = MatEle

        if self.r1:
            out += R1 * np.exp(-T/2*dE)*np.cosh( (T/2 - tau) * dE)
        if self.r2:
            out += R2 * np.exp(-T*dE)

        if self.r3:
            out /= (1 + R3 * np.exp(-T*dE))

        return out