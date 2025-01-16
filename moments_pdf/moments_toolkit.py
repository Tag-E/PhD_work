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
import h5py as h5 #to read the correlator
from tqdm import tqdm #for a nice view of for loops with loading bars
from pathlib import Path #to check whether directories exist or not
from pylatex import Document, Math, Matrix, Section, Subsection, Command, Alignat #to produce a pdf documents with the CG coeff
from pylatex.utils import NoEscape #also to produce a pdf document
import subprocess #to open pdf files
import time #to use sleep and pause the code

## custom made libraries
from building_blocks_reader import bulding_block #to read the 3p and 2p correlators
from Kfact_calculator import K_calc #to obtain the list of operators and related kinematic factors



######################## Global Variables ###############################




######################## Main Class ####################################

#each dataset corresponds to an instance of the class, its methods provide useful analysis tools
class moments_toolkit:
    """
    Create an instance of the class to setup the analysis framework associated to the given dataset
    """

    ## Global variable shared by all class instances

    #list of available structures = vector,axial,tensor
    X_list = ['V', 'A', 'T']




    ## Methods of the class

    #Initialization function
    def __init__(self, p3_folder:str, p2_folder:str,
                 tag_3p='bb',hadron='proton_3', tag_2p='hspectrum',
                 maxConf=None,max_n=3, verbose=False):
        
        """
        Input description:...
        """


        #Info Print
        if verbose:
            print("\nInitializing the moments_toolkit class instance...\n")
        

        #we store the folder variables
        self.p3_folder=p3_folder
        self.p2_folder=p2_folder

        #we initialize a list with all the available number of indices #TO DO: add input check
        self.n_list = [i for i in range(2,max_n+1)] 

        
        #First we look into the given p3 folder to see how many different subfolders we have

        #we take the path of the folders with 3 points correlators subfolder
        p = Path(p3_folder)
        #we read the avalaible list of time separations T
        self.T_list = sorted( [int(x.name[1:]) for x in p.iterdir() if x.is_dir() and x.name.startswith('T')] )
        #from that we obtain the paths of the folders containing the different building blocks
        bb_pathList = [f"{p3_folder}T{T}/" for T in self.T_list]

        
        #We now proceed to produce a  building_block class instance for each one of the times T

        #we instantiate the list with the building_block class instances
        self.bb_list = []

        #we loop over the bb paths available (loop over T)
        for i,bb_path in enumerate(bb_pathList):
            #Info Print
            if verbose:
                print(f"\n\nReading data for T = {self.T_list[i]} ...\n")
            self.bb_list.append( bulding_block(bb_path,p2_folder, maxConf=maxConf, verbose=verbose) )


        #We initialize some other class variables


        #list with operators selected for the analysis, initialized as empty
        self.selected_op = []

        ##We build the list of all the available operators

        #we initialize the list as empty
        self.operator_list = []

        #we also store the classes related to the kinematic factors, we also initialize them as empty
        self.kclass_list = []

        #we loop over X and n, store the K classes and the operators

        #we count the number of operators
        op_count=1

        #we loop over the available indices
        for n in self.n_list:
            #we loop over the V,A,T structures
            for X in self.X_list:

                #safety measure to avoid the bug present in the cg calc class: TO BE REMOVED after fixing it (TO DO)
                if n>2 and X=='T': break

                #the actual number of indices depends on X
                actual_n = n
                if X == 'T':
                    actual_n += 1
                
                #we instantiate the related Kfactor class
                self.kclass_list.append( K_calc(X,actual_n,verbose=False) )

                #we append the operators
                for op in self.kclass_list[-1].get_opList(first_op=op_count):
                    self.operator_list.append(op)

                #we update the operator count
                op_count += 4**actual_n 






    #Function used to print to pdf and show to the user all the available operators that can be chosen
    def operator_show(self, title=None, author="E.T.", doc_name=None, verbose=False, show=True, remove_pdf=False, clean_tex=True):
        """
        Function description TO DO
        """

        #we set the title param to the default value
        if title is None:
            title = "Available Operators" #TO DO: add the max number of indicices used

        #we set the document name to the default value
        if doc_name is None:
            doc_name = "operator_catalogue" #TO DO: add max number of indices
        
        #the name of the pdf file is then
        file_name = f"{doc_name}.pdf" #TO DO: add the max number of indicices used
        

        #we instantiate the .tex file
        doc = Document(default_filepath=f'{doc_name}.tex', documentclass='article',page_numbers=False)#, font_size='' )

        #create document preamble
        doc.preamble.append(Command("title", title))
        doc.preamble.append(Command("author", author))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        doc.append(NoEscape(r"\newpage"))

        doc.append(Command('fontsize', arguments = ['8', '12']))

        #we instantiate the operator count to 1
        op_count = 1

        #we loop over the available indices (TO DO: collapse this loop for another one over self.kclass_list)
        for n in self.n_list:

            #we loop over the V,A,T structures
            for X in self.X_list:

                #safety measure to avoid the bug present in the cg calc class: TO BE REMOVED after fixing it (TO DO)
                if n>2 and X=='T': break

                #the actual number of indices depends on X
                actual_n = n
                if X == 'T':
                    actual_n += 1
                
                #we instantiate the related Kfactor class
                kclass = K_calc(X,actual_n,verbose=False)

                #we add the related operator to the document
                kclass.append_operators(doc,op_count)

                #we update the operator count
                op_count += 4**actual_n #this is the number of operators written to file


        #then we generate the pdf
        
        #info print
        if verbose:
            print("\nGenerating the operators catalogue ...\n")

        #pdf generation
        #doc.generate_pdf(self.kfact_pdf_folder + '/'  + doc_name, clean_tex=clean_tex)
        doc.generate_pdf(doc_name, clean_tex=clean_tex)

        #info print
        if verbose:
            print("\nOperators catalogue generated\n")

  

        #we show to screen the generated pdf
        if show:
            #we xdg-open the pdf
            subprocess.call(["xdg-open", file_name])
            #we wait 5 seconds so that the pdf can be seen before its deletion
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
    def select_operator(self, *kwarg):
        """
        Input:
            - the int corresponding to the operators selected, as shown in the operator catalogue (accessible via the method operator_show)
        """

        #the chosen ids are
        chosen_ids = kwarg

        #for every id we append the corresponding operator to the list of chosen ones
        for id in chosen_ids:
            self.selected_op.append(self.operator_list[id-1]) #-1 because in the pdf the numbering starts from 1
                







######################## Auxiliary Functions ##########################
