######################################################
## Kfact_calculator.py                              ##
## created by Emilio Taggi - 2024/12/11             ##
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

import sympy as sym #to handle symbolic computations
from sympy import I # = imaginary unit
from sympy.tensor.array.expressions import ArraySymbol
import numpy as np #to handle just everything
from cgh4_calculator import cg_calc #hand made library to compute H(4) cg coefficients
from pathlib import Path #to check whether directories exist or not
from pylatex import Document, Math, Matrix, Section, Subsection, Command, Alignat #to produce a pdf documents with the CG coeff
from pylatex.utils import NoEscape #also to produce a pdf document
from pylatex.package import Package
import itertools as it



######################## Global Variables ################################

## Gamma Structures ##

#we instantiate the Dirac gamma matrices using the following representation
gamma1 = sym.Matrix([
                    [0,0,0,I],
                    [0,0,I,0],
                    [0,-I,0,0],
                    [-I,0,0,0]])
gamma2 = sym.Matrix([
                    [0,0,0,-1],
                    [0,0,1,0],
                    [0,1,0,0],
                    [-1,0,0,0]])
gamma3 = sym.Matrix([
                    [0,0,I,0],
                    [0,0,0,-I],
                    [-I,0,0,0],
                    [0,I,0,0]])
gamma4 = sym.Matrix([
                    [0,0,1,0],
                    [0,0,0,1],
                    [1,0,0,0],
                    [0,1,0,0]])

#we instantiate also their symbolic counterpart
gamma1_s = sym.Symbol("gamma_1")
gamma2_s = sym.Symbol("gamma_2")
gamma3_s = sym.Symbol("gamma_3")
gamma4_s = sym.Symbol("gamma_4")

#we can now instantiate the gamma_mu vector
gamma_mu = [gamma1,gamma2,gamma3,gamma4]#,gamma_5]
gamma_mu_s = [gamma1_s,gamma2_s,gamma3_s,gamma4_s]#,sym_gamma_5]

#also the identity in 4 dimension will be useful
Id_4 = sym.Matrix([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

#we construct the polarization matrix in the following way
Gamma_pol = 0.5*(Id_4 + gamma4) @ (Id_4 - I * gamma1 @ gamma2)
#and its symbolic counterpart
Gamma_pol_s = sym.Symbol("Gamma_pol")


## Some more symbols

#ground state mass
mN = sym.Symbol("m_N")

#energy
E = sym.Symbol("E(p)")

#4 momentum p_mu
p_mu = [
    sym.Symbol("p_1"),
    sym.Symbol("p_2"),
    sym.Symbol("p_3"),
    I*E # = sym.Symbol("p_4")
]

#pslash = contraction gamma_mu p_mu
pslash = np.einsum('ijk,i->jk',gamma_mu,p_mu)
#and its symbolic counterpart
pslash_s = sym.Symbol("\cancel{p}")

#denominator appearing in every kinematic factor
den = 2 * E * sym.trace( Gamma_pol * (-I*pslash + mN*Id_4) ).simplify(rational=True)






######################## Main Class ################################



#class used to compute the kinematic factors related to the basis of operator specified by the given tensor product
class K_calc:

    '''
    Create one class instance to obtain the kinematic factors related to the
    tensor product specified in input
    '''

    #global variables shared by all the class instances

    #dirs and files names

    #folders with the pdf files
    kfact_pdf_folder = 'Kfact_pdf'


    #specifics of the H(4) irreps

    #number of irreps
    n_rep = 20
    #dimensionality of the irreps
    rep_dim_list = [1,1,1,1,2,2,3,3,3,3,4,4,4,4,6,6,6,6,8,8]
    #labels of the representations
    rep_label_list = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4),(6,1),(6,2),(6,3),(6,4),(8,1),(8,2)]
    #latex labels of the irrep
    rep_latex_names = [f"\\tau^{i[0]}_{i[1]}" for i in rep_label_list]
    #dictionaries with irreps specifics
    irrep_index = dict(zip(rep_label_list, range(n_rep))) #each irrep has an index
    irrep_dim = dict(zip(rep_label_list, rep_dim_list)) #a dimensionality
    irrep_texname  = dict(zip(rep_label_list, rep_latex_names)) #and a character in latex

    def __init__(self, *kwarg, verbose=True):

        #we store in a class variable the irreps chosen for the decomposition
        self.chosen_irreps = kwarg


        ## cg coefficient computation

        #info print
        if verbose==True:
            print("\nFetching the CG coefficients for the specified tensor product\n")

        #we compute the cg coefficient using the ad hoc class and we store them in a tmp dictionary
        cg_dict_tmp = cg_calc(*self.chosen_irreps,verbose=verbose).cg_dict


        ## we now begin the computation of the kinematic factors

        #dict used to store the remapped cg coefficients
        self.cg_dict = {}

        #dict used to store the kinematic factors
        self.kin_dict = {}

        #we loop over the irrep in the decompositio of the tensor products (these are the keys in the cg dict)
        for k,v in cg_dict_tmp.items():

            #for each irrep(=key) we instantiate an empty list with len equal to the multiplicity of that irrep in the decomposition
            self.cg_dict[k] = [[] for _ in range(len(v))]
            self.kin_dict[k] = [[] for _ in range(len(v))]

            #print(k)
            #print(len(v))

            #then we loop over all the blocks we have in that irrep (i.e. we do as much iteration as the multiplicity of the current irrep)
            for imul,block in enumerate(v):

                #print(np.shape(block))
            
                #we take care of rounding errors (TO DO: take care of rounding errors directly inside cgh4 class)
                block = round_CG(block)

                #each column of the cg matrix is an operator in the new basis, so we cycle throgh the columns
                for icol in range(np.shape(block)[1]):
                
                    #we remap the column with the cg coefficient of the current operator into a matrix (best suited to do the matrix multiplications needed to compute the K factor)
                    cg_mat = cg_remapping(block[:,icol],len(self.chosen_irreps))

                    #we store the remapped cg coefficients in a dict for later use
                    self.cg_dict[k][imul].append(cg_mat)

                    #now that we have the cg matrix we compute the operator as the sum of combinations of the type: CGcoeff x gammaMat x momentum
                    operator = np.einsum('ij,ikl,j -> kl',cg_mat,gamma_mu,p_mu)

                    #we compute the kinematic factor with the function implementing its formula and we store it
                    self.kin_dict[k][imul].append( Kfactor(operator) )


    #function to print the Kinematic factors in a latex pdf
    def latex_print(self,digits=5,verbose=True, title=None, author="E.T.", clean_tex=True):

        #we set the title param to the default value
        if title is None:
            title = ' x '.join([str(ir) for ir in self.chosen_irreps]) + ": Operators and Kinematic Factors"

        #we create the folders where to store the cg pdf  files
        Path(self.kfact_pdf_folder).mkdir(parents=True, exist_ok=True)

        #that is the name we give to the document (i.e. the chosen irreps as strings)
        doc_name = "Kfact_"+''.join([str(ir) for ir in self.chosen_irreps])

        #we instantiate the .tex file
        doc = Document(default_filepath=f'{doc_name}.tex', documentclass='article')#, font_size='' )
        

        #doc.packages.append(Package("pdflscape"))

        #create document preamble
        doc.preamble.append(Command("title", title))
        doc.preamble.append(Command("author", author))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        doc.append(NoEscape(r"\newpage"))

        doc.append(Command('fontsize', arguments = ['8', '12']))
        #doc.append(Command('selectfont'))

        #we now loop over the elements of the dict with cg coefficients
        for k,v in self.kin_dict.items():

            #for every key (irrep) in the dict we make a section
            section = Section(str(self.rep_label_list[k]),numbering=False)
        
            #then we loop over the multiplicities
            for imul,base in enumerate(v):

                #we print a matrix to the pdf
                #matrix = Matrix( np.matrix(np.round(np.asarray(cgmat).astype(np.float64),digits)) , mtype="b")
                #math = Math(data=[f"M_{i+1}=", matrix])


                #we print the operator basis
                #O = sym.MatrixSymbol('A', 4, 4)
                n=len(self.chosen_irreps)
                O = ArraySymbol("O", (5,)*n)

                 

                agn = Alignat(numbering=False, escape=False)

                #for each time the given irrep appear in the decomposition we have a basis
                for iop,op in enumerate(base):
                    #math = Math(data=[f"M_{imul}_{iop}=", r"{}".format(str(op).replace('**','^'))])



                    cgmat = self.cg_dict[k][imul][iop]


                    new_op = 0

                    for indices in it.product(range(4),repeat=n):

                        shifted_indices = [sum(x) for x in zip(indices,(1,)*n)]

                        new_op += cgmat[indices] * O[shifted_indices]

                    new_op_print = str(new_op.simplify(rational=True)).replace('*','').replace('[','_{').replace(']','}')
                
                    agn.append(r"\!"*20 + r" O_{}^{} &= {} \\".format(iop+1,'{'+f"{self.rep_label_list[k]},{imul}"+'}',new_op_print))



                    op_print = str(op).replace('**','^').replace('*','').replace('I','i')

                    #if len(op_print>50):

                    if '/' in op_print:
                        op_print = "\\frac{" + op_print.split('/')[0] + "}{ " + op_print.split('/')[1]  + "}"
                        #print(op_print)

                    agn.append(r"\!"*20 + r" K_{}^{} &= {} \\\\\\".format(iop+1,'{'+f"{self.rep_label_list[k]},{imul}"+'}',op_print))
                   




                    #we append the equation to the section
                    #section.append(math)
                section.append(agn)

            #then we append the section to the document
            doc.append(section)
            doc.append(NoEscape(r"\newpage"))


        #then we generate the pdf
        doc.generate_pdf(self.kfact_pdf_folder + '/'  + doc_name, clean_tex=clean_tex)

        #info print
        if verbose==True:
            print(f"\nPdf files containing the Kinematic factors located in {self.kfact_pdf_folder}/{doc_name}\n")





######################## Auxiliary Functions ################################

#function used to remap the cg coefficients from a 4**n column to a n rank matrix of dimension 4 (with n number of tensors in the product)
def cg_remapping(raw_cg,n: int):

    #first we instantiate the new matrix empy
    cg_remapped = np.zeros(shape=(4,)*n)

    #then we create a list using the logic of the remapping
    mapping = np.asarray( [ tuple(j) for j in [str( int(np.base_repr(i,4)) +  int('1' * n) ) for i in range(4**n)] ] , dtype=int) -1

    #we map the old cg mat onto the new one
    for i in range(4**n):
        cg_remapped[*mapping[i]] = raw_cg[i]

    return cg_remapped


#function used to round the raw CG matrix obtained from the calculator
def round_CG(cgmat,digits=2):
    return np.round(np.asarray(cgmat).astype(np.float64),digits)


#function used to compute the kinematic factor for a given operator
def Kfactor(operator):

    #at the numerator of the kin factor there is the following term
    num =  sym.trace(  Gamma_pol @ (-I*pslash + mN*Id_4) @ operator @ (-I*pslash + mN*Id_4)  ).simplify(rational=True)

    #we obtain the result as numerator divided by denominator
    return num/den

