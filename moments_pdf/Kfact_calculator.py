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

#we will need also gamma 5
gamma5 = sym.Matrix([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,-1,0],
                    [0,0,0,-1]])
gamma5_s = sym.Symbol("gamma_5")

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

    def __init__(self, X: str, n: int, verbose=True):

        '''
        X = 'V', 'A', 'T'
        n = 1,2,3,...
        '''

        #TO DO: add input check

        #we store the type of structure: vector, axial or tensor
        self.structure = X

        #we also store the rank (number of indices) of the operator
        self.n = n

        #then we store in a class variable the irreps chosen for the decomposition
        self.chosen_irreps = []

        #we use the right irrep according to the structure (vector,axial or tensor)
        if X=='V':
            self.chosen_irreps.append((4,1))
        elif X=='A':
            self.chosen_irreps.append((4,4))
        elif X=='T':
            self.chosen_irreps.append((4,1))
            self.chosen_irreps.append((4,1))

        #all the other indices transform according to the fundamental
        while(len(self.chosen_irreps)!=n):
            self.chosen_irreps.append((4,1))


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
                    operator = construct_operator(cg_mat,self.chosen_irreps,self.structure)

                    #we compute the kinematic factor with the function implementing its formula and we store it
                    self.kin_dict[k][imul].append( Kfactor(operator) )


    #function to print the Kinematic factors in a latex pdf
    def latex_print(self,digits=5,verbose=True, title=None, author="E.T.", clean_tex=True):

        #we set the title param to the default value
        if title is None:
            title = f"X={self.structure}, n={self.n} : Operators and Kinematic Factors"

        #we create the folders where to store the cg pdf  files
        Path(self.kfact_pdf_folder).mkdir(parents=True, exist_ok=True)

        #that is the name we give to the document (i.e. V,A or T and the number of indices)
        doc_name = f"Kfact_{self.structure}_{self.n}"

        #we instantiate the .tex file
        doc = Document(default_filepath=f'{doc_name}.tex', documentclass='article')#, font_size='' )

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

                #for every block the given irrep we make a subsection

                #we compute the characteristics of the block
                
                #C symmetry
                C = Csymm(self.cg_dict[k][imul],self.structure,self.n)
                #trace condition
                trace = trace_symm(self.cg_dict[k][imul])
                #index symm
                symm = index_symm(self.cg_dict[k][imul],self.n)

                #we make a title for each block
                subsection = Subsection(f"(Block {imul+1})  Trace {trace}, {symm}, C = {C}",numbering=False)



                #to print to latex the operators we use an array with with n indices, ranging from 0 to 4 (and we just use 1,2,3,4 and discard 0)
                O = ArraySymbol("O", (5,)*self.n)

                 
                #we instantiate the math environmant where we print the operators
                agn = Alignat(numbering=False, escape=False)

                #for each time the given irrep appear in the decomposition we have a basis
                for iop,op in enumerate(base):

                    #we take the cg matrix for the given operator
                    cgmat = self.cg_dict[k][imul][iop]


                    #we now construct the latex symbol for the new operator using sympy

                    #we instantiate it to 0
                    new_op = 0

                    #we loop over the indicies to construct the operator
                    for indices in it.product(range(4),repeat=self.n):

                        #we shift the indices so that we can print 1234 instead of 0123
                        shifted_indices = [sum(x) for x in zip(indices,(1,)*self.n)]

                        #we construct symbolically the operator to print
                        new_op += cgmat[indices] * O[shifted_indices]

                    #we do some string manipulation to obtain a nicer output
                    new_op_print = str(new_op.simplify(rational=True)).replace('*','').replace('[','_{').replace(']','}')
                
                    #we append the output to the mathematical latex environment
                    agn.append(r"\!"*20 + r" O_{}^{} &= {} \\".format(iop+1,'{'+f"{self.structure}{self.rep_label_list[k]},{imul+1}"+'}',new_op_print))


                    #we make a nicer output also for the kinematic factor
                    op_print = str(op).replace('**','^').replace('*','').replace('I','i')

                    #if len(op_print>50): #TO DO: handle long string output

                    #we try to use \frac{}{} instead of just a slash
                    if '/' in op_print:
                        op_print = "\\frac{" + op_print.split('/')[0] + "}{ " + op_print.split('/')[1]  + "}"

                    #we append the kinematic factor to the math environment
                    agn.append(r"\!"*20 + r" K_{}^{} &= {} \\\\\\".format(iop+1,'{'+f"{self.structure}{self.rep_label_list[k]},{imul+1}"+'}',op_print))
                   

                #we append the math expression to the subsection
                subsection.append(agn)

                #and we append the subsection to the section
                section.append(subsection) 

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
    return (num/den).simplify(rational=True)


#function used to construct the operator from the matrices of cg coefficients
def construct_operator(cgmat,irreps,X: str):

    #we first get the number of irreps in the tensor product
    n = len(irreps)

    #we first instantiate the operator as the zero matrix in Dirac space
    op = np.zeros((4,4))

    #we then loop over all the possible indices combinations (we have the implicit assumption that all the irrep we are considering are 4 dim)
    for indices in it.product(range(4),repeat=n):


        #we first compute the product of the gamma matrices according to the structure of the operator
        if X=='V':
            gamma_prod = gamma_mu[indices[0]]
            start_ind = 1
        elif X=='A':
            gamma_prod = gamma_mu[indices[0]] @ gamma5
            start_ind = 1
        elif X=='T':
            gamma_prod = gamma_mu[indices[0]] @ gamma_mu[indices[1]] - gamma_mu[indices[1]] @ gamma_mu[indices[0]]
            start_ind = 2

        #then we compute the product of all the momenta
        p_prod = 1
        for ind in indices[start_ind:]:
            p_prod *= p_mu[ind]

        #then once we have these product we have the structure in dirac space, so we just have to multiply by cg and p that are numbers in dirac space
        op += cgmat[indices] * p_prod *  gamma_prod

    #we send back the operator just constructed (it is a matrix in Dirac space)
    return op


#function used to check the Charge conjugation symmetry
def Csymm(block,X, n):

    #to check the symmetry we have to cycle over all the cg coefficients and see if they respect the correct conditions

    #we cycle first over all the operators

    #to check that the C parity are all equal we take track of the last C parity computed
    C_old = ""

    #then we cycle over the operators
    for iop,cgmat in enumerate(block):

        #we initialize the charge conjugation value to be mixed
        C_new="mixed"

        #let'instantiate the cgmat matrix under C
        cgmat_C = np.empty(shape=np.shape(cgmat))

        #let's cycle over the indices to obtain the cgmat under charge conjugation
        for indices in it.product(range(4),repeat=n):

            #and we consider construct its conjugated counterpart
            if X=='V' or X=='A':
                cgmat_C[indices] = cgmat[(indices[0],)+indices[-1:0:-1]]
            else:
                cgmat_C[indices] = cgmat[(indices[0],indices[1],)+indices[-1:1:-1]] 

        
        #then we check if the two matrices are equal
        if (cgmat==cgmat_C).all():
            C_new=1
        elif (cgmat==-cgmat_C).all():
            C_new=-1
        else:
            C_new="mixed"

        #we check whether the C parity are all equal
        if iop>0:
            if C_new!=C_old:
                C_new="mixed"
                break
        C_old=C_new

           

    #now we just have to take into account the number of indices
    if C_new!="mixed":

        if X=="V":
            C_new*=(-1)**n
        elif X=="A" or X=="T":
            C_new*=(-1)**(n+1)      #TO DO: check with some literature

    #then we return the C parity
    return C_new


#function used to check the trace condition of a given block
def trace_symm(block):

    #we have to check that for all the operators the trace is the same, so we store the last computed trace
    tr_old=0

    #then we cycle over the operators
    for iop,cgmat in enumerate(block):

        #for each we compute the trace
        tr_new = np.trace(cgmat)

        #we check whether the traces are all equal
        if iop>0:
            if (tr_new!=tr_old).all():
                return "mixed"
        tr_old=tr_new

    #we look at the value of the trace
    if tr_new.all() == 0:
        tr_condition= "= 0"
    elif tr_new.all() != 0:
        tr_condition = "!= 0"
    
    #we return the trace condition
    return tr_condition


#function used to check the symmetry condition of a given block
def index_symm(block,n):

    #we have to check that all the operators have the same index symmetry, so we initialize it to a certain value
    symm_old=0

    #then we cycle over the operators
    for iop,cgmat in enumerate(block):

        
        #then we loop over all the possible permutations of the indices of the operators
        for ip,p in enumerate(it.permutations(range(n))):

            #we skip the trivial permutation as it yields no information
            if ip==0:
                continue

            
            #then we construct the permuted matrix

            #we instantiate it
            cgmat_p = np.empty(shape=np.shape(cgmat))

            #we fill it according to the permutations
            for indices in it.product(range(4),repeat=n):
                cgmat_p[indices] = cgmat[ *[indices[p[i]] for i in range(n)] ]

            #we check for a particular symmetry
            if (cgmat==cgmat_p).all() or  (cgmat==-cgmat_p).all(): #this means either symmetric or antisymmetric...
                if parity(p)==-1:                                  #... but we can only tell if the permutation is odd
                    if (cgmat==cgmat_p).all():
                        symm_new="Symmetric"
                    elif (cgmat==-cgmat_p).all():
                        symm_new="Antisymmetric"
            else:                                                  #in every other case there is mixed symmetry
                return "Mixed Symmetry"
            
            #if we are past the first iteration and the symmetry changed we conclude that it is mixed
            if symm_old!=0 and (symm_new!=symm_old):
                return "Mixed Symmetry"
            
            #we update the values of the previous symmetry so that we can check during the next iteration
            symm_old = symm_new


    #if the symm was always the same we return it
    return symm_new




#fucton used to find the parity of a permutation (credit: https://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-same-parity)
def parity(permutation):
    permutation = list(permutation)
    length = len(permutation)
    elements_seen = [False] * length
    cycles = 0
    for index, already_seen in enumerate(elements_seen):
        if already_seen:
            continue
        cycles += 1
        current = index
        while not elements_seen[current]:
            elements_seen[current] = True
            current = permutation[current]
    return (-1)**( (length-cycles) % 2 ) # even,odd are 1,-1