######################################################
## operator.py                                      ##
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


######################## Library Imports ################################

##General Libraries
import numpy as np #to handle matrices
import sympy as sym #for symbolic computations
from typing import Self #to use annotations in operator overloading
from sympy.tensor.array.expressions import ArraySymbol #to construct the symbolic expresison of the operator
import itertools as it #to have fancy loops
from pathlib import Path #to check whether directories exist or not (before saving files)
import sys #to fetch command line argument
from tqdm import tqdm #for a nice view of for loops with loading bars


##Persoal Libraries
from cgh4_calculator import cg_calc #hand made library to compute H(4) cg coefficients





#################### Global Varibales ###################################


## Imports from shared files
from kinematic_data import mN, E                             #mass of the ground state, energy
from kinematic_data import p1, p2, p3                        #spatial components of the 4 momentum
from kinematic_data import gamma_mu, gamma5                  #gamma matrices in Dirac space
from kinematic_data import p_mu                              #symbolic expression of the momentum
from kinematic_data import I                                 #complex unit in sympy
from kinematic_data import p_mu, pslash                      #symbols for p_mu and the contraction p_mu gamma_mu
from kinematic_data import den                               #symbolic expression of the denominator of the kinematic factor
from kinematic_data import Gamma_pol                         #symbolic expression of the polarization matrix in Dirac space
from kinematic_data import Id_4                              #4x4 identity matrix in Dirac space



######################## Main Class #####################################

#simple container class used to store all the information related to a given operator #TO DO: add possibility to pass O and K as optional params (to avoid recomputation)
class Operator:

    #initialization function
    def __init__(self, cgmat:np.ndarray,
                 id:int, X:str, irrep:tuple[int,int],
                 block:int, index_block:int) -> None:
        """
        Input:
            - cgmat: matrix of cg coeff
            - id: the number associated to the operator
            - X: etiher 'V', 'A' or 'T'
            - irrep: the irrep the operator belongs to, in the format (int,int) = (dimensionality, index for the given dimensionality)
            - block: the multiplicity of the selected irrep
            - index_block: the index of the operator inside its block
        """


        ## We store all the specifics of the operators we want to construct

        #we store the input parameters
        self.cgmat = cgmat[:]
        self.id = id
        self.X = X
        self.irrep = irrep 
        self.block = block 
        self.index_block = index_block 


        ## We also store specifics fo the operators derived from the input parameters

        #number of indices  and number of derivatives of the operator
        self.n = cgmat.ndim #the number of indices of the operator
        self.nder = self.n-1 #we also store the number of derivatives, which is number of indices -1 for X=V and X=A..
        if X=='T':
            self.nder -=1 #..and n derivatives = n indices -2 for X=T

        #symbolic expression of the operator and of its kinematic factor
        self.O = symO_from_Cgmat(cgmat) #the symbolical expression of the operator
        self.K = Kfactor_from_diracO( diracO_from_cgmat(cgmat, X) ) #the kinematic factor (the symbol) associated with the operator
        
        #symmetry properties of the operator
        self.C = C_parity(cgmat,X) #the C parity of the operator
        self.symm = index_symm(cgmat) #the symmetry under index exchange of the operator
        self.tr = trace_symm(cgmat) #the trace condition of the operator
        
        


    #overwrite of built in print methods
    def __repr__(self):
        return str(self.O.simplify(rational=True))
    def __str__(self):
        return str(self.O.simplify(rational=True)).replace('*','').replace('[','_{').replace(']','}')


    #we overload the addition with other operators
    def __add__(self, other_operator: Self) -> Self:
        """
        Overload of the addition of two lattice operators (operator properties that cannot be added are setted to None)
        """

        #input check: the addition is allowed only if the two operators have the same amount of indices and the same X structure
        if self.X != other_operator.X or self.n != other_operator.n:
            raise ValueError("\nAchtung: operator addition is allowed only between operator with the same number of indices and with the same X structure\n")
        
        #if the two operators also have the same irrep then the new operator also does
        new_irrep = None
        new_block = None
        if self.irrep == other_operator.irrep:
            new_irrep = self.irrep
            #the same goes for the block
            if self.block == other_operator.block:
                new_block = self.block
        
        
        #if the input is ok we instantiate the new operator
        new_op = Operator(cgmat=self.cgmat + other_operator.cgmat,
                          id = None,
                          X = self.X,
                          irrep = new_irrep,
                          block = new_block,
                          index_block = None,
                          )
        
        #we return the sum of the two operators
        return new_op
    


    #we overload the subtraction of two other operators
    def __sub__(self, other_operator: Self) -> Self:
        """
        Overload of the subtraction between two lattice operators (operator properties that cannot be subtracted are setted to None)
        """

        #input check: the subtraction is allowed only if the two operators have the same amount of indices and the same X structure
        if self.X != other_operator.X or self.n != other_operator.n:
            raise ValueError("\nAchtung: operator addition is allowed only between operator with the same number of indices and with the same X structure\n")
        
        #if the two operators also have the same irrep then the new operator also does
        new_irrep = None
        new_block = None
        if self.irrep == other_operator.irrep:
            new_irrep = self.irrep
            #the same goes for the block
            if self.block == other_operator.block:
                new_block = self.block




        
        
        #if the input is ok we instantiate the new operator
        new_op = Operator(cgmat=self.cgmat - other_operator.cgmat,
                          id = None,
                          X = self.X,
                          irrep = new_irrep,
                          block = new_block,
                          index_block = None,
                          )
        
        #we return the subtraction between the two operators
        return new_op
    



    #we overload the multiplication of an operator with a number
    def __mul__(self, coefficient: float) -> Self:
        """
        Overload of the multiplication between an operator and a numeric coefficient
        """

        #we instantiate the new operator (only cgmat, K and O change)
        new_op = Operator(cgmat=self.cgmat * coefficient,
                          id = self.id,
                          X = self.X,
                          irrep = self.irrep,
                          block = self.block,
                          index_block = self.index_block,
                          )
        
        #we return the product of operator times coefficient
        return new_op
    
    #the overload is the same for the right multiplication between number and operator
    def __rmul__(self, coefficient: float) -> Self:
        """
        Overload of the multiplication between an operator and a numeric coefficient
        """

        #it is the same as the left multiplication
        return self.__mul__(coefficient)
    

    #we overload the divison by a numeric coefficient
    def __truediv__(self, coefficient: float) -> Self:
        """
        Overload of the divison of the operator by a numeric coefficient
        """
        
        #input check
        if coefficient == 0:
            raise ValueError("\nAchtung: cannot divide by 0\n")
        
        #if the input is ok we istantiate the new operator just by changing its cgmat, K and O
        new_op = Operator(cgmat=self.cgmat / coefficient,
                          id = self.id,
                          X = self.X,
                          irrep = self.irrep,
                          block = self.block,
                          index_block = self.index_block,
                          )
        
        #we return the division of the operator by the coefficient
        return new_op
    
    
    #function used to evaluate the kinematic factor of the operator
    def evaluate_K(self, m_value:float, E_value:float, p1_value:float ,p2_value:float, p3_value:float) -> float:
        """
        Function returning the value of the kinematic factor after the input values have been substituted
        
        Input:
            - m: the value of the mass
            - E: the value of the energy
            - p1: the value of the momentum along x
            - p2: the value of the momentum along y
            - p3: the value of the momentum along z
            
        Output:
            - K: the numeric value of the kinematic factor, obtained plugging the input variables into the symbolic expression"""
        
        #we just substitute and send back the numeric value
        return complex(self.K.evalf(subs={mN:m_value, E:E_value, p1:p1_value, p2:p2_value, p3:p3_value}))
    
    #function used to save the operator in a file
    def save(self, folder:str) -> None:
        """
        Function storing the operator in a file created in the given folder
        
        Input:
            - folder: str, the path to the folder where the operator is going to be saved as a .npy file (its cgmat gets saved, and the operator specifics are in the filename)

        Output:
            - None (the operator just gets saved to file)
        """

        #we create the input folder if it does not exist
        Path(folder).mkdir(parents=True, exist_ok=True)

        #we irst construct the filename
        filename = f"{folder}/operator_{self.id}_{self.n}_{self.X}_{self.irrep[0]}_{self.irrep[1]}_{self.block}_{self.index_block}.npy"

        #we save the operator
        np.save(filename,self.cgmat)

        #we return
        return None



########### Auxiliary Functions #################

#function to obtain the functional form of an operator from its cgmat
def symO_from_Cgmat(cgmat:np.ndarray) -> sym.core.add.Add:
    """
    Input:
        - cgmat: the matrix of cg coefficients
    
    Output:
        - O: the symbolic expression of the operator
    """

    #the number of indices is given by the number of dimensions of the cgmat
    n = cgmat.ndim
    
    #we first instantiate the symbols we're going to use (we use 5 indices, from 0 to 4, so that we can discard the 0 and have the count start from 1)
    O = ArraySymbol("O", (5,)*n)

    #we instantiate the symbolic expression for the operator to 0
    operator_symbol = 0

    #we loop over the indicies to construct the operator
    for indices in it.product(range(4),repeat=n):

        #we shift the indices so that we can print 1234 instead of 0123
        shifted_indices = [sum(x) for x in zip(indices,(1,)*n)]

        #we construct symbolically the operator to print
        operator_symbol += cgmat[indices] * O[shifted_indices]

    #we return the symbolic expression of the operator
    return operator_symbol


#function used to obtain the operator representation in Dirac space from its cgmat (still symbolical)
def diracO_from_cgmat(cgmat: np.ndarray, X: str) -> sym.core.add.Add:
    """
    Function used to construct the symbolic form of the operator specified by the input
    
    Input:
        - cgmat: matrices with the Clebsch-Gordan coefficient, in the remapped form
        - X: str, either 'V', 'A' or 'T', i.e. vector, axial or tensorial, depending on the kind of structure of the operators we're dealing with
    
    Output:
        - the symbolic form of the operator we're dealing with (with a matrix structure)
    """

    #input check
    if X not in ['V','A','T']:
        raise ValueError("The input X must be either 'V', 'A' or 'T'")

    #we first get the number of irreps in the tensor product (that is the number of indices)
    n = cgmat.ndim

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


#function used to construct the symbolic expression of the kinematic factor from the matrix in dirac space reprsenting the operator under study
def Kfactor_from_diracO(operator:sym.core.add.Add) -> sym.core.mul.Mul:
    """
    Function used to construct the symbolic form of the kinematic factor given the symbolic (matrix) form of an operator
    
    Input:
        - operator: a symbolic expression (in a 4x4 Dirac matrix form) representing the operator under study
        
    Ouput:
        - K: the symbolic expression for the operator under study
    """

    #at the numerator of the kin factor there is the following term
    num =  sym.trace(  Gamma_pol @ (-I*pslash + mN*Id_4) @ operator @ (-I*pslash + mN*Id_4)  ).simplify(rational=True)

    #we obtain the result as numerator divided by denominator (we explicit the dispersion relation to obtain a nicer output)
    return (num/den).simplify(rational=True).subs({E**2:p1**2 + p2**2 + p3**2 + mN**2}).simplify(rational=True).subs({p1**2 + p2**2 + p3**2 + mN**2:E**2})



#function used to check the trace condition of a cgmat
def trace_symm(cgmat: np.ndarray) -> str:
    """
    Input:
        - cgmat: a np.array with the cg coefficients (in the remapped version)
        - X: str, either 'V', 'A' or 'T', i.e. vector, axial or tensorial, depending on the kind of structure of the operators we're dealing with

    Output:
        - a string describing the trace condition of the cgmat
    """

    #we compute the trace
    tr = np.trace(cgmat)

    #we convert the value of the trace to a string
    tr_condition = "!= 0"
    if tr.all() == 0:
        tr_condition= "= 0"
    
    #we return the trace condition
    return tr_condition

#function used to asses the C parity of an operator from its cgmat
def C_parity(cgmat: np.ndarray, X:str) -> int|str:
    """
    Input:
        - cgmat: a np.array with the cg coefficients (in the remapped version)
    
    Output:
        - C_parity: +1, -1 or "mixed", depending on the C parity of the operator
    """

    #input check
    if X not in ['V','A','T']:
        raise ValueError("The input X must be either 'V', 'A' or 'T'")
    
    ## to asses the C parity we have to construct the cgmat matrix corresponding to the charge conjugated operator

    #let'instantiate the cgmat matrix of the operator under Cunder C
    cgmat_C = np.empty(shape=np.shape(cgmat))

    #the number of indices is given by
    n = cgmat.ndim

    #let's cycle over the indices to obtain the cgmat under charge conjugation
    for indices in it.product(range(4),repeat=n):

        #and we consider construct its conjugated counterpart
        if X=='V' or X=='A':
            cgmat_C[indices] = cgmat[(indices[0],)+indices[-1:0:-1]]
        else:
            cgmat_C[indices] = cgmat[(indices[0],indices[1],)+indices[-1:1:-1]] 

    
    #then we check if the two matrices are equal
    if (cgmat==cgmat_C).all():
        C_parity=1
    elif (cgmat==-cgmat_C).all():
        C_parity=-1
    else:
        C_parity="mixed"

    #to completely asses the C parity we now just have to take into account the number of indices (if we are sure it is not mixed)
    if C_parity!="mixed":

        if X=="V":
            C_parity *= (-1)**n
        elif X=="A" or X=="T":
            C_parity *= (-1)**(n+1)  

    #we return the C parity
    return C_parity


#function used to evaluate the symmeetry under symmetry exchange of a given operator
def index_symm(cgmat: np.ndarray) -> str:
    """
    Input:
        - cgmat: a np.array with the cg coefficients (in the remapped version) representing the operator
        
    Output:
        - a string describing the symmetry of the operator under index exchange
    """

    #the number of indices of the operator is given by the dimensionality of the cgmat
    n = cgmat.ndim

    #to check the symmetry condition we have to take track of the symmetry under all permutation, so we store the symm of the previous perm in a variable
    symm_old = None

    # we loop over all the possible permutations of the indices of the operators
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
        if (symm_old is not None) and (symm_new!=symm_old):
            return "Mixed Symmetry"
        
        #we update the values of the previous symmetry so that we can check during the next iteration (next permutations)
        symm_old = symm_new

    #if the symm was always the same we return it
    return symm_new


#function used to find the parity of a permutation (credit: https://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-same-parity)
def parity(permutation: tuple) -> int:
    """
    Function used to find the parity of a permutation (credit: https://stackoverflow.com/questions/1503072/how-to-check-if-permutations-have-same-parity)
    
    Input:
        - permutation: a tuple coming out of itertools.permutation()
    
    Output:
        - parity: int, either 1 or -1, respectively for an even or for an odd permutation
    """
    #code copied from the referenced source:
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


#function used to remap the cg coefficients from a 4**n column to a n rank matrix of dimension 4 (with n number of tensors in the product)
def cg_remapping(raw_cg: np.ndarray, n: int) -> np.ndarray:
    """
    Function used to reshape a matrix of with the cg coefficients (already rounded) into a form that can be better handled
    
    Input:
        - raw_cg: column with cg coefficients, i.e. a matrix with shape (4**n,) where n is the number of indices of the operator under study
        - n: int, the number of indices of the operator under study
    
    Output:
        - cg_remapped: the matrix with cg coefficients, now with shape ((4,)**n), that is with n axis each with dimension 4
    """

    #first we instantiate the new matrix empy
    cg_remapped = np.zeros(shape=(4,)*n)

    #then we create a list using the logic of the remapping
    mapping = np.asarray( [ tuple(j) for j in [str( int(np.base_repr(i,4)) +  int('1' * n) ) for i in range(4**n)] ] , dtype=int) -1

    #we map the old cg mat onto the new one
    for i in range(4**n):
        cg_remapped[*mapping[i]] = raw_cg[i]

    #we return the remapped matrix
    return cg_remapped


#function used to construct the database of operators
def make_operator_database(operator_folder:str, max_n:int, verbose:bool=False) -> None:
    """
    Function used to construct the database of operators up to a given number of indices
    
    Input:
        - operator_folder: str, the path to the folder where the operators are going to be saved
        - max_n: int, the maximum number of indices the operators can have in the vectorial and axial case (for the tensorial case the maximum number of indices is max_n+1)
        - verbose: bool, if True ouptuts are printed to the screen
        
    Output:
        - None (the operators are saved to file)
    """
    
    #we create the folder if it does not exist
    Path(operator_folder).mkdir(parents=True, exist_ok=True)

    #we instantiate the list of structures we are going to consider
    X_list = ['V','A','T']

    #the list of indices we are going to consider is
    n_list = list(range(2,max_n+1))

    #correspondance int - irrep we're going to need later
    rep_label_list = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4),(6,1),(6,2),(6,3),(6,4),(8,1),(8,2)]

    #we use a counter to keep track of the number of operators we have constructed
    iop = 1

    #we loop over the number of indices
    for n in n_list:

        #info print
        if verbose:
            print(f"\nConstructing the operators V{n}, A{n} and T{n+1}...\n")

        #we loop over the X structures
        for X in tqdm(X_list):

            #we construct the irreps we have to use to build the operator
            chosen_irreps = []

            #we use the right irrep according to the structure (vector,axial or tensor)
            if X=='V':
                chosen_irreps.append((4,1))
            elif X=='A':
                chosen_irreps.append((4,4))
            elif X=='T':
                chosen_irreps.append((4,1))
                chosen_irreps.append((4,1))

            #all the other indices transform according to the fundamental
            while( (len(chosen_irreps)!=n and X!='T') or (len(chosen_irreps)!=n+1 and X=='T') ): 
                chosen_irreps.append((4,1))

            #we then use the cg calculator class to get the dictionary with all the cg coeff we need
            cg_dict = cg_calc(*chosen_irreps,verbose=False, force_computation=False).cg_dict

            #we loop over the irrep in the decomposition of the tensor products (these are the keys in the cg dict)
            for k,v in cg_dict.items():

                #then we loop over all the blocks we have in that irrep (i.e. we do as much iteration as the multiplicity of the current irrep)
                for imul,block in enumerate(v):

                    #print(np.shape(block))
                
                    #we round the matrix of cg coefficients (we select the number of decimal places to keep)
                    block = np.round(block,decimals=2)

                    #each column of the cg matrix is an operator in the new basis, so we cycle throgh the columns
                    for icol in range(np.shape(block)[1]):
                    
                        #we remap the column with the cg coefficient of the current operator into a matrix (best suited to do the matrix multiplications needed to compute the K factor)
                        cg_mat = cg_remapping(block[:,icol],len(chosen_irreps))

                        #we construct the operator and save it
                        Operator(cgmat=cg_mat,
                                 id = iop,
                                 X = X,
                                 irrep = rep_label_list[k],
                                 block = imul+1,
                                 index_block = icol+1
                                 ).save(operator_folder)

                        #we update the counter of the constructed operator
                        iop += 1
                        
    #info print
    if verbose:
        print(f"\nAll operators constructed and saved in the foder {operator_folder}\n")


#function used to load an operator from the file where it is stored
def Operator_from_file(filename:str) -> Operator:
    """
    Function used to load an operator from the file where it is stored
    
    Input:
        - filename: str, the path to the file where the operator is stored
        
    Output:
        - operator: the operator we loaded from file
    """

    #we load the cgmat from file
    cgmat = np.load(filename)

    #we extract the operator specifics from the filename
    filename = filename.split("/")[-1].replace(".npy","")
    id, n, X, irrep0, irrep1, block, index_block = filename.split("_")[1:]

    #we construct the operator
    operator = Operator(cgmat=cgmat,
                        id = int(id),
                        X = X,
                        irrep = (int(irrep0),int(irrep1)),
                        block = int(block),
                        index_block = int(index_block)
                        )

    #we return the operator
    return operator

###################### Execution of the Program as Main ############################

#we add the possibility to call the program as main, and in doing so create the database of operators
if __name__ == "__main__":
    
    #we read from the command line the max number of indices the operatos can have
    max_n=2 #std value
    if len(sys.argv) > 1:
        try:
            max_n = int(sys.argv[1])
        except ValueError:
            print(f"\nSpecified maximum number of indices (for V and A case)was max_n = {sys.argv[1]}, as it cannot be casted to int we proceed with max_n={max_n}\n")

    #the folder where we will store the operators is
    operator_folder = "operator_database"

    #we construct the database of operators
    make_operator_database(operator_folder=operator_folder, max_n=max_n, verbose=True)