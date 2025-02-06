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

#simple container class used to store all the information related to a given operator
class Operator:

    #initialization function
    def __init__(self, cgmat:np.ndarray,
                 id:int, X:str, irrep:tuple,
                 block:int, index_block:int,
                 C:str, symm:str, tr:str) -> None:
        """
        Input:
            - cgmat: matrix of cg coeff
            - id: the number associated to the operator
            - X: etiher 'V', 'A' or 'T'
            - irrep: the irrep the operator belongs to
            - block: the multiplicity of the selected irrep
            - index_block: the index of the operator inside its block
            - C: the C parity of the operator
            - symm: the index symmetry of the operator
            - tr: the trace condition of the operator
        """

        self.cgmat = cgmat[:]
        self.id = id
        self.K = Kfactor_from_diracO( diracO_from_cgmat(cgmat, X) ) #the kinematic factor (the symbol) associated with the operator
        self.X = X
        self.n = cgmat.ndim #the number of indices of the operator
        self.irrep = irrep
        self.block = block
        self.index_block = index_block
        self.C = C
        self.symm = symm
        self.tr = tr
        self.O = symO_from_Cgmat(cgmat, self.n) #the symbolical expression of the operator
        self.nder = self.n-1 #we also store the number of derivatives, which is number of indices -1 for X=V and X=A..
        if X=='T':
            self.nder -=1 #..and n derivatives = n indices -2 for X=T


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
            print("\nAchtung: operator addition is allowed only between operator with the same number of indices and with the same X structure\n")
            return None
        
        #if the two operators also have the same irrep then the new operator also does
        new_irrep = None
        new_block = None
        if self.irrep == other_operator.irrep:
            new_irrep = self.irrep
            #the same goes for the block
            if self.block == other_operator.block:
                new_block = self.block

        #we also do a brief symmetry analysis
        new_C = "mixed"
        if self.C == other_operator.C:
            new_C = self.C
        new_symm = "Mixed Symmetry"
        if self.symm == other_operator.symm:
            new_symm = self.symm


        
        
        #if the input is ok we instantiate the new operator
        new_op = Operator(cgmat=self.cgmat + other_operator.cgmat,
                          id = None,
                          K = self.K + other_operator.K,
                          X = self.X,
                          n = self.n,
                          irrep = new_irrep,
                          block = new_block,
                          index_block = None,
                          C = new_C,
                          symm = new_symm,
                          tr = trace_symm(self.cgmat + other_operator.cgmat),
                          O = self.O + other_operator.O)
        
        #we return the sum of the two operators
        return new_op
    


    #we overload the subtraction of two other operators
    def __sub__(self, other_operator: Self) -> Self:
        """
        Overload of the subtraction between two lattice operators (operator properties that cannot be subtracted are setted to None)
        """

        #input check: the subtraction is allowed only if the two operators have the same amount of indices and the same X structure
        if self.X != other_operator.X or self.n != other_operator.n:
            print("\nAchtung: operator addition is allowed only between operator with the same number of indices and with the same X structure\n")
            return None
        
        #if the two operators also have the same irrep then the new operator also does
        new_irrep = None
        new_block = None
        if self.irrep == other_operator.irrep:
            new_irrep = self.irrep
            #the same goes for the block
            if self.block == other_operator.block:
                new_block = self.block

        #we also do a brief symmetry analysis
        new_C = "mixed"
        if self.C == other_operator.C:
            new_C = self.C
        new_symm = "Mixed Symmetry"
        if self.symm == other_operator.symm:
            new_symm = self.symm


        
        
        #if the input is ok we instantiate the new operator
        new_op = Operator(cgmat=self.cgmat - other_operator.cgmat,
                          id = None,
                          K = self.K - other_operator.K,
                          X = self.X,
                          n = self.n,
                          irrep = new_irrep,
                          block = new_block,
                          index_block = None,
                          C = new_C,
                          symm = new_symm,
                          tr = trace_symm(self.cgmat - other_operator.cgmat),
                          O = self.O - other_operator.O)
        
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
                          K = self.K * coefficient,
                          X = self.X,
                          n = self.n,
                          irrep = self.irrep,
                          block = self.block,
                          index_block = self.index_block,
                          C = self.C,
                          symm = self.symm,
                          tr = self.tr,
                          O = self.O * coefficient)
        
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
            print("\nAchtung: cannot divide by 0\n")
            return None
        
        #if the input is ok we istantiate the new operator just by changing its cgmat, K and O
        new_op = Operator(cgmat=self.cgmat / coefficient,
                          id = self.id,
                          K = self.K / coefficient,
                          X = self.X,
                          n = self.n,
                          irrep = self.irrep,
                          block = self.block,
                          index_block = self.index_block,
                          C = self.C,
                          symm = self.symm,
                          tr = self.tr,
                          O = self.O / coefficient)
        
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



########### Auxiliary Functions #################

#function to obtain the functional form of an operator from its cgmat
def symO_from_Cgmat(cgmat:np.ndarray, n:int) -> sym.core.add.Add:
    """
    Input:
        - cgmat: the matrix of cg coefficients
        - n: the number of indices of the operator
    
    Output:
        - O: the symbolic expression of the operator
    """
    
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

    Output:
        - a string describing the trace condition of the cgmat
    """

    #we compute the trace
    tr = np.trace(cgmat)

    #we convert the value of the trace to a string
    tr_condition = "!= 0"
    if tr.all()== 0:
        tr_condition= "= 0"
    
    #we return the trace condition
    return tr_condition