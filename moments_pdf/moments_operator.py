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


######################## Main Class #####################################

#simple container class used to store all the information related to a given operator
class Operator:

    #initialization function
    def __init__(self, cgmat:np.ndarray,
                 id:int, K:sym.core.mul.Mul, X:str, n:int, irrep:tuple,
                 block:int, index_block:int,
                 C:str, symm:str, tr:str, O:sym.core.add.Add) -> None:
        """
        Input:
            - cgmat: matrix of cg coeff
            - id: the number associated to the operator
            - K: the kinematic factor (the symbol) associated with the operator
            - X: etiher 'V', 'A' or 'T'
            - n: the number of indices of the operator
            - irrep: the irrep the operator belongs to
            - block: the multiplicity of the selected irrep
            - index_block: the index of the operator inside its block
            - C: the C parity of the operator
            - symm: the index symmetry of the operator
            - tr: the trace condition of the operator
            - O: the symbolical expression of the operator
        """

        self.cgmat = cgmat[:]
        self.id = id
        self.K = K
        self.X = X
        self.n = n
        self.irrep = irrep
        self.block = block
        self.index_block = index_block
        self.C = C
        self.symm = symm
        self.tr = tr
        self.O = O #TO DO: implement the determination of O from cgmat (look code inside Kfact_calculator)
        self.nder = n-1 #we also store the number of derivatives, which is number of indices -1 for X=V and X=A..
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
    




########### Auxiliary Functions #################

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