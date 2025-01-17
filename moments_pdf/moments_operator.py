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

import numpy as np #to handle matrices
import sympy as sym #for symbolic computations



######################## Main Class #####################################

#simple container class used to store all the information related to a given operator
class Operator:

    #initialization function
    def __init__(self, cgmat:np.ndarray,
                 id:int, K:sym.core.mul.Mul, X:str, n:int, irrep:tuple,
                 block:int, index_block:int,
                 C:str, symm:str, tr:str, O:sym.core.add.Add):
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