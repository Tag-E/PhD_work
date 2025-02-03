######################################################
## kinematic_data.py                                ##
## created by Emilio Taggi - 2025/01/31             ##
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
import numpy as np #to handle just everything





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

#from moments_operator import mN,E,p1,p2,p3

#ground state mass
mN = sym.Symbol("m_N")

#energy
E = sym.Symbol("E(p)")

#4 momentum p_mu
p1=sym.Symbol("p_1")
p2=sym.Symbol("p_2")
p3=sym.Symbol("p_3")
p_mu = [
    p1,
    p2,
    p3,
    I*E # = sym.Symbol("p_4")
]

#pslash = contraction gamma_mu p_mu
pslash = np.einsum('ijk,i->jk',gamma_mu,p_mu)
#and its symbolic counterpart
pslash_s = sym.Symbol("\cancel{p}")

#denominator appearing in every kinematic factor
den = 2 * E * sym.trace( Gamma_pol * (-I*pslash + mN*Id_4) ).simplify(rational=True)