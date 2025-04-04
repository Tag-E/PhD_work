######################################################
## moments_result.py                                ##
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
from dataclasses import dataclass #to instantiate the main class as a dataclass
import gvar as gv #to handle gaussian variables (library by Lepage: https://gvar.readthedocs.io/en/latest/)


## custom made libraries
from moments_operator import Operator #to handle lattice operators



######################## Main Class #####################################


#Container dataclass used to store all the information related to a given result
@dataclass
class moments_result:
    """
    Class used to store in one place all the information related to one particular value of the moment extracted from the data analysis
    """


    ## We store in one place all the information related to one result


    # Numerical Values of the result

    #the unrenormalized value of the moment
    value              : gv._gvarcore.GVar

    #the renormalized value of the moment
    renormalized_value : gv._gvarcore.GVar



    # Labels identifying the result

    #the operator used to extract the moment
    operator : Operator

    #the momentum P at the source of the correlator used to extract the moment
    P : tuple[gv._gvarcore.GVar,gv._gvarcore.GVar,gv._gvarcore.GVar]

    #the method used to extract the moment (either 1 = summed ratios or 2 = two state fit)
    method : int

    #the value of the source sink separation the moment refers to (None is method==2)
    T : int | None


    
    # Additional information useful to have at hand

    #the renormalization factor
    Z : gv._gvarcore.GVar

    #the Dirac structure (either 'V','A' or 'T' if the structure is vectorial, axial or tensorial)
    X : str

    #the lattice spacing the moment is related to
    a : gv._gvarcore.GVar

    #the time extent of the lattice used to compute the moment
    latticeT : int