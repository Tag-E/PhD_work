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
import numpy as np #to handle computations


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



######################## Auxiliary Functions #####################################

#function used to sort a list of moments_results instances into a dictionary
def dict_from_results_list(moments_result_list:list[moments_result]) -> dict[list[moments_result]]:
    """
    Function that sorts a list of moments result into a dictionary with the keys being
    the Dirac structure of the moments and the lattice it refers to.
    
    Input:
        - moments_result_list: list of moments result
        
    Output:
        - moments_result_dict: dictionary with the same moments as in the input list, but arrangeed
                               according to the keys of the type (X,latticeT)
    """

    #we instantiate the output dictionary
    moments_result_dict = {}

    #we loop over all the result in the list
    for moment in moments_result_list:

        #if the key associated to the given moment is not yet in the dictionary we create it...
        if (moment.X,moment.latticeT) not in moments_result_dict:
            #... and we associate the corresponding value with a list contening the given result
            moments_result_dict[(moment.X,moment.latticeT)] = [moment]
        
        #if instead the key is already in the dictionary...
        else:
            #... we just append the moment result to the corresponding list
            moments_result_dict[(moment.X,moment.latticeT)].append(moment)

    #we return the moment
    return moments_result_dict

#function used to construct the weights associated to each moment starting from a dictionary containing them
def moments_weights_from_dict(moments_result_dict:dict[list[moments_result]]) -> dict[list[float]]:
    """
    Function computing the weights associated for each set of moments in the input dictionary.
    
    Input:
        - moments_result_dict: dictionary with key:value = (X,latticeT):list of moment results
    
    Output:
        - weights_dict: a dictionary with key:value = (X,latticeT):list with the weights associated to the moment results
    """

    #we instantiate the output dictionary with the weights
    weights_dict = {}

    #we loop over the keys and values of the input dicionary (i.e. (X,latticeT) and the list of moments, respectively)
    for key,value in moments_result_dict.items():

        #the weights are directly proportional to the std, and are normalized such that summing the weights at fixed method we get 1/2

        #we compute the normalization factor for the method 1 (ratio method)
        normalization_R = 2 * np.sum( [moment.renormalized_value.sdev for moment in value if moment.method==1] ) 

        #we compute the normalization factor for the method 2 (sum ratio method)
        normalization_S = 2 * np.sum( [moment.renormalized_value.sdev for moment in value if moment.method==2] )

        #we create the entry in the weights dictionary with the weights corresponding to the same entry in the input dictionary
        weights_dict[key] = [moment.renormalized_value.sdev / (normalization_R if moment.method==1 else normalization_S) for moment in value]

    #we return the weight dictionary
    return weights_dict

#function used to construct the final result of the moments starting from a dictionary containing them
def moment_final_result(moments_result_dict:dict[list[moments_result]]) -> dict[gv._gvarcore.GVar]:
    """
    Function that computes the final result of the renormalized moments, one for each key
    in the input dictionary, i.e. one for each couple (X,latticeT).
    
    Input:
        - moments_result_dict: dictionary with key:value = (X,latticeT):list of moment results
    
    Output:
        - final_moments_dict: a dictionary with key:value = (X,latticeT):gvar variable with the final result for the moment
    """

    #we first construct the dictionary with the weights associated to the moments in the input dictionary
    weights_dict = moments_weights_from_dict(moments_result_dict)

    #then we instantiate the output dictionary with the results as an empty dictionary
    result_dict = {}

    #we then loop over the keys ( = (X,latticeT) ) and values (= list of moments) of the input dictionary
    for key, value in moments_result_dict.items():

        #we compute the result by averaging the renormalized moments according to the computed weights
        result_dict[key] = np.average( [moment.renormalized_value for moment in value], weights=weights_dict[key] ) 

    #we return the dictionary containing the final results
    return result_dict

#function used to construct the systematic associateed to final result of the moments starting from a dictionary containing them
def systematic_final_result(moments_result_dict:dict[list[moments_result]]) -> dict[float]:
    """
    Function that computes the systematic uncertainty associated with the final result of the renormalized moments,
    one for each key in the input dictionary, i.e. one for each couple (X,latticeT).
    
    Input:
        - moments_result_dict: dictionary with key:value = (X,latticeT):list of moment results
    
    Output:
        - systematic_dict: a dictionary with key:value = (X,latticeT):value of the systematic uncertainty on the final result
    """

    #we first construct the dictionary with the weights associated to the moments in the input dictionary
    weights_dict = moments_weights_from_dict(moments_result_dict)

    #we also construct the dictionary with the final result of the moments
    result_dict = moment_final_result(moments_result_dict)

    #then we instantiate the output dictionary where we will put the systematics
    systematic_dict = {}

    #we then loop over the keys ( = (X,latticeT) ) and values (= list of moments) of the input dictionary
    for key, value in moments_result_dict.items():

        #we compute the systematic variance as a weighted variance, then we take the sqrt to get the systematic uncertainty
        systematic_dict[key] = np.sqrt( np.average( [ (moment.renormalized_value.mean-result_dict[key].mean)**2 for moment in value], weights=weights_dict[key] ) )

    #we return the dictionary with the systematic uncertainties
    return systematic_dict