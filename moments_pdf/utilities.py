######################################################
## utilities.py                                     ##
## created by Emilio Taggi - 2025/03/04             ##
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

import numpy as np #to handle computations
from typing import Any, Callable #to use annotations for functions
import itertools as it #for fancy iterations (groupby: used to asses the equality of elements in a list)





##################### Functions Definition ##############################

## Data Analysis Specific Routines ##

#function implementing the jackknife analysis
def jackknife(in_array_list: np.ndarray|list[np.ndarray], observable: Callable[[], Any], jack_axis_list:int|list[int|None]=0, time_axis:int|None=-1, binsize:int=1,first_conf:int=0,last_conf:int|None=None) -> list[np.ndarray]:
    """
    Function implemeneting the Jackknife mean and std estimation. The input array(s) has to match the input required by the observable function. If a list of array is given then also a list of
    jackknife axis and time axis has to be given.

    Input:
        - in_array_list: input array to be jackknifed, or a list containing such arrays
        - observable: function taking as input an array of the same shape of in_array (i.e. an observable that should be computed over it), and giving as output an array with the jackknife axis (i.e. conf axis) removed
        - jack_axis_list: the axis over which perform the jacknife analysis (from a physics p.o.v. the axis with the configurations) (or a list with such axis for every input array)
        - time_axis: axis on the output array (i.e. after observable is applyied!!) over to which look for the autocorrelation, (if None the covariance matrix is not computed)
        - binsize: binning of the jackknife procedure
        - first_conf: index of the first configuration taken into account while performing the jackknife procedure
        - last_conf: index of the last configuration taken into account while performing the jackknife procedure (if not specified then the last available configuration is used)

    Output:
        - list with [mean, std, cov] where mean and std are np array with same the same shape as the input one minus the jackknife dimension, and the cov has one extra time dimension (the new time dimension is now the last one)
    """

    #we make a check on the input to asses that the number of input_array, jackknife axes and time_axes is consistend
    if type(in_array_list) is list and (type(jack_axis_list) is not list or len(in_array_list)!=len(jack_axis_list) ):
        raise ValueError("The input array is a list, hence also the jackknife axis should be a list and have the same lenght, but that is not the case")
    
    #if the given input is just one array and not a list of arrays, then we put it in a list
    if type(in_array_list) is not list:
        in_array_list = [in_array_list]
        jack_axis_list = [jack_axis_list]

    #we set last conf to its default value
    if last_conf is None:
        last_conf = np.shape(in_array_list[0])[jack_axis_list[0]]

    #step 1: creation of the jackknife resamples (we create a jack resample for input array in the list)
    jack_resamples_list = [ np.asarray( [np.delete(in_array, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=jack_axis_list[i]) for iconf in range(first_conf,last_conf,binsize)] ) for i,in_array in enumerate(in_array_list)]#shape = (nresamp,) + shape(in_array) (with nconf -> nconf-binsize)
    #print("jack resamples")
    #for e in jack_resamples_list:
    #    print(np.shape(e))

    #the number of resamples is len(jack_resmaples[0]) or also
    #nresamp = int((last_conf-first_conf)/binsize)
    nresamp = np.shape(jack_resamples_list[0])[0] #the 0th axis now is the resample axis, (and axis has nconf-1 conf in the standard case (binsize=1 ecc.) )

    #step 2; for each resample we compute the observable of interest
    #we use the resampled input array to compute the observable we want, and we have nresamp of them
    obs_resamp = np.asarray( [observable( *[jack_resamples[i] for jack_resamples in jack_resamples_list] ) for i in range(nresamp) ] )                                                                          #shape = (nresamp,) + output_shape
    #print("obs resamples")
    #print(np.shape(obs_resamp))

    #step 3: we compute the observable also on the whole dataset
    obs_total = observable(*in_array_list)                                                                                                                                   #shape = output_shape
    #print("obs")
    #print(np.shape(obs_total))

    #step4: compute estimate, bias and std according to the jackknife method
    
    #the estimate is the mean of the resamples
    jack_mean = np.mean(obs_resamp,axis=0) #axis 0 is the resamples one                                                                                         #shape = (nresamp,) + output_shape - (nresamp,) = output_shape
    #print("jack mean")
    #print(np.shape(jack_mean))

    #the jackknife bias is given by the following formula 
    bias = (nresamp-1) * (jack_mean - obs_total)                                                                                                                     #shape = output_shape
    #print("bias")
    #print(np.shape(bias))

    #TO DO: add proper cast to real

    #the jack std is given by the following formula
    obs_std = np.sqrt( (nresamp-1)/nresamp * np.sum( (obs_resamp - jack_mean)**2, axis=0 ) ) #the axis is the resamples one                                        #shape = (nresamp,) + output_shape - (nresamp,) = output_shape
    #print("obs std")
    #print(np.shape(obs_std))

    #to obtain the final estimate we correct the jack mean by the bias
    #obs_mean = jack_mean - bias 
    obs_mean = obs_total - bias                                                                                                                                  #shape = output_shape


    #step 5: covariance matrix computation

    #we perform such a computation only if there actually see a time axis over which to look for a correlation
    if time_axis is not None:

        #to account for the fact that we have removed the jackknife dimension we change the time dimension

        #first we compute the lenght in the time dimension (by looking at the output array)
        lenT = np.shape(obs_total)[time_axis]

        #we the instantiate the covariance matrix (we add an extra time dimension so that we can compute the correlation)
        covmat = np.zeros(shape = np.shape(obs_mean) + (lenT,), dtype=float )

        #the time axis is translated to a positive value
        if time_axis<0:
            #time_axis = lenT+time_axis
            time_axis = len(np.shape(obs_total))+time_axis

        #TO DO: add cast to real values before computing covariance matrix

        #we then loop over the times and fill the covariance matrix
        for t1 in range(lenT):
            for t2 in range(lenT):

                #we do a little bit of black magic to addres the right indices combinations (credit https://stackoverflow.com/questions/68437726/numpy-array-assignment-along-axis-and-index)
                s = [slice(None)] * len(np.shape(covmat))
                axe1 = time_axis #position of the first time axis
                s[axe1] = slice(t1,t1+1)
                axe2 =  len(np.shape(covmat))-1 #because the new time axis is at the end of the array
                s[axe2] = slice(t2,t2+1)

                #we update the covariance matrix
                #covmat[tuple(s)] = np.expand_dims( (nresamp-1)/nresamp * np.sum( (  np.take(obs_resamp,t1,axis=time_axis) - np.take(obs,t1,axis=new_time_axis) ) * (  np.take(obs_resamp,t2,axis=time_axis) - np.take(obs,t2,axis=new_time_axis) ), axis=0 ),
                 #                                  [axe1,axe2])
                covmat[tuple(s)] = np.expand_dims( (nresamp-1)/nresamp * np.sum( (  np.take(obs_resamp,t1,axis=time_axis+1) - np.take(jack_mean,t1,axis=time_axis) ) * (  np.take(obs_resamp,t2,axis=time_axis+1) - np.take(jack_mean,t2,axis=time_axis) ), axis=0 ),
                                                   [axe1,axe2]) #--> obs_resamp has a +1 because it has the resample dimension at the beginning!
    #if instead there is not a time axis we just send back the std in place of the covmat
    else:
        covmat = obs_std


    #we return mean and std 
    return [obs_mean, obs_std, covmat]

#function used to obtain the jackknife resamplings for the given observable with the given input
def jackknife_resamples(in_array_list: np.ndarray|list[np.ndarray], observable: Callable[[], Any], jack_axis_list:int|list[int|None]=0, binsize:int=1,first_conf:int=0, last_conf:int|None=None) -> list[np.ndarray]:
    """
    Function returning the jackknife resamples of the given observablem that can be computed with the given inputs.

    Input:
        - in_array_list: input array to be jackknifed, or a list containing such arrays
        - observable: function taking as input an array of the same shape of in_array (i.e. an observable that should be computed over it), and giving as output an array with the jackknife axis (i.e. conf axis) removed
        - jack_axis_list: the axis over which perform the jacknife analysis (from a physics p.o.v. the axis with the configurations) (or a list with such axis for every input array)
        - binsize: binning of the jackknife procedure
        - first_conf: index of the first configuration taken into account while performing the jackknife procedure
        - last_conf: index of the last configuration taken into account while performing the jackknife procedure (if not specified then the last available configuration is used)

    Output:
        - list with [mean, std, cov] where mean and std are np array with same the same shape as the input one minus the jackknife dimension, and the cov has one extra time dimension (the new time dimension is now the last one)
    """

    #we make a check on the input to asses that the number of input_array, jackknife axes and time_axes is consistend
    if type(in_array_list) is list and (type(jack_axis_list) is not list or len(in_array_list)!=len(jack_axis_list) ):
        raise ValueError("The input array is a list, hence also the jackknife axis should be a list and have the same lenght, but that is not the case")
    
    #if the given input is just one array and not a list of arrays, then we put it in a list
    if type(in_array_list) is not list:
        in_array_list = [in_array_list]
        jack_axis_list = [jack_axis_list]

    #we set last conf to its default value
    if last_conf is None:
        last_conf = np.shape(in_array_list[0])[jack_axis_list[0]]

    #step 1: creation of the jackknife resamples (we create a jack resample for input array in the list)
    jack_resamples_list = [ np.asarray( [np.delete(in_array, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=jack_axis_list[i]) for iconf in range(first_conf,last_conf,binsize)] ) for i,in_array in enumerate(in_array_list)]#shape = (nresamp,) + shape(in_array) (with nconf -> nconf-binsize)

    #the number of resamples is len(jack_resmaples[0]) or also
    #nresamp = int((last_conf-first_conf)/binsize)
    nresamp = np.shape(jack_resamples_list[0])[0] #the 0th axis now is the resample axis, (and axis has nconf-1 conf in the standard case (binsize=1 ecc.) )

    #step 2; for each resample we compute the observable of interest
    #we use the resampled input array to compute the observable we want, and we have nresamp of them
    obs_resamp = np.asarray( [observable( *[jack_resamples[i] for jack_resamples in jack_resamples_list] ) for i in range(nresamp) ] )                                                                          #shape = (nresamp,) + output_shape

    return obs_resamp

#function used to compute the reduced chi2 of a 1D array using the covariance matrix
def redchi2_cov(in_array: np.ndarray, fit_array: np.ndarray, covmat: np.ndarray, only_sig:bool=False) -> float:
    """
    Input:
        - in_array: a 1D array, with len T
        - fit_array: a 1D array, also with len T, representing the function we want the in_arrya fitted to
        - covmat: a 2D array with shape (T,T), representing the covariance matrix of in_array
        - only_sig: bool, if True the diag of the covmat is used (the covariance) instead of the whole covmat

    Output:
        - chi2: the reduced chi2 of the fit
    """

    #then we compute the differences between the fit and input array
    deltas = in_array - fit_array

    #then we compute the number of d.o.f. (the len of the plateau)
    ndof = np.shape(in_array)[0]

    #TO DO: fix the issue with the covmat
    if only_sig==False:
        #first we invert the covariance array
        cov_inv = np.linalg.inv(covmat)
        #then we compute the reduced chi2 according to its formula and we return it
        return np.einsum( 'j,j->' , deltas, np.einsum('jk,k->j',cov_inv,deltas) ) / ndof
    else:
        sig = np.sqrt(np.diag(covmat))
        return np.sum( (deltas/sig)**2 ) / ndof

#function that given a 1D array returns the values of the indices identifying its plateau (the first and last index)
def plateau_search(in_array: np.ndarray, covmat: np.ndarray, chi2_treshold:float=1.0, only_sig:bool=True) -> tuple[int,int]:
    """
    Input:
        - in_array: the 1D array we want to search the plateau of
        - covmat: a 2D array, representing the covariance matrix of in_array
        - chi2_treshold: the treshold for the plateau determination
        - only_sig: bool, if True only the standard deviation, and not the whole cavariance matrix, is used for the plateau determination
    
    Output:
        - (start_plateau,end_plateau): indices such that in_array[start_plateau,end_plateau] is the region with the plateau
    """

    #first we compute the len of the array
    len_array = np.shape(in_array)[0]

    #we loop over all the possible plateau lenghts, starting from the biggest possible one and then diminishing it up to a plataeau of len 2
    for len_plat in range(len_array,1,-1):

        #we instantiate a tmp dictionary where we are gonna store the values of the chi2 corresponding to the different starting value of the plateau (for a fixed lenght)
        tmp_chi2_dict = {}

        #then we loop over the possible initial points of the plateau
        for start_plateau in range(0,len_array-len_plat+1,1):

            #the suggested plateau region in this case is
            plat = in_array[start_plateau:start_plateau+len_plat]

            #we also have to reshape the covariance matrix
            covmat_plat = covmat[start_plateau:start_plateau+len_plat, start_plateau:start_plateau+len_plat]

            #the value of the plateau is
            plat_value = np.average(plat, weights = np.diag(np.linalg.inv(covmat_plat)), axis=0, keepdims=True) #the weights are the inverse of the sigma squared

            #we compute the chi2 of the current plateau
            chi2 = redchi2_cov(plat, plat_value, covmat_plat,only_sig=only_sig)

            #we see if the chi2 meets the condition
            if chi2 < chi2_treshold:

                #in that case we add the values of the starting and ending point fo the plateau to a dictionary, along with the associated chi2
                tmp_chi2_dict[(start_plateau, start_plateau+len_plat)] = chi2
        
        #after looking at all the possible starting values, for a fixed plateau len, if at least one chi2 was <1, we return the smallest
        if len(tmp_chi2_dict) > 0:
            return min(tmp_chi2_dict, key=tmp_chi2_dict.get)
                
    #if by the end of the loop the chi2 condition is never met (i.e. if len_plat is 1) we return the point corresponding to the middle of the dataset
    return int(len_array/2), int(len_array/2)+1

#function that given a 1D array returns the values of the indices identifying its plateau (the first and last index), that is symmetric around its middle point
def plateau_search_symm(in_array: np.ndarray, covmat: np.ndarray, chi2_treshold:float=1.0, only_sig:bool=True) -> tuple[int,int]:
    """
    Input:
        - in_array: the 1D array we want to search the plateau of
        - covmat: a 2D array, representing the covariance matrix of in_array
        - chi2_treshold: the treshold for the plateau determination
        - only_sig: bool, if True only the standard deviation, and not the whole cavariance matrix, is used for the plateau determination
    
    Output:
        - (start_plateau,end_plateau): indices such that in_array[start_plateau,end_plateau] is the region with the plateau
    """

    #first we compute the len of the array
    len_array = np.shape(in_array)[0]

    #we loop over all the possible plateau lenghts, starting from the biggest possible one and then diminishing it up to a plataeau of len 2
    for len_plat in range(len_array,1,-2):

        start_plateau = int( (len_array-len_plat)/2 )

        #the suggested plateau region in this case is
        plat = in_array[start_plateau:start_plateau+len_plat]

        #we also have to reshape the covariance matrix
        covmat_plat = covmat[start_plateau:start_plateau+len_plat, start_plateau:start_plateau+len_plat]

        #the value of the plateau is
        plat_value = np.average(plat, weights = np.diag(np.linalg.inv(covmat_plat)), axis=0, keepdims=True) #the weights are the inverse of the sigma squared

        #we compute the chi2 of the current plateau
        chi2 = redchi2_cov(plat, plat_value, covmat_plat,only_sig=only_sig)

        #we see if the chi2 meets the condition
        if chi2 < chi2_treshold: #TO DO: in this case put the value in a list and then at the end of the inner loop search for the better one

            return start_plateau, start_plateau+len_plat
                
    #if by the end of the loop the chi2 condition is never met (i.e. if len_plat is 1) we return the point corresponding to the middle of the dataset
    return int(len_array/2), int(len_array/2)+2-len_array%2

## Auxiliary Routines ##

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

#function used to asses whether an int is a perfect square or not (credit: https://stackoverflow.com/questions/2489435/check-if-a-number-is-a-perfect-square)
def is_square(apositiveint:int) -> bool:
    """
    Function used to to asses whether the input int is a perfect square or not (credit: https://stackoverflow.com/questions/2489435/check-if-a-number-is-a-perfect-square)
    
    Input:
        - apositive int: an int bigger than 1
    
    Output:
        - bool: True if the input is a perfect square, False otherwise
    """
    
    #input check
    if type(apositiveint) is not int or apositiveint<2:
        raise ValueError("The function checking for the perfect square condition work only with integer numbers bigger than 1!")

    #algorithmic determination of perfect square condition
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

#auxiliary function used to check if all elements in an iterable are equal (credit: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-equal)
def all_equal(iterable):
    """
    Function used to check if all elements in a list are equal (credit: https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-equal)
    
    Input:
        - iterable: the list (or iterable in general) to be checked
        
    Output:
        - True if all elements are equal, False otherwise
    """
    g = it.groupby(iterable)
    return next(g, True) and not next(g, False)