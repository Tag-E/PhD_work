######################################################
## moments_toolkit.py                               ##
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

########################## Program Usage ################################
#
# ...  ...
# ... ...
# ...
#
#########################################################################



######################## Library Imports ################################

## general libraries
import numpy as np #to handle matrices
#import h5py as h5 #to read the correlator
#from tqdm import tqdm #for a nice view of for loops with loading bars
from pathlib import Path #to check whether directories exist or not
from pylatex import Document, Command  #to produce a pdf documents with the CG coeff
from pylatex.utils import NoEscape #also to produce a pdf document
import subprocess #to open pdf files
import time #to use sleep and pause the code
import matplotlib.pyplot as plt #to plot stuff
from typing import Any, Callable, List #to use annotations for functions

## custom made libraries
from building_blocks_reader import bulding_block #to read the 3p and 2p correlators
from Kfact_calculator import K_calc #to obtain the list of operators and related kinematic factors



######################## Global Variables ###############################




######################## Main Class ####################################

#each dataset corresponds to an instance of the class, its methods provide useful analysis tools
class moments_toolkit:
    """
    Create an instance of the class to setup the analysis framework associated to the given dataset
    """

    ## Global variable shared by all class instances

    #list of available structures = vector,axial,tensor
    X_list = ['V', 'A', 'T']




    ## Methods of the class

    #Initialization function
    def __init__(self, p3_folder:str, p2_folder:str,
                 tag_3p='bb',hadron='proton_3', tag_2p='hspectrum',
                 maxConf=None,max_n=3, verbose=False):
        
        """
        Input description:...
        """


        #Info Print
        if verbose:
            print("\nInitializing the moments_toolkit class instance...\n")
        

        #we store the folder variables
        self.p3_folder=p3_folder
        self.p2_folder=p2_folder

        #we initialize a list with all the available number of indices #TO DO: add input check
        self.n_list = [i for i in range(2,max_n+1)] 

        
        #First we look into the given p3 folder to see how many different subfolders we have

        #we take the path of the folders with 3 points correlators subfolder
        p = Path(p3_folder)
        #we read the avalaible list of time separations T
        self.T_list = sorted( [int(x.name[1:]) for x in p.iterdir() if x.is_dir() and x.name.startswith('T')] )
        #from that we obtain the paths of the folders containing the different building blocks
        bb_pathList = [f"{p3_folder}T{T}/" for T in self.T_list]

        
        #We now proceed to produce a  building_block class instance for each one of the times T

        #we instantiate the list with the building_block class instances
        self.bb_list = []

        #we loop over the bb paths available (loop over T)
        for i,bb_path in enumerate(bb_pathList):
            #Info Print
            if verbose:
                print(f"\n\nReading data for T = {self.T_list[i]} ...\n")
            self.bb_list.append( bulding_block(bb_path,p2_folder, maxConf=maxConf, verbose=verbose) )


        #We initialize some other class variables

        #number of configurations
        self.nconf = self.bb_list[0].nconf

        #list with operators selected for the analysis, initialized as empty
        self.selected_op = []



        ##We build the list of all the available operators

        #info print
        if verbose:
            print("\nBuilding the list of all available operators...\n")

        #we initialize the list as empty
        self.operator_list = []

        #we also store the classes related to the kinematic factors, we also initialize them as empty
        self.kclass_list = []

        #we loop over X and n, store the K classes and the operators

        #we count the number of operators
        op_count=1

        #we loop over the available indices
        for n in self.n_list:
            #we loop over the V,A,T structures
            for X in self.X_list:

                #safety measure to avoid the bug present in the cg calc class: TO BE REMOVED after fixing it (TO DO)
                if n>2 and X=='T': break

                #the actual number of indices depends on X
                actual_n = n
                if X == 'T':
                    actual_n += 1
                
                #we instantiate the related Kfactor class
                self.kclass_list.append( K_calc(X,actual_n,verbose=False) )

                #we append the operators
                for op in self.kclass_list[-1].get_opList(first_op=op_count):
                    self.operator_list.append(op)

                #we update the operator count
                op_count += 4**actual_n 

        #info print
        if verbose:
            print("\nClass Initialization Complete!\n")





    #Function used to print to pdf and show to the user all the available operators that can be chosen
    def operator_show(self, title=None, author="E.T.", doc_name=None, verbose=False, show=True, remove_pdf=False, clean_tex=True):
        """
        Function description TO DO
        """

        #we set the title param to the default value
        if title is None:
            title = "Available Operators" #TO DO: add the max number of indicices used

        #we set the document name to the default value
        if doc_name is None:
            doc_name = "operator_catalogue" #TO DO: add max number of indices
        
        #the name of the pdf file is then
        file_name = f"{doc_name}.pdf" #TO DO: add the max number of indicices used
        

        #we instantiate the .tex file
        doc = Document(default_filepath=f'{doc_name}.tex', documentclass='article',page_numbers=False)#, font_size='' )

        #create document preamble
        doc.preamble.append(Command("title", title))
        doc.preamble.append(Command("author", author))
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        doc.append(NoEscape(r"\newpage"))

        doc.append(Command('fontsize', arguments = ['8', '12']))

        #we instantiate the operator count to 1
        op_count = 1

        #we loop over the instantiated kinematic classes (=loop over n and over X)
        for kclass in self.kclass_list:

            #we add the related operator to the document
            kclass.append_operators(doc,op_count)

            #we update the operator count
            op_count += len(kclass.get_opList()) #= 4**actual_n


        #then we generate the pdf
        
        #info print
        if verbose:
            print("\nGenerating the operators catalogue ...\n")

        #pdf generation
        #doc.generate_pdf(self.kfact_pdf_folder + '/'  + doc_name, clean_tex=clean_tex)
        doc.generate_pdf(doc_name, clean_tex=clean_tex)

        #info print
        if verbose:
            print("\nOperators catalogue generated\n")

  

        #we show to screen the generated pdf
        if show:
            #we xdg-open the pdf
            subprocess.call(["xdg-open", file_name])
            #we wait 1.5 seconds so that the pdf can be seen before its deletion
            time.sleep(1.5)
            #info print
            if verbose:
                print("\nOperators catalogue shown\n")

        #we remove the pdf
        if remove_pdf:

                #file elimination
                Path(file_name).unlink()

                #info print
                if verbose:
                    print("\nOperators catalogue removed from the device\n")

        elif verbose:
            print(f"\nOperators catalogue available in {file_name}\n")


    
    #function used to select which operators we want to study
    def select_operator(self, *kwarg):
        """
        Input:
            - the int corresponding to the operators selected, as shown in the operator catalogue (accessible via the method operator_show)
        """

        #the chosen ids are
        chosen_ids = kwarg

        #first we reset the list of selected operators
        self.selected_op = []

        #then for every id we append the corresponding operator to the list of chosen ones
        for id in chosen_ids:
            self.selected_op.append(self.operator_list[id-1]) #-1 because in the pdf the numbering starts from 1

    

    #function used to compute the ratio R(T,tau)
    def get_R(self, isospin='U-D') -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Input:
            - isospin: either 'U', 'D', 'U-D' or 'U+D'
        Output:
            - R(iop,iconf,T,tau): a np array with axis (iop,iconf,T,tau) and shape [n_selected_operators, nconf, n_T, max_T] 
            - Rmean(iop,T,tau): the mean resulting from the jackknife analysis performed on R
            - Rstd(iop,T,tau): the std reasulting from the jackknife analysis performed on R
            - Rcovmat(iop,T,tau1, tau2): the whole covariance matrix reasulting from the jackknife analysis performed on R
        """
        
        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

        #the axis have dimensionalities
        nop = len(self.selected_op)
        nT = len(self.T_list)
        maxT = np.max(self.T_list) #this is the dimensionality of the tau axis (so for T<maxT there is a padding with zeros from taus bigger than their max value)

        #we initialize the output array with zeros
        R = np.zeros(shape=(nop, self.nconf, nT, maxT+1),dtype=complex) #+1 because tau goes from 0 to T included

        #we now fill the array using the method of the building block class to extract R
        #we have to loop over the dimensionalities to fill

        #loop over the selected operators
        for iop,op in enumerate(self.selected_op):

            #we extract the relevant info from the operator
            cgmat = op.cgmat #mat with cg coeff
            X = op.X # 'V','A' or 'T'
            nder = op.nder #number of derivatives in the operators = number of indices -1 (for V and A) or -2 (for T)

            #loop over the available times T
            for iT,T in enumerate(self.T_list):

                #we compute the ratio R (that is just the building block normalized to the 2 point correlator)
                R[iop,:,iT,:T+1] = self.bb_list[iT].operatorBB(cgmat,X,isospin,nder) #the last axis is padded with zeros

        #we perform the jackknife analysis (the observable being the avg over the configuration axis)
        Rmean, Rstd, Rcovmat = jackknife(R, lambda x: np.mean(x,axis=1), jack_axis=1, time_axis=-1)

        #we return the ratios just computed and the results of the jackknife analysis
        return R, Rmean, Rstd, Rcovmat
    

    #function used to plot the ratio R for all the selected operators
    def plot_R(self, isospin='U-D', show=True, save=False, figname='plotR') -> None:
        """
        Input:
            - isospin: either 'U', 'D', 'U-D' or 'U+D
            - show: bool, if True the plot with R is shown
            - save: bool, if True the plot is saved to .png
        Output:
            - None (the function is used just to have the plots with R printed to screen)
        """
        
        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

        
        #we first fetch R using the dedicate method
        R, Rmean, Rstd, Rcovmat = self.get_R(isospin=isospin)

        #TO DO: add check on selected op



        #loop over selected operators (for each we make a plot)
        for iop,op in enumerate(self.selected_op):
            
            #instantiate figuer
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(32, 14))

            #we loop over T and each time we add a graph to the plot
            for iT, T in enumerate(self.T_list):

                times = np.arange(-T/2+1,T/2)
                 
                ratio = R[iop,:,iT,:T+1]

                ratio_err = np.abs(ratio[:,1:-1]).std(axis=0)/np.sqrt(np.shape(ratio)[0]-1)
                ratio = ratio.mean(axis=0) #mean over cfg axis

                ratio = Rmean[iop,iT,:T+1]
                ratio_err = Rstd[iop,iT,:T+1]
                #ratio
                #ratio = ratio.real #cast to real
                #ratio = ratio.imag
                ratio = np.abs(ratio)

                #we discard the endpoints
                r = ratio[1:-1]
                r_err = ratio_err[1:-1]


                #_=plt.plot(times,r,marker = 'o', linewidth = 0.3, linestyle='dashed',label=i)
                #ax.errorbar(times, r,yerr=ratio_err, marker = 'o', linewidth = 0.3, linestyle='dashed',label=f"T{T}")
                ax.errorbar(times, r,yerr=r_err, marker = 'o', linewidth = 0.3, linestyle='dashed',label=f"T{T}")
                ax.legend()

                ax.set_title(r"R(T,$\tau$) - Operator " + str(op.id))
                ax.set_xlabel(r"$\tau$")
                ax.set_ylabel('R')

            #we save the plot if the user asks for it
            if save:
                plt.savefig(f"{figname}_operator{op.id}.png")
            
            #we show the plot if the user asks for it
            if show:
                plt.show()


    #function used to to compute the sum of ratios S #TO DO: add jackknife analysis
    def get_S(self, tskip: int, isospin='U-D'):
        """
        Input:
            - tskip = \tau_skip = gap in time when performing the sum of ratios
            - isospin: either 'U', 'D', 'U-D' or 'U+D'
        Output:
            - S: the sum of ration given by S(T,tskip) = sum_(t=tskip)^(T-tskip) R(T,t) (with shape (nop,nconf,nT))
            - Smean, Sstd: mean and std from jackknife procedure applied on S (with shape (nop, nT))
        """
        
        #input control
        if isospin not in ['U', 'D', 'U-D', 'U+D']:
            print("Selected isospin not valid, defaulting to 'U-D'")
            isospin='U-D'

        
        #we first fetch R using the dedicate method
        R,_,_,_= self.get_R(isospin=isospin) #shape = (nop, nconf, nT, ntau)

        #then based on the shape of R we instantiate S
        #S = np.zeros(shape=np.shape(R)[:-1], dtype=complex) #shape = (nop, nconf, nT)

        #we compute S
        #for iT,T in enumerate(self.T_list):
            #S[:,:,iT] = np.sum(R[:,:,iT,tskip:T+1-tskip], axis =-1)

        #ratio = ok.operatorBB(op,'V','U-D',1)
        #ratio = ratio.mean(axis=0) #mean over cfg axis
        #ratio = np.abs(ratio)

        #we compute S for each configuration
        S = sum_ratios(R,Tlist=self.T_list, tskip=tskip)

        #we compute S with the jackknife
        Smean, Sstd, _ = jackknife(R, lambda x: np.mean( sum_ratios(x,Tlist=self.T_list, tskip=tskip), axis=1), jack_axis=1, time_axis=None)

        #we return S
        return S, Smean, Sstd




        

        

######################## Auxiliary Functions ##########################


#function implementing the jackknife analysis
def jackknife(in_array: np.ndarray, observable: Callable[[], Any], jack_axis=0, time_axis=-1, binsize=1,first_conf=0,last_conf=None) -> List[np.ndarray]:
    """
    Input:
        - in_array: input array to be jackknifed
        - observable: function taking as input an array of the same shape of in_array (i.e. an observable that should be computed over it), and giving as output an array with the jackknife axis (i.e. conf axis) removed
        - jack_axis: the axis over which perform the jacknife analysis (from a physics p.o.v. the axis with the configurations),
        - time_axis: axis over to which look for the autocorrelation
        - binsize: binning of the jackknife procedure
        - first_conf: index of the first configuration taken into account while performing the jackknife procedure
        - last_conf: index of the last configuration taken into account while performing the jackknife procedure (if not specified then the last available configuration is used)

    Output:
        - list with [mean, std, cov] where mean and std are np array with same the same shape as the input one, and the cov has one extra time dimension (the new time dimension is now the last one)
    """

    #we set last conf to its default value
    if last_conf is None:
        last_conf = np.shape(in_array)[jack_axis]

    #step 1: creation of the jackknife resamples
    jack_resamples = np.asarray( [np.delete(in_array, list(range(iconf,min(iconf+binsize,last_conf))) ,axis=jack_axis) for iconf in range(first_conf,last_conf,binsize)] ) #shape = (nresamp,) + shape(in_array) (with nconf -> nconf-binsize)
    #print("jack resamples")
    #print(np.shape(jack_resamples))

    #the number of resamples is len(jack_resmaples[0]) or also
    #nresamp = int((last_conf-first_conf)/binsize)
    nresamp = np.shape(jack_resamples)[0] #the 0th axis now is the resample axis, (and axis has nconf-1 conf in the standard case (binsize=1 ecc.) )

    #step 2; for each resample we compute the observable of interest
    #we use the resampled input array to compute the observable we want, and we have nresamp of them
    obs_resamp = np.asarray( [observable(jack_resamples[i]) for i in range(nresamp) ] )                                                                          #shape = (nresamp,) + shape(in_array) - jack_dimension   (jack dimension replaced by replica dimension) (the observable function removes the jackdimension -> jack_resamp[i] has the same shape as in_array)
    #print("obs resamples")
    #print(np.shape(obs_resamp))

    #step 3: we compute the observable also on the whole dataset
    obs = observable(in_array)                                                                                                                                   #shape = shape(in_array) - jack_dimension (the observable function removes the jackdimension)
    #print("obs")
    #print(np.shape(obs))

    #step4: compute estimate, bias and std according to the jackknife method
    
    #the estimate is the mean of the resamples
    jack_mean = np.mean(obs_resamp,axis=0) #axis 0 is the resamples one                                                                                         #shape = shape(in_array) - jack_dimension
    #print("jack mean")
    #print(np.shape(jack_mean))

    #the jackknife bias is given by the following formula 
    bias = (nresamp-1) * (jack_mean - obs)                                                                                                                     #shape = shape(in_array) - jack_dimension
    #print("bias")
    #print(np.shape(bias))

    #TO DO: add proper cast to real

    #the jack std is given by the following formula
    obs_std = np.sqrt( (nresamp-1)/nresamp * np.sum( (obs_resamp - jack_mean)**2, axis=0 ) ) #the axis is the resamples one                                        #shape = shape(in_array) - jack_dimension
    #print("obs std")
    #print(np.shape(obs_std))

    #to obtain the final estimate we correct the jack mean by the bias
    obs_mean = jack_mean - bias                                                                                                                                  #shape = shape(in_array) - jack_dimension


    #step 5: covariance matrix computation

    #we perform such a computation only if there actually see a time axis over which to look for a correlation
    if time_axis is not None:

        #to account for the fact that we have removed the jackknife dimension we change the time dimension

        #first we compute the lenght in the time dimension
        lenT = np.shape(in_array)[time_axis]

        #the time axis is translated to a positive value
        if time_axis<0:
            #time_axis = lenT+time_axis
            time_axis = len(np.shape(in_array))+time_axis

        #then we check if the time dimension has to be reduced by one (i.e. if the just deleted jack axis causes the time axis to be smaller by 1)
        if jack_axis < time_axis :
            new_time_axis = time_axis - 1

        #we the instantiate the covariance matrix
        covmat = np.zeros(shape = np.shape(obs_mean) + (lenT,), dtype=float )

        #TO DO: add cast to real values before computing covariance matrix

        #we then loop over the times and fill the covariance matrix
        for t1 in range(lenT):
            for t2 in range(lenT):

                #we do a little of black magic to addres the right indices combinations (credit https://stackoverflow.com/questions/68437726/numpy-array-assignment-along-axis-and-index)
                s = [slice(None)] * len(np.shape(covmat))
                axe1 = new_time_axis #position of the first time axis
                s[axe1] = slice(t1,t1+1)
                axe2 =  len(np.shape(covmat))-1 #because the new time axis is at the end of the array
                s[axe2] = slice(t2,t2+1)

                #we update the covariance matrix
                #covmat[tuple(s)] = np.expand_dims( (nresamp-1)/nresamp * np.sum( (  np.take(obs_resamp,t1,axis=time_axis) - np.take(obs,t1,axis=new_time_axis) ) * (  np.take(obs_resamp,t2,axis=time_axis) - np.take(obs,t2,axis=new_time_axis) ), axis=0 ),
                 #                                  [axe1,axe2])
                covmat[tuple(s)] = np.expand_dims( (nresamp-1)/nresamp * np.sum( (  np.take(obs_resamp,t1,axis=time_axis) - np.take(jack_mean,t1,axis=new_time_axis) ) * (  np.take(obs_resamp,t2,axis=time_axis) - np.take(jack_mean,t2,axis=new_time_axis) ), axis=0 ),
                                                   [axe1,axe2])
    #if instead there is not a time axis we just send back the std in place of the covmat
    else:
        covmat = obs_std


    #we return mean and std 
    return [obs_mean, obs_std, covmat]





#function translating R to S (i.e. the array with ratios to the array where the tau dimension has been summed appropiately)
def sum_ratios(Ratios: np.ndarray, Tlist: list[int], tskip: int) -> np.ndarray:
    """
    Input:
        - Ratios: the array R, with shape (nop,nconf,nT,max(ntau))
        - Tlist: a list of all the available T corresponding the third dimensionality
        - tskip: the tau skip used while summing the ratios
    Output:
        - S: the sum of ratios, with dimensionalities (nop,nconf,nT)
    """
    #then based on the shape of R we instantiate S
    S = np.zeros(shape=np.shape(Ratios)[:-1], dtype=complex) #shape = (nop, nconf, nT)

    #we compute S
    for iT,T in enumerate(Tlist):
        S[:,:,iT] = np.sum(Ratios[:,:,iT,tskip:T+1-tskip], axis =-1)

    #we return S
    return S