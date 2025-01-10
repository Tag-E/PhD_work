######################################################
## building_blocks_reader.py                        ##
## created by Emilio Taggi - 2025/01/09             ##
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

import numpy as np #to handle matrices
import h5py as h5 #to read the correlator
from tqdm import tqdm #for a nice view of for loops with loading bars
from pathlib import Path #to check whether directories exist or not



######################## Global Variables ###############################

#gamma vector
gamma_mu = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
#identity in Dirac space
identity = 'id'
#gamma5 matrix
gamma5 = 'gamma5'

#dictionary translating the gamma matrices into the corresponding list of integers (in the adopted convention)
gamma_to_list = {'gamma1': [1,0,0,0], 'gamma2': [0,1,0,0], 'gamma3': [0,0,1,0], 'gamma4': [0,0,0,1],
                 'id': [0,0,0,0], 'gamma5': [1,1,1,1]}
#dictionary translating the gamma matrices into the corresponding key (in the adopted convention)
gamma_to_key = {'gamma1': 'g1', 'gamma2': 'g2', 'gamma3': 'g4', 'gamma4': 'g8',
                 'id': 'g0', 'gamma5': 'g15'}



#The adopted convention for the gamma matrices is explained in the following table (with g_n being the key in the h5 file):
#g_n  ->  n (base 10)  -> n (base 2, with rightmost digit most significant one) ->       list       ->      gamma matrix
#g0   ->  0            ->        0000                                           ->    [0, 0, 0, 0]  ->  (γ₁)⁰ (γ₂)⁰ (γ₃)⁰ (γ₄)⁰ = 1
#g1   ->  1            ->        1000                                           ->    [1, 0, 0, 0]  ->  (γ₁)¹ (γ₂)⁰ (γ₃)⁰ (γ₄)⁰ = γ₁
#g2   ->  2            ->        0100                                           ->    [0, 1, 0, 0]  ->  (γ₁)⁰ (γ₂)¹ (γ₃)⁰ (γ₄)⁰ = γ₂
#g3   ->  3            ->        1100                                           ->    [1, 1, 0, 0]  ->  (γ₁)¹ (γ₂)¹ (γ₃)⁰ (γ₄)⁰ = γ₁ γ₂
#g4   ->  4            ->        0010                                           ->    [0, 0, 1, 0]  ->  (γ₁)⁰ (γ₂)⁰ (γ₃)¹ (γ₄)⁰ = γ₃
#g5   ->  5            ->        1010                                           ->    [1, 0, 1, 0]  ->  (γ₁)¹ (γ₂)⁰ (γ₃)¹ (γ₄)⁰ = γ₁ γ₃
#g6   ->  6            ->        0110                                           ->    [0, 1, 1, 0]  ->  (γ₁)⁰ (γ₂)¹ (γ₃)¹ (γ₄)⁰ = γ₂ γ₃ 
#g7   ->  7            ->        1110                                           ->    [1, 1, 1, 0]  ->  (γ₁)¹ (γ₂)¹ (γ₃)¹ (γ₄)⁰ = γ₁ γ₂ γ₃
#g8   ->  8            ->        0001                                           ->    [0, 0, 0, 1]  ->  (γ₁)⁰ (γ₂)⁰ (γ₃)⁰ (γ₄)¹ = γ₄
#g9   ->  9            ->        1001                                           ->    [1, 0, 0, 1]  ->  (γ₁)¹ (γ₂)⁰ (γ₃)⁰ (γ₄)¹ = γ₁ γ₄
#g10  ->  10           ->        0101                                           ->    [0, 1, 0, 1]  ->  (γ₁)⁰ (γ₂)¹ (γ₃)⁰ (γ₄)¹ = γ₂ γ₄
#g11  ->  11           ->        1101                                           ->    [1, 1, 0, 1]  ->  (γ₁)¹ (γ₂)¹ (γ₃)⁰ (γ₄)¹ = γ₁ γ₂ γ₄
#g12  ->  12           ->        0011                                           ->    [0, 0, 1, 1]  ->  (γ₁)⁰ (γ₂)⁰ (γ₃)¹ (γ₄)¹ = γ₃ γ₄
#g13  ->  13           ->        1011                                           ->    [1, 0, 1, 1]  ->  (γ₁)¹ (γ₂)⁰ (γ₃)¹ (γ₄)¹ = γ₁ γ₃ γ₄ 
#g14  ->  14           ->        0111                                           ->    [0, 1, 1, 1]  ->  (γ₁)⁰ (γ₂)¹ (γ₃)¹ (γ₄)¹ = γ₂ γ₃ γ₄
#g15  ->  15           ->        1111                                           ->    [1, 1, 1, 1]  ->  (γ₁)¹ (γ₂)¹ (γ₃)¹ (γ₄)¹ = γ₁ γ₂ γ₃ γ₄

#dictionary with the full correspondence
int_to_gamma = {0: '1', 1: 'gamma1', 2: 'gamma2', 3: 'gamma1gamma2',
                4: 'gamma3', 5: 'gamma1gamma3', 6: 'gamma2gamma3', 7: 'gamma1gamma2gamma3',
                8: 'gamma4', 9: 'gamma1gamma4', 10: 'gamma2gamma4', 11: 'gamma1gamma2gamma4',
                12: 'gamma3gamma4',13: 'gamma1gamma3gamma4', 14: 'gamma2gamma3gamma4', 15: 'gamma1gamma2gamma3gamma4'}




######################## Main Class ###############################

#class used to read the building blocks from the h5 files
class bulding_block:
    """
    Create one instance of the class to read the building blocks from the given folders
    """

    ## Global variable shared by all class instances



    #Initialization function
    def __init__(self, bb_folder,
                 tag='bb',hadron='proton_3',
                 maxConf=None, verbose=False):
        
        """
        Input description:...
        """


        #Info Print
        if verbose:
            print("\nInitializing the building block class instance...\n")
        
        
        #First we look into the given folder to see how many configurations we have

        #we store the folder for later use
        self.bb_folder = bb_folder

        #Path file to the data folder
        p = Path(bb_folder).glob('**/*')
        files = [x for x in p if x.is_file()]

        #we store the number of configurations into a class variable (nconf is given by the number of data files in the given folder)
        #if maxConf is not specified we take all the configurations, otherwise we read a smaller amount
        if maxConf is None or maxConf > len(files):
            self.nconf = len(files)
        else:
            self.nconf = maxConf

        #we store the list with all the configuration names
        self.conflist = [file.name for file in files]


        #Now looking only at the first configuration we read the keys of the h5 file structure

        #first conf is
        firstconf = self.conflist[0]

        #we open the h5 file corresponding too the first configuration
        with h5.File(bb_folder+firstconf, 'r') as h5f:

            #we read all the available keys and store them in lists

            #id of the given configuration (this is equal to firstconf)
            cfgid = list(h5f.keys())[0]

            #we store all the keys of the nested dictionaries structure of the h5 file
            self.tag_list = list(h5f[cfgid])
            self.smearing_list = list(h5f[cfgid][self.tag_list[0]])
            self.mass_list = list(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]])
            self.hadron_list = list(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]][self.mass_list[0]])
            self.qcontent_list = list(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]][self.mass_list[0]][self.hadron_list[0]])
            self.momentum_list = list(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]][self.mass_list[0]][self.hadron_list[0]][self.qcontent_list[0]])
            self.displacement_list = list(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]][self.mass_list[0]][self.hadron_list[0]][self.qcontent_list[0]][self.momentum_list[0]])
            self.dstructure_list = list(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]][self.mass_list[0]][self.hadron_list[0]][self.qcontent_list[0]][self.momentum_list[0]][self.displacement_list[0]])
            self.insmomementum_list = list(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]][self.mass_list[0]][self.hadron_list[0]][self.qcontent_list[0]][self.momentum_list[0]][self.displacement_list[0]][self.dstructure_list[0]])

            #we store the time extent of the correlator we have
            self.T = len(h5f[cfgid][self.tag_list[0]][self.smearing_list[0]][self.mass_list[0]][self.hadron_list[0]][self.qcontent_list[0]][self.momentum_list[0]][self.displacement_list[0]][self.dstructure_list[0]][self.insmomementum_list[0]])


        #we choose a default values for the different keys we want to be specified at reading time (for the other keys instead we loop over and read them)
        self.tag = tag
        self.smearing = self.smearing_list[0]
        self.mass = self.mass_list[0]
        self.hadron = hadron
        self.momentum = [mom for mom in self.momentum_list if mom.startswith('PX0_PY0_PZ0')][0] #we take the 0 forward momentum as default
        self.insmomentum = 'qx0_qy0_qz0' #same for the insertion momentum (this is the only choice available for every other key choice)

        #we store the dimensionality of the dimension we have not fixed and that we want to read
        self.nquarks = len(self.qcontent_list)
        self.ndisplacements = len(self.displacement_list)
        self.ndstructures = len(self.dstructure_list)


        #Now we loop over the configurations and we convert the nconf h5 files into a single dictionary

        #info print
        if verbose:
            print("\nLooping over the configurations to read the building blocks from the h5 files...\n")

        #we initialize the np array with the building blocks
        self.bb_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndisplacements, self.ndstructures, self.T),dtype=complex)

        #we loop over the configurations
        for iconf, file in enumerate(tqdm(files[:self.nconf])):

            #we open the h5 file corresponding too the current configuration
            with h5.File(bb_folder+file.name, 'r') as h5f:

                #id of the given configuration (this is equal to firstconf)
                cfgid = list(h5f.keys())[0]

                #we loop over the keys that have not been set:
                for iq, qcontent in enumerate(self.qcontent_list):
                    for idisp, displacement in enumerate(self.displacement_list):
                        for idstruct, dstructure in enumerate(self.dstructure_list):

                            #we read the building block and store it in the np array
                            self.bb_array[iconf, iq, idisp, idstruct] = h5f[cfgid][self.tag][self.smearing][self.mass][self.hadron][qcontent][self.momentum][displacement][dstructure][self.insmomentum]




    #function used to print the information regarding the keys in the dictionary
    def print_keys(self):


        #First we print the currently chosen values for the keys
        print("\nSelected Keys:\n")
        print(f"    -tag: {self.tag}")
        print(f"    -smearing: {self.smearing}")
        print(f"    -mass: {self.mass}")
        print(f"    -hadron: {self.hadron}")
        #print(f"    -qcontent: {self.qcontent}")
        print(f"    -momentum: {self.momentum}")
        #print(f"    -displacement: {self.displacement}")
        #print(f"    -dstructure: {self.dstructure}")
        print(f"    -insmomentum: {self.insmomentum}")


        #Then we print the lists with all the available keys
        print("\nAvailable Keys:\n")
        print(f"tag_list: {self.tag_list}")
        print(f"smearing_list: {self.smearing_list}")
        print(f"mass_list: {self.mass_list}")
        print(f"hadron_list: {self.hadron_list}")
        print(f"qcontent_list: {self.qcontent_list}")
        print(f"momentum_list: {self.momentum_list}")
        print(f"displacement_list: {self.displacement_list}")
        print(f"dstructure_list: {self.dstructure_list}")
        print(f"insmomementum_list: {self.insmomementum_list}")

        print("\nACHTUNG: not every combination of the above keys may be available\n")

    
    #function used to change the keys and reread the h5 files accordingly
    def update_keys(self, tag=None, smearing=None, mass=None, hadron=None, momentum=None, insmomentum=None, verbose=False):

        #we update the keys the user wants to change

        #auxiliary lists
        keys = [tag,smearing,mass,hadron,momentum,insmomentum]
        keys_list = [self.tag_list, self.smearing_list, self.mass_list, self.hadron_list, self.momentum_list, self.insmomementum_list]
        keys_name = ['tag','smearing','mass','hadron','momentum','insertion momementum']

        #flag used to signal an update in the keys
        update=False

        #we loop over the keys and update them if the user has specified a new value
        for i,k in enumerate(keys):
            if k in keys_list[i]:
                match i:
                    case 0:
                        self.tag = k
                    case 1:
                        self.smearing = k
                    case 2:
                        self.mass = k
                    case 3:
                        self.hadron = k
                    case 4:
                        self.momentum = k
                    case 5:
                        self.insmomentum = k
                update=True
            elif k is not None:
                print(f"Error: {keys_name[i]} not valid (look at the available keys with the print_keys method)")
        
        
        #if one of the keys has been updated we reread the h5 files
        if update is True:

            #info print
            if verbose:
                print("\nReading the building blocks from the h5 files according to the new keys...\n")


            #this is a copy of the reading procedure from the __init__ function

            #Path file to the data folder
            p = Path(self.bb_folder).glob('**/*')
            files = [x for x in p if x.is_file()]
            
            self.bb_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndisplacements, self.ndstructures, self.T),dtype=complex)

            #we loop over the configurations
            for iconf, file in enumerate(tqdm(files[:self.nconf])):

                #we open the h5 file corresponding too the current configuration
                with h5.File(self.bb_folder+file.name, 'r') as h5f:

                    #id of the given configuration (this is equal to firstconf)
                    cfgid = list(h5f.keys())[0]

                    #we loop over the keys that have not been set:
                    for iq, qcontent in enumerate(self.qcontent_list):
                        for idisp, displacement in enumerate(self.displacement_list):
                            for idstruct, dstructure in enumerate(self.dstructure_list):

                                #we read the building block and store it in the np array
                                self.bb_array[iconf, iq, idisp, idstruct] = h5f[cfgid][self.tag][self.smearing][self.mass][self.hadron][qcontent][self.momentum][displacement][dstructure][self.insmomentum]



    #function used to construct the right covariant derivative of the building blocks (with the chosen keys)
    def covD_r1(self) -> np.ndarray:

        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4), 4 being the dimensionality of the index mu of the covariant derivative
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_r1_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, self.T,4),dtype=complex) #the last dimension is the mu index of the covariant derivative

        #for each mu we compute the right derivative of the building block
        for i, mu in enumerate(mu_list):

            #the displacements we have to use are the following
            disp1 = 'l1_'+mu.upper()
            disp2 = 'l1_'+mu.lower()
            #with indices given by
            idisp1 = self.displacement_list.index(disp1)
            idisp2 = self.displacement_list.index(disp2)

            #knowing which displacements to use we can now compute the right covariant derivative as follows

            #       conf,quarks,dstruct,T,mu            conf,quarks,displacements,dstruct,T
            covD_r1_array[:, :, :, :, i] = self.bb_array[:, :, idisp1, :, :] - self.bb_array[:, :, idisp2, :, :]

        #we return the right covariant derivative
        return covD_r1_array
    


    #function used to construct the left covariant derivative of the building blocks (with the chosen keys)
    def covD_l1(self) -> np.ndarray:

        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4), 4 being the dimensionality of the index mu of the covariant derivative
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_l1_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, self.T,4),dtype=complex) #the last dimension is the mu index of the covariant derivative

        #for each mu we compute the right derivative of the building block
        for i, mu in enumerate(mu_list):

            #the displacements we have to use are the following (reversed wrt the r1 case)
            disp1 = 'l1_'+mu.lower()
            disp2 = 'l1_'+mu.upper()
            #with indices given by
            idisp1 = self.displacement_list.index(disp1)
            idisp2 = self.displacement_list.index(disp2)

            #knowing which displacements to use we can now compute the right covariant derivative as follows

            #the procedure is different for spatial and temporal components:
            #(in particular for the temporal component the temporal axis has to be shifted by -1 for backward displacement and by +1 for forward displacement)

            #for the spatial components we have
            if i<3:
                #       conf,quarks,dstruct,T,mu            conf,quarks,displacements,dstruct,T
                covD_l1_array[:, :, :, :, i] = self.bb_array[:, :, idisp1, :, :] - self.bb_array[:, :, idisp2, :, :]
            #for the temporal component we have
            elif i==3:
                #       conf,quarks,dstruct,T,mu            conf,quarks,displacements,dstruct,T
                covD_l1_array[:, :, :, :, i] = np.roll(self.bb_array[:, :, idisp1, :, :], shift=-1, axis=-1) - np.roll(self.bb_array[:, :, idisp2, :, :], shift=1, axis=-1) #axis=-1 is the time axis
            

        #we return the right covariant derivative
        return covD_l1_array


    #function used to obtain the building block of the relevant operators
    def get_bb(self, X:str, isospin:str, n_mu:int) -> np.ndarray:
        """
        Input:
            - X: either 'V', 'A' or 'T' (for vector, axial or tensorial operators)
            - isospin: either 'U' or 'D' (for up or down quarks) !!!! TO DO: add support for the 'isovector' choice (i.e. U-D) (or U+D??)
            - n_mu: the number of mu indices of the operator (either 1 or 2) (the indices after the ones due to the gamma matrices, i.e. the number of covariant derivatives) !!!! TO DO: add support for n_mu=0 

        Output:
            - the building block (a np array) of the operator with the features specified in input
        """

        #Input check

        #input check on X
        if X not in ['V','A','T']:
            print("Error: X must be either 'V', 'A' or 'T'")
            return None
        
        #input check on isospin
        if isospin not in ['U','D', 'U+D', 'U-D']:
            print("Error: isospin must be either 'U', 'D', 'U+D' or 'U-D'")
            return None
        
        #input check on n_mu
        if n_mu not in [1,2]:
            print("Error: n_mu must be either 1 or 2")
            return None
        

        #We now initialize the np array where we will store the building block of the operator

        #the number of indices of the operator depends on X and n_mu
        n_indices = 1+n_mu #1 gamma matrix + n_mu indices (if X is 'V' or 'A')
        if X == 'T':
            n_indices += 1 #2 gamma matrices + n_mu indices (if X is 'T')

        #we now instantiate the output array
        #the first two axis are nconf and time, the other ones are the indices of the operator, and they have dimensionality 4
        bb_operator = np.zeros(shape=(self.nconf, self.T, ) + (4,)*n_indices, dtype=complex)


        #We take now care of the quark content

        #first thing first we retrieve the right minus left covariant derivative
        covD = self.covD_r1() - self.covD_l1() #shape = (nconf, nquarks, ndstructures, T, 4), the last dimension being the index of the covariant derivative

        #now we select the isospin component requested by the user
        #the shape will change: (nconf, nquarks, ndstructures, T, 4) -> (nconf, ndstructures, T, 4) 

        #if the isospin is just 'U' or 'D' we select the corresponding quark content
        if isospin in ['U','D']:
            qcontent = self.qcontent_list.index(isospin)
            covD = covD[:,qcontent,:,:,:]             
        #if otherwise it is 'U+D' we sum the two components
        elif isospin == 'U+D':
            qcontent_U = self.qcontent_list.index('U')
            qcontent_D = self.qcontent_list.index('D')
            covD = covD[:,qcontent_U,:,:,:] + covD[:,qcontent_D,:,:,:]
        #if otherwise it is 'U-D' we subtract the two components
        elif isospin == 'U-D':
            qcontent_U = self.qcontent_list.index('U')
            qcontent_D = self.qcontent_list.index('D')
            covD = covD[:,qcontent_U,:,:,:] - covD[:,qcontent_D,:,:,:]      #now covD has shape (nconf, ndstructures, T, 4)



        
        #We now proceed with the actual construction of the building block, that depends on the number of covariant derivatives (=n_mu)

        #case with 1 covariant derivative
        if n_mu==1:
            
            #We now build the axis corresponding to the gamma structure
            #the shape will change in the following way:
            #       (nconf, ndstructures, T, 4) -> (nconf, T, 4, 4)  for X='V' or 'A'
            #       (nconf, ndstructures, T, 4) -> (nconf, T, 4, 4, 4)  for X='T'

            #we have to look and understand which gamma matrix goes into which position, and this depends on X

            #vectorial case
            if X == 'V':

                #for the vectorial case we just loop over the components of gamma_mu and add them to the building block
                for mu, gamma in enumerate(gamma_mu):

                    #the key corresponding to the gamma_mu matrix is 
                    key = gamma_to_key[gamma]
                    #the index of the key is 
                    index = self.dstructure_list.index(key)

                    #we now just assign the dstructure axis we want to the mu axis

                    # (nconf,T,4,4)              (nconf,ndirac,T,4)
                    bb_operator[:,:,mu,:] = covD[:,index,:,:]

            #axial case
            elif X == 'A':
                
                #for the axial case we have to take the product of each gamma mu with gamma5
                for mu, gamma in enumerate(gamma_mu):

                    #we look at what is the result of the product gamma_mu * gamma5 (both in overall sign and in gamma structure)
                    sign_result, list_result = gamma_prod( gamma_to_list[gamma], gamma_to_list[gamma5]  )

                    #from the result about the dirac structure we find the key
                    key = list_to_key(list_result)

                    #and from the key the index
                    index = self.dstructure_list.index(key)

                    #we now just assign the dstructure axis we want to the mu axis, taking also into account the sign

                    # (nconf,T,4,4)                          (nconf,ndirac,T,4)
                    bb_operator[:,:,mu,:] = sign_result * covD[:,index,:,:]

            #tensorial case
            elif X == 'T':
                
                #for the tensor case the gamma matrices are gamma1gamma2, gamma1gamma3, gamma1gamma4, gamma2gamma3, gamma2gamma4, gamma3gamma4
                
                #so we have to loop two times over gamma_mu
                for mu1, gamma_mu1 in enumerate(gamma_mu):
                    for mu2, gamma_mu2 in enumerate(gamma_mu):

                        #we look at what is the result of the product gamma_mu1 * gamma_mu2 and of gamma_mu2 * gamma_mu1 (both in overall sign and in gamma structure)
                        sign_result1, list_result1 = gamma_prod( gamma_to_list[gamma_mu1], gamma_to_list[gamma_mu2]  ) #gammma_mu1 * gamma_mu2
                        sign_result2, list_result2 = gamma_prod( gamma_to_list[gamma_mu2], gamma_to_list[gamma_mu1]  ) #gammma_mu2 * gamma_mu1

                        #from the results about the dirac structure we find the key
                        key1 = list_to_key(list_result1)
                        key2 = list_to_key(list_result2)

                        #and from the key the index
                        index1 = self.dstructure_list.index(key1)
                        index2 = self.dstructure_list.index(key2)

                        #we now just assign the dstructure axis we want to the mu1 and mu2 axis, taking also into account the sign
                        #we also use the fact that the tensor structure has the gamma structure sigma_mu1_mu2 = i/2 * (gamma_mu1 gamma_mu2 - gamma_mu2 gamma_mu1)

                        # (nconf,T,4,4,4)                          (nconf,ndirac,T,4)
                        bb_operator[:,:,mu1,mu2,:] = 1j/2 * ( sign_result1 * covD[:,index1,:,:] - sign_result2 * covD[:,index2,:,:] )



        #case with 2 covariant derivatives
        elif n_mu==2:
            pass #TO BE IMPLEMENTED


        #we return the building block as calculated
        return bb_operator






################ Auxiliary Functions ###############################

#function used to implement the product of two gamma matrices
def gamma_prod(g1,g2):

    """
    - the inputs are g1 and g2 , lists of 4 bits (with the rightmost being the most significat)
    - the output is a tuple with (sign, gamma structure), where the gamma structure is in the same format (list of 4 bits)
    """

    #the gamma multiplication is just a bit by bit addition modulo 2
    gstructure = [ (e1+e2)%2 for e1,e2 in zip(g1,g2)]

    #the sign is the tricky part, to compute it we have to count the number of swaps occuring
    sign = 0
    for i,e1 in enumerate(g1):
        sign += e1 * sum(g2[:i]) #the multiplication to e1 is equivalent to count the swaps only when e1 is 1 (since e1 can only be 0 or 1)

    return (-1)**sign, gstructure


#function translating one list encoding the gamma structure into the relevant key
def list_to_key(list):
    return 'g'+ str(int( ''.join([str(e) for e in reversed(list)]), 2))