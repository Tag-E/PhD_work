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
import json #to store parameters file



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

    #list of available structures = vector,axial,tensor
    X_list = ['V', 'A', 'T']

    #list of the available isospin structures
    isospin_list = ['U', 'D', 'U-D', 'U+D']

    #list with the available number of derivatives
    n_mu_list = [1,2]



    ## Methods of the class

    #Initialization function
    def __init__(self, #bb_folder: str,
                 p3_folder:str, p2_folder: str,
                 tag_3p:str='bb', hadron:str='proton_3', tag_2p:str='hspectrum',
                 insertion_momentum:str='qx0_qy0_qz0', momentum:str='PX0_PY0_PZ0',
                 smearing_index:int = 0,
                 fast_data_folder:str='fast_data', force_reading:bool=False,
                 T_no_read_list:list[int]=[], skip3p:bool=False, 
                 maxConf:int|None=None, verbose:bool=False) -> None:
        
        """
        Initialization of the class used to read and perform analysis routines on a 3point correlator, for a given T, and on to the realated 2p correlator

        Input:
            - bb_folder: folder with the 3-point correlators
            - p2_folder: folder with the 2-point correlators (related to the 3 point ones)
            - tag_3p: tag of the 3-point correlator
            - hadron: hadron type we want to read from the dataset (for both 3-points and 2-points)
            - insertion_momentum: str, the momentum of the operator inserted in the 3 point correlator
            - momentum: str, the momentum of the hadron
            - tag_2p: tag of the 2-points correlator
            - smearing_index: int, index of the chose smearing in the list of available ones, default is 0 (i.e. the first smaring type is chosen)
            - fast_data_folder: str, the path to the folder where the dataset gets saved in a reduced format suitable for fast access
            - force_reading: bool, if True the correlators will be read from the complete dataset even if a fast access dataset is available
            - T_no_read_list: list of the times T (hence 3 point correlators) for which the reading can be avoided
            - skip3p: bool, if True the reading of the 3 point correlators is skipped alltogether
            - maxConf: maximum number of configuration to be red
            - verbose: bool, if True info print are provided while the class instance is being constructed

        Output:
            - None (an instance of the building_block class is created)
        """

        #Input check on the folders with the corelators (the class is not initialized if one of the folders does not exist, and hence the dataset can't be read)
        if Path(p3_folder).is_dir() == False:
            raise RuntimeError("\nAchtung; the bb_folder specified does not exist, the building block class can't be initialized\n")
        if Path(p2_folder).is_dir() == False:
            raise RuntimeError("\nAchtung; the p2_folder specified does not exist, the building block class can't be initialized\n")


        #Info Print
        if verbose:
            print("\nInitializing the building block class instance...\n")
        

        ## First we look into the given p3 folder to see how many different subfolders we have

        #we store the three point correlators folder
        self.p3_folder = p3_folder

        #we take the path of the folders with 3 points correlators subfolder
        p = Path(p3_folder)

        #we read the avalaible list of time separations T
        self.T_list = sorted( [int(x.name[1:]) for x in p.iterdir() if x.is_dir() and x.name.startswith('T')] )

        #we remove the times the user specified
        for T_to_remove in T_no_read_list:
            if T_to_remove in self.T_list:
                self.T_list.remove(T_to_remove)


        #we store the number of source-sink separation values T we are using in the analysis
        #self.nT = len(self.T_list)

        #from that we obtain the paths of the folders containing the different building blocks
        self.bb_pathList = [f"{p3_folder}T{T}/" for T in self.T_list]


        #Info Print
        if verbose:
            print("\nReading the the keys of the dataset ...\n")

        
        ## Then we look into the first 3p folder to see how many configurations we have

        #the folder we want to access for make a first inspection is
        bb_folder = self.bb_pathList[0]

        #Path file to the data folder
        p = Path(bb_folder).glob('**/*')
        files_3p = sorted( [x for x in p if x.is_file()] ) #we sort the configurations according to ascending ids


        #we store the number of configurations into a class variable (nconf is given by the number of data files in the given folder)
        #if maxConf is not specified we take all the configurations, otherwise we read a smaller amount
        if maxConf is None or maxConf > len(files_3p):
            self.nconf = len(files_3p)
        else:
            self.nconf = maxConf

        #we store the list with all the configuration names
        self.conflist = [file.name for file in files_3p]


        ## Now looking only at the first configuration of the first 3p file we read the keys of the h5 file structure

        #first conf is
        firstconf = self.conflist[0]

        #we open the h5 file corresponding too the first configuration
        with h5.File(bb_folder+firstconf, 'r') as h5f:

            #we read all the available keys and store them in lists

            #id of the given configuration (this is equal to firstconf)
            cfgid = list(h5f.keys())[0]

            ## We now do three things
            #  1) we store all the keys of the nested dictionaries structure of the h5 file
            #  2) we choose a default values for the different keys we want to be specified at reading time (for the other keys instead we loop over and read them)
            #  3) we raise an error if the chosen key is not available

            #tag of 3p corr
            self.tag_list = list(h5f[cfgid])
            self.tag_3p = tag_3p if tag_3p in self.tag_list else None
            if self.tag_3p is None: raise ValueError(f"The chosen value for tag_3p is not allowed, please choose one value from the following list: {self.tag_list}")

            #smearing both for 3p and 2p corr
            self.smearing_list = list(h5f[cfgid][self.tag_3p])
            self.smearing = self.smearing_list[smearing_index]

            #quark mass, same for 3p and 2p
            self.mass_list = list(h5f[cfgid][self.tag_3p][self.smearing])
            self.mass = self.mass_list[0]
            
            #hadron, same for 2p and 3p
            self.hadron_list = list(h5f[cfgid][self.tag_3p][self.smearing][self.mass])
            self.hadron = hadron if hadron in self.hadron_list else None
            if self.hadron is None: raise ValueError(f"The chosen value for hadron is not allowed, please choose one value from the following list: {self.hadron_list}")

            #quark content
            self.qcontent_list = list(h5f[cfgid][self.tag_3p][self.smearing][self.mass][self.hadron])

            #momentum
            self.momentum_list = list(h5f[cfgid][self.tag_3p][self.smearing][self.mass][self.hadron][self.qcontent_list[0]])
            self.momentum = [mom for mom in self.momentum_list if mom.startswith(momentum)][0] if len([mom for mom in self.momentum_list if mom.startswith(momentum)])>0 else None
            if self.momentum is None: raise ValueError(f"The chosen value for momentum is not allowed, please choose one value from the following list: {self.momentum_list}")

            #displacement
            self.displacement_list = list(h5f[cfgid][self.tag_3p][self.smearing][self.mass][self.hadron][self.qcontent_list[0]][self.momentum])

            #diract structure
            self.dstructure_list = list(h5f[cfgid][self.tag_3p][self.smearing][self.mass][self.hadron][self.qcontent_list[0]][self.momentum][self.displacement_list[0]])

            #insertion momentum
            self.insmomentum_list = list(h5f[cfgid][self.tag_3p][self.smearing][self.mass][self.hadron][self.qcontent_list[0]][self.momentum][self.displacement_list[0]][self.dstructure_list[0]])
            self.insmomentum = insertion_momentum if insertion_momentum in self.insmomentum_list else None
            if self.insmomentum is None: raise ValueError(f"The chosen value for insertion_momentum is not allowed, please choose one value from the following list: {self.insmomentum_list}")


        #we update the momentum list and the momentum by stripping the T from the string
        self.momentum_list = [mom.split('_T')[0] for mom in self.momentum_list]
        self.momentum = self.momentum.split('_T')[0]

        #we store the dimensionality of the dimension we have not fixed and that we want to read
        self.nquarks = len(self.qcontent_list)
        self.ndisplacements = len(self.displacement_list)
        self.ndstructures = len(self.dstructure_list)


        ## We do the same thing for the 2 point function, and to read it later we first look now at the available keys

        #we store the folder for later use
        self.p2_folder = p2_folder

        #Path file to the data folder
        p = Path(p2_folder).glob('**/*')
        files_2p = sorted( [x for x in p if x.is_file()] ) #we sort the configurations, so that index by index the configurations match the ones of the 3 points correlators

        #we open the first configuration just to read the lenght of the lattice
        firstconf = files_2p[0].name
        with h5.File(p2_folder+firstconf, 'r') as h5f:

            #id of the given configuration (this is equal to firstconf)
            cfgid = list(h5f.keys())[0]

            #we store the list of available tags
            self.tag2p_list = list(h5f[cfgid])
            self.tag_2p = tag_2p if tag_2p in self.tag2p_list else None
            if self.tag_2p is None: raise ValueError(f"The chosen value for tag_2p is not allowed, please chose a value from the following list: {self.tag2p_list}")

            #for the momentum we choose the same as the 3p correlator
            self.momentum_2p = self.momentum

            #we store the time extent of the 2 point correlator we have
            self.latticeT = len(h5f[cfgid][self.tag_2p][self.smearing][self.mass][self.hadron][self.momentum_2p]) #momentum is in the right format to read the 2 point correlator


        ## We perform an extra check on the ids of the configuration to be sure that they are the same for the 2 point and the 3 point correlators

        #we first grep the ids of 2 and 3 point correlators
        ids_2p = [file.name.split("/")[-1].split(".")[1] for file in files_2p]
        ids_3p = [file.name.split("/")[-1].split(".")[1] for file in files_2p]

        #we raise an error if the configuration ids are not equal for 2 point and 3 point correlators
        if ids_2p != ids_3p:
            raise RuntimeError(f"\nAchtung: the 2 point and 3 point correlators in the dataset do not have the same configuration ids.\n2 point ids: {ids_2p}\n3 point ids: {ids_2p}\n")


        ## We now decide whether we can avoid reading the whole dataset by looking into the fast access folder

        #first we construct a dictionary with all the parameters related to this particular reading
        self.params_dict = {'p3_folder': self.p3_folder, 'p2_folder': self.p2_folder,
                       'nconf': self.nconf, 'T_list': self.T_list, 'skip3p': skip3p, 
                       'tag_3p': self.tag_3p, 'smearing': self.smearing, 'mass': self.mass, 'hadron': self.hadron, 'momentum': self.momentum, 'insmomentum': self.insmomentum,
                       'tag_2p': self.tag_2p, 'momentum_2p': self.momentum_2p
                       }
        
        #the name of the files where we store params, p2_corr and p3_corr are
        self.params_file_name = "params"
        self.p2corr_file_name = "p2_corr"
        self.p3corr_file_prefix = "p3_corr"

        #we store also the folder where we are going to save the fast access dataset
        self.fast_data_folder = fast_data_folder
        
        #Case 1: all the conditions to avoid reading from the whole dataset are met
        if  force_reading==False  and  Path(f"{fast_data_folder}/{self.params_file_name}.json").exists()  and  self.params_dict==json.loads(open(f"{fast_data_folder}/{self.params_file_name}.json").read()) :
            
            #info print
            if verbose:
                print("\nReading the 2 point and 3 point correlators from the fast access dataset ...\n")

            #we read the 3 point correlators in the fast way

            #we instantiate a dict for the 3p correlators
            self.bb_array_dict = {}

            #we loop over the source-sink separations T (loop over the different 3 points correlators)
            for iT, T in enumerate(self.T_list):

                #we read the 3 point correlators
                self.bb_array_dict[T] = np.load(f"{fast_data_folder}/{self.p3corr_file_prefix}_T{T}.npy")


            #we read the 2 point correlator in the fast way
            self.p2_corr = np.load(f"{fast_data_folder}/{self.p2corr_file_name}.npy")

        #Case 2: not all the conditions are met and we have to read again from the main dataset
        else:

            #info print
            if verbose:
                print("\nReading the 2 point and 3 point correlators from the complete dataset ...\n")

            ## Now we loop 3p files and over the configurations and we convert the nconf h5 files into a single dictionary

            #we instantiate a dict for the 3p correlators
            self.bb_array_dict = {}

            #we loop over the files
            for iT, T in enumerate(self.T_list):


                #info print
                if verbose:
                    print(f"\n3 Points Correlators, T = {T}: looping over the configurations to read the building blocks from the h5 files...\n")

                #we select the file we have to read
                bb_folder = self.bb_pathList[iT]

                #Path file to the data folder
                p = Path(bb_folder).glob('**/*')
                files_3p = sorted( [x for x in p if x.is_file()] ) #we sort the configurations according to ascending ids

                #we initialize the np array with the building blocks (for the three-point correlators)
                bb_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndisplacements, self.ndstructures, T+1),dtype=complex)

                #we select the momentum we want to read
                momentum_3p = self.momentum+f'_T{T}'

                #we skip the reading of the 3point if the user asks for it
                if skip3p==False:

                    #we loop over the configurations
                    for iconf, file in enumerate(tqdm(files_3p[:self.nconf])):

                        #we open the h5 file corresponding too the current configuration
                        with h5.File(bb_folder+file.name, 'r') as h5f:

                            #id of the given configuration (this is equal to firstconf)
                            cfgid = list(h5f.keys())[0]

                            #we loop over the keys that have not been set:
                            for iq, qcontent in enumerate(self.qcontent_list):
                                for idisp, displacement in enumerate(self.displacement_list):
                                    for idstruct, dstructure in enumerate(self.dstructure_list):

                                        #we read the building block and store it in the np array
                                        bb_array[iconf, iq, idisp, idstruct] = h5f[f"{cfgid}/{self.tag_3p}/{self.smearing}/{self.mass}/{self.hadron}/{qcontent}/{momentum_3p}/{displacement}/{dstructure}/{self.insmomentum}"] #TO DO: conf index last

                #we store the 3p correlators in a dict of the type T:3p_corr
                self.bb_array_dict[T] = bb_array[:]



            ## Now we now read the 2p correlator

            #info print
            if verbose:
                print("\n2 Points Correlators: looping over the configurations to read the h5 files...\n")

            #we initialize the np array with the 2 point correlators
            self.p2_corr = np.zeros(shape=(self.nconf, self.latticeT),dtype=complex)

            #we loop over the configurations
            for iconf, file in enumerate(tqdm(files_2p[:self.nconf])):

                #we open the h5 file corresponding too the current configuration
                with h5.File(p2_folder+file.name, 'r') as h5f:

                    #id of the given configuration (this is equal to firstconf)
                    cfgid = list(h5f.keys())[0]

                    #we read the 2-point correlator
                    self.p2_corr[iconf] = h5f[f"{cfgid}/{self.tag_2p}/{self.smearing}/{self.mass}/{self.hadron}/{self.momentum_2p}"]


            ## We now store the reduced database into a folder for later use (to have a faster access to the data)

            #we store the variable where we want the fast data to be saved
            self.fast_data_folder = fast_data_folder
            #we create such folder if it does not exist
            Path(fast_data_folder).mkdir(parents=True, exist_ok=True)

            #we store the parameter dict into a json file
            with open(f"{fast_data_folder}/{self.params_file_name}.json", "w") as f:
                #Write it to file
                json.dump(self.params_dict, f)

            #we save the 2 point correlator to file
            np.save(f"{fast_data_folder}/{self.p2corr_file_name}.npy", self.p2_corr)

            #we save the 3 point correlators to file
            for T, corr in self.bb_array_dict.items():
                np.save(f"{fast_data_folder}/{self.p3corr_file_prefix}_T{T}.npy", corr)

            
        ## We construct first the numer vectors of the momentum and of the insertion momentum and we store them to class variables

        #we initialize the two vectors as empty lists
        self.n_P_vec = np.zeros(shape=(3,), dtype=int) #number vectors
        self.n_q_vec = np.zeros(shape=(3,), dtype=int)
        self.P_vec = np.zeros(shape=(3,), dtype=float) #actual momentum vectors (in lattice units)
        self.q_vec = np.zeros(shape=(3,), dtype=float)

        #we construct the P vector and the q vector from the value of the momentum string (something like 'PX-2_PY0_PZ0' and 'qx-1_qy-1_qz-1')
        for i, (Pi,qi) in enumerate(zip(self.momentum.split('_'), self.insmomentum.split('_'))):
            self.n_P_vec[i] = int( Pi[2:] )
            self.n_q_vec[i] = int( qi[2:] )

        #we then construct the 3-vectors for P and q just by adding the right prefactors
        self.P_vec = (2 * np.pi / self.latticeT ) * self.n_P_vec
        self.q_vec = (2 * np.pi / self.latticeT ) * self.n_q_vec




    #function used to print the information regarding the keys in the dictionary
    def print_keys(self) -> None:
        """
        Function printing to screen the specifics of the building block class instance
        """


        #First we print the currently chosen values for the keys
        print("\nSelected Keys:\n")
        print(f"    -tag_3point: {self.tag_3p}")
        print(f"    -smearing: {self.smearing}")
        print(f"    -mass: {self.mass}")
        print(f"    -hadron: {self.hadron}")
        print(f"    -momentum: {self.momentum}")
        print(f"    -insmomentum: {self.insmomentum}")
        print(f"    -tag_2point: {self.tag_2p}")


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
        print(f"insmomentum_list: {self.insmomentum_list}")
        print(f"tag2p_list: {self.tag2p_list}")

        #additional warning message for the user
        print("\nACHTUNG: not every combination of the above keys may be available\n")

    
    #function used to change the keys and reread the h5 files accordingly
    def update_keys(self, tag_3p:str|None=None, smearing:str|None=None, mass:str|None=None, hadron:str|None=None, momentum:str|None=None, insmomentum:str|None=None,
                    tag2p:str|None=None, verbose:bool=False) -> None:
        """
        Update reading keys by specifing them again here.

        Input:
            - tag_3p: new tag key for the 3 points correlators
            - smearing: new smearing key 
            - momentum: new momentum key
            - insmomentum: new insertion momentum key
            - tag2p: new tag key for the two point correlators
            - verbose: bool, if True info print are given while updating the keys

        Output:
            - None (the keys are updated and the h5 files are read again)
        """

        #we update the keys the user wants to change

        #auxiliary lists
        keys = [tag_3p,smearing,mass,hadron,momentum,insmomentum,tag2p]
        keys_list = [self.tag_list, self.smearing_list, self.mass_list, self.hadron_list, self.momentum_list, self.insmomentum_list,self.tag2p_list]
        keys_name = ['tag_3p','smearing','mass','hadron','momentum','insmomentum', 'tag_2p']

        #flag used to signal an update in the keys
        update=False

        #we loop over the keys and update them if the user has specified a new value
        for i,k in enumerate(keys):
            if k in keys_list[i]:
                match i:
                    case 0:
                        self.tag_3p = k
                    case 1:
                        self.smearing = k
                    case 2:
                        self.mass = k
                    case 3:
                        self.hadron = k
                    case 4:
                        self.momentum = k
                        self.momentum_2p = '_'.join(self.momentum.split('_')[:-1])
                        self.params_dict["momentum_2p"] = self.momentum_2p #we update the dict
                    case 5:
                        self.insmomentum = k
                    case 6:
                        self.tag_2p = k               #TO DO: this case shoulds be treated separately, as in this case only the 2point correlators need to be read again
                update=True
                self.params_dict[keys_name[i]] = k #we update the dict
            elif k is not None:
                raise ValueError(f"Error: {keys_name[i]} not valid (look at the available keys with the print_keys method)")
        
        
        #if one of the keys has been updated we reread the h5 files
        if update is True:

            #info print
            if verbose:
                print("\nReading the building blocks from the h5 files according to the new keys...\n")


            ## this is a copy of the reading procedure from the __init__ function


            #we loop over the files
            for iT, T in enumerate(self.T_list):

                #we select the file we have to read
                bb_folder = self.bb_pathList[iT]

                #Path file to the data folder
                p = Path(bb_folder).glob('**/*')
                files_3p = sorted( [x for x in p if x.is_file()] )

                bb_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndisplacements, self.ndstructures, T+1),dtype=complex)

                #we select the momentum we want to read
                momentum = self.momentum+f'_T{T}'

                #we loop over the configurations
                for iconf, file in enumerate(tqdm(files_3p[:self.nconf])):

                    #we open the h5 file corresponding too the current configuration
                    with h5.File(bb_folder+file.name, 'r') as h5f:

                        #id of the given configuration (this is equal to firstconf)
                        cfgid = list(h5f.keys())[0]

                        #we loop over the keys that have not been set:
                        for iq, qcontent in enumerate(self.qcontent_list):
                            for idisp, displacement in enumerate(self.displacement_list):
                                for idstruct, dstructure in enumerate(self.dstructure_list):

                                    #we read the building block and store it in the np array
                                    bb_array[iconf, iq, idisp, idstruct] = h5f[cfgid][self.tag_3p][self.smearing][self.mass][self.hadron][qcontent][momentum][displacement][dstructure][self.insmomentum]

                #we store the 3p correlators in a dict of the type T:3p_corr
                self.bb_array_dict[T] = bb_array[:]


            ## same procedure for the 2point correlator, with few adjustments

            #info print
            if verbose:
                print("\nLooping over the configurations to read the 2-point correlators from the h5 files...\n")

            #Path file to the data folder
            p = Path(self.p2_folder).glob('**/*')
            files_2p = sorted( [x for x in p if x.is_file()] ) #we sort the configurations, so that index by index the configurations match the ones of the 3 points correlators

            #we loop over the configurations
            for iconf, file in enumerate(tqdm(files_2p[:self.nconf])):

                #we open the h5 file corresponding too the current configuration
                with h5.File(self.p2_folder+file.name, 'r') as h5f:

                    #id of the given configuration (this is equal to firstconf)
                    cfgid = list(h5f.keys())[0]

                    #we read the 2-point correlator
                    self.p2_corr[iconf] = h5f[cfgid][self.tag_2p][self.smearing][self.mass][self.hadron][self.momentum_2p]


            ## We also update the reduced database

            #we store the parameter dict into a json file
            with open(f"{self.fast_data_folder}/{self.params_file_name}.json", "w") as f:
                #Write it to file
                json.dump(self.params_dict, f)

            #we save the 2 point correlator to file
            np.save(f"{self.fast_data_folder}/{self.p2corr_file_name}.npy", self.p2_corr)

            #we save the 3 point correlators to file
            for T, corr in self.bb_array_dict.items():
                np.save(f"{self.fast_data_folder}/{self.p3corr_file_prefix}_T{T}.npy", corr)



    #The below functions to compute the covariant derivatives are based on three rules connecting the gauge link
    #expression of the covariant derivatives to the displacements expression
    #   1) the order of the mu indices is reversed when passing from gauge links to displacements
    #   2) U = forward = capital mu, U dagger = backward = small mu
    #   3) the time argument of the displacement is that of the psibar part of the gauge link expression,
    #      so whenever psibar is shifted in time also the displacements must be, to do that in the code we
    #      use np.roll(), which we use to shift the time axis accordingly, so e.g. to get the t+1 argument
    #      we have to roll with a shift of -1 (so that t+1 is now in the t position) and vice vers
    #      (the use of roll is possible because we leverage PBC conditions in time)



    #function used to construct the right covariant derivative of the building blocks (with the chosen keys)
    def covD_r1(self, T:int) -> np.ndarray:

        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4), 4 being the dimensionality of the index mu of the covariant derivative
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we fetch the 3p correlator corresponding to the given T
        bb_array = self.bb_array_dict[T]

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_r1_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, T+1,4),dtype=complex) #the last dimension is the mu index of the covariant derivative

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
            covD_r1_array[:, :, :, :, i] = 1/2 * ( bb_array[:, :, idisp1, :, :] - bb_array[:, :, idisp2, :, :] )

        #we return the right covariant derivative
        return covD_r1_array
    


    #function used to construct the left covariant derivative of the building blocks (with the chosen keys)
    def covD_l1(self, T:int) -> np.ndarray:

        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4), 4 being the dimensionality of the index mu of the covariant derivative
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we fetch the 3p correlator corresponding to the given T
        bb_array = self.bb_array_dict[T]

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_l1_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, T+1,4),dtype=complex) #the last dimension is the mu index of the covariant derivative

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

            #smart of way of getting the shift only for i==3
            covD_l1_array[:, :, :, :, i] = 1/2 * ( np.roll(bb_array[:, :, idisp1, :, :], shift=-(i//3), axis=-1) - np.roll(bb_array[:, :, idisp2, :, :], shift=i//3, axis=-1) ) #axis=-1 is the time axis
            

        #we return the right covariant derivative
        return covD_l1_array



    #function used to construct the double right-right covariant derivative of the building blocks
    def covD_r2(self, T:int) -> np.ndarray:
        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4, 4), 4 being the dimensionality of each one of the two indices of the covariant derivatives
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we fetch the 3p correlator corresponding to the given T
        bb_array = self.bb_array_dict[T]

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_r2_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, T+1, 4, 4),dtype=complex) #the last dimensions are the mu,vu indices of the covariant derivatives

        #we now loop over the two indices of the double covariant derivative
        for i1, mu1 in enumerate(mu_list):
            for i2, mu2 in enumerate(mu_list):

                #the displacements we have to use are the following
                disp1 = 'l2_'+mu2.upper()+mu1.upper()
                if i1!=i2:                                  #the mixed cases have to be treated with care: tT,Tt,xX,Xx,.. ecc. all give rise to UUdagger=1, hence l0_
                    disp2 = 'l2_'+mu2.lower()+mu1.upper()
                    disp3 = 'l2_'+mu2.upper()+mu1.lower()
                else:
                    disp2 = 'l0_'
                    disp3 = 'l0_'
                disp4 = 'l2_'+mu2.lower()+mu1.lower()

                #with indices given by
                idisp1 = self.displacement_list.index(disp1)
                idisp2 = self.displacement_list.index(disp2)
                idisp3 = self.displacement_list.index(disp3)
                idisp4 = self.displacement_list.index(disp4)

                 #knowing which displacements to use we can now compute the right-right covariant derivative as follows

                #       conf,quarks,dstruct,T,mu            conf,quarks,displacements,dstruct,T
                covD_r2_array[:, :, :, :, i1, i2] = 1/4 * ( bb_array[:, :, idisp1, :, :] - bb_array[:, :, idisp2, :, :] - bb_array[:, :, idisp3, :, :] + bb_array[:, :, idisp4, :, :] )

        #we return the right-right covariant derivative
        return covD_r2_array

    
    #function used to construct the double left-left covariant derivative of the building blocks
    def covD_l2(self, T:int) -> np.ndarray:
        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4, 4), 4 being the dimensionality of each one of the two indices of the covariant derivatives
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we fetch the 3p correlator corresponding to the given T
        bb_array = self.bb_array_dict[T]

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_l2_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, T+1, 4, 4),dtype=complex) #the last dimensions are the mu,vu indices of the covariant derivatives

        #we now loop over the two indices of the double covariant derivative
        for i1, mu1 in enumerate(mu_list):
            for i2, mu2 in enumerate(mu_list):

                #the displacements we have to use are the following
                disp1 = 'l2_'+mu2.lower()+mu1.lower()
                if i1!=i2:                                  #the mixed cases have to be treated with care: tT,Tt,xX,Xx,.. ecc. all give rise to UUdagger=1, hence l0_
                    disp2 = 'l2_'+mu2.lower()+mu1.upper()
                    disp3 = 'l2_'+mu2.upper()+mu1.lower()
                else:
                    disp2 = 'l0_'
                    disp3 = 'l0_'
                disp4 = 'l2_'+mu2.upper()+mu1.upper()
                #with indices given by
                idisp1 = self.displacement_list.index(disp1)
                idisp2 = self.displacement_list.index(disp2)
                idisp3 = self.displacement_list.index(disp3)
                idisp4 = self.displacement_list.index(disp4)
                #and the related building blocks have to be shifted on the time axis by (smart way of computing the shift)
                shift1 = -(i1//3) -(i2//3)
                shift2 = +i1//3 -(i2//3)
                shift3 = -(i1//3) +i2//3
                shift4 = +i1//3 +i2//3

                 #knowing which displacements to use we can now compute the left-left covariant derivative as follows

                #       conf,quarks,dstruct,T,mu            conf,quarks,displacements,dstruct,T
                covD_l2_array[:, :, :, :, i1, i2] = 1/4 * ( np.roll( bb_array[:, :, idisp1, :, :], shift=shift1, axis=-1) \
                                                     -  np.roll( bb_array[:, :, idisp2, :, :], shift=shift2, axis=-1) \
                                                     -  np.roll( bb_array[:, :, idisp3, :, :], shift=shift3, axis=-1) \
                                                     +  np.roll( bb_array[:, :, idisp4, :, :], shift=shift4, axis=-1) )
        #we return the left-left covariant derivative
        return covD_l2_array
    


    #function used to construct the double left-right covariant derivative of the building blocks
    def covD_l1_r1(self, T:int) -> np.ndarray:
        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4, 4), 4 being the dimensionality of each one of the two indices of the covariant derivatives
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we fetch the 3p correlator corresponding to the given T
        bb_array = self.bb_array_dict[T]

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_l1_r1_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, T+1, 4, 4),dtype=complex) #the last dimensions are the mu,vu indices of the covariant derivatives

        #we now loop over the two indices of the double covariant derivative
        for i1, mu1 in enumerate(mu_list):
            for i2, mu2 in enumerate(mu_list):

                #the displacements we have to use are the following
                disp1 = 'l2_'+mu2.upper()+mu1.upper()
                if i1!=i2:                                  #the mixed cases have to be treated with care: tT,Tt,xX,Xx,.. ecc. all give rise to UUdagger=1, hence l0_
                    disp2 = 'l2_'+mu2.upper()+mu1.lower()
                    disp3 = 'l2_'+mu2.lower()+mu1.upper()
                else:
                    disp2 = 'l0_'
                    disp3 = 'l0_'
                disp4 = 'l2_'+mu2.lower()+mu1.lower()
                #with indices given by
                idisp1 = self.displacement_list.index(disp1)
                idisp2 = self.displacement_list.index(disp2)
                idisp3 = self.displacement_list.index(disp3)
                idisp4 = self.displacement_list.index(disp4)
                #and the related building blocks have to be shifted on the time axis by (smart way of computing the shift)
                shift1 = +i1//3
                shift2 = -(i1//3)
                shift3 = +i1//3
                shift4 = -(i1//3)

                 #knowing which displacements to use we can now compute the left-left covariant derivative as follows

                #       conf,quarks,dstruct,T,mu            conf,quarks,displacements,dstruct,T
                covD_l1_r1_array[:, :, :, :, i1, i2] = - 1/4 * ( np.roll( bb_array[:, :, idisp1, :, :], shift=shift1, axis=-1) \
                                                            -  np.roll( bb_array[:, :, idisp2, :, :], shift=shift2, axis=-1) \
                                                            -  np.roll( bb_array[:, :, idisp3, :, :], shift=shift3, axis=-1) \
                                                            +  np.roll( bb_array[:, :, idisp4, :, :], shift=shift4, axis=-1) )
        #we return the left-left covariant derivative
        return covD_l1_r1_array
    

    #function used to construct the double right-left covariant derivative of the building blocks
    def covD_r1_l1(self, T:int) -> np.ndarray:
        """
        The function takes no argument since the keys are all specified and the combination of displacements to be used is fixed

        The output is an object with shape (nconf, nquarks, ndstructures, T, 4, 4), 4 being the dimensionality of each one of the two indices of the covariant derivatives
        
        (the code assumes the momentum to be 0)
        """

        #list with the mu indices
        mu_list = ['x','y','z','t']

        #we fetch the 3p correlator corresponding to the given T
        bb_array = self.bb_array_dict[T]

        #we instatiate the np array where we will store the covariant derivative of the building blocks
        covD_r1_l1_array = np.zeros(shape=(self.nconf, self.nquarks, self.ndstructures, T+1, 4, 4),dtype=complex) #the last dimensions are the mu,vu indices of the covariant derivatives

        #we now loop over the two indices of the double covariant derivative
        for i1, mu1 in enumerate(mu_list):
            for i2, mu2 in enumerate(mu_list):

                #the displacements we have to use are the following
                disp1 = 'l2_'+mu2.upper()+mu1.upper()
                if i1!=i2:                                  #the mixed cases have to be treated with care: tT,Tt,xX,Xx,.. ecc. all give rise to UUdagger=1, hence l0_
                    disp2 = 'l2_'+mu2.lower()+mu1.upper()
                    disp3 = 'l2_'+mu2.upper()+mu1.lower()
                else:
                    disp2 = 'l0_'
                    disp3 = 'l0_'
                disp4 = 'l2_'+mu2.lower()+mu1.lower()
                #with indices given by
                idisp1 = self.displacement_list.index(disp1)
                idisp2 = self.displacement_list.index(disp2)
                idisp3 = self.displacement_list.index(disp3)
                idisp4 = self.displacement_list.index(disp4)
                #and the related building blocks have to be shifted on the time axis by (smart way of computing the shift)
                shift1 = +i2//3
                shift2 = -(i2//3)
                shift3 = +i2//3
                shift4 = -(i2//3)

                 #knowing which displacements to use we can now compute the left-left covariant derivative as follows

                #       conf,quarks,dstruct,T,mu            conf,quarks,displacements,dstruct,T
                covD_r1_l1_array[:, :, :, :, i1, i2] = - 1/4 * ( np.roll( bb_array[:, :, idisp1, :, :], shift=shift1, axis=-1) \
                                                             -  np.roll( bb_array[:, :, idisp2, :, :], shift=shift2, axis=-1) \
                                                             -  np.roll( bb_array[:, :, idisp3, :, :], shift=shift3, axis=-1) \
                                                             +  np.roll( bb_array[:, :, idisp4, :, :], shift=shift4, axis=-1) )
        #we return the left-left covariant derivative
        return covD_r1_l1_array





    #function used to obtain the building block of the relevant operators
    def get_bb(self, T:int, X:str, isospin:str, n_mu:int) -> np.ndarray:
        """
        Input:
            - X: either 'V', 'A' or 'T' (for vector, axial or tensorial operators)
            - isospin: either 'U', 'D', 'U+D' or 'U-D' (with U and D meaning up and down quark)
            - n_mu: the number of mu indices of the operator (either 1 or 2) (the indices after the ones due to the gamma matrices, i.e. the number of covariant derivatives) !!!! TO DO: add support for n_mu=0 

        Output:
            - the building block (a np array) of the operator with the features specified in input
              (if normalize==True then the output is the ratio giving the matrix element)
        """

        #Input check

        #input check on X
        if X not in self.X_list:
            raise ValueError(f"Error: the value of X must be one in the list {self.X_list}, but instead X={X} was chosen.")
        
        #input check on isospin
        if isospin not in self.isospin_list:
            raise ValueError(f"The isospin value must be one in the list {self.isospin_list}, but instead isospin={isospin} was chosen.")
 
        #input check on n_mu
        if n_mu not in self.n_mu_list:
            raise ValueError(f"Error: the value of n_mu must be one of the values in the list {self.n_mu_list}, but instead n_mu={n_mu} was chosen.")
        

        #we fetch the 3p correlator corresponding to the given T
        #bb_array = self.bb_array_dict[T]


        #We now initialize the np array where we will store the building block of the operator

        #the number of indices of the operator depends on X and n_mu
        n_indices = 1+n_mu #1 gamma matrix + n_mu indices (if X is 'V' or 'A')
        if X == 'T':
            n_indices += 1 #2 gamma matrices + n_mu indices (if X is 'T')

        #we now instantiate the output array
        #the first two axis are nconf and time, the other ones are the indices of the operator, and they have dimensionality 4
        bb_operator = np.zeros(shape=(self.nconf, T+1, ) + (4,)*n_indices, dtype=complex)


        #We take now care of the quark content

        #first thing first we retrieve the right minus left covariant derivative
        if n_mu==1:
            covD = 1/2 * ( self.covD_r1(T) - self.covD_l1(T) ) #shape = (nconf, nquarks, ndstructures, T, 4), the last dimension being the index of the covariant derivative
        elif n_mu==2:
            covD = 1/4 * ( self.covD_l2(T) + self.covD_r2(T) - self.covD_r1_l1(T) - self.covD_l1_r1(T) ) #shape = (nconf, nquarks, ndstructures, T, 4, 4)

        #in the following we can just ignore the dimensionalities due to the mu index and everything is the same for the n+mu =1 and =2 case
        
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
            covD = covD[:,qcontent_U,:,:,:] - covD[:,qcontent_D,:,:,:]      #now covD has shape (nconf, ndstructures, T, 4) (extra 4 if n_mu=2)



        
        #We now proceed with the actual construction of the building block, that depends on the number of covariant derivatives (=n_mu)

        ##case with 1 covariant derivative
        #if n_mu==1:
            
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

                #we look at what is the result of the product gamma_mu * gamma5 (both in overall sign and in gamma structure) #TO DO: check convention: gamma gamma5 or gamma5 gamma ? 
                sign_result, list_result = gamma_prod( gamma_to_list[gamma], gamma_to_list[gamma5]  )
                #sign_result, list_result = gamma_prod( gamma_to_list[gamma5], gamma_to_list[gamma]  )

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



        ##case with 2 covariant derivatives
        #elif n_mu==2:
        #    pass #TO BE IMPLEMENTED

        
        #This was wrong: before this normalization the gauge average should be performed
        ##we normalize the operator if the user requested so (in this case the output will be the ratio giving the matrix element)
        #if normalize==True:
        #    for iconf in range(self.nconf):
        #        bb_operator[iconf] /= self.p2_corr[iconf,T].real #to obtain the matrix element we have to divide by the 2pcorr computed at the sink position (= to source-sink separation, being the sink at t=0) #TO DO: check cast to real here


        #to do: reverse index


        #we return the building block as calculated
        return bb_operator
    


################ Auxiliary Functions ###############################

#function used to implement the product of two gamma matrices
def gamma_prod(g1: list[int], g2: list[int]) -> tuple[int, list]:

    """
    Function implementing the product of two gamma matrices

    Input:
        - g1 and g2: lists of 4 bits (with the rightmost being the most significat)

    Output:
        - (sign, gamma structure): a tuple, where the gamma structure is in the same format (list of 4 bits)
    """

    #the gamma multiplication is just a bit by bit addition modulo 2
    gstructure = [ (e1+e2)%2 for e1,e2 in zip(g1,g2)]

    #the sign is the tricky part, to compute it we have to count the number of swaps occuring
    sign = 0
    for i,e1 in enumerate(g1):
        sign += e1 * sum(g2[:i]) #the multiplication to e1 is equivalent to count the swaps only when e1 is 1 (since e1 can only be 0 or 1)

    return (-1)**sign, gstructure


#function translating one list encoding the gamma structure into the relevant key
def list_to_key(list: list[int]) -> str:
    """
    Function translating a gamma structure to the related key to access the dataset.
    
    Input:
        - list: the list with the gamma structure (e.g. [0,1,0,1])
        
    Output
        - gamma_key: the key to access the dictionary (e.g. 'g10')
    """

    #we do the conversion and return the key
    return 'g'+ str(int( ''.join([str(e) for e in reversed(list)]), 2))