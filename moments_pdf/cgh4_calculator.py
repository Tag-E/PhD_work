######################################################
## cgh4_calculator.py                               ##
## created by Emilio Taggi - 2024/12/09             ##
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
import sympy as sym #to handle symbolic computations
from tqdm import tqdm #for a nice view of for loops with loading bars
from pathlib import Path #to check whether directories exist or not

######################## Global Variables ###############################


## specifics of the H(4) irreps ##

#number of elements in S(4) = 4!
n_ele_s4 = 4*3*2

#number of elements in H(4) = 4! * 2**4
n_ele_h4 = 2**4 * 4*3*2

#number of irreps of H(4) (and also of S(4))
n_rep = 20

#dimensionality of the irreps
rep_dim_list = [1,1,1,1,2,2,3,3,3,3,4,4,4,4,6,6,6,6,8,8]

#index corresponding to the fundamental representation irrep (i.e. the (4,1) irrep)
fund_index = 10

#list with the representations (k,l) ordered: k is the dimensionality, l the index of the l-th k-dimensional irrep
rep_label_list = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4),(6,1),(6,2),(6,3),(6,4),(8,1),(8,2)]

#latex labels of the irrep
rep_latex_names = [f"\tau^{i[0]}_{i[1]}" for i in rep_label_list]

#dictionaries with irreps specifics
irrep_index = dict(zip(rep_label_list, range(n_rep))) #each irrep has an index
irrep_dim = dict(zip(rep_label_list, rep_dim_list)) #a dimensionality
irrep_texname  = dict(zip(rep_label_list, rep_latex_names)) #and a character in latex


#character table of H(4)
char_table = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, 1],
    [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
    [2, 2, 0, 0, 2, -1, 0, 0, 2, -1, 0, 0, 2, 2, 0, -1, -1, 2, 0, 2],
    [2, -2, 0, 0, 2, 1, 0, 0, -2, -1, 0, 0, 2, -2, 0, -1, 1, 2, 0, 2],
    [3, 3, 1, 1, 3, 0, 1, 1, 3, 0, -1, -1, -1, -1, 1, 0, 0, -1, 1, 3],
    [3, -3, -1, 1, 3, 0, 1, -1, -3, 0, 1, -1, -1, 1, -1, 0, 0, -1, 1, 3],
    [3, 3, -1, -1, 3, 0, -1, -1, 3, 0, 1, 1, -1, -1, -1, 0, 0, -1, -1, 3],
    [3, -3, 1, -1, 3, 0, -1, 1, -3, 0, -1, 1, -1, 1, 1, 0, 0, -1, -1, 3],
    [4, 2, 2, 2, 0, 1, 0, 0, -2, 1, 0, 0, 0, 0, -2, -1, -1, 0, -2, -4],
    [4, -2, -2, 2, 0, -1, 0, 0, 2, 1, 0, 0, 0, 0, 2, -1, 1, 0, -2, -4],
    [4, 2, -2, -2, 0, 1, 0, 0, -2, 1, 0, 0, 0, 0, 2, -1, -1, 0, 2, -4],
    [4, -2, 2, -2, 0, -1, 0, 0, 2, 1, 0, 0, 0, 0, -2, -1, 1, 0, 2, -4],
    [6, 0, 2, 0, -2, 0, 0, -2, 0, 0, 0, 0, 2, 0, 2, 0, 0, -2, 0, 6],
    [6, 0, -2, 0, -2, 0, 0, 2, 0, 0, 0, 0, 2, 0, -2, 0, 0, -2, 0, 6],
    [6, 0, 0, 2, -2, 0, -2, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 2, 2, 6],
    [6, 0, 0, -2, -2, 0, 2, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 2, -2, 6],
    [8, 4, 0, 0, 0, -1, 0, 0, -4, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, -8],
    [8, -4, 0, 0, 0, 1, 0, 0, 4, -1, 0, 0, 0, 0, 0, 1, -1, 0, 0, -8]
    ],dtype=int)

#number of elements in the 20 conjugacy classes of H(4)
class_orders = [1,4,12,12,6,32,24,24,4,32,48,48,12,24,12,32,32,12,12,1]



## matrix representations for the generators of H(4) ## 


#first for alpha
alpha_list = []

#(1,1)
a = np.array([1],dtype=float)
alpha_list.append(a)
#(1,2)
a = np.array([1],dtype=float)
alpha_list.append(a)
#(1,3)
a = np.array([-1],dtype=float)
alpha_list.append(a)
#(1,4)
a = np.array([-1],dtype=float)
alpha_list.append(a)


#(2,1)
a = np.array([[1, 0],
              [0, -1]],dtype=float)
alpha_list.append(a)
#(2,2)
a = np.array([[1, 0],
              [0, 1]],dtype=float)
alpha_list.append(a)


#(3,1)
a = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, -1]],dtype=float)
alpha_list.append(a)
#(3,2)
a = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, -1]],dtype=float)
alpha_list.append(a)
#(3,3)
a = np.array([[-1, 0, 0],
              [0, -1, 0],
              [0, 0, 1]],dtype=float)
alpha_list.append(a)
#(3,4)
a = np.array([[-1, 0, 0],
              [0, -1, 0],
              [0, 0, 1]],dtype=float)
alpha_list.append(a)


#(4,1)
a = np.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]],dtype=float)
alpha_list.append(a)
#(4,2)
a = np.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]],dtype=float)
alpha_list.append(a)
#(4,3)
a = np.array([[0, -1, 0, 0],
              [-1, 0, 0, 0],
              [0, 0, -1, 0],
              [0, 0, 0, -1]],dtype=float)
alpha_list.append(a)
#(4,4)
a = np.array([[0, -1, 0, 0],
              [-1, 0, 0, 0],
              [0, 0, -1, 0],
              [0, 0, 0, -1]],dtype=float)
alpha_list.append(a)


#(6,1)
a = np.array([[-1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1]],dtype=float)
alpha_list.append(a)
#(6,2)
a = np.array([[-1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1]],dtype=float)
alpha_list.append(a)
#(6,3)
a = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1]],dtype=float)
alpha_list.append(a)
#(6,4)
a = np.array([[-1, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 0],
              [0, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, -1, 0],
              [0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0, -1]],dtype=float)
alpha_list.append(a)

#(8,1)
a = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, -1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, -1]],dtype=float)
alpha_list.append(a)
#(8,2)
a = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, -1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, -1]],dtype=float)
alpha_list.append(a)



#then for beta
beta_list = []

s2 = np.sqrt(2)
s3 = np.sqrt(3)
s6 = np.sqrt(6)
s8 = np.sqrt(8)

#(1,1)
b = np.array([1],dtype=float)
beta_list.append(b)
#(1,2)
b = np.array([1],dtype=float)
beta_list.append(b)
#(1,3)
b = np.array([-1],dtype=float)
beta_list.append(b)
#(1,4)
b = np.array([-1],dtype=float)
beta_list.append(b)


#(2,1)
b = np.array([[-1/2, -s3/2],
              [-s3/2, 1/2]],dtype=float)
beta_list.append(b)
#(2,2)
b = np.array([[-1/2, -s3/2],
              [-s3/2, 1/2]],dtype=float)
beta_list.append(b)


#(3,1)
b = np.array([[-1/3, s8/3, 0],
              [-s2/3, -1/6, s3/2],
              [-s6/3, -s3/6, -1/2]],dtype=float)
beta_list.append(b)
#(3,2)
b = np.array([[-1/3, s8/3, 0],
              [-s2/3, -1/6, s3/2],
              [-s6/3, -s3/6, -1/2]],dtype=float)
beta_list.append(b)
#(3,3)
b = np.array([[1/3, -s8/3, 0],
              [s2/3, 1/6, -s3/2],
              [s6/3, s3/6, 1/2]],dtype=float)
beta_list.append(b)
#(3,4)
b = np.array([[1/3, -s8/3, 0],
              [s2/3, 1/6, -s3/2],
              [s6/3, s3/6, 1/2]],dtype=float)
beta_list.append(b)


#(4,1)
b = np.array([[0, 0, 0, 1],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]],dtype=float)
beta_list.append(b)
#(4,2)
b = np.array([[0, 0, 0, 1],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0]],dtype=float)
beta_list.append(b)
#(4,3)
b = np.array([[0, 0, 0, -1],
              [-1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, -1, 0]],dtype=float)
beta_list.append(b)
#(4,4)
b = np.array([[0, 0, 0, -1],
              [-1, 0, 0, 0],
              [0, -1, 0, 0],
              [0, 0, -1, 0]],dtype=float)
beta_list.append(b)


#(6,1)
b = np.array([[0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, -1, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1]],dtype=float)
beta_list.append(b)
#(6,2)
b = np.array([[0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, -1, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1]],dtype=float)
beta_list.append(b)
#(6,3)
b = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1]],dtype=float)
beta_list.append(b)
#(6,4)
b = np.array([[0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, -1, 0],
              [-1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, -1],
              [0, -1, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 1]],dtype=float)
beta_list.append(b)

#(8,1)
b = np.array([[0, 0, 0, -1/2, 0, 0, 0, -s3/2],
              [-1/2, 0, 0, 0, -s3/2, 0, 0, 0],
              [0, -1/2, 0, 0, 0, -s3/2, 0, 0],
              [0, 0, -1/2, 0, 0, 0, -s3/2, 0],
              [0, 0, 0, -s3/2, 0, 0, 0, 1/2],
              [-s3/2, 0, 0, 0, 1/2, 0, 0, 0],
              [0, -s3/2, 0, 0, 0, 1/2, 0, 0],
              [0, 0, -s3/2, 0, 0, 0, 1/2, 0]],dtype=float)
beta_list.append(b)
#(8,2)
b = np.array([[0, 0, 0, -1/2, 0, 0, 0, -s3/2],
              [-1/2, 0, 0, 0, -s3/2, 0, 0, 0],
              [0, -1/2, 0, 0, 0, -s3/2, 0, 0],
              [0, 0, -1/2, 0, 0, 0, -s3/2, 0],
              [0, 0, 0, -s3/2, 0, 0, 0, 1/2],
              [-s3/2, 0, 0, 0, 1/2, 0, 0, 0],
              [0, -s3/2, 0, 0, 0, 1/2, 0, 0],
              [0, 0, -s3/2, 0, 0, 0, 1/2, 0]],dtype=float)
beta_list.append(b)





#same thing for gamma
gamma_list = []

#(1,1)
g = np.array([1],dtype=float)
gamma_list.append(g)
#(1,2)
g = np.array([-1],dtype=float)
gamma_list.append(g)
#(1,3)
g = np.array([1],dtype=float)
gamma_list.append(g)
#(1,4)
g = np.array([-1],dtype=float)
gamma_list.append(g)


#(2,1)
g = np.diag([1,1])
gamma_list.append(g)
#(2,2)
g = np.diag([-1,-1])
gamma_list.append(g)


#(3,1)
g = np.diag([1,1,1])
gamma_list.append(g)
#(3,2)
g = np.diag([-1,-1,-1])
gamma_list.append(g)
#(3,3)
g = np.diag([1,1,1])
gamma_list.append(g)
#(3,4)
g = np.diag([-1,-1,-1])
gamma_list.append(g)


#(4,1)
g = np.diag([-1,1,1,1])
gamma_list.append(g)
#(4,2)
g = np.diag([1,-1,-1,-1])
gamma_list.append(g)
#(4,3)
g = np.diag([-1,1,1,1])
gamma_list.append(g)
#(4,4)
g = np.diag([1,-1,-1,-1])
gamma_list.append(g)


#(6,1)
g = np.diag([-1,-1,1,-1,1,1])
gamma_list.append(g)
#(6,2)
g = np.diag([1,1,-1,1,-1,-1])
gamma_list.append(g)
#(6,3)
g = np.diag([-1,-1,1,-1,1,1])
gamma_list.append(g)
#(6,4)
g = np.diag([1,1,-1,1,-1,-1])
gamma_list.append(g)

#(8,1)
g = np.diag([-1,1,1,1,-1,1,1,1])
gamma_list.append(g)
#(8,2)
g = np.diag([1,-1,-1,-1,1,-1,-1,-1])
gamma_list.append(g)



#list with all the inverse beta matrix representations (needed to construct the elements of H(4))
beta_inv_list = [np.linalg.inv(beta) if np.shape(beta)[0]>1 else np.array(beta) for beta in beta_list]

#we construct also the matrix responsible for the axis flip
gamma1_list = gamma_list

#other axis
gamma2_list = []
gamma3_list = []
gamma4_list = []

for ir in range(n_rep):
    b = np.reshape(beta_list[ir],(dim_rep[ir],dim_rep[ir]))
    binv = np.reshape(beta_inv_list[ir],(dim_rep[ir],dim_rep[ir]))
    g = np.reshape(gamma_list[ir],(dim_rep[ir],dim_rep[ir]))
    gamma2_list.append( binv @ binv @ binv @ g @ b @ b @ b )
    gamma3_list.append( binv @ binv @ g @ b @ b )
    gamma4_list.append( binv @ g @ b )





######################## Main Class ###############################


#class used to compute the clabsch gordan coefficients related to the H(4) group
class cg_calc:

    '''
    Create one class instance to obtain the cg coeff realted to the
    tensor product specified in input
    '''

    #global variables shared by all the class instances

    #dirs and files names

    #folders containing all the matrix representations for all the elements of H(4)
    h4_ele_folder = 'h4_ele'


    #specifics of the H(4) irreps

    #number of elements in S(4) = 4!
    n_ele_s4 = 4*3*2
    #number of elements in H(4) = 4! * 2**4
    n_ele_h4 = 2**4 * 4*3*2
    #number of irreps
    n_rep = 20
    #dimensionality of the irreps
    rep_dim_list = [1,1,1,1,2,2,3,3,3,3,4,4,4,4,6,6,6,6,8,8]
    #index corresponding to the fundamental representation irrep (i.e. the (4,1) irrep)
    fund_index = 10
    #labels of the representations
    rep_label_list = [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4),(6,1),(6,2),(6,3),(6,4),(8,1),(8,2)]
    #latex labels of the irrep
    rep_latex_names = [f"\tau^{i[0]}_{i[1]}" for i in rep_index_list]
    #dictionaries with irreps specifics
    irrep_index = dict(zip(rep_label_list, range(n_rep))) #each irrep has an index
    irrep_dim = dict(zip(rep_label_list, rep_dim_list)) #a dimensionality
    irrep_texname  = dict(zip(rep_label_list, rep_latex_names)) #and a character in latex

    def __init__(self, *kwarg, force_computation=False, force_h4gen=False, verbose=True):

        #we store in a class variable the irreps chosen for the decomposition
        self.chosen_irreps = kwarg
        
        ## first we have to either load or compute all the elements of H(4) ##

        #we do the computation if the elements are not saved or if the user force it
        if (Path(self.h4_ele_folder).exists() == False) or (force_h4gen==True): 

            #info print
            if verbose==True:
                print("\nConstructing matrix representations for all the elements of H(4) ...\n")

            #we initialize first the elements of S(4)
            self.s4_mat = [ [np.eye(d)] for d in self.rep_dim_list]

            ## Algorithm to find all the elements of S(4) from the two generators alpha and beta ##

            #variable used to start generation of new elements from a different starting point
            self.start=0

            #while we don't have all elements of s4 we continue the search
            while len(self.s4_mat[0])!=self.n_ele_s4: 
            
                #order of the two generators (2 for alpha and 4 for beta)
                for i in range(2):
                    for k in range(4):
                    
                        #we start by looking at the fundamental to see if we have a new matrix (the fundamental because it is a faithful rep)

                        #starting point
                        res = self.s4_mat[self.fund_index][self.start]

                        #we try to construct a new element
                        for _ in range(i):
                            res =   res @ alpha_list[self.fund_index]

                        for _ in range(k):
                            res =  res @ beta_list[self.fund_index]


                        #check if we have a duplicate or the matrix (group element) is new
                        duplicate = 0
                        for l in self.s4_mat[self.fund_index]:
                            if (l==res).all():
                                duplicate = 1
                                break

                        #if we don't have a duplicate we compute the new group element for each irrep
                        if duplicate == 0:
                            for ir,rep in enumerate(self.s4_mat):
                            
                                #starting point
                                res = rep[self.start]
                                res = np.reshape(res,(self.rep_dim_list[ir],self.rep_dim_list[ir]))

                                #we construct a new element
                                for _ in range(i):
                                    res =   res @ alpha_list[ir]
                                    res = np.reshape(res,(self.rep_dim_list[ir],self.rep_dim_list[ir]))

                                for _ in range(k):
                                    res =  res @ beta_list[ir]
                                    res = np.reshape(res,(self.rep_dim_list[ir],self.rep_dim_list[ir]))

                                #we add the new element to the list (the reshape is needed to make sure that 1 dimensiona irrep are saved as matrices)
                                self.s4_mat[ir].append(np.reshape(res,(self.rep_dim_list[ir],self.rep_dim_list[ir])))

                #start has to change so that at each iteration of the while loop we can try to find new elements of s4
                self.start = self.start+1


            #let's now construct the element of H(4)

            #array initialization
            self.h4_mat = [ [] for _ in range(self.n_rep)]

            #for every irrep
            for ir in range(self.n_rep):
                #for every s4 element in the given irrep
                for s4_ele in self.s4_mat[ir]:
                    #we construct 2**4 elements
                    for a1 in range(2):
                        for a2 in range(2):
                            for a3 in range(2):
                                for a4 in range(2):
                                
                                    #the starting point is the element of s4
                                    new_ele = s4_ele

                                    #we then apply the 2**4 possible axis inversions
                                    for _ in range(a1):
                                        new_ele = new_ele @ gamma1_list[ir]
                                    for _ in range(a2):
                                        new_ele = new_ele @ gamma2_list[ir]
                                    for _ in range(a3):
                                        new_ele = new_ele @ gamma3_list[ir]
                                    for _ in range(a4):
                                        new_ele = new_ele @ gamma4_list[ir]

                                    #we add the new element
                                    self.h4_mat[ir].append(new_ele)

            #once we have done the computation we save to file all the matric representations for all the elements of H(4)

            #info print
            if verbose==True:
                print("\nSaving to file the matrix representations ...\n")

            #first we loop over the representations
            for ir in range(self.n_rep):

                #for each irrep we create a folder
                rep_folder = self.h4_ele_folder + f"/{str(self.rep_label_list[ir])}"
                Path(rep_folder).mkdir(parents=True, exist_ok=True)

                #then for each irrep we save all the matrix representations of the group elements
                for i,ele in enumerate(self.h4_mat[ir]):
                    with open(f'{rep_folder}/{i}.npy', 'wb') as f:
                        np.save(f,ele)
        
        #if instead the H(4) elements are already computed we just load it
        elif Path(self.h4_ele_folder).exists() == True: 

            #info print
            if verbose==True:
                print("\nLoading matrix representations for all the elements of H(4) ...\n")

            #array initialization
            self.h4_mat = [ [] for _ in range(self.n_rep)]

            #to load the matrices first we loop over the representations
            for ir in range(self.n_rep):

                #the folder we have to look into is
                rep_folder = self.h4_ele_folder + f"/{str(self.rep_label_list[ir])}"

                #then for the current irrep we load all the 384 matrices
                for i in range(self.n_ele_h4):
                    with open(f'{rep_folder}/{i}.npy', 'rb') as f:
                         self.h4_mat[i].append( np.load(f,allow_pickle=True) )


        ## once the matrix representations for the elements of H(4) are loaded we can do the actual computation of the CG coeff ##

        #first we get the multiplicities and we store them in a class variable for later use
        self.mul_list = get_multiplicities(*kwarg)

        #we then compute the dimension of the bases
        dim=1
        for rep in self.chosen_irreps:
            dim *= irrep_dim[rep]

        #then we instantiate the auxiliary symbolic matrix
        A = sym.MatrixSymbol('A', dim, dim)

        #and we instantiate the matrix with the results
        res_mat = np.zeros((dim,dim))

        #then we start to iterate over the group elements

        #info print
        if verbose: print("\nLooping over group elements...\n")

        #loop over elements of H(4) (with a nice loading bar)
        for ih in tqdm(range(self.n_ele_h4)):

            #for each element first we find the matrix in the kronecker product form
            Tp = np.eye(1)
            for rep in kwarg:
                Tp = np.kron(Tp,self.h4_mat[irrep_index[rep]][ih])

            #then we find the matrix in the block diagonal form
            nblocks = np.sum(self.mul_list) #number of block in the matrix
            rows = [[] for _ in range(nblocks)] #array containing the rows of this matrix
            #count_rep = mul_list[:]

            #let's get a list of index of the rep to use
            rep_to_use = []
            for irep,mul in enumerate(self.mul_list):
                for _ in range(mul):
                    rep_to_use.append(irep) #in this list there is the index of the irre as many time as the multiplicity

            #let's loop over the block rows and block columns of this mat
            for ib in range(nblocks):
                #let's determine the i-th block
                rep_i = rep_to_use[ib]
                for jb in range(nblocks):
                    rep_j = rep_to_use[jb]
                    #on the diagonal we add the block
                    if ib==jb:
                        rows[ib].append(self.h4_mat[rep_i][ih])
                    else:
                        rows[ib].append(np.zeros((rep_dim_list[rep_i],rep_dim_list[rep_j]))) #off diag there are 0s blocks with the right dim

            #matrix in the block diagonal form
            Bd = np.block(rows)

            #then we do the computation
            res_mat = res_mat + Tp @ A @ Bd.T

        #we then eliminate rounding errors
        rounded_res = np.empty((dim,dim),dtype=object)

        eps=10**(-10)
        infth=10**20

        for i in range(dim):
            for j in range(dim):
                entry = sym.Add.make_args(res_mat[i,j])
                new_entry = 0
                for element in entry:
                    coeff = element.as_coeff_Mul()[0]
                    monom = element.as_coeff_Mul()[1]
                    if np.abs(coeff)<eps or np.abs(coeff)>infth :
                        coeff=0
                    new_entry = new_entry + coeff * monom
                rounded_res[i,j] = new_entry

        #we now construct the output dict
        self.cg_dict = {}
        #to obtain the output we loop over the multiplicities
        d = 0 #we take note of the current dim
        for irep,mul in enumerate(self.mul_list):
            for m in range(mul):
                if irep not in self.cg_dict.keys(): #we initialize the output as an empty array
                    self.cg_dict[irep] = []
                #then we append to the empty list the block with the CG coeff
                self.cg_dict[irep].append( CGmat_from_block( rounded_res[:,d:d+rep_dim_list[irep]], m, mul ) )
                d += rep_dim_list[irep]



    #function returning the multiplicities in the decomposition
    def get_multiplicities(self):
        return self.mul_list





######################## Auxiliary Functions ###############################

#function that given the irreps in the tensor products returns the multiplicity in their decomoposition
def get_multiplicities(*kwarg):
    #the inputs are the label of the irrep i want to multiply together
    
    #the output will be a list with the multiplicity for each irrep
    mult_list = []
    
    #implementation of formula for multiplicities (see attached documentation in pdf format, if any)
    for ir in range(n_rep):
        mul = 0
        for ic, class_ord in enumerate(class_orders):
            tmp = class_ord * char_table[ir,ic]
            for k in kwarg:
                tmp *=  char_table[irrep_index[k],ic]
            mul += tmp
        mul /= n_ele_h4
        mult_list.append(int(mul))

    return mult_list


#function to obtain the correctly normalized matrix with cg coefficients from the raw block obtained from the cg formula
def CGmat_from_block(block : np.ndarray, m = 0, mul=1) -> np.ndarray:

    #matrix in which cg coefficients hide
    mat = block.copy()

    #shape adjustement for 1 dim array
    if len(np.shape(mat))==1:
        mat = np.reshape(mat,(np.shape(mat)[0],1))

    #we now count wich monomial in the symbolic matrix appears more often in the first column
    monom_counts = {}
    monom_rows = {}
    for i in range(np.shape(mat)[0]):
        entry = sym.Add.make_args(mat[i,0])
        for element in entry:
            monom = element.as_coeff_Mul()[1]
            if (monom not in monom_counts.keys()) and (type(monom)!= sym.core.numbers.One):
                monom_counts[monom] = 1
                monom_rows[monom] = [i]
            elif (monom in monom_counts.keys()) and (type(monom)!= sym.core.numbers.One):
                monom_counts[monom] += 1
                monom_rows[monom].append(i)


    #we take note of its index because this monomial will be set to 1 
    index = min(monom_counts, key=monom_counts.get)

    #in this dict we store the number we will assign to each A monomial
    monom_dict = {}
    found = 0 #we just need to find one new element per groups of columns
    target=mul
    total_found=0
    for j in range(np.shape(mat)[1]):
        for i in range(np.shape(mat)[0]):
            entry = sym.Add.make_args(mat[i,j])
            for element in entry:
                coeff = element.as_coeff_Mul()[0]
                monom = element.as_coeff_Mul()[1]
                if (found==0) and (monom not in monom_dict.keys()) and (type(monom)!= sym.core.numbers.One):
                    monom_dict[monom] = 1.0
                    found = 1
                    total_found+=1
                elif (found==1) and (monom not in monom_dict.keys()):
                    monom_dict[monom] = 0.0
            if total_found<target:
                found=0

    newmat = mat.copy()

    index = min(monom_counts, key=monom_counts.get)
    index_list=[]
    index_list.append(min(monom_counts, key=monom_counts.get))
    for _ in range(target-1):
        for k in monom_counts:
            if monom_rows[k] == monom_rows[index]:
                monom_counts[k] = max(monom_counts.values())+1
        index = min(monom_counts, key=monom_counts.get)
        index_list.append(min(monom_counts, key=monom_counts.get))


    #anyway, monom_dict was just there to count all the monomial, in the end we put to 1 the
    #one appearing less in the first column
    for k,v in monom_dict.items():
        if k == index_list[m]:
            monom_dict[k]=1
        else:
            monom_dict[k]=0
    
    newmat = mat.copy()

    for j in range(np.shape(mat)[1]):
        for i in range(np.shape(mat)[0]):
    
    
            tmp = newmat[i,j]
            for k,v in monom_dict.items():
                tmp = tmp.subs(k,v)
            newmat[i,j] = tmp
        

    #we normalize each column to the value of is maximum entry (so such that the max number appearing is 1)
    for j in range(np.shape(newmat)[1]):
    
        index = (newmat[:,j]!=0).argmax(axis=0)
        norm = newmat[index,j]
    
        newmat[:,j] /= norm
    

    #then we make the matrix columns orthogonal using gram schmidt
    for j in range(np.shape(newmat)[1]):
        for jp in range(j):
            newmat[:,j] = newmat[:,j] - np.dot(newmat[:,jp],newmat[:,j])/np.dot(newmat[:,jp],newmat[:,jp]) * newmat[:,jp]

        newmat[:,j]/=newmat[ (newmat[:,j]!=0).argmax(axis=0) , j]

    return newmat


#function used to print in a nice way the matrix with cg coefficientss
def print_CGmat(cgmat,digits=5):
    display( sym.Matrix(np.round(np.asarray(cgmat).astype(np.float64),digits)) )
