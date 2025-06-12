#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np 
from tqdm import tqdm

# DATA IMPORT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# We import the Transition Matrices
# To reload the array with the transition matrices we need to reshape it
transition_matrices_reshaped = np.loadtxt("Working Data/triangles_transition_matrices.csv",
                         delimiter=",")  
# Changing the array shape back to 3D
transition_matrices = transition_matrices_reshaped.reshape(transition_matrices_reshaped.shape[0], 
                                      int(np.sqrt(transition_matrices_reshaped.shape[1])), 
                                      int(np.sqrt(transition_matrices_reshaped.shape[1])))

# We import the Transition Probabilities
# To reload the array with the transition probabilities we need to reshape it
transition_probabilities_reshaped = np.loadtxt("Working Data/triangles_transition_probabilities.csv",
                         delimiter=",")  
# Changing the array shape back to 3D
transition_probabilities = transition_probabilities_reshaped.reshape(transition_probabilities_reshaped.shape[0], 
                                      int(np.sqrt(transition_probabilities_reshaped.shape[1])), 
                                      int(np.sqrt(transition_probabilities_reshaped.shape[1])))

average_transition_matrix_path = 'Working Data/triangles_average_transition_matrix.csv'
average_transition_probabilities_path = 'Working Data/triangles_average_transition_probabilities.csv'
# We import the average matrices
average_transition_matrix = pd.read_csv(average_transition_matrix_path,
                                        delimiter=",",
                                        index_col='motif')
average_transition_probabilities = pd.read_csv(average_transition_probabilities_path,
                                               delimiter=",",
                                               index_col='motif')

# We import the Triangle Abundances
triangle_abundances_path = "Working Data/triangle_abundances.csv"
triangle_abundances = pd.read_csv(triangle_abundances_path,
                                  delimiter=",",
                                  index_col='motif')
# We import the Triangle Densities
triangle_densities_path = "Working Data/triangle_densities.csv"
triangle_densities = pd.read_csv(triangle_densities_path,
                                 delimiter=",",
                                 index_col='motif')
# We import the Triangle Densities Errors
error_triangle_densities_path = "Working Data/error_triangle_densities.csv"
error_triangle_densities = pd.read_csv(error_triangle_densities_path,
                                       delimiter=",",
                                       index_col='motif')

# We define the wave numbers we will use
wave_numbers = np.array([0,1,2,3,4,5,6,7,8,9])

# We are interested in having the relevant link types appearing always in the same order
order = pd.read_csv('Working Data/triangles_order_string.csv', index_col='Unnamed: 0')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

'''
There are two ways of computing the stationary abundances from the transition 
matrices. The first one is to take the average transition matrix and extract
the eigenvector associated with the eigenvalue 1. The other option is to extract
the stationary predicted by each individual transition matrix, put them together
and average. In principle, if the matrix is truly stationary, both methods should 
give the same results.
'''

# Strategy 1
#----------------------------------------------------------------------
# We define the average transition matrix as an array
M = np.array(average_transition_probabilities)
# We compute the eigenvalues and eigenvectors of its transpose
eigenvalues, eigenvectors = np.linalg.eig(M.T)
# We get the position of the eigenvector associated with the eigenvalue 1
indices = (np.abs(eigenvalues - 1)).argsort()
r = 0
i = indices[r]
while sum(abs(eigenvectors[:,i])==1)!=0:
    r += 1
    i = indices[r]
    
# We obtain the stationary abundances
average_stationary_edge_abundances = abs(eigenvectors[:,i])
average_stationary_edge_densities = average_stationary_edge_abundances/average_stationary_edge_abundances.sum()

average_stationary_edge_densities = average_stationary_edge_densities
average_eigenvalue_closest_to_1 = eigenvalues[i]

# Strategy 2
#----------------------------------------------------------------------

eigenvalue_closest_to_1 = np.zeros(len(wave_numbers)-1)
stationary_triangle_densities = np.zeros((len(wave_numbers)-1,len(triangle_densities)))


for wave in tqdm(range(len(wave_numbers)-1)):
    
    M = transition_probabilities[wave]
    # We compute the eigenvalues and eigenvectors of its transpose
    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    # We get the position of the eigenvector associated with the eigenvalue 1
    indices = (np.abs(eigenvalues - 1)).argsort()
    r = 0
    i = indices[r]
    while sum(abs(eigenvectors[:,i])==1)!=0:
        r += 1
        i = indices[r]
    # We obtain the stationary abundances
    stationary_triangle_abundances = abs(eigenvectors[:,i])
    densities = stationary_triangle_abundances/stationary_triangle_abundances.sum()
        
    stationary_triangle_densities[wave] = densities
    eigenvalue_closest_to_1[wave] = np.real(eigenvalues[i])


‘’’
—CAVEAT—

To replicate this analysis, it is important to consider a subtle detail. When certain states have a 
very low probability of occurring (for example, links +2-2, or extremely high degrees), the transition 
probabilities involving these rare states in the corresponding transition matrices are very difficult 
to estimate reliably. This means that, in cases of extremely sparse statistics, we might observe only 
a single link in the network that is in the +2-2 state, which remains unchanged in the next snapshot. 
When calculating the transition probability from +2-2 to itself, since there was only one such link 
and it did not change state, the resulting transition probability is 1. This leads to a problem: when 
computing the expected stationary distribution (e.g., via the corresponding eigenvector of the transition matrix), 
we observe abnormally high stationary values for this state, values that clearly do not reflect the 
empirical distribution. A simple way to understand this is that, no matter how unlikely the +2-2 state 
is, once a link enters it, the transition matrix imposes it cannot leave. Over time, this causes the 
stationary distribution to accumulate weight in that state, which is a spurious effect caused by poor 
statistical estimation due to insufficient sample size. For this reason, transitions involving these 
extremely rare states with self-loops should be excluded to avoid distorting the long-term dynamics.

‘’’



