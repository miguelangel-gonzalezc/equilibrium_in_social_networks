#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np 
from tqdm import tqdm

distribution = 'out_degree_1p'

# DATA IMPORT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# We import the Transition Matrices
# To reload the array with the transition matrices we need to reshape it
transition_matrices_reshaped = np.loadtxt(f"Working Data/degrees_transition_matrices_{distribution}.csv",
                         delimiter=",")  
# Changing the array shape back to 3D
transition_matrices = transition_matrices_reshaped.reshape(transition_matrices_reshaped.shape[0], 
                                      int(np.sqrt(transition_matrices_reshaped.shape[1])), 
                                      int(np.sqrt(transition_matrices_reshaped.shape[1])))

# We import the Transition Probabilities
# To reload the array with the transition probabilities we need to reshape it
transition_probabilities_reshaped = np.loadtxt(f"Working Data/degrees_transition_probabilities_{distribution}.csv",
                         delimiter=",")  
# Changing the array shape back to 3D
transition_probabilities = transition_probabilities_reshaped.reshape(transition_probabilities_reshaped.shape[0], 
                                      int(np.sqrt(transition_probabilities_reshaped.shape[1])), 
                                      int(np.sqrt(transition_probabilities_reshaped.shape[1])))

average_transition_matrix_path = f'Working Data/degrees_average_transition_matrix_{distribution}.csv'
average_transition_probabilities_path = f'Working Data/degrees_average_transition_probabilities_{distribution}.csv'
# We import the average matrices
average_transition_matrix = pd.read_csv(average_transition_matrix_path,
                                        delimiter=",",
                                        index_col='Unnamed: 0')
average_transition_probabilities = pd.read_csv(average_transition_probabilities_path,
                                               delimiter=",",
                                               index_col='Unnamed: 0')

# We import the Degree Abundances
degree_abundances_path = f"Working Data/degree_abundances_{distribution}.csv"
degree_abundances = pd.read_csv(degree_abundances_path,
                                delimiter=",",
                                index_col='Unnamed: 0')
# We import the Degree Densities
degree_densities_path = f"Working Data/degree_densities_{distribution}.csv"
degree_densities = pd.read_csv(degree_densities_path,
                               delimiter=",",
                               index_col='Unnamed: 0')
# We import the Degree Densities Errors
error_degree_densities_path = f"Working Data/error_degree_densities_{distribution}.csv"
error_degree_densities = pd.read_csv(error_degree_densities_path,
                                     delimiter=",",
                                     index_col='Unnamed: 0')

# We define the wave numbers we will use
wave_numbers = np.array([0,1,2,3,4,5,6,7,8,9])

# Define the order variable, which is an array of integers from 0 to the max degree measured in the network
#order 

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
average_stationary_degree_abundances = abs(eigenvectors[:,i])
average_stationary_degree_densities = average_stationary_degree_abundances/average_stationary_degree_abundances.sum()

average_stationary_degree_densities = average_stationary_degree_densities
average_eigenvalue_closest_to_1 = eigenvalues[i]

# Strategy 2
#----------------------------------------------------------------------

eigenvalue_closest_to_1 = np.zeros(len(wave_numbers)-1)
stationary_degree_densities = np.zeros((len(wave_numbers)-1,len(degree_densities)))


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
    stationary_degree_abundances = abs(eigenvectors[:,i])
    densities = stationary_degree_abundances/stationary_degree_abundances.sum()
        
    stationary_degree_densities[wave] = densities
    eigenvalue_closest_to_1[wave] = np.real(eigenvalues[i])
        
