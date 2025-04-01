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



# We define the wave numbers we will use
wave_numbers = np.array([0,1,2,3,4,5,6,7,8,9])

# We are interested in having the relevant link types appearing always in the same order
order = pd.read_csv('Working Data/triangles_order_string.csv', index_col='Unnamed: 0')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

transition_ids = np.array([0,1,2,3,4,5,6,7,8]) #!!!

# Total sum of the average matrix
NT = np.array(average_transition_matrix).sum()
# Flatten the average matrix
flat_matrix = np.array(average_transition_matrix).flatten()
# Compute probabilities (normalize frequencies)
probabilities = flat_matrix / NT

# We create the random samples of the transition matrices
randomizations = 100
random_transition_matrices = np.zeros((randomizations,len(transition_ids),len(order),len(order)))
random_transition_probabilities = np.zeros((randomizations,len(transition_ids),len(order),len(order)))
for randomization in tqdm(range(randomizations)): 
        
    # We compute the matrix for each transition
    for c in transition_ids:
        
        # We are resampling the matrix of a specific transition - we mimic its total count
        Nt = int(transition_matrices[c].sum())
        
        # Perform resampling: sample 'Nt' times from the flattened matrix according to probabilities
        # We resample 'Nt' indices and then use these indices to count how often each element was picked
        resampled_indices = np.random.choice(len(flat_matrix), size=Nt, p=probabilities)
        
        # Count the number of times each element was chosen (this is the frequency in the resampled matrix)
        counts = np.bincount(resampled_indices, minlength=len(flat_matrix))
        
        # Reshape the counts back into the original matrix shape (N x N)
        resampled_matrix = counts.reshape(average_transition_matrix.shape)
        
        # Compute the row sums
        row_sums = resampled_matrix.sum(axis=1, keepdims=True)

        # Normalize the matrix by row sums, avoiding division by zero
        resampled_probabilities = np.divide(resampled_matrix, row_sums, where=row_sums != 0)
        
        random_transition_matrices[randomization,c] = resampled_matrix
        random_transition_probabilities[randomization,c] = resampled_probabilities


# Now wo compare with the random samples
N = len(transition_ids)
distance_matrices = np.zeros((N,len(order),len(order)))
random_distance_matrices = np.zeros((randomizations,N,len(order),len(order)))


# Distance for the real matrices
for i in transition_ids:
        
        A = np.array(average_transition_probabilities)
        B = transition_probabilities[i]
        
        difference = A-B
        distance_matrices[i] = difference
            
            
# Distance for the random matrices
for randomization in tqdm(range(randomizations)):
    for i in transition_ids:
            
            A = np.array(average_transition_probabilities)
            B = random_transition_probabilities[randomization,i]
            
            difference = A-B
            random_distance_matrices[randomization,i] = difference
            

# Finally, we will construct p-values matrices and effect-size matrices
p_values = np.zeros((N,len(order),len(order)))
z_scores = np.zeros((N,len(order),len(order)))


for i in range(N):
    for j in range(len(order)):
        for k in range(len(order)):
    
            real_distance = distance_matrices[i,j,k]
            random_distances = np.zeros(randomizations)
            for randomization in range(randomizations):
                
                random_distances[randomization] = random_distance_matrices[randomization,i,j,k]
                
            if np.std(random_distances,ddof=1)!=0:
                z_scores[i,j,k] = (real_distance-np.mean(random_distances))/np.std(random_distances,ddof=1)
            else:
                z_scores[i,j,k] = 0
            
            
            p_values[i,j,k] = (abs(random_distances)>=abs(real_distance)).astype(int).sum()/randomizations
    



p_values_reshaped = p_values.reshape(p_values.shape[0], -1)
np.savetxt("Working Data/edge_p_values.csv", p_values_reshaped, delimiter=",", fmt='%s')
z_scores_reshaped = z_scores.reshape(z_scores.shape[0], -1)
np.savetxt("Working Data/edge_z_scores.csv", z_scores_reshaped, delimiter=",", fmt='%s')


# Final reported result with applied Bonferroni correction
for i in range(N):
    ps = p_values[i].flatten()
    zs = z_scores[i].flatten()
    
    significance = 0.05/(len(ps))
    #print(zs[ps<significance])
    #significance = 0.05
    print((ps<significance).sum()/len(ps))
    print((ps<significance).sum())
    
    
    





