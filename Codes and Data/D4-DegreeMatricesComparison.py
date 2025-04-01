#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True) #!!!


degree_distributions = ['out_degree_2p','out_degree_1p', 'out_degree_1n', 'out_degree_2n',
                        'in_degree_2p','in_degree_1p', 'in_degree_1n', 'in_degree_2n']

for distribution in degree_distributions:
    print(distribution)

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
    
    # We import the Degree Transitions
    degrees_transitions_path = f"Working Data/degrees_transitions_{distribution}.csv"
    # We import the degrees_transitions DataFrame
    degrees_transitions = pd.read_csv(degrees_transitions_path,
                                   delimiter=",",
                                   index_col='Unnamed: 0',
                                   dtype={'transition_number': str})
    
    # We define the wave numbers we will use
    wave_numbers = np.array([0,1,2,3,4,5,6,7,8,9])
    
    degrees_path = "Working Data/degrees.csv"
    # We import the degrees DataFrame
    degrees = pd.read_csv(degrees_path,
                          delimiter=",",
                          index_col='Unnamed: 0')
    order = np.arange(degrees[distribution].max()+1)
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    # We create the random samples of the transition matrices
    random_transitions = degrees_transitions.copy()
    randomizations = 1000
    random_transition_matrices = np.zeros((randomizations,len(degrees_transitions['transition_number'].unique()),len(order),len(order)))
    random_transition_probabilities = random_transition_matrices.copy()
    for randomization in tqdm(range(randomizations)): 
        random_transitions['transition_number'] = np.random.permutation(random_transitions['transition_number'])
        
        random_transitions_count = random_transitions.value_counts().reset_index()
        random_transitions_count.set_index('transition',inplace=True)
        
        transition_numbers = degrees_transitions['transition_number'].unique()
        
        # We compute the matrix for each transition
        for c in range(len(transition_numbers)):
     
            # We obtain the part of the count we are interested in
            transition_number = transition_numbers[c]
            partial_count = random_transitions_count.copy() 
            partial_count = partial_count[partial_count['transition_number']==transition_number].copy()
    
            # We define a temporal transition matrix
            transition_matrix = pd.DataFrame(index=order,columns=order)
            
            # We follow the same strategy
            for transition in partial_count.index: 
                pre, post = transition.split('-')
                
                transition_matrix.loc[int(pre),int(post)] = partial_count.loc[transition,'count']
                
            transition_matrix.fillna(0,inplace=True)
                      
            transition_probability = transition_matrix.copy()
            transition_probability = transition_probability.astype(float)
            
            for transition_row in transition_matrix.index: 
                total_count = np.sum(transition_matrix.loc[transition_row])
                if total_count!=0:
                    for transition_col in transition_matrix.columns: 
                        transition_probability.loc[transition_row,transition_col] = transition_matrix.loc[transition_row,transition_col] / total_count
            
            random_transition_matrices[randomization,c] = transition_matrix.copy()
            random_transition_probabilities[randomization,c] = transition_probability.copy()
    
    
    # Now wo compare with the random samples
    N = len(transition_numbers)
    distance_matrices = np.zeros((N,len(order),len(order)))
    random_distance_matrices = np.zeros((randomizations,N,len(order),len(order)))
    
    
    # Distance for the real matrices
    for i in range(len(transition_numbers)):
            
            A = np.array(average_transition_probabilities)
            B = transition_probabilities[i]
            
            difference = A-B
            distance_matrices[i] = difference
                
                
    # Distance for the random matrices
    for randomization in tqdm(range(randomizations)):
        for i in range(len(transition_numbers)):
                
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
    np.savetxt(f"Working Data/degrees_p_values_{distribution}.csv", p_values_reshaped, delimiter=",", fmt='%s')
    z_scores_reshaped = z_scores.reshape(z_scores.shape[0], -1)
    np.savetxt(f"Working Data/degrees_z_scores_{distribution}.csv", z_scores_reshaped, delimiter=",", fmt='%s')
    
    
    # Final reported result with applied Bonferroni correction
    for i in range(N):
        print(f'Transition: {i}')
        ps = p_values[i].flatten()
        zs = z_scores[i].flatten()
        
        significance = 0.05/(len(ps))
        print(zs[ps<significance])
        #significance = 0.05
        print((ps<significance).sum()/len(ps))
        
        print(zs[abs(zs)>=3])
        
        
        
    




