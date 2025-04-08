#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np 


# DATA IMPORT
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# We import the Transition Matrices
# To reload the array with the transition matrices we need to reshape it
transition_matrices_reshaped = np.loadtxt("Working Data/edge_transition_matrices.csv",
                         delimiter=",")  
# Changing the array shape back to 3D
transition_matrices = transition_matrices_reshaped.reshape(transition_matrices_reshaped.shape[0], 
                                      int(np.sqrt(transition_matrices_reshaped.shape[1])), 
                                      int(np.sqrt(transition_matrices_reshaped.shape[1])))

# We import the Transition Probabilities
# To reload the array with the transition probabilities we need to reshape it
transition_probabilities_reshaped = np.loadtxt("Working Data/edge_transition_probabilities.csv",
                         delimiter=",")  
# Changing the array shape back to 3D
transition_probabilities = transition_probabilities_reshaped.reshape(transition_probabilities_reshaped.shape[0], 
                                      int(np.sqrt(transition_probabilities_reshaped.shape[1])), 
                                      int(np.sqrt(transition_probabilities_reshaped.shape[1])))

average_transition_matrix_path = 'Working Data/edges_average_transition_matrix.csv'
average_transition_probabilities_path = 'Working Data/edges_average_transition_probabilities.csv'
# We import the average matrices
average_transition_matrix = pd.read_csv(average_transition_matrix_path,
                                        delimiter=",",
                                        index_col='Unnamed: 0')
average_transition_probabilities = pd.read_csv(average_transition_probabilities_path,
                                               delimiter=",",
                                               index_col='Unnamed: 0')

# We import the Edge Abundances
edge_abundances_path = "Working Data/edge_abundances.csv"
edge_abundances = pd.read_csv(edge_abundances_path,
                               delimiter=",",
                               index_col='Unnamed: 0')
# We import the Edge Densities
edge_densities_path = "Working Data/edge_densities.csv"
edge_densities = pd.read_csv(edge_densities_path,
                             delimiter=",",
                             index_col='Unnamed: 0')
# We import the Edge Densities Errors
error_edge_densities_path = "Working Data/error_edge_densities.csv"
error_edge_densities = pd.read_csv(error_edge_densities_path,
                                   delimiter=",",
                                   index_col='Unnamed: 0')

# We define the wave numbers we will use
wave_numbers = np.array([0,1,2,3,4,5,6,7,8,9])

# We are interested in having the relevant link types appearing always in the same order
order = ['+2+2','+2+1','+1+2','+1+1','+2+0','+0+2','+1+0','+0+1','+0+0',
         '+0-1','-1+0','+0-2','-2+0','-1-1','-1-2','-2-1','-2-2',
         '-1+1','+1-1','-2+1','+1-2','-1+2','+2-1','-2+2','+2-2']

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

        

N = transition_matrices.shape[0]
Rij = np.zeros((N,len(order),len(order)))
errorRij = np.zeros((N,len(order),len(order)))
detailed_balance = np.zeros((N,len(order),len(order)))

violations = pd.DataFrame(columns=['edge_i','edge_j','wave','n_edges_i','n_edges_j','n_transitions_ij','n_transitions_ji','relative_error'])
v = 0
for k in range(N):
    
    E = edge_abundances[str(k)]
    e = edge_densities[str(k)]

    P = np.array(transition_matrices[k].copy())
    p = np.array(transition_probabilities[k].copy())


    for i in range(len(order)):
        for j in range(len(order)):
            
            if np.sum(P[i])!=0: error_pl = 1.96*np.sqrt(p[i,j]*(1-p[i,j])/np.sum(P[i]))
            else: error_pl = 0
            if np.sum(E)!=0: error_el = 1.96*np.sqrt(e.iloc[i]*(1-e.iloc[i])/np.sum(E))
            else: error_el = 0
            if np.sum(P[j])!=0: error_pr = 1.96*np.sqrt(p[j,i]*(1-p[j,i])/np.sum(P[j]))
            else: error_pr = 0
            if np.sum(E)!=0: error_er = 1.96*np.sqrt(e.iloc[j]*(1-e.iloc[j])/np.sum(E))
            else: error_er = 0
            
            
            
            left = p[i,j]*e.iloc[i]
            right = p[j,i]*e.iloc[j]
            
            error_left = error_pl*e.iloc[i]+p[i,j]*error_el
            error_right = error_pr*e.iloc[j]+p[j,i]*error_er
            
            Rij[k,i,j] = abs(left-right)
            errorRij[k,i,j] = error_left + error_right
    

            detailed_balance[k,i,j] = ((Rij[k,i,j] + errorRij[k,i,j]) >=0) * ((Rij[k,i,j] - errorRij[k,i,j])<=0)
            if detailed_balance[k,i,j] == 0:
                if j>i:
                    violations.loc[v,'edge_i'] = order[i]
                    violations.loc[v,'edge_j'] = order[j]
                    violations.loc[v,'wave'] = k
                    violations.loc[v,'n_edges_i'] = E.iloc[i]
                    violations.loc[v,'n_edges_j'] = E.iloc[j]
                    violations.loc[v,'n_transitions_ij'] = transition_matrices[k,i,j]
                    violations.loc[v,'n_transitions_ji'] = transition_matrices[k,j,i]
                    violations.loc[v,'relative_error'] = 1.96*Rij[k,i,j]/(errorRij[k,i,j])
                    
                    v+=1
            


 






