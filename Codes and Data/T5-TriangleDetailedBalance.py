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



N = transition_matrices.shape[0]
Rij = np.zeros((N,len(order),len(order)))
errorRij = np.zeros((N,len(order),len(order)))
detailed_balance = np.zeros((N,len(order),len(order)))

violations = pd.DataFrame(columns=['triangle_i','triangle_j','wave','n_triangles_i','n_triangles_j','n_transitions_ij','n_transitions_ji','relative_error'])
v = 0
for k in range(N):
    
    E = triangle_abundances[str(k)]
    e = triangle_densities[str(k)]

    P = np.array(transition_matrices[k].copy())
    p = np.array(transition_probabilities[k].copy())


    for i in tqdm(range(len(order))):
        for j in range(len(order)):
            
            if np.sum(P[i])!=0: error_pl = 1.96*np.sqrt(p[i,j]*(1-p[i,j])/np.sum(P[i]))
            else: error_pl = 0
            if np.sum(E)!=0: error_el = 1.96*np.sqrt(e.loc[order.iloc[i]['motif']]*(1-e.loc[order.iloc[i]['motif']])/np.sum(E))
            else: error_el = 0
            if np.sum(P[j])!=0: error_pr = 1.96*np.sqrt(p[j,i]*(1-p[j,i])/np.sum(P[j]))
            else: error_pr = 0
            if np.sum(E)!=0: error_er = 1.96*np.sqrt(e.loc[order.iloc[j]['motif']]*(1-e.loc[order.iloc[j]['motif']])/np.sum(E))
            else: error_er = 0
            
            
            
            left = p[i,j]*e.loc[order.iloc[i]['motif']]
            right = p[j,i]*e.loc[order.iloc[j]['motif']]
            
            error_left = error_pl*e.loc[order.iloc[i]['motif']]+p[i,j]*error_el
            error_right = error_pr*e.loc[order.iloc[j]['motif']]+p[j,i]*error_er
            
            Rij[k,i,j] = abs(left-right)
            errorRij[k,i,j] = error_left + error_right
    

            detailed_balance[k,i,j] = ((Rij[k,i,j] + errorRij[k,i,j]) >=0) * ((Rij[k,i,j] - errorRij[k,i,j])<=0)
            if detailed_balance[k,i,j] == 0:
                if j>i:
                    violations.loc[v,'triangle_i'] = order.iloc[i]['motif']
                    violations.loc[v,'triangle_j'] = order.iloc[j]['motif']
                    violations.loc[v,'wave'] = k
                    violations.loc[v,'n_triangles_i'] = E.loc[order.iloc[i]['motif']]
                    violations.loc[v,'n_triangles_j'] = E.loc[order.iloc[j]['motif']]
                    violations.loc[v,'n_transitions_ij'] = transition_matrices[k,i,j]
                    violations.loc[v,'n_transitions_ji'] = transition_matrices[k,j,i]
                    violations.loc[v,'relative_error'] = 1.96*Rij[k,i,j]/(errorRij[k,i,j])
                    
                    v+=1
            
‘’’
--COMMENT--

The z-scores of the detailed balance violations in transitions between triangles are inflated. The intuition behind this 
is the following: suppose we detect that the link-level transition from +0+0 to +2+2 violates detailed balance. When 
measuring transitions between triangles, we will also observe that transitions such as +0+0+0+0+0+0 → +0+0+0+0+2+2 
violate detailed balance. However, when computing the z-score for this triangle-level transition, we compare the observed 
deviation to the width of the confidence interval. In these cases, the confidence interval is abnormally narrow. This is 
because the triangle counts include many triangles that share the same individual link responsible for the detailed balance 
violation. The intuition here is that there are far more triangles of type +0+0+0+0+0+0 or +0+0+0+0+2+2 than there are 
individual links of type +0+0 or +2+2. Even if the number of violations is the same at the link level, triangles are counted 
multiple times for the same violating link, artificially increasing the signal without increasing the effective sample size. 
A more general conclusion from this observation is that the confidence intervals for triangle-level statistics are underestimated. 
To correct this bias, it would be necessary to account for the network’s structure, specifically, the dependency between triangle 
counts caused by overlapping links.

‘’’

 



