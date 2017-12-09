
# coding: utf-8

# In[1]:


import numpy as np


# In[35]:


#funtion to compute the eigenvalues of uncontrolled bike
def eig_values(M, C1, K0, K2, v):
    #gravity
    g = 9.81
    
    #compute C and K matrix
    C = v*C1
    K = g*K0 + (v**2)*K2
    
    #Get the coefficients for each order
    fourth_order = M[0][0]*M[1][1] - M[0][1]*M[1][0]
    
    third_order = M[0][0]*C[1][1] + C[0][0]*M[1][1]                - M[0][1]*C[1][0] - C[0][1]*M[1][0]
        
    second_order = M[0][0]*K[1][1] + C[0][0]*C[1][1] + K[0][0]*M[1][1]                 - M[0][1]*K[1][0] - C[0][1]*C[1][0] - K[0][1]*M[1][0]
    
    first_order = C[0][0]*K[1][1] + K[0][0]*C[1][1]                - C[0][1]*K[1][0] - K[0][1]*C[1][0]
        
    constant = K[0][0]*K[1][1] - K[0][1]*K[1][0]
    
    #compute the eigenvalues
    coeff = [fourth_order, third_order, second_order, first_order, constant]
    eigs = np.roots(coeff)
    
    return eigs
    


# In[40]:


M = np.array([[80.81722, 2.31941332208709],              [2.31941332208709, 0.29784188199686]])

K0 = np.array([[-80.95, -2.59951685249872],               [-2.59951685249872, -0.80329488458618]])

K2 = np.array([[0, 76.59734589573222],               [0, 2.65431523794604]])

C1 = np.array([[0, 33.86641391492494],               [-0.85035641456978, 1.68540397397560]])

ans = eig_values(M,C1, K0, K2, 0.0)
print(ans)

