import numpy as np

OUTPUT: bool = True
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

######################################################################################
# Problem 2.2
######################################################################################

# Declare the matrix
transition_matrix = np.array([
    [0.1, 0, 0.2, 0.3, 0.4],
    [0, 0.6, 0, 0.4, 0],
    [0.2, 0, 0, 0.4, 0.4],
    [0, 0.4, 0, 0.5, 0.1],
    [0.6, 0, 0.3, 0.1, 0]
])

# Goal: we want to do eigenvalue decomposition T = D diag(l) D^-1

# Get the eigenvalues and eigenvectors
eigenvalue, eigenvector = np.linalg.eig(transition_matrix)

# Diagonalize the eigenvalues
diagonizable_eigenvalues = np.diag(eigenvalue)

# Get the inverse of the eigenvectors
eigenvector_inverse = np.linalg.inv(eigenvector)

if OUTPUT:
    print("Eigenvalue: ", eigenvalue)
    print("\nEigenvector: ", eigenvector)
    print("\nEigenvector inverse: ", eigenvector_inverse)
    print("\nDiagonizable eigenvalues: ", diagonizable_eigenvalues)


######################################################################################
# Problem 2.2 - part b
######################################################################################

# Goal: P(X_2 = 2, X_4 = 5)
# Solution: Calculate D(l^2)D^(-1)
diagonalized_to_2nd = diagonizable_eigenvalues @ diagonizable_eigenvalues
transition_to_2nd = eigenvector @ diagonalized_to_2nd @ eigenvector_inverse

if OUTPUT:
    print("\nT^2: ", transition_to_2nd)

######################################################################################
# Problem 2.2 - part c
######################################################################################
    
# Goal: P(X_7 = 3 | X_3 = 4)
# Solution: Calculate D(l^4)D^(-1)
    
diagonalized_to_4th = diagonalized_to_2nd @ diagonizable_eigenvalues @ diagonizable_eigenvalues
transition_to_4th = eigenvector @ diagonalized_to_4th @ eigenvector_inverse

if OUTPUT:
    print("\nT^4: ", transition_to_4th)

######################################################################################
# Problem 2.2 - part d
######################################################################################
    
# Goal: P(X_1 in {1, 2, 3}, X_2 in {4, 5})
# For simplicity, define A = {1, 2, 3}, B = {4, 5}
# Solution: calculate sum_{k in S} sum_{i in A} sum_{i in B} p_{i,j} p_{k,i} P(X_0 = k)

starting_state = [0.5, 0, 0, 0, 0.5]
state_space = {1, 2, 3, 4, 5}
A = {1, 2, 3}
B = {4, 5}

total_probability = 0

for k in state_space:
    prob_start_k = starting_state[k - 1]

    # If there is no probability of starting, just skip the entire section
    # Do this since 3/5 of the starting spots are 0
    if prob_start_k == 0:
        continue

    for i in A:
        p_ki = transition_matrix[k - 1, i - 1]
        for j in B:
            p_ij = transition_matrix[i - 1, j - 1]
            
            probability_of_event = prob_start_k * p_ki * p_ij

            if OUTPUT:
                print("i: ", i)
                print("j: ", j)
                print("k: ", k)
                print("P(X_0 = k): ", prob_start_k)
                print("p_ki: ", p_ki)
                print("p_ij: ", p_ij)
                print("Probability of event: ", probability_of_event)

            total_probability = total_probability + probability_of_event

if OUTPUT:
    print("\nTotal probability: ", total_probability)      