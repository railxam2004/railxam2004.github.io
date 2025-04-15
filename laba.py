import numpy as np

# Given matrix A and vector b
A = np.array([[22, 0, 23, 23],
              [3, 3, 0, 23],
              [2, 5, 3, 17],
              [17, 40, 17, 1]])

b = np.array([2, 2, 2, 7])

# Define the stopping criterion
epsilon = 1e-3

def simple_iteration(A, b, epsilon):
    n = A.shape[0]
    x = np.zeros(n)
    max_iterations = 1000

    # Checking if the method converges
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    norm_L_plus_U = np.linalg.norm(np.linalg.inv(D) @ L_plus_U, np.inf)

    if norm_L_plus_U < 1:
        print("Метод простых итераций сходится.")
    else:
        print("Метод простых итераций не сходится.")

    for k in range(max_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        # Check for convergence
        if np.linalg.norm(x_new - x, np.inf) < epsilon:
            return x_new

        x = x_new

    return x

def seidel_method(A, b, epsilon):
    n = A.shape[0]
    x = np.zeros(n)
    max_iterations = 1000

    # Checking if the method converges
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    norm_L_plus_U = np.linalg.norm(np.linalg.inv(D) @ L_plus_U, np.inf)

    if norm_L_plus_U < 1:
        print("Метод Зейделя сходится.")
    else:
        print("Метод Зейделя не сходится.")

    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]

        # Check for convergence
        if np.linalg.norm(x_new - x, np.inf) < epsilon:
            return x_new

        x = x_new

    return x

# Solving the system using both methods
solution_simple_iteration = simple_iteration(A, b, epsilon)
solution_seidel = seidel_method(A, b, epsilon)

solution_simple_iteration, solution_seidel
