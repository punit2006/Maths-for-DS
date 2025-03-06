#Maths-for-DS
Assignment - Matrix and Vectors

Objective - Develop a deeper understanding of matrices and vectors, including advanced operations, practical linear algebra concepts, and their application in data science.

1. Matrix and Vector Operations
    1. Create a 3 × 3 matrix A and a 3 × 1 vector B :

   import numpy as np
   A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   B = np.array([1, 2, 3])

   Tasks -
   - Perform matrix-vector multiplication  A X B.
   - Calculate the trace of matrix  A  (sum of diagonal elements).
   - Find the eigenvalues and eigenvectors of A.

    2. Replace the last row of matrix A with [10, 11, 12] and:
   - Compute the determinant of the updated matrix A.
   - Identify if the updated matrix is singular or non-singular.

2. Invertibility of Matrices
    1. Verify the invertibility of the updated matrix A:
   - Check if the determinant is non-zero.
   - If invertible, calculate the inverse of A.

    2. Solve a system of linear equations A x X = B, where:
   - A is the updated matrix.
   -  B = np.array([[1],  [2],  [3]])	 

3. Practical Matrix Operations
    1. Create a 4 × 4 matrix C with random integers between 1 and 20:
  
   C = np.random.randint(1, 21, size=(4, 4))

   Tasks
   - Compute the rank of C.
   - Extract the submatrix consisting of the first 2 rows and last 2 columns of C.
   - Calculate the Frobenius norm of C.

    2. Perform matrix multiplication between A (updated to 3 × 3) and C (trimmed to 3 × 3):
   - Check if the multiplication is valid. If not, reshape C to make it compatible with A.

4. Data Science Context
    1. Create a dataset as a 5 × 5 matrix D, where each column represents a feature, and each row represents a data point:


   D = np.array([[3, 5, 7, 9, 11],
                 [2, 4, 6, 8, 10],
                 [1, 3, 5, 7, 9],
                 [4, 6, 8, 10, 12],
                 [5, 7, 9, 11, 13]])


   Tasks
   - Standardize D column-wise (mean = 0, variance = 1).
   - Compute the covariance matrix of D.
   - Perform Principal Component Analysis (PCA):
     - Find the eigenvalues and eigenvectors of the covariance matrix.
     - Reduce D to 2 principal components.
