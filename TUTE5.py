#1
#1A
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([1, 2, 3])

result = np.dot(A, B)
print("Matrix-vector multiplication A x B : ", result)

trace_A = np.trace(A)
print("Trace of matrix A : ", trace_A)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A : ", eigenvalues)
print("Eigenvectors of A : ", eigenvectors)

#1B
A[2] = [10, 11, 12]
print("Updated matrix A : ", A)

det_A = np.linalg.det(A)
print("Determinant of updated matrix A : ", det_A)

if det_A == 0:
    print("Matrix A is singular.")
else:
    print("Matrix A is non-singular.")

#2
#2A
if det_A != 0:
    inverse_A = np.linalg.inv(A)
    print("Inverse of matrix A : ", inverse_A)
else:
    print("Matrix A is singular, cannot calculate inverse.")

#2B
if det_A != 0:
    X = np.linalg.solve(A, B)
    print("Solution to A x X = B : ", X)
else :
    print("The Matrix A is Singular")

#3
#3A
C = np.random.randint(1, 21, size=(4, 4))
print("Matrix C : ", C)

rank_C = np.linalg.matrix_rank(C)
print("Rank of matrix C : ", rank_C)

submatrix_C = C[:2, -2:]
print("Submatrix of C (first 2 rows, last 2 columns) : ", submatrix_C)

frobenius_norm_C = np.linalg.norm(C, 'fro')
print("Frobenius norm of matrix C : ", frobenius_norm_C)

#3B
C_trimmed = C[:3, :3]
if A.shape[1] == C_trimmed.shape[0]:
    result_mult = np.dot(A, C_trimmed)
    print("Matrix multiplication result (A x C) : ", result_mult)
else:
    print("Matrix dimensions do not match for multiplication.")

#4
D = np.array([[3, 5, 7, 9, 11],
              [2, 4, 6, 8, 10],
              [1, 3, 5, 7, 9],
              [4, 6, 8, 10, 12],
              [5, 7, 9, 11, 13]])

print("Dataset D : ", D)

D_standardized = (D - D.mean(axis=0)) / D.std(axis=0)
print("Standardized matrix D (column-wise) : ", D_standardized)

cov_matrix_D = np.cov(D.T)
print("Covariance matrix of D : ", cov_matrix_D)

eigenvalues_cov, eigenvectors_cov = np.linalg.eig(cov_matrix_D)
print("Eigenvalues of the covariance matrix : ", eigenvalues_cov)
print("Eigenvectors of the covariance matrix : ", eigenvectors_cov)

principal_components = eigenvectors_cov[:, :2]
D_reduced = np.dot(D_standardized, principal_components)
print("Reduced D to 2 principal components : ", D_reduced)
