from common import *

m11 = np.matrix([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0]
])
m12 = np.matrix([
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0]
])
m13 = np.matrix([
    [0, 1, 1, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0]
])
m14 = np.matrix([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0]
])
m15 = np.matrix([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0]
])

m4 = np.matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
m6 = np.matrix([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
m8 = np.matrix([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])

small_motifs = [m4, m6, m8]
motifs = [m11, m12, m13, m14, m15]

for i, m in enumerate(small_motifs):
    V = count_noiseless_eigenvalues(m)
    print(V)

    print(f"Eigh of m{2*i+4}: {np.linalg.eigh(m)[1]}")

for i, m in enumerate(motifs):
    V = count_noiseless_eigenvalues(m)
    print(V)

    print(f"Eigh of m1{i+1}: {np.linalg.eigh(m)[1]}")

m12_p = np.matrix([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

print(np.linalg.eigh(m12_p))

M = np.linalg.eigh(m13)
print(abs(M[1].sum(axis=0)) < tol)
print(abs(np.sum(M[1], axis=0)))
print(abs(M[1].sum(axis=1)))
# print(abs(sum(np.asarray(M[1].T[3])[0])) < tol)
