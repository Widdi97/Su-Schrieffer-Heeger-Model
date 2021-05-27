import numpy as np
import matplotlib.pyplot as plt

alpha = 0.58 # intra-cell-hopping
beta = 1 # inter-cell-hopping
N = 6 # Unit cells for alpha != beta

H = np.zeros(( 2*N,2*N ),dtype=np.float64)

#populate H:
for k in range(N):
    H[2*k-1,2*k] = beta
    H[2*k,2*k-1] = beta
    H[2*k,2*k+1] = alpha
    H[2*k+1,2*k] = alpha
        
    H[-1,0] = 0
    H[0,-1] = 0

eigensystem = np.linalg.eigh(H)
print("Eigenenergies:\n",*[ round(k,4) for k in list(eigensystem[0])])


abscissa = [k*0.5+0.75 for k in range(2*N)]


# plot eigenstates:
for k in range(2*N):
    plt.bar([k+1 for k in range(2*N)][::2],eigensystem[1][:,k:k+1].T[0][::2],label="Sublattice A")
    plt.bar([k+1 for k in range(2*N)][1:][::2],eigensystem[1][:,k:k+1].T[0][1:][::2],label="Sublattice B")
    plt.title("$E={}$".format( round(list(eigensystem[0])[k],4)))
    plt.xlabel("Lattice position")
    plt.ylabel("Wavefunction")
    plt.legend()
    plt.show()
