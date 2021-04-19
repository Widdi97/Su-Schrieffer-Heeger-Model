import numpy as np
try:
    import cupy as cp  # effiziernte Eigenwertberechnung
except:
    class cupyPlaceholder:
        def __init__(self):
            pass
        
        def asnumpy(self,x):
            return x
    cp = cupyPlaceholder()
import matplotlib.pyplot as plt


alpha = 0.58 # intra-cell hopping (first chain)
beta = 1 # inter-cell hopping (first chain)
coupling = 1 # coupling of the two chains
N = 4  #unit cells per side

useGpuAccel = False

if useGpuAccel:
    H = cp.zeros(( 4*N,4*N ),dtype=np.float64)
else:
    H = np.zeros(( 4*N,4*N ),dtype=np.float64)

#populate H:
for k in range(N):
    H[2*k-1,2*k] = beta
    H[2*k,2*k-1] = beta
    H[2*k,2*k+1] = alpha
    H[2*k+1,2*k] = alpha
        
    H[-1,0] = 0
    H[0,-1] = 0
for k in range(N,2*N):
    H[2*k-1,2*k] = alpha
    H[2*k,2*k-1] = alpha
    H[2*k,2*k+1] = beta
    H[2*k+1,2*k] = beta
        
    H[-1,0] = 0
    H[0,-1] = 0
H[2*N-1,2*N] = coupling
H[2*N,2*N-1] = coupling


if useGpuAccel:
    eigensystem = cp.linalg.eigh(H)
else:
    eigensystem = np.linalg.eigh(H)

abscissa = [k*0.5+0.75 for k in range(4*N)]

# plot eigenstates
for k in range(4*N):
    plt.bar([k+1 for k in range(2*N)][::2],np.absolute(cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[:2*N][::2]),color="tab:orange")
    plt.bar([k+1 for k in range(2*N)][1:][::2],np.absolute(cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[:2*N][1:][::2]),color="tab:blue")
    plt.bar([k+1 for k in range(2*N,4*N)][::2],np.absolute(cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[2*N:4*N][::2]),color="tab:blue")
    plt.bar([k+1 for k in range(2*N,4*N)][1:][::2],np.absolute(cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[2*N:4*N][1:][::2]),color="tab:orange")
    plt.title("$E={}$".format( round(list(cp.asnumpy(eigensystem[0]))[k],4)))
    plt.xlabel("Lattice position")
    plt.ylabel("Wavefunction")
    plt.show()




