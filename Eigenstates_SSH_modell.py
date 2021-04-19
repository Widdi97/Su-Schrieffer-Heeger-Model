import numpy as np
try:
    import cupy as cp  # effiziernte Eigenwertberechnung mit gpu
except:
    class cupyPlaceholder:
        def __init__(self):
            pass
        
        def asnumpy(self,x):
            return x
    cp = cupyPlaceholder()
from sympy.physics.quantum.dagger import Dagger
import IPython.display as disp
import matplotlib.pyplot as plt
import time
import sympy as sp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import string

alpha = 0.58 # intra-cell-coupling
beta = 1 # inter-cell-coupling
N = 6 # Unit cells for alpha != beta


useGpuAccel = False

if useGpuAccel:
    H = cp.zeros(( 2*N,2*N ),dtype=np.float64)
else:
    H = np.zeros(( 2*N,2*N ),dtype=np.float64)

t0 = time.time()
#populate H:
for k in range(N):
    H[2*k-1,2*k] = beta
    H[2*k,2*k-1] = beta
    H[2*k,2*k+1] = alpha
    H[2*k+1,2*k] = alpha
        
    H[-1,0] = 0
    H[0,-1] = 0

if useGpuAccel:
    eigensystem = cp.linalg.eigh(H)
else:
    eigensystem = np.linalg.eigh(H)
print("Eigenenergies:\n",*[ round(k,4) for k in list(cp.asnumpy(eigensystem[0]))])

print("calculation time: ",time.time()-t0,"s")

abscissa = [k*0.5+0.75 for k in range(2*N)]


# plot eigenstates:
for k in range(2*N):
    plt.bar([k+1 for k in range(2*N)][::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[::2],label="Sublattice A")
    plt.bar([k+1 for k in range(2*N)][1:][::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[1:][::2],label="Sublattice B")
    plt.title("$E={}$".format( round(list(cp.asnumpy(eigensystem[0]))[k],4)))
    plt.xlabel("Lattice position")
    plt.ylabel("Wavefunction")
    plt.legend()
    plt.show()
