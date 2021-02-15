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
from sympy.physics.quantum.dagger import Dagger
import IPython.display as disp
import matplotlib.pyplot as plt
import time
import sympy as sp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import string

# XvXwXvXwXvXwXvXwXwXvXwXvXwXvXwXvX   v<w

v = 0.58 #0.58
w = 1#0.8
coupling = 1
N = 4  #unit cells per side!

energy_niveau = 7  # {0,1, ... , 2*N-1}


useGpuAccel = False

#a,b = sp.symbols("alpha, beta")
#H_sp = sp.zeros(2*N)
#for k in range(N):
#    H_sp[2*k-1,2*k] = b
#    H_sp[2*k,2*k-1] = b
#    H_sp[2*k,2*k+1] = a
#    H_sp[2*k+1,2*k] = a
#        
#    H_sp[-1,0] = 0
#    H_sp[0,-1] = 0
#print(sp.latex(H_sp))

if useGpuAccel:
    H = cp.zeros(( 4*N,4*N ),dtype=np.float64)
else:
    H = np.zeros(( 4*N,4*N ),dtype=np.float64)


t0 = time.time()
#populate H:
for k in range(N):
    H[2*k-1,2*k] = w
    H[2*k,2*k-1] = w
    H[2*k,2*k+1] = v
    H[2*k+1,2*k] = v
        
    H[-1,0] = 0
    H[0,-1] = 0
for k in range(N,2*N):
    H[2*k-1,2*k] = v
    H[2*k,2*k-1] = v
    H[2*k,2*k+1] = w
    H[2*k+1,2*k] = w
        
    H[-1,0] = 0
    H[0,-1] = 0
H[2*N-1,2*N] = coupling
H[2*N,2*N-1] = coupling


if useGpuAccel:
    eigensystem = cp.linalg.eigh(H)
else:
    eigensystem = np.linalg.eigh(H)
# print("Eigenenergies:\n",*[ round(k,4) for k in list(cp.asnumpy(eigensystem[0]))])
 
# print("calculation time: ",time.time()-t0,"s")

abscissa = [k*0.5+0.75 for k in range(4*N)]

def symmetrize(array): # symmetrization like in the 
    pass

# # ==== Alle Wellenfunktionen plotten:
# for k in range(4*N):
#     plt.bar([k+1 for k in range(4*N)][::2],np.absolute(cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[::2]),label="Basisatom A")
#     plt.bar([k+1 for k in range(4*N)][1:][::2],np.absolute(cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[1:][::2]),label="Basisatom B")
#     plt.title("Wellenfunktion der Eigenenergie {}".format( round(list(cp.asnumpy(eigensystem[0]))[k],4)))
#     plt.xlabel("Gitterplatz")
#     plt.ylabel("Wellenfunktion [a.u.]")
#     plt.legend()
#     plt.show()

# ==== Interface State schön plotten:
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.set_xlabel("Gitterplatz")
ax.set_ylabel("Wellenfunktion [a.u.]")

intensity = np.absolute(cp.asnumpy(eigensystem[1][:,8:9].T[0]))
latticeIndex = [k+1 for k in range(4*N)]

ax.bar(latticeIndex[:8][::2],intensity[:8][::2],color="tab:orange")
ax.bar(latticeIndex[:8][1::2],intensity[:8][1::2],color="tab:blue")
ax.bar(latticeIndex[8:][::2],intensity[8:][::2],color="tab:blue")
ax.bar(latticeIndex[8:][1::2],intensity[8:][1::2],color="tab:orange")
ax.set_yscale("log")
ax.minorticks_on()
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.tick_params(which='major', width=1.00,length=4,direction="in")
ax.tick_params(which='minor', width=0.75,length=2,direction="in",bottom=0,top=0)
ax.set_ylim((np.min(intensity)/1.5,1.5*np.max(intensity)))
ax.vlines(8.5,1e-3,1e1,linewidth=1,color="lime",linestyles="--")
ax.set_xlim((0.3,16.7))
# ax.set_ylim((1e-4,1.6e2))
ax.legend([],title="$\\alpha=0.58$, $\\beta=1$, $\\gamma=1$")
plt.tight_layout()
plt.savefig("interfaceStatesTheory.pdf")
plt.show()



