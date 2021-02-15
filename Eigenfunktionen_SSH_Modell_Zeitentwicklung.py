import numpy as np
from scipy.linalg import expm, sinm, cosm
from sympy.physics.quantum.dagger import Dagger
import IPython.display as disp
import matplotlib.pyplot as plt
import time

"""
README:
    - Zeitentwicklung rotiert nur die Wellenfunktion und dreht jeden Vektoreintrag um den selben, zeitabhängigen
      Winkel in der komplexen Ebene

"""


v = 0.5#0.5797101449275363 # 0.46
w = 1
N = 6

energy_niveau = 0  # {0,1, ... , 2*N-1}

H = np.zeros(( 2*N,2*N ),dtype=np.float64)


t0 = time.time()
#populate H:
for k in range(N):
    H[2*k-1,2*k] = w
    H[2*k,2*k-1] = w
    H[2*k,2*k+1] = v
    H[2*k+1,2*k] = v
        
    H[-1,0] = 0
    H[0,-1] = 0

eigensystem = np.linalg.eigh(H)
print("Eigenenergies:\n",*[ round(k,4) for k in list(eigensystem[0])])
print("calculation time: ",time.time()-t0,"s")

# ==== time evolution
def U(t):  #t0 = 0
    return expm(-1j*H*t)
maximum = np.max(np.absolute(eigensystem[1][:,energy_niveau:energy_niveau+1]))
for t in range(0,1000):
    U_ = U(t/10)
    eigenvector = np.dot(U_,eigensystem[1][:,energy_niveau:energy_niveau+1])
    plt.bar([k+1 for k in range(2*N)][::2],np.real(eigenvector.T[0][::2]),label="Basisatom A")
    plt.bar([k+1 for k in range(2*N)][1:][::2],np.real(eigenvector.T[0][1:][::2]),label="Basisatom B")
    plt.xlabel("Gitterplatz")
    plt.ylabel("Realteil der Wellenfunktion")
    plt.ylim((-maximum,maximum))
    plt.title("$t = {}$".format(round(t/10,5)))
    plt.legend()
    plt.show()
    print("A: ",eigenvector.T[0][::2])
    print("B: ",eigenvector.T[0][1:][::2])



## ==== Alle Wellenfunktionen plotten:
#for k in range(2*N):
#    plt.bar([k+1 for k in range(2*N)][::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[::2],label="Basisatom A")
#    plt.bar([k+1 for k in range(2*N)][1:][::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[1:][::2],label="Basisatom B")
#    plt.title("Wellenfunktion der Eigenenergie {}".format( round(list(cp.asnumpy(eigensystem[0]))[k],4)))
#    plt.xlabel("Gitterplatz")
#    plt.ylabel("Wellenfunktion [a.u.]")
#    plt.legend()
#    plt.show()



## ==== Überlagerte SSH Mode
#compositionFactor = 0.3
#superposition = cp.asnumpy(eigensystem[1][:,N:N+1].T[0]) + compositionFactor*cp.asnumpy(eigensystem[1][:,N-1:N].T[0])
#plt.bar([k+1 for k in range(2*N)][::2],superposition[::2],label="Basisatom A")
#plt.bar([k+1 for k in range(2*N)][1:][::2],superposition[1:][::2],label="Basisatom B")
#plt.xlabel("Gitterplatz")
#plt.ylabel("Wellenfunktion [a.u.]")
#plt.title("Superposition aus symmetrischer und\nantisymmetrischer Wellenfunktion")
#plt.legend()
#plt.show()



# ==== Vordefinierte Wellenfunktion plotten
plt.bar([k+1 for k in range(2*N)][::2],eigensystem[1][:,energy_niveau:energy_niveau+1].T[0][::2],label="Basisatom A")
plt.bar([k+1 for k in range(2*N)][1:][::2],eigensystem[1][:,energy_niveau:energy_niveau+1].T[0][1:][::2],label="Basisatom B")
plt.xlabel("Gitterplatz")
plt.ylabel("Wellenfunktion [a.u.]")
plt.legend()
plt.show()