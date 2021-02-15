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

v = 0.58 #0.58
w = 1#0.8
N = 6

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
    H = cp.zeros(( 2*N,2*N ),dtype=np.float64)
else:
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

if useGpuAccel:
    eigensystem = cp.linalg.eigh(H)
else:
    eigensystem = np.linalg.eigh(H)
print("Eigenenergies:\n",*[ round(k,4) for k in list(cp.asnumpy(eigensystem[0]))])
 
print("calculation time: ",time.time()-t0,"s")

abscissa = [k*0.5+0.75 for k in range(2*N)]


# ==== Alle Wellenfunktionen plotten:
for k in range(2*N):
    plt.bar([k+1 for k in range(2*N)][::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[::2],label="Basisatom A")
    plt.bar([k+1 for k in range(2*N)][1:][::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[1:][::2],label="Basisatom B")
    plt.title("Wellenfunktion der Eigenenergie {}".format( round(list(cp.asnumpy(eigensystem[0]))[k],4)))
    plt.xlabel("Gitterplatz")
    plt.ylabel("Wellenfunktion [a.u.]")
    plt.legend()
    plt.show()



# ==== Überlagerte SSH Mode
compositionFactor = 0.3
superposition = cp.asnumpy(eigensystem[1][:,N:N+1].T[0]) + compositionFactor*cp.asnumpy(eigensystem[1][:,N-1:N].T[0])
plt.bar([k+1 for k in range(2*N)][::2],np.absolute(superposition[::2]),label="Basisatom A")
plt.bar([k+1 for k in range(2*N)][1:][::2],np.absolute(superposition[1:][::2]),label="Basisatom B")
plt.xlabel("Gitterplatz")
plt.ylabel("Wellenfunktion [a.u.]")
plt.title("Superposition aus symmetrischer und\nantisymmetrischer Wellenfunktion")
plt.legend()
plt.show()



## ==== Vordefinierte Wellenfunktion plotten
#plt.bar([k+1 for k in range(2*N)][::2],cp.asnumpy(eigensystem[1][:,energy_niveau:energy_niveau+1].T[0])[::2],label="Basisatom A")
#plt.bar([k+1 for k in range(2*N)][1:][::2],cp.asnumpy(eigensystem[1][:,energy_niveau:energy_niveau+1].T[0])[1:][::2],label="Basisatom B")
#plt.xlabel("Einheitszelle")
#plt.ylabel("Wellenfunktion [a.u.]")
#plt.legend(framealpha=1)
#plt.show()


## ==== Vordefinierte Wellenfunktion schön plotten
#fig, ax = plt.subplots()
#ax.bar(abscissa[::2],cp.asnumpy(eigensystem[1][:,energy_niveau:energy_niveau+1].T[0])[::2],label="Basisatom A",width=0.4)
#ax.bar(abscissa[1:][::2],cp.asnumpy(eigensystem[1][:,energy_niveau:energy_niveau+1].T[0])[1:][::2],label="Basisatom B",width=0.4)
#locs, labels = plt.xticks([k+1 for k in range(N)])
#ax.autoscale(False)
#a = [plt.vlines(k+1.5,-1,1,linewidth=1,linestyles='--',color="grey") for k in range(N-1)]
#plt.xlim((0.5,N+0.5))
#plt.xlabel("Einheitszelle")
#plt.ylabel("Wellenfunktion [a.u.]")
#plt.legend(framealpha=1)
#fig.tight_layout()
#plt.show()



# ==== Alle Wellenfunktionen schön subplotten:
fig = plt.figure(figsize=(9,4))
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
ax.set_xlabel('Einheitszelle')
ax.set_ylabel('Wellenfunktion [a.u.]')
if N == 4:
    for k in range(2*N):
        en = round(list(cp.asnumpy(eigensystem[0]))[k],4)
        ax = fig.add_subplot(2,4,k+1)
        ax.bar(abscissa[::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[::2],label="Basisatom A",width=0.4)
        ax.bar(abscissa[1:][::2],cp.asnumpy(eigensystem[1][:,k:k+1].T[0])[1:][::2],label="Basisatom B",width=0.4)
        ax.set_xticks([k+1 for k in range(N)])
        ax.set_ylim((-0.65,0.65))
        ax.set_xlim((0.5,N+0.5))
        ax.autoscale(False)
        [plt.vlines(k+1.5,-1,1,linewidth=1,linestyles='--',color="grey") for k in range(N-1)]
        ax.text(0.07, 1.06, "("+string.ascii_lowercase[k]+"):  $E = "+str(en)+"$", transform=ax.transAxes, 
            size=11)
    plt.tight_layout()
    plt.legend()
    plt.savefig("Wellenfunktionen;N=4;alpha=0,58.pdf")
    plt.show()



## ==== exponentieller Abfall schön subplotten:
#vList = [0.1,0.2,0.4,0.6]
#fig = plt.figure(figsize=(9,5))
#ax = fig.add_subplot(111)
#ax.spines['top'].set_color('none')
#ax.spines['bottom'].set_color('none')
#ax.spines['left'].set_color('none')
#ax.spines['right'].set_color('none')
#ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
#ax.set_xlabel('Einheitszelle')
#ax.set_ylabel('Betrag der Wellenfunktion [a.u.]')
#for v_ in vList:
#    for k in range(N):
#        H[2*k-1,2*k] = w
#        H[2*k,2*k-1] = w
#        H[2*k,2*k+1] = v_
#        H[2*k+1,2*k] = v_
#        H[-1,0] = 0
#        H[0,-1] = 0
#    eigensystem = np.linalg.eigh(H)
#    ax = fig.add_subplot(2,2,vList.index(v_)+1)
#    ax.bar(abscissa[::2],np.absolute(cp.asnumpy(eigensystem[1][:,N:N+1].T[0])[::2]),label="Basisatom A",width=0.4)
#    ax.bar(abscissa[1:][::2],np.absolute(cp.asnumpy(eigensystem[1][:,N:N+1].T[0])[1:][::2]),label="Basisatom B",width=0.4)
#    ax.set_xticks([k+1 for k in range(N)])
#    ax.autoscale(False)
#    [plt.vlines(k+1.5,-1,1,linewidth=1,linestyles='--',color="grey") for k in range(N-1)]
#    ax.set_xlim((0.5,N+0.5))
#    ax.set_ylim((0,0.75))
#    ax.text(0.4, 1.03, "("+string.ascii_lowercase[vList.index(v_)]+"):  $α = "+str(v_)+"$", transform=ax.transAxes,size=11)
#plt.tight_layout()
#plt.legend()
#plt.savefig("ExponentiellerAbfallWellenfunktionen.pdf")
#plt.show()






