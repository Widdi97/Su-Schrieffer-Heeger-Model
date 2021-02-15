import sympy as sp
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
import time


w = 1
N = 4

steps = 2000
v_list = [ 0.001 + 3*1/steps*k for k in range(steps+1)] # bei v=0: Err, da degenerierte EW
graphs = [ [] for k in range(2*N) ]

useGpuAccel = False

#H_bulk = sp.zeros(2*N)
if useGpuAccel:
    H = cp.zeros( ( 2*N,2*N ) )
else:
    H = np.zeros(( 2*N,2*N ))


t0 = time.time()
#populate H:
for v in v_list:
#    for k in range(N):
#        H_bulk[2*k-1,2*k] = w
#        H_bulk[2*k,2*k-1] = w
#        H_bulk[2*k,2*k+1] = v
#        H_bulk[2*k+1,2*k] = v
    
    for k in range(N):
        H[2*k-1,2*k] = w
        H[2*k,2*k-1] = w
        H[2*k,2*k+1] = v
        H[2*k+1,2*k] = v
        
    H[-1,0] = 0
    H[0,-1] = 0

    if useGpuAccel:
        energies = cp.linalg.eigvalsh(H)
    else:
        energies = np.linalg.eigvalsh(H)#[ sp.re(key.evalf()) for key in H.eigenvals().keys() ] # realteil, da floatingpoint errors zu kleinen imaginären Anteilen führen
    energies.sort()
    for k in range(2*N):
        graphs[k].append(energies[k])
        
print("calculation time: ",time.time()-t0,"s")



##  ===== NORMAL PLOTTEN
#for k in range(2*N):
#    plt.plot(v_list,graphs[k],color="k",linewidth=0.5)
#plt.xlim(0,3)
#plt.ylim(-3,3)
#plt.xlabel("$α$")
#plt.ylabel("Energie $E$")
#plt.text(0.2, -2.6, '$β={}$'.format(w))
#plt.title("Energiespektrum des SSH Modells für\nvariable Kopplungsparameter $α$")
#plt.vlines(1,-3,3,color="grey",linestyle="-.",linewidth=0.5)
#plt.show()


#  ===== SCHÖN PLOTTEN
fig, ax = plt.subplots(figsize=(6,4.5))
for k in range(2*N):
    ax.plot(v_list,graphs[k],color="k",linewidth=0.5)
plt.xlim(0,3)
plt.ylim(-3,3)
plt.xlabel("$α$")
plt.ylabel("Energie $E$")
plt.text(0.2, -2.6, '$β={}$'.format(w))
plt.vlines(1,-3,3,color="grey",linestyle="-.",linewidth=0.5)
plt.vlines(0.58,-3,3,color="red",linestyle="--",linewidth=0.5)
plt.text(0.2,2.3,"α = 0.58",color="red")
plt.plot([0.34, 0.57], [2.2, 1.9], 'red', lw=1)
fig.tight_layout()
plt.savefig("Energiespektrum.pdf")
plt.show()
