import numpy as np
import matplotlib.pyplot as plt


beta = 1 # inter-cell coupling
N = 8 # Unit cells

steps = 2000 # resolution
v_list = [ 0.001 + 3*1/steps*k for k in range(steps+1)] # bei v=0: Err, da degenerierte EW
graphs = [ [] for k in range(2*N) ]

H = np.zeros(( 2*N,2*N ))


#populate H:
for alpha in v_list:
    for k in range(N):
        H[2*k-1,2*k] = beta
        H[2*k,2*k-1] = beta
        H[2*k,2*k+1] = alpha
        H[2*k+1,2*k] = alpha
        
    H[-1,0] = 0
    H[0,-1] = 0

    energies = np.linalg.eigvalsh(H)#[ sp.re(key.evalf()) for key in H.eigenvals().keys() ] # realteil, da floatingpoint errors zu kleinen imaginären Anteilen führen
    energies.sort()
    for k in range(2*N):
        graphs[k].append(energies[k])


fig, ax = plt.subplots(figsize=(6,4.5))
for k in range(2*N):
    ax.plot(v_list,graphs[k],color="k",linewidth=0.5)
plt.xlim(0,3)
plt.ylim(-3,3)
plt.xlabel("$α$")
plt.ylabel("Energy $E$")
plt.text(0.2, -2.6, '$β={}$'.format(beta))
plt.vlines(1,-3,3,color="grey",linestyle="-.",linewidth=0.5)
fig.tight_layout()
plt.show()
