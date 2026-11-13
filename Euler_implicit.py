import numpy as np
import matplotlib.pyplot as plt

#Definim primer les condicions inicials
T_cos = 36.5 + 273 #K, temperatura inicial i a la frontera
k = 0.56 #W/(m*K)
cv = 3698 #J/(kg*K)
rho = 1081 #kg/m^3
sigma = 0.472 #S/m
alpha = k/(cv*rho)
Delta_V = 40 #V

T0 = sigma * (Delta_V)**2 / (alpha*k)

"""
EULER IMPLÍCIT
"""

#Mallats
h=[1, 0.5]

#Intervals
N = 101
Delta_x = 1/(N-1) #perquè al normalitzar, el domini de x és [0,1]

dt = [h[0]*Delta_x**2, h[1]*Delta_x**2]

k = 0

while k<=1:
    Delta_t = dt[k]

    t_max = 0.025
    t_passes = int(t_max/Delta_t) #nombre de passes (quants intervals de t hi ha)

    T = np.zeros((t_passes+1,N))


    #condicions inicials i de contorn
    T[0] = T_cos/T0
    T[:,0] = T_cos/T0
    T[:,-1] = T_cos/T0


    gamma = Delta_t / Delta_x**2

    #escrivim l'edo trobada i fem el bucle
    
    for j in range(t_passes):
        for i in range(1, N-1):
            T[j+1,i] = 1/(1+2*gamma) *(T[j-1,i] + gamma*(T[j, i+1] +T[j, i-1]) + Delta_t)



    #print(T[t_passes]) #aquesta serà la matriu

    #per obtenir els valors de les temperatures a t_a (t_max), li demano que em tregui la última fila.
    #tindré les temperatures finals per cada posicio (N)
    T_final = T[-1, :]

    #Per cada h, li demano que m'imprimeixi aquest vector T.
    print('Pel cas h= ',h[k], 'el vector temperatura final és: ', T_final)

   
    #fem el plot per treure el gràfic
    x_espai=np.linspace(0, 1, N)
    llista_col = ['mediumseagreen', 'darksalmon']
    plt.plot(x_espai, T_final, label='h= ' + str(h[k]), color = llista_col[k]) #aquesta linia ha d'estar dins del bucle per tenir un sol grafic
    k+= 1



"""
SOLUCIÓ ANALÍTICA
"""

# Definim la funció per calcular la solució analítica:
def T_analitica(x_tilde, t_tilde, Tcos, T0, n_termes):
    
    #fem un for per calcular el sumatori:
    for n in range(2, n_termes + 1):

        #Posem el terme inicial T, perquè així té un valor a partir del qual anar sumant els altres termes.
        p1 = np.pi
        ter_temp1 = (1.0 - np.exp(-(p1**2) * t_tilde)) / (p1**2)
        ter_pos1 = np.sin(p1 * x_tilde)

        T = (Tcos / T0) + (4 / p1) * ter_temp1 * ter_pos1 #T inicial n=1

        p=(2*n-1)*np.pi
        ter_temp = (1.0 - np.exp(-(p**2) * t_tilde)) / (p**2)
        ter_pos = np.sin(p * x_tilde)

        T+=Tcos/T0 + (4/p * ter_temp * ter_pos)

    return T

x_espai=np.linspace(0, 1, N)  # Espai normalitzat
# Gràfica per a l'instant t_max:

plt.plot(
    x_espai,
    T_analitica(x_espai, t_max, T_cos, T0,1000000),
    color="royalblue",
    linewidth=2,
    label=f"$\\tilde{{t}}_a = {t_max}$",
)


plt.title(f"Perfil de temperatura a $\\tilde{{t}}_a = {t_max}$")
plt.xlabel("$\\tilde{x}$")
plt.ylabel("$\\tilde{T}(\\tilde{x}, \\tilde{t}_a)$")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
