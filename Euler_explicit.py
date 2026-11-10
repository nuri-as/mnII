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

#Mallats
h1 = 0.51
h2 = 0.49
h3 = 0.25
N = 101

#Intervals
Delta_x = 1/(N-1) #perquè al normalitzar, el domini de x és [0,1]


h=[h1, h2, h3]
dt = [h1*Delta_x**2, h2*Delta_x**2, h3*Delta_x**2]

k = 0


while k<=2:
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
            T[j+1,i] = T[j,i] + gamma*(T[j, i+1] -2*T[j,i]+T[j,i-1]) + Delta_t



    #print(T[t_passes]) #aquesta serà la matriu

    #per obtenir els valors de les temperatures a t_a (t_max), li demano que em tregui la última fila.
    #tindré les temperatures finals per cada posicio (N)
    T_final = T[-1, :]

    #Per cada h, li demano que m'imprimeixi aquest vector T.
    print('Pel cas h= ',h[k], 'el vector temperatura final és: ', T_final)

    k+= 1


