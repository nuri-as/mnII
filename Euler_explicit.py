import numpy as np
import matplotlib.pyplot as plt

#Definim primer les condicions inicials
T_cos = 36.5 + 273 #K, temperatura inicial i a la frontera
z = 0.56 #W/(m*K)
cv = 3698 #J/(kg*K)
rho = 1081 #kg/m^3
sigma = 0.472 #S/m
alpha = z/(cv*rho)
Delta_V = 40 #V

#Constants de normalització
T0 = sigma * (Delta_V)**2 / (2*z)
d = 0.02 #m, gruix de paret entre els ventricles
t0 = d**2/alpha

#Mallats
h1 = 0.51
h2 = 0.49
h3 = 0.25
N = 101

#Intervals
Delta_x = 1/(N-1) #perquè al normalitzar, el domini de x és [0,1]


h=[h1, h2, h3]
dt = [h1*Delta_x**2, h2*Delta_x**2, h3*Delta_x**2]


for k in range(len(h)):
    Delta_t = dt[k]

    t_max = 0.025
    t_passes = int(round(t_max/Delta_t)) #nombre de passes (quants intervals de t hi ha). Important arrodonir

    #El que passa aquí és que per h=0.25 arrodoneix cap a dalt (1000 passes) i llavors 1000*delta t> 0.025, pero NO passa el mateix pels altres h's perquè arrodoneix cap abaix. haurem de definir-nos un t nou--> t_real
    t_real = t_passes*Delta_t
    T = np.zeros((t_passes+1,N))

    #condicions inicials i de contorn
    T[0] = T_cos/T0
    T[:,0] = T_cos/T0
    T[:,-1] = T_cos/T0

    gamma = Delta_t / Delta_x**2

    for j in range(t_passes):
        for i in range(1, N-1):
            T[j+1,i] = T[j,i] + gamma*(T[j, i+1] -2*T[j,i]+T[j,i-1]) + Delta_t

    #per obtenir els valors de les temperatures a t_a (t_max), li demano que em tregui la ultima fila, aixi tinc les temperatures finals per cada posicio (N)
    T_final = T[-1, :]

    #print('Pel cas h= ',h[k], 'el vector temperatura final és: ', T_final)

    #ara fem el plot de la distribució de temperatura per cada h:

    #Generem les x, que com estan uniformement distribuïdes, les podem generar amb linspace, amb N punts entre 0 i 1.
    x_espai=np.linspace(0, 1, N)

    #"Desnormalitzem"
    T_expl = T_final*T0
    x_expl = x_espai*d

    colors = ['aquamarine', 'mediumturquoise', 'teal']
    plt.plot(x_expl, T_expl, label='h= ' + str(h[k]), color=colors[k])


    plt.xlabel('x (m)')
    plt.ylabel('T (k)')
    plt.title('Distribució de temperatura')
    plt.legend()

    plt.show()





