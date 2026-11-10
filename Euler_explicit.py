import numpy as np

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
Delta_t = h1*Delta_x**2

t_max = 0.025
t_passes = int(t_max/Delta_t) #nombre de passes
T = np.zeros((t_passes+1,N+1))

#condicions inicials i de contorn
T[0] = T_cos/T0
T[:,0] = T_cos/T0
T[:,-1] = T_cos/T0


gamma = Delta_t / Delta_x**2


for i in range(t_passes):
    for j in range(N-1):
        T[i, j+1] = T[i,j] + gamma*(T[i+1,j] -2*T[i,j]+T[i-1, j]) + Delta_t
        print(T[i, j+1], i, j)

print(T[t_passes])

