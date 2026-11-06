import numpy as np

#Definim primer les condicions inicials
T_cos = 36.5 + 273 #K, temperatura inicial i a la frontera
k = 0.56 #W/(m*K)
cv = 3698 #J/(kg*K)
rho = 1081 #kg/m^3
sigma = 0.472 #S/m
alpha = k/(cv*rho)

#Constants de normalització
l_0 = np.sqrt(k*T_cos)
t_0 = l_0**2/alpha
