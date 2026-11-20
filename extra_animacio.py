import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#Definim primer les condicions inicials
T_cos = 36.5 + 273 #K, temperatura inicial i a la frontera
z = 0.56 #W/(m*K)
cv = 3698 #J/(kg*K)
rho = 1081 #kg/m^3
sigma = 0.472 #S/m
alpha = z/(cv*rho)
Delta_V = 40 #V

#enunciat
# t_max=0.025 #aquí ja no podem posar aquest temps ja que no ens donava bé
N=101


#Constants de normalització
T0 = sigma * (Delta_V)**2 / (2*z)
d = 0.02 #m, gruix de paret entre els ventricles
t0 = d**2/alpha

"""
SOLUCIÓ ANALÍTICA
"""

# Definim la funció per calcular la solució analítica:
def T_analitica(x_tilde, t_tilde, Tcos, T0, n_termes):
  T= Tcos / T0 #terme de fora del sumatori
  
  #fem un for per calcular el sumatori:
  for n in range(1, n_termes + 1):
    p=(2*n-1)*np.pi
    ter_temp = (1.0 - np.exp(-(p**2) * t_tilde)) / (p**2)
    ter_pos = np.sin(p * x_tilde)

    T+= (4/p * ter_temp * ter_pos)
  return T

"""
DISCRETITZACIÓ DEL TEMPS, trobem un temps que faci complir les condicions de l'enunciat
"""
t_max = 0
T_sa_max = 50 + 273 #K, límit regió sana
T_limit  = 80 + 273 #K, límit absolut
x_front  = 0.0075 / d #normalitzat, 1a frontera entre el teixit sa i malalt

#busquem t_tilde tal que T(x_front, t_tilde) = 323 K perquè aixi el teixit malalt estarà a la màxima temperatura el major temps possible

def f(t_tilde): #definim una funcio que ens retorni la diferència entre la temperatura a la 1a frontera i els 50ºC
    return T_analitica(x_front, t_tilde, T_cos, T0, 200)*T0 - T_sa_max

t_inf, t_sup = 0.0, 0.025 #ens definim un interval de temps de 0 a 0.025 ja que a 0.025 sobrepassavem llavors la solució ha d'estar entremig
for _ in range(100):
    t_mit = 0.5*(t_inf + t_sup) #per moltes iteracions anem calculant la mitjana entre els extrems superior i inferior de l'interval
    if f(t_mit) < 0: #el temps serà màxim quan la diferència sigui més petita que 0
        t_inf = t_mit
    else:
        t_sup = t_mit #si el temps fa que la temperatura sobrepassi Tsa max agafem el nou valor superior per acotar l'interval
t_max = t_inf


"""
EULER IMPLÍCIT
"""
#mallats
h = 0.5 #ja que hem definit delta_t = h*(delta_x)**2, agafem només una discretització perquè només necessitem una solució

Delta_x = 1/(N-1)
dt = h*Delta_x**2

#El que farem en aquest cas és aplicar moltes iteracions, ja que tenim un sistema d'equacions i hem d'aplicar Gauss-Seidel
error_tolerat = 1e-10 #marquem un màxim de l'error que volem obtenir al fer les iteracions.
max_iter = 10000 #posem un límit al nombre d'iteracions que volem fer.


#com que només tenim una discretització, no cal fer un bucle
Delta_t = dt
gamma = Delta_t / Delta_x**2

t_passes = int(t_max / Delta_t)
T = np.zeros((t_passes + 1, N)) #definim una matriu buida on anirem guardant els valors de la temperatura, acabem a ta+1 perquè necessitem termes superiors per trobar ta amb l'implicit

# condicions inicials i de contorn
T[0] = T_cos/T0
T[:, 0] = T_cos/T0
T[:, -1] = T_cos/T0

for j in range(1, t_passes + 1): #comencem a l'1 perque tenim les condicions inicials

    #fem un ansatz: suposem inicialment que la temperatura a t+1 és la mateixa que a t, i després anirem actualitzant-la amb Gauss-Seidel
    for i in range(N):
        T[j][i] = T[j-1][i] #així tenim una matriu on tots els punts inicialment valen Tcos/T0

    #iteracions
    for it in range(max_iter):
        error_max = 0.0
        for i in range(1, N - 1): #anem de 1 a N-1 per no esborrar les condicions de contorn
            T_nou = (gamma * (T[j][i+1] + T[j][i-1]) + T[j-1][i] + Delta_t) / (1 + 2*gamma) #euler implícit

            #calculem la diferencia entre el nou valor i l'antic, i comparem l'error d'aquesta iteracio amb el de la iteracio anterior
            error_max = max(error_max, np.abs(T_nou - T[j][i])) #aixi ens quedem amb l'error més gran de tota la malla, no només del punt i
            T[j][i] = T_nou   #actualitzem en la mateixa iteracio el terme Tji (tal i com diu Gauss-Seidel)

        if error_max < error_tolerat:
            break
    

"""
MAPA DE TEMPERATURES
"""

plt.figure(figsize=(7, 5))

#per crear una imatge colorejada, li diem que representi les temperatures (desnormalitzades), 
plt.imshow(T*T0, aspect='auto', cmap='inferno', origin='lower', extent=[0, d, 0, t_max*t0])

plt.colorbar(label='Temperatura (K)')

plt.xlabel('x(m)')
plt.ylabel('t(s)')
plt.title(rf"Evolució del camp de temperatures durant {t_max*t0:.2f} s")
plt.show()
