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

#enunciat
t_max=0.025
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
EULER IMPLÍCIT
"""
#mallats
h = [1, 0.5] #ja que hem definit delta_t = h*(delta_x)**2

Delta_x = 1/(N-1)
dt = [h[0]*Delta_x**2, h[1]*Delta_x**2]

#El que farem en aquest cas és aplicar moltes iteracions, ja que tenim un sistema d'equacions i hem d'aplicar Gauss-Seidel
error_tolerat = 1e-10 #marquem un màxim de l'error que volem obtenir al fer les iteracions.
max_iter = 10000 #posem un límit al nombre d'iteracions que volem fer.


llista_col = ['darksalmon', 'mediumvioletred']

for k in range(len(h)):
    Delta_t = dt[k]
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


    t_fin = (len(T) - 1) * Delta_t           # temps real assolit (adimensional)
    x_espai = np.linspace(0, 1, N)
    T_anal = T_analitica(x_espai, t_fin, T_cos, T0, 200)
    T_final = T[-1, :]
    

    #"Desnormalitzem"
    x_desnorm = x_espai * d
    T_desnorm = T_anal*T0
    T_impl = T_final*T0
    t_desnorm = t_fin * t0


    plt.plot(x_desnorm, T_impl, '--', color= llista_col[k], linewidth=2,
               label=f'Euler implícit, h={h[k]}')

# la solució analítica és la mateixa per a tots dos (mateix t_fin)
plt.plot(x_desnorm, T_desnorm, 'k', linewidth=1.5, label='Analítica')


plt.title(rf"Perfil de temperatura a $t \approx {t_desnorm:.2f}$ s")
plt.xlabel('x (m)')
plt.ylabel('T (K)')
plt.legend()

plt.show()
