import numpy as np
from scipy.integrate import quad, solve_ivp
import matplotlib.pyplot as plt
#pip install numpy scipy matplotlib sympy

# Integral de sin(x) entre 0 i pi (hauria de donar 2)
valor, error = quad(np.sin, 0, np.pi)
print("Integral:", valor)

# EDO: y' = -y, y(0) = 1  (solució exacta: e^-t)
sol = solve_ivp(lambda t, y: -y, [0, 5], [1], dense_output=True)
t = np.linspace(0, 5, 100)
plt.plot(t, sol.sol(t)[0], label="numèrica")
plt.plot(t, np.exp(-t), "--", label="exacta")
plt.legend()
plt.show()