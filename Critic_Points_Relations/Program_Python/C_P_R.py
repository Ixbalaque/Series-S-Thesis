import numpy as np
import random 
import matplotlib.pyplot as plt

def g_n(x, a, s, theta):
    return a / (1 + np.exp(s * (x + theta)))

def S_func(x, a, s, theta, K):
    return np.sum([g_n(x, a[n], s[n], theta[n]) for n in range(len(a))], axis=0) + K

# Ajuste lineal: y = m*x + b    
def linear_fit(x, y): 
    m, b   = np.polyfit(x, y, 1)
    y_fit  = np.polyval([m, b], x)
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return m, b, r2, y_fit

#Ajuste logarítmico: y ≈ A * ln(x) + B
def log_fit(x, y):

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    X_log = np.log(x)

    A, B = np.polyfit(X_log, y, 1)
    y_fit = A * np.log(x) + B

    # Calcular R²
    ss_res = np.sum((y - y_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0

    return A, B, r2, y_fit


p = 0.95
Lp  = np.log(1/p - 1)      # para máximos
L1p = np.log(1/(1-p) - 1)  # para mínimos

# Parámetros
theta0 = 0
theta1 = -3
a0     = 1
a1     = 5
s0     = -4
s1     = -Lp / ((theta0 - theta1) - Lp/s0)

theta  = [theta0, theta1]
a      = [a0, a1]
s      = [s0, s1]

####
####  Primer Prueba
####

#--- Caso 1 a0=var a1=const s0=const s1=const theta0-theta1=const ----# 

# Número de casos
Na   = 30
a1   = 5
a[1] = a1

# Dominio de integración y evaluación
x  = np.linspace(-theta0, -theta1, 2000)

# Listas para almacenar resultados
a0_vals    = []
a1_vals    = []
integrals  = []
x_max_vals = []
S_max_vals = []

#listas para alamacenar los fit.
"""---Para la intgral en funcion a0----"""
mI_a0  = []
bI_a0  = []
r2I_a0 = []
yI_fit = []

"""---Para la x_max en funcion a0----"""
mx_a0  = []
bx_a0  = []
r2x_a0 = []
yx_fit = []

"""---Para la S_max en funcion a0----"""
mS_a0  = []
bS_a0  = []
r2S_a0 = []
yS_fit = []

#--- Configuracion para graficar N_graficas de las Na pruebas que se hicieron ----# 

N_graf = 10             # Numero de curvas
a_graf = random.sample(range(1, Na+1), N_graf)
S_graf_list = []        # valores S para cada curva
labels_graf = []        # etiquetas para cada curva
crit_points = []        # puntos criticos para cada curva


for a0 in range(1, Na+1):
    
    a[0] = a0
    K = 0
    #K = -a0/2 - a1 

    S = S_func(x, a, s, theta, K)    
     
    # Integral en [-theta0, -theta1]
    integral_val = np.trapz(S, x)

    # Máximo y mínimo
    idx_max = np.argmax(S)
  
    x_max, S_max = x[idx_max], S[idx_max]

    if a0 in a_graf:
        S_graf_list.append(S)                   # guardamos las datos de S
        labels_graf.append(f"a0={a0}")          # guardamos la leyenda para a
        crit_points.append([x_max,S_max])       # guardamos el punto critico para S
  
    # Guardar resultados
    a0_vals.append(a0)
    integrals.append(integral_val)
    x_max_vals.append(x_max)
    S_max_vals.append(S_max)
    


# Ajuste lineal para Imtegral
m,b,r2,y_fit =linear_fit(a0_vals,integrals);
mI_a0.append(m)
bI_a0.append(b)
r2I_a0.append(r2)
yI_fit = y_fit

print("Ajuste lineal para Imtegral")
print(f"y ≈ {mI_a0[0]:.4f} x + {bI_a0[0]:.4f}")
print(f"R² = {r2I_a0[0]:.4f}")
       
#  Ajuste logarítmico para x_max
m,b,r2,y_fit = log_fit(a0_vals, x_max_vals)
mx_a0.append(m)
bx_a0.append(b)
r2x_a0.append(r2)
yx_fit = y_fit

print("Ajuste logarítmico para x_max")    
print(f"y ≈ {mx_a0[0]:.4f} ln(x) + {bx_a0[0]:.4f}")
print(f"R² = {r2x_a0[0]:.4f}")

# Ajuste lineal para S_max
m,b,r2,y_fit =linear_fit(a0_vals, S_max_vals);
mS_a0.append(m)
bS_a0.append(b)
r2S_a0.append(r2)
yS_fit = y_fit

print("Ajuste lineal para S_max")
print(f"y ≈ {mS_a0[0]:.4f} x + {bS_a0[0]:.4f}")
print(f"R² = {r2S_a0[0]:.4f}")

        
# --- Graficar S para a0 ---
plt.figure(figsize=(8,5))

for i, S in enumerate(S_graf_list):
    plt.plot(x, S, linestyle="--", label=labels_graf[i])

# Marcar con X los máximos
crit_points = np.array(crit_points)   # lo pasamos a numpy
plt.scatter(crit_points[:,0], crit_points[:,1], 
            color="red", marker="x", s=100, label="Máximos")

plt.xlabel("x")
plt.ylabel("S(x)")
plt.legend()
plt.grid(True)
plt.title("Curvas S(x) para distintos a0 aleatorios")
plt.show()


# -------- Graficar resúmenes --------
plt.figure(figsize=(12,5))

# --- Integral ---
plt.subplot(1,3,1)
plt.plot(a0_vals, integrals, "bo", label="Integral S(x)")
plt.plot(a0_vals, yI_fit, "orange", linestyle="--", label="Fit lineal")
plt.xlabel("a0")
plt.ylabel("Integral S(x)")
plt.title("Integral de S(x) en [-θ0, -θ1]")
plt.legend()
plt.grid(True)
plt.text(0.25, 0.15, f"y ≈ {mI_a0[0]:.2f} x + {bI_a0[0]:.2f}\nR²={r2I_a0[0]:.4f}",
         transform=plt.gca().transAxes, color="black", fontsize=10, verticalalignment='top')

# --- x_max ---
plt.subplot(1,3,2)
plt.plot(a0_vals, x_max_vals, "ro", label="x_max")
plt.plot(a0_vals, yx_fit, "orange", linestyle="--", label="Fit log")
plt.xlabel("a0")
plt.ylabel("x de máximo")
plt.title("Posición de extremos de S(x)")
plt.legend()
plt.grid(True)
plt.text(0.25, 0.15, f"y ≈ {mx_a0[0]:.3f} ln(x) + {bx_a0[0]:.3f}\nR²={r2x_a0[0]:.4f}",
         transform=plt.gca().transAxes, color="black", fontsize=10, verticalalignment='top')

# --- S_max ---
plt.subplot(1,3,3)
plt.plot(a0_vals, S_max_vals, "go", label="S_max")
plt.plot(a0_vals, yS_fit, "orange", linestyle="--", label="Fit lineal")
plt.xlabel("a0")
plt.ylabel("Valor de S(x)")
plt.title("Valores extremos de S(x)")
plt.legend()
plt.grid(True)
plt.text(0.25, 0.15, f"y ≈ {mS_a0[0]:.2f} x + {bS_a0[0]:.2f}\nR²={r2S_a0[0]:.4f}",
         transform=plt.gca().transAxes, color="black", fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()
