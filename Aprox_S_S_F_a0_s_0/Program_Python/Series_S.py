import numpy as np
import matplotlib.pyplot as plt
from analyze_signal import AnalyzeSignal

# --------- Definir funciones ---------
def f_1(x):
    return -3*x**4 + x**3 + 2*x**2 - x + 1

def f_2(x):
    return 2.0 * (1.0 - 1.547 * np.exp(-1.5 * x) * np.sin(2.5981 * x + 1.0472))

def f_3(x):
    return np.sin(4*x)

def g_n(x, a, s, theta):
    return a / (1 + np.exp(s * (x + theta)))

def S_func(x, a, s, theta, K):
    return np.sum([g_n(x, a[n], s[n], theta[n]) for n in range(len(a))], axis=0) + K


def Calculate_Param_S(inc_point, fin_point, infl_points, crit_points, a0, s0, p=0.95):
    N   = len(crit_points)
    y_p = np.zeros(N+1)
    theta = np.zeros(N+1)
    a = np.zeros(N+1)
    s = np.zeros(N+1)
    a[0] = a0
    s[0] = s0
    theta[0] = -1*inc_point[0]
    theta[N] = -1*fin_point[0] 
    y_p[0]   = inc_point[1]
    y_p[N]   = fin_point[1]
    K = 0
    
    Lp  = np.log(1/p - 1)      # para máximos
    L1p = np.log(1/(1-p) - 1)  # para mínimos
 
    for n in range(0,N-1):
        theta[n+1] =  -1*infl_points[n, 0]
        y_p[n+1]   =  infl_points[n, 1]
     
    for n in range(1,N+1):
        if crit_points[n-1,2] == "max":
            s[n] = -Lp  / ((theta[n-1] - theta[n]) - Lp  / s[n-1])
            a[n] = a[n-1] - 2*(y_p[n] - y_p[n-1])
            if n == 1:  K += - a[n-1]/2  - a[n] + y_p[0]
            else: K += -a[n]
        else: 
            s[n] = -L1p / ((theta[n-1] - theta[n]) - L1p / s[n-1])
            a[n] = a[n-1] + 2*(y_p[n] - y_p[n-1])   
            if n == 1: K += -a[n-1]/2 + y_p[0] 

    return theta, a, s, K


# --------- Función general ---------
def run_Series_S(f, N, t_i, t_f, a0, s0, p=0.95, title="Ajuste con sigmoides"):
    # Crear señal
    signal = [[t, f(t)] for t in np.linspace(t_i, t_f, N)]

    # Analizar señal
    analyzer    = AnalyzeSignal(signal, normalize=True)
    f_n         = analyzer.get_normalized_signal()  
    crit_points = analyzer.get_critical_points()
    infl_points = analyzer.get_inflexion_points()
    max_point   = analyzer.get_max_point()
    inc_point   = f_n[0]
    fin_point   = f_n[-1]

    # Calcular parámetros
    theta, a, s, K = Calculate_Param_S(inc_point, fin_point, infl_points, crit_points, a0, s0, p)

    # Graficar
    x_vals = np.linspace(t_i, t_f, 6000)
    y_vals = np.array([S_func(x, a, s, theta, K) for x in x_vals])

    plt.figure(figsize=(8,5))
    plt.plot(f_n[:,0], f_n[:,1], label="f(x) normalizada", color="blue")  
    plt.plot(x_vals, y_vals, label="S(x) sigmoides", color="red", linestyle="--")
    plt.scatter(infl_points[:,0], infl_points[:,1], color="green",  label="Inflexión")
    plt.scatter(crit_points[:,0], crit_points[:,1], color="orange", label="Críticos")
    plt.scatter([max_point[0]], [max_point[1]], color="purple", label="Pico global")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# --------- Ejemplos ---------
# Para f_1
run_Series_S(f=f_1, N=2000, t_i=-1, t_f=1, a0=2.2166, s0=-8.559, title="f_1")

# Para f_2
run_Series_S(f=f_2, N=20000, t_i=-0.8061, t_f=2.8214, a0=2.2, s0=4.88, title="f_2")

# Para f_3
run_Series_S(f=f_3, N=20000, t_i=-2*np.pi, t_f=2*np.pi, a0=1.2, s0=-10, title="f_3")
