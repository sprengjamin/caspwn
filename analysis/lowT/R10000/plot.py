import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k, hbar, c, Boltzmann
import sys
sys.path.append("/home/benjamin/wd/casimir_data/src")
from polylogarithm import polylog2
from scipy.integrate import quad, dblquad
from numba import njit

def integrand(K, k):
    kappa = np.sqrt(k**2+K**2)
    return k/kappa*polylog2(np.exp(-2*kappa))*2

def PFA2(R, L, T):
    f = lambda k: integrand(0., k)
    en = quad(f, 0., np.inf)[0]/L

    K_matsubara = 2*np.pi*Boltzmann*T*L/(hbar*c)
    n = 1
    while(True):
        f = lambda k: integrand(K_matsubara*n, k)
        term = 2*quad(f, 0., np.inf)[0]/L
        en += term
        if term/en < 1.e-16:
            break
        n += 1
    print("nmax", n)
    return -R*0.25*Boltzmann*T*en

def PFA_zero(R, L):
    en = dblquad(integrand, 0, np.inf, lambda x: 0., lambda x: np.inf)[0]
    return R/L*en
        

R = 1
L = 1e-4

# xi = 0 term
Phi0 = -3003.887089368429+-2993.191818497945
Phi0 = -3003.8870889806353+-2993.1918181101632

data = np.loadtxt("energy_lowT_R10000_eta12.dat").T

pfa_zero = -np.pi**3*R/L/720.*hbar*c/L

Tvalues = data[0,:]
en = data[1,:]

print("pfa_zero", PFA_zero(R, L))
for i, T in enumerate(Tvalues):
    print(T)
    pfa2 =  PFA2(R, L, T)
    print("pfa2", pfa2)
    lambda_T = hbar*c/k/T
    fen = (0.5*k*T*Phi0 + en[i])
    print(en[i], fen)
    print("r", fen)
    #plt.plot(L/lambda_T, np.abs((fen-pfa2)*L/hbar/c/pfa_zero+45/np.pi**3*L**2/lambda_T/R*np.log(L/R)**2), "bx")
    #plt.loglog(L/lambda_T, np.abs((fen-pfa2)*L/hbar/c/pfa_zero), "bx")
    try1 = (fen-pfa2)/pfa_zero- L/R*(1/3-20/np.pi**2)
    #plt.loglog(L/lambda_T, try1, "bx")
    #plt.semilogx(L/lambda_T, pfa2, "rx")
    #plt.semilogx(L/lambda_T, pfa_zero, "rx")
    #plt.semilogx(L/lambda_T, np.abs(np.log(L/lambda_T)**2), "gx")
    #plt.plot(L/lambda_T, L/R*(1/3-20/np.pi**2), "gx")
    #plt.loglog(L/lambda_T, np.abs(-45/np.pi**3*L**2/lambda_T/R*np.log(L/R)**2), "kx")
    #plt.loglog(L/lambda_T, np.abs(90/np.pi**3*L**2/lambda_T/R*np.log(L/lambda_T)**2), "rx")
    try2 = -45/np.pi**3*L**2/lambda_T/R*np.log(L/R)**2+90/np.pi**3*L**2/lambda_T/R*np.log(L/lambda_T)**2
    try21 = -45/np.pi**3*L**2/lambda_T/R*np.log(L/R)**2
    try22 = 90/np.pi**3*L**2/lambda_T/R*np.log(L/lambda_T)**2
    plt.loglog(L/lambda_T, np.abs(try1-try2), "cx")
    plt.loglog(L/lambda_T, np.abs(try1-try21), "gx")
    plt.loglog(L/lambda_T, np.abs(try1-try22), "bx")
    plt.loglog(L/lambda_T, np.abs(try1), "yx")

l_T = hbar*c/k/Tvalues
x = L/l_T
plt.loglog(x, 0.02*x, "k-")
plt.loglog(x, 0.004*x, "k-")
plt.xlabel(r"$L/\lambda_T$")
plt.ylabel(r"correction ")
#plt.axhline(y = 2.6*(L/R)**1.5)
#plt.axhline(y = pfa_zero)
plt.show()
