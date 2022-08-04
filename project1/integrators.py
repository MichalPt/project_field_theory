from sympy import Symbol, diff, integrate, exp, sqrt, factorial, oo
from .states import *


def h1_1(bra, ket, var, Z=2, eta=1):
    prefactor = 1
    
    h_diff = -1/2 * diff(ket(var), var, 2) - 1/var * diff(ket(var), var, 1) 
    h = ket.l * (ket.l + 1)/(2 * var**2) - eta/var + 1/2
    
    integral = integrate(var**2 * bra(var) * (h_diff + h * ket(var)), (var, 0, oo))
    return integral.doit() * prefactor


def h1(bra1, bra2, ket1, ket2, var1, var2, Z=2, eta=1):
    prefactor = 1
    
    part1 = h1_1(bra1, ket1, var1, eta=eta) * integrate(var2**2 * bra2(var2) * ket2(var2), (var2, 0, oo))
    part2 = integrate(var1**2 * bra1(var1) * ket1(var1), (var1, 0, oo)) * h1_1(bra2, ket2, var2, eta=eta)
    
    return prefactor * (part1 + part2)


def h2(bra1, bra2, ket1, ket2, l, var1 = r1, var2=r2, Z=2):    
    prefactor = 1 /(2*l + 1)
    
    # branch r1 > r2
    integral1 = integrate(var1**2 * var2**2 * (bra1 * bra2) * var2**l * var1**(-l-1) * (ket1 * ket2), (var2, 0, var1), (var1, 0, oo))
    
    # branch r1 < r2
    integral2 = integrate(var1**2 * var2**2 * (bra1 * bra2) * var1**l * var2**(-l-1) * (ket1 * ket2), (var2, var1, oo), (var1, 0, oo))
    
    return prefactor * (integral1 + integral2)

