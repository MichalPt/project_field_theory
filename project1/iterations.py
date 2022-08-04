import numpy as np
from sympy import Symbol, factorial, diff, integrate, exp, sqrt, oo
from sympy.physics.quantum.cg import CG
from functools import partial  
from IPython.utils.io import capture_output


def f58(vect, enlist, l, sign=1, eta=[1,1,1,1], order=np.array([3,2,1,0])):
    order = max(order) - order
    vect = vect.astype(float)
    n = enlist
    
    out0 = vect.copy()
    out1 = vect.copy()
    out2 = vect.copy()
    out3B = vect.copy()
    
    out0[order[2]] += 1
    out1[order[2]] += 2
    out2[order[2]] += 1
    out2[order[3]] += 1
    out3B[order[2]] += 1
    
    if sign == 1:
        L = -l - 1
    elif sign == -1:
        L = l
        
    l1 = vect[order[3]]
    l2 = vect[order[2]] + 1
    
    ci = -eta[order[2]] * np.sqrt((n[order[2]])**2 - (l2**2))/l2 * (-1 + (l2 -1 - l1 - L)/(2*l2 + 1))
    c0 = (enlist[order[3]] * eta[order[3]] / (l1 + 1)) - (n[order[2]] * eta[order[2]] / l2) + (eta[order[2]] * n[order[2]] * (l2 -1 - l1 - L))/(l2*(l2 + 1))
    c1 = eta[order[2]] * (l2 - 1 - l1 - L)/(2*l2 + 1) * np.sqrt((n[order[2]])**2 - (l2+1)**2)/(l2 + 1)
    c2 = eta[order[3]] * np.sqrt((n[order[3]])**2 - (l1 + 1)**2)/(l1 + 1)
    print("   c0 = {} , c1 = {} , c2 = {} , ci = {}".format(c0,c1,c2,ci))
    
    for out,c in zip([out0,out1,out2,out3B],[c0,c1,c2,-sign]): 
        out[4] *= c/ci
    
    return [out0, out1, out2], out3B


def f59(vect, enlist, l, sign=1, eta=[1,1,1,1], order=np.array([3,2,1,0])):
    order = max(order) - order
    vect = vect.astype(float)
    n = enlist
    
    out0 = vect.copy()
    out1 = vect.copy()
    out2 = vect.copy()
    out3B = vect.copy()
    
    out0[order[0]] += 1
    out1[order[0]] += 2
    out2[order[0]] += 1
    out2[order[1]] += 1
    out3B[order[0]] += 1
        
    if sign == -1:
        L = -l - 1
    elif sign == 1:
        L = l
        
    l3 = vect[order[1]]
    l4 = vect[order[0]] + 1
    
    ci = -eta[order[0]] * np.sqrt(n[order[0]]**2 - l4**2)/l4 * (-1 + (l4 - 1 - l3 - L)/(2*l4 + 1))
    c0 = eta[order[1]] * n[order[1]]/(l3 + 1) - eta[order[0]] * n[order[0]]/l4 + eta[order[0]] * n[order[0]] * (l4 - 1 - l3 - L)/(l4*(l4 + 1))
    c1 = eta[order[0]] * (l4 - 1 - l3 - L)/(2*l4 + 1) * np.sqrt((n[order[0]])**2 - (l4 + 1)**2)/(l4 + 1)
    c2 = eta[order[1]] * np.sqrt((n[order[1]])**2 - (l3 + 1)**2)/(l3 + 1)
    
    for out,c in zip([out0,out1,out2,out3B],[c0,c1,c2,sign]):
        out[4] *= c/ci
    
    return [out0, out1, out2], out3B


def f60(vect, enlist, eta=[1,1,1,1], order=np.array([3,2,1,0])):
    order = max(order) - order
    vect = vect.astype(float)
    n = enlist
    
    out0B = vect.copy()
    out1B = vect.copy()
    out2B = vect.copy()
    out3B = vect.copy()
    out4B = vect.copy()
    
    out0B[order[0]] += 1
    out1B[order[0]] += 2
    out2B[order[0]] += 1
    out2B[order[1]] += 1
    out3B[order[0]] += 1
    out3B[order[2]] += 1
    out4B[order[0]] += 1
    out4B[order[3]] += 1
    
    l1 = vect[order[3]]
    l2 = vect[order[2]]
    l3 = vect[order[1]]
    l4 = vect[order[0]] + 1
    
    ci = - eta[order[0]] * np.sqrt((n[order[0]])**2 - l4**2)/l4 * (-1 + (l4 - l1 - l2 - l3 - 2)/(2*l4 + 1))
    c0 = eta[order[1]] * n[order[1]] /(l3+1) + eta[order[2]] * n[order[2]] /(l2 + 1) + eta[order[3]] * n[order[3]] /(l1 + 1) - eta[order[0]] * n[order[0]] / l4 + eta[order[0]] * n[order[0]] * (l4 - l3 - l2 - l1 - 2)/(l4 * (l4 + 1))
    c1 = eta[order[0]] * (l4 - l3 - l2 - l1 - 2)/(2*l4 + 1) * np.sqrt((n[order[0]])**2 - (l4 + 1)**2)/(l4 + 1) 
    c2 = eta[order[1]] * np.sqrt((n[order[1]])**2 - (l3 + 1)**2)/(l3 + 1)
    c3 = eta[order[2]] * np.sqrt((n[order[2]])**2 - (l2 + 1)**2)/(l2 + 1) * (-1)
    c4 = eta[order[3]] * np.sqrt((n[order[3]])**2 - (l1 + 1)**2)/(l1 + 1)
    
    for out,c in zip([out0B,out1B,out2B,out3B,out4B],[c0,c1,c2,c3,c4]):
        out[4] *= c/ci
    
    return [out0B, out1B, out2B, out3B, out4B]


def iterator(elist, enlist, l, sign=1, eta2=[1,1,1,1], silent=True):
    with capture_output() as captured:
        delta = enlist - elist[:4]
        assert (np.array(elist[:4]) >= 0).all(), "All values of 'l' needs to be non-negative"
        #delta = [ delta[i] for i in [2,3,0,1] ]

        elevate1 = partial(f58, enlist=enlist, l=l, sign=sign, eta=eta2, order=np.array([3,2,0,1]))
        elevate2 = partial(f58, enlist=enlist, l=l, sign=sign, eta=eta2, order=np.array([3,2,1,0]))
        elevate3 = partial(f59, enlist=enlist, l=l, sign=sign, eta=eta2, order=np.array([2,3,1,0]))
        elevate4 = partial(f59, enlist=enlist, l=l, sign=sign, eta=eta2, order=np.array([3,2,1,0]))
        elevate = [elevate2, elevate1, elevate4, elevate3]

        elevate1B = partial(f60, enlist=enlist, eta=eta2, order=np.array([0,2,1,3]))
        elevate2B = partial(f60, enlist=enlist, eta=eta2, order=np.array([1,2,3,0]))
        elevate3B = partial(f60, enlist=enlist, eta=eta2, order=np.array([2,3,1,0]))
        elevate4B = partial(f60, enlist=enlist, eta=eta2, order=np.array([3,2,1,0]))
        elevateB = [elevate2B, elevate1B, elevate4B, elevate3B]

        newlist = list()
        vectlistA = [np.array(elist),]
        vectlistB = list()

        for i, (eleA, eleB) in zip([2,3,0,1], zip(elevate, elevateB)):
            genlistB = list()
            setsthesameA = False
            setsthesameB = False
            
            print("\n>>> PART A ({}, {}) <<<  ................................\n".format(i,sign))
            k = 0
            while not setsthesameA:
                print("K = {}".format(k))
                k += 1
                genlistA = list()

                for vectorA in vectlistA:
                    generatedA = list()
                    generatedB = list()

                    if (enlist - vectorA[:4])[i] > 1:
                        generatedA, generatedB = eleA(vectorA)
                        genlistB.append(generatedB) if not (enlist - generatedB[:4] < 1).any() or generatedB[4] != 0 else None
                    else:
                        generatedA.append(vectorA)

                    print("\n genB {}\n".format(generatedB))
                    print("  {}  -->  {}\n".format(vectorA, generatedA))
                    genlistA = genlistA + [ v for v in generatedA if not (enlist - v[:4] < 1).any() or v[4] != 0]
                    
                setsthesameA = True if np.array_equal(vectlistA, genlistA) else False
                print("\ncomparison: {} == {} is same: {}".format(vectlistA, genlistA, setsthesameA))
                vectlistA = genlistA.copy()

            vectlistB = vectlistB + genlistB
            
            print("\n   >>> resulted B: {} \n".format(vectlistB))

            print("\n>>> PART B ({}, {}) <<<  .............\n".format(i, sign))
            k = 0
            while (not setsthesameB):
                print("K = {}".format(k))
                k += 1
                genlistB = list()

                for vectorB in vectlistB:
                    generatedB = list()

                    if (enlist - vectorB[:4])[i] > 1:
                        generatedB = eleB(vectorB)
                    else:
                        generatedB.append(vectorB)

                    genlistB = genlistB + [ v for v in generatedB if not (enlist - v[:4] < 1).any() or v[4] != 0 ]
                    print("  {}  -->  {}\n".format(vectorB, generatedB))

                setsthesameB = True if np.array_equal(vectlistB, genlistB) else False
                print("\ncomparison: {} == {} is same: {}".format(vectlistB, genlistB, setsthesameB))
                vectlistB = genlistB.copy()

            print("   >>> resulted B: {}".format(vectlistB))

        print(">>>>> TOTAL RESULTS <<<<<\n   A: {} \n   B: {}".format(vectlistA,vectlistB))
        if (np.unique(np.array(vectlistA)[:,:4],axis=0).shape[0] == 1):
            outputA = np.array(vectlistA)[:,4].sum()
        if len(vectlistB) > 0 and (np.unique(np.array(vectlistB)[:,:4],axis=0).shape[0] == 1):
            outputB = np.array(vectlistB)[:,4].sum() 
        else:
            outputB = 0
    
    if not silent:
        print(captured)
    
    #return vectlistA, vectlistB
    return outputA, outputB


###########################################

def one_particle_integral(n11, n12, l1, n21, n22, l2, eta=1, particle=1):
    assert particle in [1, 2], "the value of argument particle has to be either 1 or 2"
    particle -= 1
    p = particle
    q = 1 - p
    
    bra = [n11, n12, l1]
    ket = [n21, n22, l2]
    
    if (bra[2] != ket[2]) or (bra[p] != ket[p]):
        print("no one particle contribution")
        return 0
    
    prefactor = ket[p] - eta
    bracket0 = ket[q] if ket[q] == bra[q] else 0
    bracket1 = -sqrt((ket[q] - ket[2])*(ket[q] + ket[2] + 1)) / 2 if (bra[q] == ket[q] + 1) else 0
    bracket2 = -sqrt((ket[q] + ket[2])*(ket[q] - ket[2] - 1)) / 2 if (bra[q] == ket[q] - 1) else 0
    
    return prefactor * (bracket0 + bracket1 + bracket2)


def integral_J(alpha, beta, a, b, l):
    """
    Arguments:
        alpha ... scaling factor x 2 !!!)
        beta ... scaling factor x 2 !!!)
        a, b ... exponent of r1, r2 respectively, from jacobian and wavefunctions only (do not include multipole expansion term)
    """
    prefactor = factorial(a - l - 1)
    out = 0
    
    for q in range(0, a-l):
        out0 = factorial(b + l + q) / factorial(q) * alpha**int(-a + l + q) * (alpha + beta)**int(-b - l - q - 1)
        out += out0
        
    return prefactor * out


def integral_I(alpha, beta, a, b, l):
    return integral_J(alpha, beta, a, b, l) + integral_J(beta, alpha, b, a, l) 

def integral_square_bracket(l4,l3,l2,l1):
    ltot = int(l1+l2+l3+l4)
    return 1 / sqrt(factorial(2*l4 + 1) * factorial(2*l3 + 1) * factorial(2*l2 + 1) * factorial(2*l1 + 1)) * 2**(-ltot-4) * factorial(ltot + 3)

def coeff(l, alpha=1):
    return 2 / sqrt(factorial(2*l + 1)) * (2 * alpha)**l


def two_particle_integral(n11, n12, l1, n21, n22, l2, eta=1, Z=2, typ="c", silent=True):
    assert typ in ["c", "e"], "the value of typ argument can be either 'c' (Coulomb) or 'e' (exchange)"
    
    bra = [n11, n12, l1]
    ket = [n21, n22, l2]
    
    if typ == "c":
        #       l1      l2       l1      l2
        els = [bra[2], ket[2], bra[2], ket[2]]
        #       n11     n21      n12     n22
        ens = [bra[0], ket[0], bra[1], ket[1]]
        
    elif typ == "e":
        els = [bra[2], ket[2], bra[2], ket[2]]
        ens = [bra[0], ket[1], bra[1], ket[0]]
    else:
        raise ValueError()
    
    ens = np.array(ens)
    reduce = partial(iterator, elist=np.array(els+[1],dtype=float), enlist=ens, silent=silent)
    
    prefactor = (-1)**(bra[2] + ket[2]) * sqrt((2*bra[2] + 1)*(2*ket[2] + 1)) * eta / Z
    
    l_max = bra[2] + ket[2]
    l_min = abs(ket[2] - bra[2])
    
    out = 0
    
    for l in range(l_min, l_max+1):
        if (l - l_min) > 0 and (l - l_min) % 2 == 1:
            continue
            
        out_plus = 0
        out_minus = 0
        
        plusA, plusB = reduce(l=l,sign=1)
        minusA, minusB = reduce(l=l,sign=-1)
        
        nfree = ens - 1
        
        print("\n>>> A:\n PLUS:",plusA, "MINUS:", minusA, " L:",l)
        print("\n>>> B:\n PLUS:",plusB, "MINUS:", minusB, " L:",l)
        
        cf = coeff(nfree[0]) * coeff(nfree[1]) * coeff(nfree[2]) * coeff(nfree[3])        
        
        out_plus += cf * plusA * integral_J(2, 2, nfree[2] + nfree[3] + 2, nfree[0] + nfree[1] + 2, l)
        out_minus += cf * minusA * integral_J(2, 2, nfree[0] + nfree[1] + 2, nfree[2] + nfree[3] + 2, l)
        
        out_plus += plusB * integral_square_bracket(nfree[2], nfree[3], nfree[0], nfree[1])
        out_minus += minusB * integral_square_bracket(nfree[0], nfree[1], nfree[2], nfree[3])

        out += (out_plus + out_minus) * (CG(ket[2], 0, bra[2], 0, l, 0).doit())**2 / (2*l + 1)
        
    return float(out * prefactor)

    