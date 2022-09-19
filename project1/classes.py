from sympy import Symbol, sqrt, exp, factorial
from .iterations import *
from IPython.utils.io import capture_output
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

class Config:
    def __init__(self, n1, n2, l):
        assert (l <= np.array([n1, n2])).all(), "l needs to be smaller than both n1 and n2"
        self.n1 = n1
        self.n2 = n2
        self.l = l
        self.x_repre = {1:MState(n1, l), 2:MState(n2,l)}
        
    def get_numbers(self):
        return [self.n1, self. n2, self.l]
    
    def get_x_repre(self, r1, a, r2, b):
        return self.x_repre[1](r1,a) * self.x_repre[2](r2,b)

    
def overlap(n11,n12,l1,n21,n22,l2,Z=2):
    if l1 != l2:
        return 0
    else:
        p1 = n21 if n11 == n21 else 0
        p2 = -sqrt((n21 - l1)*(n21 + l1 + 1))/2 if n11 == n21 +1 else 0 
        p3 = -sqrt((n21 + l1)*(n21 - l1 - 1))/2 if n11 == n21 -1 else 0 
        
        q1 = n22 if n12 == n22 else 0
        q2 = -sqrt((n22 - l2)*(n22 + l2 + 1))/2 if n12 == n22 +1 else 0 
        q3 = -sqrt((n22 + l2)*(n22 - l2 - 1))/2 if n12 == n22 -1 else 0 
        
        p = (p1 + p2 + p3)
        q = (q1 + q2 + q3)
        return p*q
    
    
class Configurations:
    def __init__(self, n12, Z=2, eta=1):
        assert n12 >= 2
        self.n12 = n12
        self.configs = list()
        self.Z = Z
        self.eta = eta
        self.N = 0
        
        self.hamiltonian = None
        self.overlap = None
        self.eigenvalues = None
        self.energy_gs = 0
        self.eng_ref = {2:-2.847656, 3:-2.847656, 4:-2.895444, 5:-2.897109, 6:-2.900714, 7:-2.901452, 8:-2.902341, 9:-2.902654, 10:-2.902975, 11:-2.903127}
        self.eta_ref = {2:32/27, 3:32/27, 4:0.971, 5:0.940, 6:0.796, 7:0.760, 8:0.682, 9:0.648, 10:0.595, 11:0.566}
        
        for n2 in range(1, n12):
            for n1 in range(n2, n12-n2+1):
                for l in range(0, min([n1,n2])):
                    self.N += 1
                    self.configs.append(Config(n2,n1,l))
                    
        print("States:", [x.get_numbers() for x in self.configs])
        print("\nTotal number of states:", self.N)

    def build_hamiltonian_matrix(self, silent=True):
        hamiltonian = np.zeros([len(self.configs)]*2)
        hamiltonian0 = np.zeros([len(self.configs)]*2)
        hamiltonianW = np.zeros([len(self.configs)]*2)
        if self.overlap is None:
            self.build_overlap_matrix()
        
        with capture_output() as captured:
            for i1, c1 in enumerate(self.configs):
                for i2, c2 in enumerate(self.configs):
                    print("\n<{}|...|{}>  ...........\n".format(c1.get_numbers(), c2.get_numbers()))
                    sign = 1
                    NC1 = c1.get_numbers()
                    NC2 = c2.get_numbers()
                    revNC1 = [NC1[i] for i in [1,0,2]]
                    revNC2 = [NC2[i] for i in [1,0,2]]
                    one = one_particle_integral
                    
                    E1 = one(*NC1, *NC2, eta=self.eta, particle=1) + one(*NC1, *NC2, eta=self.eta, particle=2) + one(*revNC1, *revNC2, eta=self.eta, particle=1) + one(*revNC1, *revNC2, eta=self.eta, particle=2) + sign * (one(*NC1, *revNC2, eta=self.eta, particle=1) + one(*NC1, *revNC2, eta=self.eta, particle=2) + one(*revNC1, *NC2, eta=self.eta, particle=1) + one(*revNC1, *NC2, eta=self.eta, particle=2))
                    
                    E2c = two_particle_integral(*c1.get_numbers(), *c2.get_numbers(), eta=self.eta, Z=self.Z, typ='c', silent=silent) 
                    E2e = two_particle_integral(*c1.get_numbers(), *c2.get_numbers(), eta=self.eta, Z=self.Z, typ='e', silent=silent)
                    
                    hamiltonian0[i1,i2] = E1 / 4
                    hamiltonianW[i1,i2] = (E2c + E2e)/2

                    #print("\n >>>  E1 = {:.5f}   E2c = {:.5f}  E2e = {:.5f}  ...  E = {:.6f}\n".format(*[float(eng) for eng in [E1,E2c,E2e,Etotal]]))
                    
        if not silent:
            print(captured)
        print("ETA: {}".format(self.eta))
        ham0 = hamiltonian0 - self.overlap
        self.hamiltonian0 = ham0
        self.hamiltonianW = hamiltonianW
        self.hamiltonian = self.hamiltonian0 + self.hamiltonianW
        
    def build_overlap_matrix(self):
        S = np.zeros([len(self.configs)]*2)
        
        for i1, c1 in enumerate(self.configs):
            for i2, c2 in enumerate(self.configs):
                S[i1,i2] = overlap(*c1.get_numbers(), *c2.get_numbers()) + overlap(*c1.get_numbers(), *[c2.get_numbers()[i] for i in [1,0,2]])
        self.overlap = S / 2
                          
    def get_energy(self, verb=True):
        if self.hamiltonian is None:
            self.build_hamiltonian_matrix()
        if self.overlap is None:
            self.build_overlap_matrix()
        
        #self.hamiltonian = self.hamiltonian0 + self.hamiltonianW
        #self.eigenvalues = eigh(self.hamiltonian, self.overlap, eigvals_only=True)
        #self.energy_gs = (self.eigenvalues[0] - 1) * self.Z**2 / self.eta**2
        
        self.hamiltonian = self.hamiltonian0 + self.hamiltonianW
        self.eigenvalues = eigh(self.hamiltonian * self.Z**2 / self.eta**2, self.overlap, eigvals_only=True)
        self.energy_gs = self.eigenvalues[0] 
        
        if verb:
            print("\nResults:  E = {:.6f} ,  eta = {:.5f} ".format(self.energy_gs, self.eta))
            print("Reference optimized values from the book:   E_ref = {:.6f}  ,  eta_ref = {:.5f}  ,  dE = {:.7f}".format(self.eng_ref[self.n12], self.eta_ref[self.n12], self.eng_ref[self.n12] - self.energy_gs))
        
        return self.energy_gs
    
    def optimize_eta(self, silent=True):
        def get_E(eta, obj):
            obj.eta = eta
            obj.build_hamiltonian_matrix()
            obj.build_overlap_matrix()
            obj.eigenvalues = eigh(obj.hamiltonian * obj.Z**2 / obj.eta**2, obj.overlap, eigvals_only=True)
            obj.energy_gs = obj.eigenvalues[0]
            return obj.energy_gs
        
        with capture_output() as captured:
            results = minimize_scalar(get_E, args=(self),bracket=(0.5,1.0,1.5))
            eta_opt = results['x']
            E_opt = results['fun']
            
        if not silent:
            print(captured)
        
        print("\nMinimized E = {:.6f}   for eta = {:.6f}".format(E_opt, eta_opt))
        return E_opt, eta_opt 
                       
        
        
class MState:
    def __init__(self, n, l):
        self.n = n
        self.l = l
        
    def __call__(self, variable, alpha=1):
        if self.n == self.l + 1:
            return 2*alpha / sqrt(factorial(2*self.l + 1)) * (2*alpha * variable)**self.l * exp(-alpha * variable)
        elif self.n == 2 and self.l == 0:
            return 2*sqrt(2)*(1 - variable) * exp(-variable)
        elif self.n == 3 and self.l == 0:
            return 2 * sqrt(3) * (1 - 2 * variable + 2 * variable**2 / 3) * exp(- variable)
            #return 2 * sqrt(3/7) * (1 - 2 * variable**2 / 3) * exp(- variable) 
        else:
            raise Exception("Only n, n-1 states are supported for now")

class State:
    def __init__(self, n, l):
        self.n = n
        self.l = l
        
    def __call__(self, variable, alpha, c0=0, c1=0):
        if self.n==1 and self.l==0:
            return 2*sqrt(alpha**3) * exp(-alpha*variable)                                        # pp. 149
        elif self.n==2 and self.l==1:
            return sqrt(alpha**5 / factorial(4)) * variable * exp(-alpha * variable / 2)          # pp. 150
        elif self.n==3 and self.l==2:
            return sqrt((2*alpha/3)**7 /factorial(6)) * variable**2 * exp(-alpha*variable / 3)    # pp. 157, úkol 14
        elif self.n==2 and self.l==0:
            c0 = sqrt(2) * alpha**(3/2) / 2
            c1 = - sqrt(2) * alpha**(5/2) / 4
            return (c0 + c1 * variable) * exp(- alpha * variable / 2)                             # pp. 157, úkol 15
        
