import random as rand
import math
from math import exp
from mpl_toolkits.mplot3d import Axes3D as plt3d
import numpy as np
import time
from scipy.integrate import quad as integrate

class Atom():
    def __init__(self, pos, sigma, epsilon, vdw, mass, charge):
        '''
        '''
        #default parameters = argon
        self.kb = 1.38064852e-23
        self.mass = mass
        self.vdw = vdw #angstrom...just random value of previous proposed
        self.sigma = sigma # mourits and rummens. Can. J. Chem. 55, 3007 (1977)
        self.epsilon = epsilon #joules, mourits and rummens. Can. J. Chem. 55, 3007 (1977)
        self.V = 0
        self.pos = pos #np.array([x,y]) #center position, which is the same as position
        self.charge = 1.60217662e-19 * charge
        self.vel = np.array([0, 0], dtype = float) #starts at 0. to save computation, only assign velocities during recombination
        #self.x = x
        #self.y = y
    
    def getPosc(self):
        #redundant but creating this method so that certain lines of codes are universal regardless of
        #Atom or Diatomic type objects.
        return self.pos

class Iodide(Atom):
    def __init__(self, pos):
        self.kb = 1.38064852e-23
        #initialized with argon parameters...will change later
        Atom.__init__(self, pos, sigma = 3.465, epsilon = 113.5*self.kb, vdw = 1.98, mass = 2.1072981168e-25, charge = 0)
    
    def Translate(self, configuration, lim):
        low = -0.01
        high = 0.01
        pos_trial = self.pos + [rand.uniform(low, high), rand.uniform(low, high)]
        if pos_trial[0] < 0:
            pos_trial[0] = abs(pos_trial[0] + lim[0])
        elif pos_trial[0] > lim[0]:
            pos_trial[0] = pos_trial[0] - lim[0]
        if pos_trial[1] < 0:
            pos_trial[1] = abs(pos_trial[1] + lim[1])
        elif pos_trial[1] > lim[1]:
            pos_trial[1] = pos_trial[1] - lim[1]
        return pos_trial