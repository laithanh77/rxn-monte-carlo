import random as rand
import math
from math import exp
from mpl_toolkits.mplot3d import Axes3D as plt3d
import numpy as np
import time
from scipy.integrate import quad as integrate
from atom import Atom, Iodide

class Diatomic():
    def __init__(self, pos, sigma, epsilon, bond_length, vdw, mass, charge):
        #default parameters = I2
        #I2 bond length = 2.666 A
        self.kb = 1.38064852e-23
        self.sigma = sigma 
        self.epsilon = epsilon
        self.bond_length = bond_length #Angstrom
        self.vdw = vdw
        self.pos = pos #dictionary. key = atom, value = np.array([x_coord, y_coord])
        self.V = 0 #potential energy
        self.mass = mass
        self.charge = 1.60217662e-19 * charge
        self.vel = np.array([0, 0], dtype = float) #starts at 0. to save computation, only assign velocities during recombination
    
    def getPosc(self):
        #return center position coordinates
        return np.array([(self.pos[0][0] + self.pos[1][0])/2,(self.pos[0][1] + self.pos[1][1])/2]) 
    
    def __ccw(self, A,B,C):
        # source: https://stackoverflow.com/questions/70528/why-are-pythons-private-methods-not-actually-private
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def __If_Intersect(self, A,B,C,D):
        # source: https://stackoverflow.com/questions/70528/why-are-pythons-private-methods-not-actually-private
        # Return true if line segments AB and CD intersect
        return self.__ccw(A,C,D) != self.__ccw(B,C,D) and self.__ccw(A,B,C) != self.__ccw(A,B,D)
    
    
    def __If_Collision(self, pos_test, configuration):
        #PROBAAABBLY dont need to check collision bc if overlap then r = 0 and the probability acceptance ratio is infinity
        #and so the translation will never be accepeted
        #if any(np.array_equal(self.pos[0], otherParticle.pos[0]) for otherParticle in configuration if otherParticle is not self) or \
        #    any(np.array_equal(self.pos[1], otherParticle.pos[1]) for otherParticle in configuration if otherParticle is not self):
        #    print("Collision! Atom overlap")
        #    return True
        if any(self.__If_Intersect(self.pos[0], self.pos[1], otherParticle.pos[0], otherParticle.pos[1]) \
               for otherParticle in configuration if otherParticle is not self and isinstance(otherParticle, Diatomic)):
            print("Collision! Bond overlap")
            return True
        return False
    
    def Translate(self, configuration, lim):
        #return trial coordinates after translation. does not update self's coordinates
        low = -0.01
        high = 0.01
        dx = rand.uniform(low, high)
        dy = rand.uniform(low, high)
        pos_trial = {0: self.pos[0] + [dx, dy], 1: self.pos[1] + [dx, dy]}
        attempts = 0
        while self.__If_Collision(pos_trial, configuration) or \
                any(pos_trial[0] > lim) or any(pos_trial[1] > lim) or \
                any(pos_trial[0] < [0,0]) or any(pos_trial[1] < [0,0]) or attempts < 100:
            #print("Rerolling translation coordinates")
            dx = rand.uniform(low, high)
            dy = rand.uniform(low, high)
            pos_trial = {0: self.pos[0] + [dx, dy], 1: self.pos[1] + [dx, dy]}
            attempts += 1
        return pos_trial
        
    def Rotate(self, configuration, lim):
        #returns trial coordinates after rotation. does not update self's coordinates
        cx = (self.pos[0][0] + self.pos[1][0])/2 #center x
        cy = (self.pos[0][1] + self.pos[1][1])/2 #center y
        th = rand.uniform(0, 2*math.pi) #random theta
        
        #trial x coordinate for atom 1
        x1t = ((self.pos[0][0] - cx) * math.cos(th) + (self.pos[0][1] - cy) * math.sin(th) ) + cx 
        #trial y coordinate for atom 1
        y1t = (-(self.pos[0][0] - cx) * math.sin(th) + (self.pos[0][1] - cy) * math.cos(th) ) + cy 
        
        #trial x coordinate for atom 2
        x2t = ((self.pos[1][0] - cx) * math.cos(th) + (self.pos[1][1] - cy) * math.sin(th) ) + cx 
        #trial y coordinate for atom 2
        y2t = (-(self.pos[1][0] - cx) * math.sin(th) + (self.pos[1][1] - cy) * math.cos(th) ) + cy 
        
        pos_trial = {0: np.array([x1t, y1t]), 1: np.array([x2t, y2t])}
        attempts = 0
        while self.__If_Collision(pos_trial, configuration) or \
                any(pos_trial[0] + [self.vdw, self.vdw] > lim) or any(pos_trial[1] + [self.vdw,  self.vdw] > lim) or \
                any(pos_trial[0] < [0,0]) or any(pos_trial[1] < [0,0]) or attempts < 100:
            #print("Rerolling rotation")
            th = rand.uniform(0, 2*math.pi) #random theta
        
            #trial x coordinate for atom 1
            x1t = ((self.pos[0][0] - cx) * math.cos(th) + (self.pos[0][1] - cy) * math.sin(th) ) + cx 
            #trial y coordinate for atom 1
            y1t = (-(self.pos[0][0] - cx) * math.sin(th) + (self.pos[0][1] - cy) * math.cos(th) ) + cy 

            #trial x coordinate for atom 2
            x2t = ((self.pos[1][0] - cx) * math.cos(th) + (self.pos[1][1] - cy) * math.sin(th) ) + cx 
            #trial y coordinate for atom 2
            y2t = (-(self.pos[1][0] - cx) * math.sin(th) + (self.pos[1][1] - cy) * math.cos(th) ) + cy 

            pos_trial = {0: np.array([x1t, y1t]), 1: np.array([x2t, y2t])}
            attempts += 1
        return pos_trial
    


class H2(Diatomic):
    def __init__(self, pos):
        self.kb = 1.38064852e-23
        #LJ parameters from: Michels, Graaf, Seldam 1960 physica 26 393-408
        #suggested LJ parameters as "best" agreement, but still off from experimental.
        #sigma converted from reported units to Angstroms
        Diatomic.__init__(self, pos, sigma = 2.953, epsilon = 36.7 * self.kb, bond_length = 0.74, vdw = 1.20, \
                          mass = 3.34711e-27, charge = 0)
        
class HI(Diatomic):
    def __init__(self, pos):
        self.kb = 1.38064852e-23
        #vdw is average of H and I for now..will change later. vdw is just to make plot look better a bit better
        #pos 1 is H, pos 2 is I
        Diatomic.__init__(self, pos, sigma = 4.080, epsilon = 333.6 * self.kb, bond_length = 1.609, vdw = 1.59, \
                          mass = 2.12403e-25, charge = 0)


class I2(Diatomic):
    def __init__(self, pos):
        #sigma & epsilon values: Mourits and Rummens. Can. J. Chem. 55, 3007 (1977)
        self.kb = 1.38064852e-23
        Diatomic.__init__(self, pos, sigma = 4.630, epsilon = 577.4 * self.kb, bond_length = 2.666, vdw = 1.98, \
                          mass = 4.2145962337e-25, charge = 0)
        
    def Dissociate(self, system, lim):
        for particle in system.configuration:
            if isinstance(particle, H2):
                #add tolerance
                m1 = (particle.pos[0][1] - self.pos[0][1])/(self.pos[1][1]-self.pos[0][1])
                m2 = (particle.pos[1][1] - self.pos[0][1])/(self.pos[1][1]-self.pos[0][1])
                b1 = (particle.pos[0][0] - self.pos[0][0])/(self.pos[1][0]-self.pos[0][0])
                b2 = (particle.pos[1][0] - self.pos[0][0])/(self.pos[1][0]-self.pos[0][0])
                tol = 1.98 #vdw radius...a very poor approximation since nuceli must collide, but how sparse the
                            #system is, its a good idea to make the tolerance a little bit higher
                if (m1-tol <= b1 <= m1+tol) or (m2-tol <= b2 <= m2+tol):
                    #if H2 center is colinear with I2
                    H_1 = particle.pos[0] #hydrogen 1
                    H_2 = particle.pos[1] #hydrogen 2

                    d = (H_1 - H_2)/np.linalg.norm(H_1-H_2) #normalized vector distance between the two hydrogens
                    #0: H, 1: I
                    system.configuration = np.append(system.configuration, (HI({0:H_1, 1:H_1+d*1.609})))
                    system.configuration = np.append(system.configuration, (HI({0:H_2, 1:H_2+d*-1.609})))
                    system.configuration = np.delete(system.configuration, np.argwhere(system.configuration == particle))
                    system.configuration = np.delete(system.configuration, np.argwhere(system.configuration == self))
                    print("Hit H2!")
                    return
        else:
            #if no colinear H2, reaction does not proceed and I radical is formed
            m = (self.pos[0][1] - self.pos[1][1])/(self.pos[0][0] - self.pos[1][0])
            b = self.pos[0][1]-(self.pos[0][0]*m)
            #y = mx + b
            #check case intersection y = 0, x = -b/m
            xt = -b/m
            if 0 <= xt <= lim[0]:
                x1 = xt
                y1 = 0
                
                #check every other case to determine other radicals coordinates
                yt = b
                if 0 <= yt <= lim[1]:
                    x2 = 0
                    y2 = yt
                yt = m*lim[0] + b
                if 0 <= yt <= lim[1]:
                    x2 = lim[0]
                    y2 = yt
                xt = (lim[1]-b)/m
                if 0 <= xt <= lim[0]:
                    x2 = xt
                    y2 = lim[1]
                
            
            #check case intersection x = 0, y = b
            yt = b
            if 0 <= yt <= lim[1]:
                x1 = 0
                y1 = yt
                
                xt = -b/m
                if 0 <= xt <= lim[0]:
                    x2 = xt
                    y2 = 0
                yt = m*lim[0] + b
                if 0 <= yt <= lim[1]:
                    x2 = lim[0]
                    y2 = yt
                xt = (lim[1]-b)/m
                if 0 <= xt <= lim[0]:
                    x2 = xt
                    y2 = lim[1]
            
            #check case intersection x = lim
            yt = m*lim[0] + b
            if 0 <= yt <= lim[1]:
                x1 = lim[0]
                y1 = yt
                
                yt = b
                if 0 <= yt <= lim[1]:
                    x2 = 0
                    y2 = yt
                xt = -b/m
                if 0 <= xt <= lim[0]:
                    x2 = xt
                    y2 = 0
                xt = (lim[1]-b)/m
                if 0 <= xt <= lim[0]:
                    x2 = xt
                    y2 = lim[1]
                
            #check case intersection y = lim
            xt = (lim[1]-b)/m
            if 0 <= xt <= lim[0]:
                x1 = xt
                y1 = lim[1]
                
                yt = b
                if 0 <= yt <= lim[1]:
                    x2 = 0
                    y2 = yt
                yt = m*lim[0] + b
                if 0 <= yt <= lim[1]:
                    x2 = lim[0]
                    y2 = yt
                xt = -b/m
                if 0 <= xt <= lim[0]:
                    x2 = xt
                    y2 = 0
            
            system.configuration = np.append(system.configuration, Iodide(np.array([x1, y1])))
            system.configuration = np.append(system.configuration, Iodide(np.array([x2, y2])))
            system.configuration = np.delete(system.configuration, np.argwhere(system.configuration == self))
            print("Formed 2I!")
        