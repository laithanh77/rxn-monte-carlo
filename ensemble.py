import random as rand
import math
from math import exp
from mpl_toolkits.mplot3d import Axes3D as plt3d
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import quad as integrate
from atom import Atom, Iodide
from diatomic import Diatomic, H2, I2, HI

dt = 2.9116223400000000865
def pulse(t):
    #guassian form
    return 100*np.exp(-(1/500)*(t-60)**2)
total_pulse_area = integrate(pulse, 0, 120)[0]
def rand_vel_comp(M, T):
    #https://scicomp.stackexchange.com/questions/19969/how-do-i-generate-maxwell-boltzmann-variates-using-a-uniform-distribution-random
    #M = mass
    #T = temperature
    '''
    Uses uniform random number generator to return a velocity component from maxwell boltzmann distribution
    '''
    kb = 1.38064852e-23
    
    #did some test and found that the results for most probable velocity is off by a factor of sqrt(2)
    #the distribution does not fully represent the tail end of actual distributions...? not sure
    return math.sqrt((2*kb*T)/M) * rand.uniform(0,1)

def distance(atom1, atom2):
    #gives distance between atom
    return ((atom1.x - atom2.x)**2 + (atom1.y - atom2.y)**2)**(1/2)

def rand_sign():
    #returns a random sign
    if rand.random() < 0.5:
        return 1
    else:
        return -1

class Ensemble():
    #basically a numpy array containing objects but with more functionalities
    def __init__(self, N_I2, N_H2, lim = (20, 20), T = 300):
        self.configuration = np.array([])
        self.lim = lim
        self.T = T
        self.B = ((1.38064852e-23) * T)**-1
        #for i in range(N):
        #    self.Spawn(N)
        self.Spawn(N_I2, N_H2)
        #during sampling, any accepted configuration will have its energy sotred in this list.
        self.configuration_total_energies = np.array([self.Energy_Total()])
    def __getitem__(self, key):
        return self.configuration[key]
    
    def __setitem__(self, key, value):
        self.configuration[key] = value

    def append(self, value):
        self.configuration = np.append(self.configuration, value)
    
    def Distance(self, atom1, atom2):
    #gives distance between atom
        return ((atom1.x - atom2.x)**2 + (atom1.y - atom2.y)**2)**(1/2)
    
    def Check_VDW_Collision(self, testAtom, atom, k = 1):
        '''
        Parameters: k is a scaling factor for collision
        returns True if there exists a atom within testAtom's vdw radius
        '''
        return self.Distance(testAtom, atom) <= (testAtom.vdw + atom.vdw)*k

    def Spawn_Diatomic(self, N):
        pos1 = np.array((rand.uniform(0+10, self.lim[0]-10), rand.uniform(0+10, self.lim[1]-10)))
        dx = rand.uniform(-2.666, 2.666)
        pos2 = np.array((pos1[0]+dx,(2.666**2-(dx)**2)**0.5+pos1[1]))
        
        
        self.configuration = np.append(self.configuration, I2({0: pos1, 1: pos2}))
    
    def Spawn(self, N_I2, N_H2):
        spawn_tol = 2 #temporary variable to restrict spawn area
        for i in range(N_I2):
            pos1 = np.array((rand.uniform(0+spawn_tol, self.lim[0]-spawn_tol), rand.uniform(0+spawn_tol, self.lim[1]-spawn_tol)))
            dx = rand.uniform(-2.666, 2.666)
            pos2 = np.array((pos1[0]+dx,(2.666**2-(dx)**2)**0.5+pos1[1]))
            self.configuration = np.append(self.configuration, I2({0: pos1, 1: pos2}))
        
        for i in range(N_H2):
            pos1 = np.array((rand.uniform(0+spawn_tol, self.lim[0]-spawn_tol), rand.uniform(0+spawn_tol, self.lim[1]-spawn_tol)))
            dx = rand.uniform(-0.74, 0.74)
            pos2 = np.array((pos1[0]+dx,(0.74**2-(dx)**2)**0.5+pos1[1]))
            self.configuration = np.append(self.configuration, H2({0: pos1, 1: pos2}))
    
    def Update_Energy(self):
        #pairwise calculation & i != j
        e0 = 8.85418782e-12 #permititivy of free space in m^-3 kg^-1 s^4 Amphere^2
        for i in self.configuration:
            i.V = 0
            for j in self.configuration:
                if i == j:
                    continue
                d = np.linalg.norm(i.getPosc() - j.getPosc())
                #Lorentz-Berthelot mixing rule
                si = (i.sigma + j.sigma)/2 
                ep = (i.epsilon * j.epsilon)**1/2
                i.V += ((4 * ep * ((si/d)**12 - (si/d)**6))) #joules, LJ
                d *= 1e-10
                i.V += ((i.charge*j.charge)/(4*math.pi*e0*d)) #joules, electrostatics
                
    
    def Energy_Total(self):
        self.Update_Energy()
        return sum(atom.V for atom in self.configuration)
    
    def AcceptanceRule(self, Uo, Ut):
        #Canonical acceptance function
        try:
            return min(1, exp((-1)*self.B*(Ut - Uo)))
        except OverflowError:
            print("Overflow error! - Ensemble.AcceptanceRule()")
            return 1
    
    def Cycle(self, time_pulse, time_total):
        #check dissociation
        for particle in self.configuration:
            chance = rand.uniform(0, 1)
            if isinstance(particle, I2) and chance <= (integrate(pulse, time_pulse, time_pulse+dt)[0]/total_pulse_area):
                print("Dissociation!")
                print("Integral chance:", (integrate(pulse, time_pulse, time_pulse+dt)[0]/total_pulse_area)*100, "%")
                print("Random chance:", chance * 100, "%")
                print("Time pulse:",time_pulse)
                print("Time total:", time_total)
                print()
                particle.Dissociate(self, self.lim)
                
        
        #this boolean is to check if velocities has been assigned this cycle. Velocities are only assigned
        #if theres a possibility of recombination to save computational time
        assign_vel = False
        #check for recombination
        
        #I2 bond energy = 2.51542e-19 joules per bond
        for particle in self.configuration:
            if isinstance(particle, Iodide):
                for otherParticle in self.configuration:
                    if otherParticle != particle and isinstance(otherParticle, Iodide):
                        #if distance between two atoms are less than/equal to I2 bond length, check if recombination
                        #is possible
                        d = ((otherParticle.pos[0] - particle.pos[0])**2 + (otherParticle.pos[1] - particle.pos[1])**2)**0.5
                        if d <= 2.666:
                            #assign velocities
                            if assign_vel == False:
                                for all_particle in self.configuration:
                                    all_particle.vel[0] = rand_sign()*rand_vel_comp(all_particle.mass, self.T)
                                    all_particle.vel[1] = rand_sign()*rand_vel_comp(all_particle.mass, self.T)
                                assign_vel = True
                            center = (particle.getPosc() + otherParticle.getPosc())/2
                            for thirdParticle in self.configuration:
                                d = ((thirdParticle.getPosc()[1] - center[0])**2 + (thirdParticle.getPosc()[1] - center[1])**2)**0.5
                                if thirdParticle != particle and thirdParticle != otherParticle and d <= 2.666:
                                    #set spawn diatomic with total velocity
                                    #make thirdParticle have its velocity + the bond energy
                                    Vel = (2*(2.51542e-19)*(thirdParticle.mass**-1))**0.5 #total bond energy converted to velocity
                                    thirdParticle.vel += [Vel/math.sqrt(2), Vel/math.sqrt(2)]
                                    
                                    Vel2 = particle.vel + otherParticle.vel#velocity for diatomic formed
                                    u = (particle.pos - otherParticle.pos)/np.linalg.norm((particle.pos - otherParticle.pos))
                                    self.configuration = np.append(self.configuration, I2({0:particle.pos, 1:particle.pos+(u*2.666)}))
                                    self.configuration = np.delete(self.configuration, np.argwhere(self.configuration == particle))
                                    self.configuration = np.delete(self.configuration, np.argwhere(self.configuration == otherParticle))
                                    self.configuration[-1].vel = Vel2
                                    print("Recombination!")
                                    return
        particle = self.configuration[rand.randint(0, len(self.configuration) -1)] #select random particle
        pos_old = particle.pos
        
        if isinstance(particle, Diatomic):
            if rand.uniform(0, 1.0) < 0.5: #50% chance for translation
                #translation
                #print("Translation")
                pos_trial = particle.Translate(self.configuration, self.lim)
            else:
                #rotation
                #print("Rotation")
                pos_trial = particle.Rotate(self.configuration, self.lim)
        else:
            pos_trial = particle.Translate(self.configuration, self.lim)
        Uo = self.Energy_Total() # Uo is total system of OLD/current configuration
        particle.pos = pos_trial 
        Ut = self.Energy_Total()
        if rand.uniform(0,1) < self.AcceptanceRule(Uo, Ut):
            #accept Ut
            self.configuration_total_energies = np.append(self.configuration_total_energies, Ut)
            return
        else:
            #reject Ut
            particle.pos = pos_old
            #could it be double-counting if this is uncommented?
            #self.configuration_total_energies = np.append(self.configuration_total_energies, Uo)
    
    def Average(self):
        #takes in an array containing energies corresponding to a sampled configuration
        #evaluates average energy by probaility weighting
        numerator = 0
        denominator = 0
        for i in range(len(self.configuration_total_energies)):
            numerator += (self.configuration_total_energies[i] * exp((-1)*self.B*self.configuration_total_energies[i]))
            denominator += exp((-1)*self.B*self.configuration_total_energies[i])
        return numerator/denominator

    def Plot(self, title = ""):
        fig, ax = plt.subplots(figsize=(20,10))
        for atom in self.configuration:
            if isinstance(atom, Diatomic):
                plt.scatter(atom.pos[0][0], atom.pos[0][1], c = "black", s = 5)
                plt.scatter(atom.pos[1][0], atom.pos[1][1], c = "red", s = 10)
                plt.plot([atom.pos[0][0], atom.pos[1][0]], [atom.pos[0][1], atom.pos[1][1]])
                ax.add_artist(plt.Circle((atom.pos[0][0], atom.pos[0][1]), atom.vdw, fill = False))
                ax.add_artist(plt.Circle((atom.pos[1][0], atom.pos[1][1]), atom.vdw, fill = False))
            else:
                plt.scatter(atom.pos[0], atom.pos[1], c = "black", s = 1)
                ax.add_artist(plt.Circle((atom.pos[0], atom.pos[1]), atom.vdw, fill = False))
        plt.xlim([0, self.lim[0]])
        plt.ylim([0, self.lim[1]])
        plt.title(title)
        ax.set_aspect('equal')
