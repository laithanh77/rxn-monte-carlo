import random as rand
import math
from math import exp
import numpy as np
import time
from scipy.integrate import quad as integrate
import matplotlib.pyplot as plt
from ensemble import Ensemble


dt = 2.9116223400000000865
def run(ncyc, N = 1, lim = (20, 20), T = 300, ensemble = None, animation = False, dframe = 0.001):
    #initialize system
    time_total = 0 #total "time" of the system
    time_pulse = 0 #time of the pulse
    if ensemble == None:
        ensemble = Ensemble(N, lim, T)
        ensemble.Plot("Initial Configuration")
    else:
        ensemble.Plot("Initial Configuration")
    #start simulation
    y = [ensemble.Energy_Total()]
    start_time = time.time()
    #for i in trange(ncyc):
    for i in range(ncyc):
        if (i in list(range(0, ncyc, int(ncyc/20)))):
            print("{0} cycles: {1} s".format(i, time.time()-start_time))
        ensemble.Cycle(time_pulse = time_pulse, time_total = time_total)
        y.append(ensemble.Energy_Total())
        time_total += dt
        if time_pulse + dt > 120:
            time_pulse += dt - 120
        else:
            time_pulse += dt
    print("Elapsed time:", time.time() - start_time, "(s)")
    print("Initial energy:", y[0], "(J)")
    print("Final energy:", y[-1], "(J)")
    print("Average energy:", ensemble.Average(),"(J)")

    ensemble.Plot("Final Configuration")
    x = range(0, len(y))
    fig, ax = plt.subplots(figsize=(20,10))
    plt.plot(x,y)
    plt.xlim([0, ncyc])
    plt.ylim([min(y), max(y)])
    plt.title("Total Energy vs. Cycle")
    return ensemble


def main():
    system = Ensemble(3, 3, lim = (30, 30), T = 300)
    run(5000, ensemble = system)

if __name__ == "__main__":
    main()