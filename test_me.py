import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import OpenCOR as oc


# Experiment data -- has to be regularly sampled
expt_data = np.loadtxt('test_data/hatakeyama_2003_test_data.csv', delimiter=',')
print(expt_data)
expt_time = expt_data[...,0]
expt_points = expt_data[...,1:]
print(expt_points.shape[1])

# The state variable in the model that the data represents
expt_state_uri = ['variables/Akt_PIPP','variables/ShGS','variables/ERKPP']


# Load and initialise the simulation
simulation = oc.openSimulation('hatakeyama_2003_decomposed/hatakeyama_2003_main.cellml')

# In case we have reloaded an open simulation
simulation.resetParameters()
simulation.clearResults()

# Reference some simulation objects
initial_data = simulation.data()
constants = initial_data.constants()
results = simulation.results()
times = expt_time.astype(int)

# Simulation time points must match experiment data points
initial_data.setStartingPoint(0.0)
initial_data.setEndingPoint(1800)
initial_data.setPointInterval(1)


try:
    simulation.run()
except RuntimeError:
    print("Runtime error:")
    #for n, v in enumerate(params):
    #    print('  {}: {}'.format(parameter_names[n], v))
    raise
for i in range(0,expt_points.shape[1]):
    print(np.isclose(simulation.results().states()[expt_state_uri[i]].values()[times],expt_points[:,i]))
    print(simulation.results().states()[expt_state_uri[i]].values()[times])