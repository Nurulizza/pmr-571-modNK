import numpy as np

import OpenCOR as oc


def simulate_cellml(filename, start_time, end_time, time_interval):
    # Load and initialise the simulation
    simulation = oc.openSimulation(filename)
    # In case we have reloaded an open simulation
    simulation.resetParameters()
    simulation.clearResults()

    # Reference some simulation objects
    initial_data = simulation.data()
    constants = initial_data.constants()
    results = simulation.results()

    # Simulation time points must match experiment data points
    initial_data.setStartingPoint(start_time)
    initial_data.setEndingPoint(end_time)
    initial_data.setPointInterval(time_interval)


    try:
        simulation.run()
    except RuntimeError:
        print("Runtime error:")
        print(simulation_cellml + " did not run sucessfully")
        raise
    return simulation.results()
    
def check_all_same(model,data,expt_state_uri,times):
    output = False
    count_true = 0
    for i in range(0,data.shape[1]):
        if np.isclose(model.states()[expt_state_uri[i]].values()[times],data[:,i]).all:
            count_true = count_true + 1
    if(count_true == data.shape[1]):
        output = True
    return  output
    
def report_test_result(testname,testflag, countpasses,counttotal):
    if(testflag):
        print("Test " + testname + " passed")
        countpasses = countpasses + 1
        counttotal = counttotal + 1
    else:
        print("*** Test " + testname + " Failed")
        counttotal = counttotal + 1
    return countpasses, counttotal
    
def test_hatakeyama_2003():
    # The state variable in the model that the data represents
    expt_state_uri = ['variables/Akt_PIPP','variables/ShGS','variables/ERKPP']
    simulation_cellml = 'hatakeyama_2003_decomposed/hatakeyama_2003_main.cellml'
    start_time = 0.0
    end_time = 1800.0
    time_interval = 1.0
    # Experimental data
    expt_data = np.loadtxt('test_data/hatakeyama_2003_test_data.csv', delimiter=',')
    expt_time = expt_data[...,0]
    expt_points = expt_data[...,1:]
    myresults = simulate_cellml(simulation_cellml, start_time, end_time, time_interval)
    times = expt_time.astype(int)  

    are_same = check_all_same(myresults,expt_points, expt_state_uri,times)
    return are_same
    
################################################################  
countpasses = 0
counttotal = 0  
test1 = test_hatakeyama_2003()
[countpasses,counttotal] = report_test_result("Hatakeyama 2003",test1, countpasses,counttotal)

print("-------------------------------")
print("        TESTS COMPLETED        ")
print("-------------------------------")

print("Passed: " + str(countpasses) + "  Failed: " + str(counttotal-countpasses))