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
    
def check_all_same_state(model,data,expt_state_uri,times):
    output = False
    count_true = 0
    for i in range(0,data.shape[1]):
        if np.isclose(model.states()[expt_state_uri[i]].values()[times],data[:,i]).all:
            count_true = count_true + 1
        else:
            print('Data and model not alike: ' + str(expt_state_uri[i]))
            print('Model: ')
            print(model.states()[expt_state_uri[i]].values()[times])
            print('Data: ')
            print(data[:,i])
            
    if(count_true == data.shape[1]):
        output = True
    return  output
    
def check_all_same_algebraic(model,data,expt_algebraic_uri,times):
    output = False
    count_true = 0
    for i in range(0,data.shape[1]):
        if np.isclose(model.algebraic()[expt_algebraic_uri[i]].values()[times],data[:,i]).all:
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

    are_same = check_all_same_state(myresults,expt_points, expt_state_uri,times)
    return are_same
    
def test_cooling_2009_tomida():
    # The state variable in the model that the data represents
    expt_algebraic_uri = ['variables/nuclearNFATpercent','variables/NFATN_c_percent']
    simulation_cellml = 'cooling_2009/cooling_2009_main.cellml'
    start_time = 0.0
    end_time = 1800.0
    time_interval = 1.0
    # Experimental data
    expt_data = np.loadtxt('test_data/cooling_2009_test_data.csv', delimiter=',')
    expt_time = expt_data[...,0]
    expt_points = expt_data[...,1:]
    myresults = simulate_cellml(simulation_cellml, start_time, end_time, time_interval)
    times = expt_time.astype(int)  
    are_same = check_all_same_algebraic(myresults,expt_points, expt_algebraic_uri,times)
    return are_same
    
    
def test_dupont_1997():
    # The state variable in the model that the data represents
    expt_state_uri = ['variables/IP3','variables/IP4']
    simulation_cellml = 'dupont_1997_decomposed/dupont_1997_main.cellml'
    start_time = 0.0
    end_time = 100.0
    time_interval = 0.01
    # Experimental data
    expt_data = np.loadtxt('test_data/dupont_1997_test_data.csv', delimiter=',')
    expt_time = expt_data[...,0]
    expt_points = expt_data[...,1:]
    myresults = simulate_cellml(simulation_cellml, start_time, end_time, time_interval)
    times = expt_time.astype(int)  
    are_same = check_all_same_state(myresults,expt_points, expt_state_uri,times)
    return are_same
    
def test_FCepsilon_to_Grb():
    # The state variable in the model that the data represents
    expt_state_uri = ['variables/pGrb2','variables/Grb2']
    simulation_cellml = 'HLAG_to_PI3K/FCepsilonRI_to_Grb_main.cellml'
    start_time = 0.0
    end_time = 1800.0
    time_interval = 1.0
    # Experimental data
    expt_data = np.loadtxt('test_data/FCepsilonRI_to_Grb_test_data.csv', delimiter=',')
    expt_time = expt_data[...,0]
    expt_points = expt_data[...,1:]
    myresults = simulate_cellml(simulation_cellml, start_time, end_time, time_interval)
    times = expt_time.astype(int)  
    are_same = check_all_same_state(myresults,expt_points, expt_state_uri,times)
    return are_same
################################################################  
countpasses = 0
counttotal = 0  
##TEST 1 - Hatakeyama (2003) 
test1 = test_hatakeyama_2003()
[countpasses,counttotal] = report_test_result("Hatakeyama 2003",test1, countpasses,counttotal)
## TEST 2 - Cooling (2009) Tomida protocol
test2 = test_cooling_2009_tomida()
[countpasses,counttotal] = report_test_result("Cooling 2009 Tomida protocol",test2, countpasses,counttotal)
## TEST 3 - Dupont & Erneaux (1997)
test3 = test_dupont_1997()
[countpasses,counttotal] = report_test_result("Dupont 1997 protocol",test3, countpasses,counttotal)

test4 = test_FCepsilon_to_Grb()
[countpasses,counttotal] = report_test_result("FCepsilon to Grb",test4, countpasses,counttotal)

print("-------------------------------")
print("        TESTS COMPLETED        ")
print("-------------------------------")

print("Passed: " + str(countpasses) + "  Failed: " + str(counttotal-countpasses))