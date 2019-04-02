from collections import OrderedDict
import numpy as np
import OpenCOR as oc
from SALib.sample import saltelli
import math
from matplotlib import pyplot as plt


class Simulation(object):
    def __init__(self,*args):
        self.args = dict(args)
        self.simulation = oc.simulation(self.args.get('Sim_file'))
        self.simulation.data().setStartingPoint(self.args.get('start_time'))
        self.simulation.data().setEndingPoint(self.args.get('end_time'))
        self.simulation.data().setPointInterval(self.args.get('time_step'))
        self.simulation.resetParameters()
        self.constants = self.simulation.data().constants()
        self.variables = self.simulation.data().states()
        self.all_variable_names = sorted(list(self.variables.keys()))
        self.fitted_variable_names = sorted(list(self.variables.keys()))
        self.fitted_parameter_names = sorted(list(self.constants.keys()))
        self.all_parameter_names = sorted(list(self.constants.keys()))

        
    def assign_param_ranges(self,bounds_dictionary,fit_parameters_exclude,ic_bounds_dictionary,fit_ics_exclude, num_samples):
        print('Assigning parameter ranges:')
        for i in range(0,len(fit_parameters_exclude)):
            self.fitted_parameter_names.remove(fit_parameters_exclude[i])
        for i in range(0,len(fit_ics_exclude)):
            self.fitted_variable_names.remove(fit_ics_exclude[i])

			
        self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.fitted_parameter_names})
											
        
        # default the parameter bounds to something sensible, needs to be set directly
        bounds = []
        for c in self.fitted_parameter_names:
            v = self.constants[c];
            bounds.append([bounds_dictionary[c][0], bounds_dictionary[c][1]])
        for c in self.fitted_variable_names:
            print(c,self.variables)
            v = self.variables[c];
            bounds.append([ic_bounds_dictionary[c][0], ic_bounds_dictionary[c][1]])

        # define our sensitivity analysis problem
        self.problem = {
                   'num_vars': len(self.fitted_parameter_names) + len(self.fitted_variable_names),
                   'names': [self.fitted_parameter_names, self.fitted_variable_names],
                   'bounds': bounds
                   }
        self.samples = saltelli.sample(self.problem, num_samples)
        #print(self.samples)
        np.savetxt("self.samples.txt", self.samples)
        
    def set_initial_conditions(self,ic_dictionary):
        for i in self.all_variable_names:
            self.simulation.data().states()[i]=ic_dictionary[i]
    #def set_parameters(self,parameter_values):

    def run_once(self):
        self.simulation.run()
        
    def evaluate_ssq(self,model,time_indices,fit_data,expt_state_uri,call_num):
        num_series = len(fit_data)
        trial = np.zeros([num_series,len(time_indices)])
        ssq = np.ones(num_series+1)*1e10#
        try:
            self.simulation.run()
        except RuntimeError:
            print("Runtime error, skipping")
            return ssq
        
        for i in range(0,num_series):
            trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[time_indices]
            ssq[i+1] = math.sqrt(np.sum((fit_data[i,:]-trial[i,:])**2))
        ssq[0] = np.sum(ssq[1:num_series+1])
        return ssq 

    def define_individual_run(self,ic_dictionary,parameter_values):
        #Reset results
        self.simulation.clearResults()
        self.simulation.resetParameters()
        # Assign fixed initial conditions
        self.set_initial_conditions(ic_dictionary)
        for i, k in enumerate(self.fitted_parameter_names):
            self.constants[k] = 10.0**parameter_values[i]
        for i, k in enumerate(self.fitted_variable_names):
            self.simulation.data().states()[k]= 10.0**parameter_values[i+len(self.fitted_parameter_names)]
        
            
            
    def reset_sim_export_run_conditions(self, ic_dictionary, parameter_values, filename):
        self.simulation.clearResults()
        self.simulation.resetParameters()
        myfile = open(filename,"w+") 
        ### Fixed parameters
        for i, k in enumerate(self.fitted_parameter_names):
            self.constants[k] = 10.0**parameter_values[i]
        for i, k in enumerate(self.all_parameter_names):
            if k in self.fitted_parameter_names:
                myfile.write(k + ' ' + str(self.constants[k]) +'\n')
            else:
                myfile.write(k + ' ' + str(self.constants[k]) +' (not fitted)\n')
        #revert to original initial consitions
        self.set_initial_conditions(ic_dictionary)        
        for i, k in enumerate(self.fitted_variable_names):
            self.simulation.data().states()[k] = 10.0**parameter_values[i+len(self.fitted_parameter_names)]
        for i, k in enumerate(self.all_variable_names):
            if k in self.fitted_variable_names:
                myfile.write(k + ' ' + str(self.simulation.data().states()[k]) +'\n')
            else:
                myfile.write(k + ' ' + str(self.simulation.data().states()[k]) +' (not fitted)\n')
                
                
        myfile.close()
        
    def plot_current_best(self,time_indices,time_step,fit_data,expt_state_uri,plottitle):
        
        self.simulation.run()
        num_series = len(fit_data)
        trial = np.zeros([num_series,len(time_indices)])
        for i in range(0,num_series):
            trial[i,:] = self.simulation.results().states()[expt_state_uri[i]].values()[time_indices]
            plt.plot(time_indices*time_step,trial[i,:])
            plt.plot(time_indices*time_step,fit_data[i,:],'*')
            plt.xlabel(expt_state_uri)
            plt.ylabel('Time')
            plt.title(plottitle)
            plt.show()
        
    def run_parameter_sweep(self,num_retain,ic_dictionary,model,time_indices,fit_data,expt_state_uri):
        num_series =len(fit_data) 
        num_cols = num_series + 1 + self.samples.shape[1]
        num_rows = num_retain+1
        Y = np.zeros([num_rows,num_cols])
        for i, X in enumerate(self.samples):
            self.define_individual_run(ic_dictionary,X)
            ssq = self.evaluate_ssq(model,time_indices,fit_data,expt_state_uri,i)            
            j = i
            print('Run number: ' + str(i) + ' Overall SSQ: ' + str(ssq[0]))
            if j < num_retain:
                Y[j,0] = ssq[0]
                for k in range(0,num_series):
                    Y[j,k+1] = ssq[k+1]
                Y[j,(k+2):num_cols]=X
            else:
                Y[num_retain,0] = ssq[0]
                for k in range(0,num_series):
                    Y[num_retain,k+1] = ssq[k+1]
                Y[num_retain,(k+2):num_cols]=X
                ind = np.argsort(Y[:,0])
                Y=Y[ind]
                #print(Y)
				
			#Want to retain top N here
        ind = np.argsort(Y[:,0])
        Z=Y[ind]

        return Z


filename = '../HLAG_to_PI3K/FCepsilonRI_to_GRB_main.cellml'
start_time = 0
end_time = 3900
time_step = 1

parameter_list = ['geometry/J_FCsource', 'parameters/FCepsilonRI_k_f1', 'parameters/FCepsilonRI_k_f2', 'parameters/FCepsilonRI_k_f3', 'parameters/FCepsilonRI_k_f4', 'parameters/FCepsilonRI_k_f5', 'parameters/FCepsilonRI_k_f6', 'parameters/FCepsilonRI_k_f7', 'parameters/FCepsilonRI_k_r1', 'parameters/FCepsilonRI_k_r4', 'parameters/FCepsilonRI_k_r6', 'variables/Pi']
variable_list = ['variables/FC', 'variables/Grb2', 'variables/Lyn', 'variables/Syk', 'variables/pFC', 'variables/pFCLyn', 'variables/pFCSyk', 'variables/pGrb2', 'variables/pLyn', 'variables/pSyk', 'variables/pSykGrb2']

parameter_bounds_dictionary = {'parameters/FCepsilonRI_k_f1': [-3,2], 'parameters/FCepsilonRI_k_f2': [-3,2],'parameters/FCepsilonRI_k_f3': [-3,2], 'parameters/FCepsilonRI_k_f4': [-3,2],
	 'parameters/FCepsilonRI_k_f5': [-3,2], 'parameters/FCepsilonRI_k_f6': [-3,2], 'parameters/FCepsilonRI_k_f7': [-3,2], 
	 'parameters/FCepsilonRI_k_r1': [-3,2], 'parameters/FCepsilonRI_k_r4': [-3,2], 'parameters/FCepsilonRI_k_r6': [-3,2], 
         'variables/Pi': [-3,2]}
     
#List of parameters you want to exclude from fit
fit_parameters_exclude = ['geometry/J_FCsource']#, 'parameters/FCepsilonRI_k_f2', 'parameters/FCepsilonRI_k_f4', 'parameters/FCepsilonRI_k_f6', 'parameters/FCepsilonRI_k_f7']


ic_bounds_dictionary = {'variables/Lyn': [-3,3], 'variables/Grb2': [3,4]}

fit_ics_exclude = ['variables/FC', 'variables/Syk', 'variables/pFC', 'variables/pFCLyn', 'variables/pFCSyk', 'variables/pGrb2', 'variables/pLyn', 'variables/pSyk', 'variables/pSykGrb2']   
  
initial_conditions_tsang = { 'variables/FC': 1000.0, 'variables/Grb2': 6470.0, 'variables/Lyn': 6500, 'variables/Syk': 5.0, 
    'variables/pFC': 0.0, 'variables/pFCLyn': 0.0, 'variables/pFCSyk': 0.0, 'variables/pGrb2': 0.0, 'variables/pLyn': 0.0, 'variables/pSyk': 0.0, 'variables/pSykGrb2':0.0}
    
initial_consitions_faeder = { 'variables/FC': 47.4, 'variables/Grb2': 0.0, 'variables/Lyn': 0.0, 'variables/Syk': 25.0, 
    'variables/pFC': 0.0, 'variables/pFCLyn': 0.0, 'variables/pFCSyk': 0.0, 'variables/pGrb2': 0.0, 'variables/pLyn': 0.0, 'variables/pSyk': 0.0, 'variables/pSykGrb2':0.0}
    


# The state variable  or variables in the model that the data represents
num_series = 1
expt_state_uri = ['variables/pGrb2']


                          

#Number of samples to generate for each parameter
num_samples = 2

#Number of results to retain, if we store too many in high res parameter sweeps we can have memory issues
num_retain = 10


#Some example output that we are maybe aiming for
time_indices_tsang = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                  2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600])#note this is not time in seconds it is the time indices
exp_data_tsang = np.zeros([num_series,len(time_indices_tsang)])
exp_data_tsang[0,:] = np.array([0.0, 0.5, 0.97, 1.35, 1.8, 2.58, 2.95, 3.55, 3.82, 4.43, 4.7, 4.96, 5.15, 5.38, 5.6, 5.68, 5.87, 5.98,
                          6.02, 6.1, 6.11, 6.13, 6.24, 6.27, 6.12, 6.29, 6.44, 6.31, 6.34, 6.38, 6.37, 6.37, 6.36, 6.34, 6.35, 6.43, 6.47])*1000.0 #pGrb2 (picoM)
s = Simulation(("Sim_file", filename), ('start_time',start_time),('end_time',end_time),('time_step',time_step))

#s.set_initial_conditions(initial_conditions_tsang)

#s.run_once()

s.assign_param_ranges(parameter_bounds_dictionary,fit_parameters_exclude, ic_bounds_dictionary, fit_ics_exclude,num_samples)
Z = s.run_parameter_sweep(num_retain,initial_conditions_tsang,'FCep_to_Grb',time_indices_tsang,exp_data_tsang,expt_state_uri)

s.reset_sim_export_run_conditions(initial_conditions_tsang,Z[0,num_series+1:],'parameter_sweep_best.txt')
plottitle = 'Best fit after parameter sweep Tsang data. SSQ: ' + str(Z[0,0])
s.plot_current_best(time_indices_tsang,time_step,exp_data_tsang,expt_state_uri,plottitle)

