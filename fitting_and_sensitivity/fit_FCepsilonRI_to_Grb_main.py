import model_fit_analyse

bounds_dictionary = {'FCepsilonRI/k_f1': [-1.5,2], 'FCepsilonRI/k_f3': [-3,2], 'FCepsilonRI/k_f5': [1,2],  
	 'FCepsilonRI/k_r1': [-3,1], 'FCepsilonRI/k_r4': [-3,0.5], 'FCepsilonRI/k_r6': [-3,0], 'FCepsilonRI/Pi': [-3,2]}

# The state variable  or variables in the model that the data represents
num_series = 2
expt_state_uri = ['FCepsilonRI/pFC','FCepsilonRI/pSyk']

#Some example output that we are maybe aiming for
times = np.array([0,  480, 960, 1920, 3840])
exp_data = np.zeros([num_series,len(times)])
exp_data[0,:] = np.array([0.0, 0.0408, 0.136, 0.105, 0.136])*.474 #pFC
exp_data[1,:] = np.array([0.0,  0.05437, 0.0644, 0.0518, 0.04373])*.474 #pSyk

#Number of samples to generate for each parameter
num_samples = 500

#Number of results to retain, if we store too many in high res parameter sweeps we can have memory issues
num_retain = 10