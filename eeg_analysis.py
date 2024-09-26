import numpy as np
import multiprocess as mp
from datetime import datetime
import matplotlib.pyplot as plt

from eeg_utils import(
    eeg,
)

##### Set analysis parameters

number_of_subjects = 108 #shouldn't be larger that 108, dataset has 109 but subject 109 has some not valid values at the end
filt_mode = 'filt' # 'raw' or 'filt', for considering only the alpha band
word_length = 3 # Word length
lag = 1 # Spatial lag
experiment_run = 1 #Run of the experiment, 1 for Eyes Open, and 2 to Eyes Closed

analysis_mode = 'spatial' # Controls the symbol construction (temporal or spatial) 
# and how the averages are performed
# 'ensemble' : spatial analysis, but for each time, the mean value of SPE in all subjects if provided
# This analysis corresponds to the one of Boaretto et al. (2023), and outputs the data used to
# plot Fig.4 of Gancio et al. (2024). It provides SPE for diferente arrengement:
# linear: straight foward, as come from the dataset, ordering (first approa of Boaretto et al. (2023))
# boaretto's best: best arrengement of electrodes find by Boaretto et al. (2023)
# horizontal symbols
# vertical symbosl
#
# 'spatial': Spatial analysis as performed by Gancio et al. (2024), an average in time of the SPE values is provided 
# (one averaged quantity for each subject). This provied the data for Fig. 5,and partial of Figs. 7 and 8 of  Gancio et al. (2024)
#
# 'temporal':  Temporal analysis as performed by Gancio et al. (2024), an average in space of the Permutation Entropy (PE)
#  values is provided (one averaged quantity for each subject). This provied the data for Fig. 6, and the additional
# data for Figs. 7 and 8 of  Gancio et al. (2024)


obj=eeg(number_of_subjects,filt_mode,experiment_run)
obj.L=word_length
obj.lag=lag
obj.file_path='/Volumes/T7/data_eeg'
obj.load_data()


if analysis_mode == 'ensemble':

    startTime = datetime.now()
    if __name__ == '__main__':  
        pool = mp.Pool(mp.cpu_count())

        obj.set_mode('horizontal')  
        ensemble_hor = pool.map(obj.par_ensemble_spe, range(obj.max_time))    

        obj.set_mode('vertical')
        ensemble_ver = pool.map(obj.par_ensemble_spe, range(obj.max_time)) 

        ensemble_lin = pool.map(obj.par_ensemble_linear, range(obj.max_time)) 
        ensemble_boaretto = pool.map(obj.par_ensemble_boaretto, range(obj.max_time)) 

        pool.close()  
        pool.join() 

    print('Time elapsed:' + str(datetime.now() - startTime))
    np.savetxt('eeg_processed/ensemble_linear_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',ensemble_lin, delimiter=",")
    np.savetxt('eeg_processed/ensemble_best_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',ensemble_boaretto, delimiter=",")
    np.savetxt('eeg_processed/ensemble_hor_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',ensemble_hor, delimiter=",")
    np.savetxt('eeg_processed/ensemble_ver_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',ensemble_ver, delimiter=",")
    print('Process completed.')


elif analysis_mode == 'spatial':
    
    startTime = datetime.now()
    if __name__ == '__main__':  
        pool = mp.Pool(mp.cpu_count()) 

        obj.set_mode('horizontal')
        spe_hor = pool.map(obj.par_spatial, range(obj.subjects))
        spe_hor_std = pool.map(obj.par_spatial_std, range(obj.subjects))
        pspe_hor = pool.map(obj.par_pool_SPE, range(obj.subjects))  

        obj.set_mode('vertical')
        spe_ver = pool.map(obj.par_spatial, range(obj.subjects))
        spe_ver_std = pool.map(obj.par_spatial_std, range(obj.subjects))  
        pspe_ver = pool.map(obj.par_pool_SPE, range(obj.subjects))

        spe_boa = pool.map(obj.par_spatial_boaretto, range(obj.subjects))  

        pool.close()  
        pool.join() 

    print('Time elapsed:' + str(datetime.now() - startTime))
    np.savetxt('eeg_processed/spe_hor_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',spe_hor, delimiter=",")
    np.savetxt('eeg_processed/spe_ver_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',spe_ver, delimiter=",")
    np.savetxt('eeg_processed/std_spe_hor_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',spe_hor_std, delimiter=",")
    
    np.savetxt('eeg_processed/std_spe_ver_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',spe_ver_std, delimiter=",")
    np.savetxt('eeg_processed/pspe_hor_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',pspe_hor, delimiter=",")
    np.savetxt('eeg_processed/pspe_ver_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',pspe_ver, delimiter=",")
    
    np.savetxt('eeg_processed/spe_boa_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',spe_boa, delimiter=",")
    
    print('Process completed.')
    

elif analysis_mode == 'temporal':
    startTime = datetime.now()
    if __name__ == '__main__':  
        pool = mp.Pool(mp.cpu_count())  
       
        pe=pool.map(obj.par_PE, range(obj.subjects))  
        pe_std=pool.map(obj.par_PE_std, range(obj.subjects)) 
        ppe=pool.map(obj.par_PPE, range(obj.subjects))   

        pool.close()  
        pool.join() 

    np.savetxt('eeg_processed/pe_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',pe, delimiter=",")
    np.savetxt('eeg_processed/ppe_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',ppe, delimiter=",")
    np.savetxt('eeg_processed/pe_std_L_'+str(word_length)+'_lag_'+str(lag)+'_run_'+str(experiment_run)+'_'+filt_mode+'.csv',pe_std, delimiter=",")
    
    print('Time elapsed:' + str(datetime.now() - startTime))
    print('Process completed.')

else:
    raise Exception("Analysis mode not specified or incorrect, mode has to be 'endemble', 'temporal' or 'spatial'")
