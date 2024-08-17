# eeg-spatial-analysis

Python scrips for the spatial analysis of EEG signals. 
These scrips reproduce the analysis performed and published in the work: ["Permutation entropy analysis of EEG signals for distinguishing eyes-open and eyes-closed brain states: Comparison of different approaches"](https://doi.org/10.1063/5.0200029).

## Scripts

Summary of the scripts included:

eeg_analysis.py: Main analysis 

eeg_utils.py: Supporting functions for eeg_analysis.py

RF_single_feature.py: Random forest classification of the data

make_figs.m: Matlab script that outputs the figures of Gancio et al. (2024)

## eeg_analysis.py
This script performs the main analysis publish in Gancio et al. (2024). 
The following parameters have to be set:

number_of_subjects: shouldn't be larger that 108, dataset has 109 but subject 109 has some not valid values at the end

filt_mode:'raw' or 'filt', for considering only the alpha band

word_length: Word length of the symbols in the Ordinal Pattern analysis. For a 64 channel EEG, we recommend only set it word_length=3

lag : Spatial lag of the symbols in the Ordinal Pattern analysis. For a 64 channel EEG, we recommend only set it lag=1

experiment_run = 1 #Run of the experiment, 1 for Eyes Open, and 2 to Eyes Closed

analysis_mode: Controls the symbol construction (temporal or spatial) and how the averages are performed.

'ensemble' : spatial analysis, but for each time, the mean value of SPE in all subjects if provided. This analysis corresponds to the one of Boaretto et al. (2023), and outputs the data used to plot Fig.4 of Gancio et al. (2024). It outputs SPE for diferente arrangement:

	linear: straight forward, as come from the dataset, ordering (first approach of Boaretto et al. (2023))

	boaretto's best: best arrangement of electrodes find by Boaretto et al. (2023)

	horizontal symbols

	vertical symbols


'spatial': Spatial analysis as performed by Gancio et al. (2024), an average in time of the SPE values is provided (one averaged quantity for each subject). This provied the data for Fig. 5,and partial of Figs. 7 and 8 of  Gancio et al. (2024).

'temporal':  Temporal analysis as performed by Gancio et al. (2024), an average in space of the Permutation Entropy (PE) values is provided (one averaged quantity for each subject). This provied the data for Fig. 6, and the additional data for Figs. 7 and 8 of  Gancio et al. (2024).