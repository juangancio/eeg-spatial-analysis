import mne
import math
import numpy as np



class eeg:

    def __init__(self,subjects,mode,run):
        self.subjects = subjects #number of subjects
        self.mode=mode #raw or filt
        self.run=run #number of experiment: 1 corresponds to Eyes Open, and 2 to Eyes Closed
        self.max_time=9600 #maximum time fo the experiment
        self.i=[]
        self.j=[]
        self.structured_data=[]
        self.L=3
        self.lag = 1
        self.Lx=self.L
        self.Ly=1
        self.file_path='/Volumes/T7/data_eeg'

    def load_data(self):
        R=self.run
        self.data=[]
        if self.mode=='raw':
            for subject_number in range(self.subjects):
                        
                if subject_number == 96:
                    
                    continue
                if subject_number>=99:
                    name=self.file_path+"/S"+str(subject_number+1)+"/S"+str(subject_number+1)+"R0"+str(R)+".edf"
                elif subject_number>=9:
                    name=self.file_path+"/S0"+str(subject_number+1)+"/S0"+str(subject_number+1)+"R0"+str(R)+".edf"
                else:
                    name=self.file_path+"/S00"+str(subject_number+1)+"/S00"+str(subject_number+1)+"R0"+str(R)+".edf"
                raw = mne.io.read_raw_edf(name,verbose=None)

                self.data=self.data+[raw.get_data()]

        elif self.mode=='filt':
            for subject_number in range(self.subjects):
                        
                if subject_number == 96:
                  
                    continue
                if subject_number>=99:
                    name=self.file_path+"/S"+str(subject_number+1)+"/S"+str(subject_number+1)+"R0"+str(R)+".edf"
                elif subject_number>=9:
                    name=self.file_path+"/S0"+str(subject_number+1)+"/S0"+str(subject_number+1)+"R0"+str(R)+".edf"
                else:
                    name=self.file_path+"/S00"+str(subject_number+1)+"/S00"+str(subject_number+1)+"R0"+str(R)+".edf"
                raw = mne.io.read_raw_edf(name,verbose=None)

                self.data=self.data+[mne.filter.filter_data(data=raw.get_data(), sfreq=160, l_freq=8, h_freq=12)]
        
        else:
            raise Exception("Load mode not specified or incorrect, Mode has to be 'raw' or 'filt'")
            
        if self.subjects>=96:
             self.subjects=self.subjects-1
             
             print('Subjects number changed to: '+str(self.subjects))


    def set_mode(self,mode):
        if mode == 'vertical':
            self.Lx=1
            self.Ly=self.L
        elif mode == 'horizontal':
            self.Lx=self.L
            self.Ly=1
        else:
            raise Exception("Analysis mode not specified or incorrect, Mode has to be 'horizontal' or 'vertical'")
      


    def create_data_struc(self,data):
        #This function gives the grid arrangement as in 
        #Gancio, J., Masoller, C., & Tirabassi, G. (2024). Permutation entropy analysis of EEG signals for distinguishing eyes-open and eyes-closed brain states: Comparison of different approaches. Chaos: An Interdisciplinary Journal of Nonlinear Science, 34(4).
        
        row_len=[3,5,9,9,11,9,9,5,3,1]
        new_order=[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
                   1,2,3,4,5,6,7,40,43,41,8,9,10,11,12,13,14,42,44,45,15,
                   16,17,18,19,20,21,46,47,48,49,50,51,52,53,54,55,56,57,
                   58,59,60,61,62,63,64]
        
        new_data=[math.nan]*(len(row_len)*max(row_len))
        new_data=np.array(new_data).reshape(len(row_len),max(row_len))
        mid=(max(row_len)-1)/2-1
        counter=0
        for j in range(len(row_len)):
            for i in range(row_len[j]):
                new_data[j,int(mid-(row_len[j]-1)/2+i+1)]=data[new_order[counter]-1]
                counter+=1
        return new_data
    
    def boaretto_best(self,data):
        #This function orders the data according to the best ordering reported in 
        #Boaretto, B. R., Budzinski, R. C., Rossi, K. L., Masoller, C., & Macau, E. E. (2023). Spatial permutation entropy distinguishes resting brain states. Chaos, Solitons & Fractals, 171, 113453.
        
        new_order=[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
                   1,2,3,4,5,6,7,40,43,41,8,9,10,11,12,13,14,42,44,45,15,
                   16,17,18,19,20,21,46,47,48,49,50,51,52,53,54,55,56,57,
                   58,59,60,61,62,63,64]
        
        new_data=[math.nan]*(len(new_order))

        for i in range(len(data)):
            
            new_data[i]=data[new_order[i]-1]
                
        return new_data
    
    def spatial_code(self,data):
        code=[]
        for j in range(10-(self.Ly-1)*self.lag):
            for i in range(11-(self.Lx-1)*self.lag):

                word=data[j+np.arange(self.Ly)*self.lag,i+np.arange(self.Lx)*self.lag]

                if not(np.isnan(word).any()):
                    code.extend(perm_indices(word,self.L,lag=1))
             
        return code
    
    def par_spatial(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc(new_data)
                    
            code=self.spatial_code(structured_data)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return np.mean(Ht)
    
    def par_spatial_boaretto(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            
            structured_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.boaretto_best(structured_data)
                    
            code=perm_indices(structured_data,self.L,self.lag)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return np.mean(Ht)
    
    def par_spatial_std(self,j):
        #Gets mean SPE from subject j
        Ht=[]
        
        for t in range(self.max_time):
            
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc(new_data)
                    
            code=self.spatial_code(structured_data)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        
        return np.std(Ht)

    def par_PE(self,j):
        # Gets average usual permutation entropy (PE) of subject j
        Ht=[]
        for i in range(64):
            selected_data=self.data[j][i,:] #Get channels i for all times 
            code=perm_indices(selected_data,self.L,self.lag)

            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        return np.mean(Ht)
    
    def par_PE_std(self,j):
        # Gets average usual permutation entropy (PE) of subject j
        Ht=[]
        for i in range(64):
            selected_data=self.data[j][i,:] #Get channels i for all times 
            code=perm_indices(selected_data,self.L,self.lag)

            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]
        return np.std(Ht)

    def par_PPE(self,j):
        # Gets average pooled usual permutation entropy (PPE) of subject j

        code=[]

        for i in range(64):

            selected_data=self.data[j][i,:] #Get channels i for all times 
            code.extend(perm_indices(selected_data,self.L,self.lag))

        probs=probabilities(code,self.L)
            
        return entropy(probs)/np.log(math.factorial(self.L))
    
    def par_pool_SPE(self,j):
        #Gets pooled spatial entropy (PSPE) of subject j
                    
        code=[]
        
        for t in range(self.max_time):
            
            new_data=self.data[j][:,t] #Get channels for time t
            structured_data=self.create_data_struc(new_data)
            code.extend(self.spatial_code(structured_data))
        
        probs=probabilities(code,self.L)
        
        return entropy(probs)/np.log(math.factorial(self.L))
    
    def par_ensemble_spe(self,t):

        Ht=[]
        for s in range(self.subjects):
        
            structured_data=self.data[s][:,t] #Get channels for time t
            structured_data=self.create_data_struc(structured_data)
                    
            code=self.spatial_code(structured_data)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]

        return np.mean(Ht)
    
    def par_ensemble_linear(self,t):

        Ht=[]
        for s in range(self.subjects):

            code=perm_indices(self.data[s][:,t],self.L,self.lag)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]

        return np.mean(Ht)
    
    def par_ensemble_boaretto(self,t):

        Ht=[]
        for s in range(self.subjects):
        
            structured_data=self.data[s][:,t] #Get channels for time t
            structured_data=self.boaretto_best(structured_data)
                    
            code=perm_indices(structured_data,self.L,self.lag)
            
            probs=probabilities(code,self.L)
            Ht=Ht+[entropy(probs)/np.log(math.factorial(self.L))]

        return np.mean(Ht)



def perm_indices(ts, wl, lag):
    m = len(ts) - (wl - 1)*lag
    indcs = np.zeros(m, dtype=int)
    for i in range(1,wl):
        st = ts[(i - 1)*lag : m + ((i - 1)*lag)]
        for j in range(i,wl):
            zipped=zip(st,ts[j*lag : m+j*lag])
            indcs += [x > y for (x, y) in zipped]
        indcs*= wl - i
    return indcs + 1


def entropy(probs):
    h=0
    for i in range(len(probs)):
        if probs[i]==0:
            continue
        else:
            h=h-probs[i]*np.log(probs[i])
    return h

def probabilities(code,L):
    get_indexes = lambda x, xs: [k for (y, k) in zip(xs, range(len(xs))) if x == y]
    probs=[]
    for i in range(1,math.factorial(L)+1):
            
                
        probs=probs + [len(get_indexes(i,code))/len(code)
                   ]
        
        #print(self.entropy(probabilities))
    return probs