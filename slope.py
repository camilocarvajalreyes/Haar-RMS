import pandas as pd
import numpy as np
import matplotlib as mtpl
import matplotlib.pyplot as plt
font = {'family' : 'serif','weight' : 'ultralight','size'   : 14};mtpl.rc('font', **font)
import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m*60
    if m < 60:
        return '%dm %ds' % (m, s)
    else:
        h = math.floor(m/60)
        m -= h*60
        return '%dh %dm %ds' % (h, m, s)


class Haar:
    def __init__(self,t,x,smooth=False,bins=20):
        self.t = np.array(t)
        self.x = np.array(x)

        self.smooth = smooth
        self.bins = bins

        self.deltas_t = []
        self.epsilons = []

        self.ep_min = None
        self.ep_max = None
        self.calib = None
        self.Hs = None
    
    @staticmethod
    def Smooth121(x):
        signal = x
        length = len(signal)
        output = np.zeros(length-2)
        coef= np.array([1,2,1])
        for i in range(length - 2):
            output[i]= np.sum(signal[i:i+3] * coef / 4)
        return output
    
    def compute_deltas(self):
        for i in range(1,int(len(self.t)/2)+1):  # steps: i from 1 to n/2 
            self.deltas_t.append(self.t[i:]-self.t[:-i])  # 0= Delta_t1, 1= Delta_t2, ..., [n/2]-1 = Delta_tn/2
    
    def compute_epsilons(self):
        for i in range(1,len(self.deltas_t)+1):  # for all my deltas_t (all steps-differences, de 1 en 1, de 2 en 2, etc)
            self.epsilons.append(self.deltas_t[i-1][:-i]/(self.deltas_t[i-1][:-i]+self.deltas_t[i-1][i:]))
    
    def fluctuations(self,ep_min=0.25,calib=2,verbose=True,prop_print=500):
        """
        Computing fluctuations
        
        Arguments:
            ep_min: float
                default 0.25
                
            calib: int
                default 2
                
            verbose: boolean
                whether or not to print progress and deleted fluctuations
            
            prop_print: int
                how often do we print the progress, default 500
        """
        self.x = self.x[:-1]
        self.t = self.t[:-1]
        #Cálculo fluctuaciones
        self.ep_min = ep_min
        self.ep_max = 1 - self.ep_min
        self.calib = calib

        self.Hs = []
        self.delta_t = [] 
        counter = 0

        start_time = time.time()

        for H in range(2,len(self.x)-1,2):
            for start in range(len(self.x)-H-1):
                int1 = np.sum(self.x[start:start+int(H/2)]*self.deltas_t[0][start:start+int(H/2)]/self.deltas_t[int(H/2)-1][start])
                int2 = np.sum(self.x[start+int(H/2):start+H]*self.deltas_t[0][start+int(H/2):start+H]/self.deltas_t[int(H/2)-1][start+int(H/2)])
                counter += 1
                if self.epsilon_range(H,start):
                    # Hs.append(calib*abs(int2-int1))  # S_1
                    self.Hs.append((calib*(int2 - int1))**2)  # S_2 (falta despues sacar raíz)
                    self.delta_t.append(self.deltas_t[int(H/2)-1][start] + self.deltas_t[int(H/2)-1][start+int(H/2)])  # No logaritmicos
            if verbose:
                prop = 100*H / (len(self.x) - 1)
                if H % prop_print  == 0:
                    print("Progress: {}%, time elapsed {}".format('%.3f'%(prop),timeSince(start_time)))
        
        if verbose:
            print("Finished computations in {}".format(timeSince(start_time)))
            perct = (counter - len(self.Hs))/counter*100
            perct = '%.3f'%(perct)
            print("{} fluctuaciones eliminadas ({}%)".format(counter - len(self.Hs),perct))

    def epsilon_range(self,H,start):
        min_condition = self.ep_min  < self.epsilons[int(H/2)-1][start]
        max_condition = self.epsilons[int(H/2)-1][start] < self.ep_max
        return min_condition and max_condition

    @property
    def data_df(self):
        if self.Hs is None:
            raise ValueError("Hs not yet defined")
        #Paso a dataframes
        df = pd.DataFrame(data={'delta t':self.delta_t , 'Hs': self.Hs})
        return df.sort_values('delta t',axis=0).reset_index(drop=True)
    
    def algo(self):
        self.data_sorted = self.data_df  # get df
        max_t = max(np.log10(self.data_sorted['delta t']))
        min_t = min(np.log10(self.data_sorted['delta t']))
        self.min_t = min_t
        self.n_bins = int(((max_t)-(min_t))*self.bins)
        self.rango_int = (max_t-min_t)/self.n_bins

        self.time = np.array([])
        self.ave_values=np.array([])
        self.upper=np.array([])
        self.lower=np.array([])

        for i in range(self.n_bins):
            interval = self.data_sorted[(i*self.rango_int+min_t <= np.log10(self.data_sorted['delta t']))&(np.log10(self.data_sorted['delta t']) < min_t+(i+1)*self.rango_int)]
            self.time = np.append(self.time,((i*self.rango_int+min_t)+((i+1)*self.rango_int+min_t))/2)
            self.ave_values = np.append(self.ave_values,np.sqrt(interval.mean()[1]))        ## Tomo el sqrt del <[2*(x1-x2)]^2> ...
            self.upper = np.append(self.upper,np.sqrt(interval.mean()[1]+interval.std()[1]/np.sqrt(len(interval))))
            self.lower = np.append(self.lower,np.sqrt(interval.mean()[1]-interval.std()[1]/np.sqrt(len(interval))))

    def do_smoothing(self):
        if not self.smooth:
            self.smooth_val = self.ave_values
        else:
            self.smooth_val = self.Smooth121(self.ave_values)
            upper = self.Smooth121(self.upper)
            lower = self.Smooth121(self.lower)
            self.time = self.time[1:-1]
        self.val_mask = np.isfinite(self.smooth_val)
        
        self.time = self.time[self.val_mask]  # ??
        self.smooth_val = self.smooth_val[self.val_mask]  # ??
        self.upper = upper[self.val_mask]
        self.lower = lower[self.val_mask]
    
    def plot(self):
        plt.subplots(figsize=(15, 7))
        plt.title(' Sin test Haar fluctuation',fontsize=20)
        plt.xlabel('Log\u2081\u2080\u0394 t (Ky)',fontsize=18)
        plt.ylabel('Log\u2081\u2080 S\u2082(\u0394 t)^1/2',fontsize=18)
        for i in range(self.n_bins):
            plt.axvline(x = i*self.rango_int + self.min_t,linestyle='-.',linewidth=0.5)
        plt.plot(self.time,np.log10(self.smooth_val),'.-',color='black',linewidth=2,label='Log binning')
        plt.legend()
        plt.show()
    
    @property
    def delta_t_values(self):
        return self.data_sortedd['delta t']
    
    @property
    def Hs_values(self):
        return self.data_sorted['Hs']


def load_data(dust_file):
    id_columns=['Name','Data id','Latitud','Longitud','Age units','Data units','Data length']
    id_data = pd.read_excel(dust_file,sheet_name=0,usecols=id_columns)
    df_data = pd.read_excel(dust_file,sheet_name=1,skiprows=1)  # skip row 1 o 0

    #SEPARAR COLUMNAS
    columns = df_data.size/len(df_data)
    lis = []
    new_length = np.array([])

    for i in range(int(columns)):
        if (i+1)%2==1: 
            dupla = df_data[[df_data.columns[i],df_data.columns[i+1]]].dropna() 
            dupla.index=[j for j in range(0, len(dupla))] 
            lis.append(dupla)
            new_length=np.append(new_length,np.shape(dupla)[0])
    
    return id_data, lis, new_length
