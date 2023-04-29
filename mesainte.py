# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 2023

@author: Bensi
"""

# =========================================================================
# =========================================================================
#   Documentation: Interpolation of mesa binaries
#   Why is somebody else thinking about using this shitty little homebrew?
#   
#   Future work:
#   Find some better way to use the thermal eq EEPS or optamize them better
#   
# ========================================================================= 
# =========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import sys
# =========================================================================
# =========================================================================
pd.options.display.max_rows = 2000
class mp_overflow:

    def __init__(self,file):
        self.file = file

    # =========================================================================

    def clean(self,k):
        ''' Cleans up given mesa history.data file returning prelevant values as 10 numpy arrays, 0 for ms, any other int for post ms'''
        history= pd.read_fwf(self.file)
        history=history.iloc[3:] # makes the history file readable for pandas, !!One known problem, if a column starts positive and becomes negative, the minus will not be included when the column is read!!
        history.columns=history.iloc[0]
        history=history.iloc[1:]
        #print(history)
        LogT = history["log_Teff"].to_numpy(dtype=float)
        LogL = history["log_L"].to_numpy(dtype=float)
        star1_mass = history["star_1_mass"].to_numpy(dtype=float)
        m_dot1 = history["lg_mstar_dot_1"].to_numpy(dtype=float)
        LogL_nuc = history["log_Lnuc"].to_numpy(dtype=float)
        radius = history["log_R"].to_numpy(dtype=float)
        rl_radius = history["rl_1"].to_numpy(dtype=float)
        log_rl_radius = np.log10(rl_radius) # everything is better in log scale
        centerh1 = history["center_h1"].to_numpy(dtype=float)
        age = history["star_age"].to_numpy(dtype=float)
        centerh1 = centerh1[(centerh1 > 10**(-12))]
        centerh0 = centerh1[(centerh1 > centerh1[0]-0.0015)]
        Thermal_diff = history["thermal_eq_difference"].to_numpy(dtype=float)
        P = history["period_days"].to_numpy(dtype=float)
        LogP = np.log10(P)
        ######################################################
        # We use known conditions to define the ms and post ms
        if k==0: 
            LogL = LogL[len(centerh0):len(centerh1)]
            LogT = LogT[len(centerh0):len(centerh1)]
            star1_mass = star1_mass[len(centerh0):len(centerh1)]
            m_dot1 = m_dot1[len(centerh0):len(centerh1)]
            LogL_nuc = LogL_nuc[len(centerh0):len(centerh1)]
            radius = radius[len(centerh0):len(centerh1)]
            log_rl_radius = log_rl_radius[len(centerh0):len(centerh1)]
            age = age[len(centerh0):len(centerh1)]
            LogP = LogP[len(centerh0):len(centerh1)]
            Thermal_diff = Thermal_diff[len(centerh0):len(centerh1)]
        ######################################################
        # The given simulation history data file ends at the end of core Helium burning, if not, use centerhe4 = 10**-4 as the cutting point
        else : 
            LogL = LogL[len(centerh1):len(LogP)]
            LogT = LogT[len(centerh1):len(LogP)]
            star1_mass = star1_mass[len(centerh1):len(LogP)]
            m_dot1 = m_dot1[len(centerh1):len(LogP)]
            LogL_nuc = LogL_nuc[len(centerh1):len(LogP)]
            radius = radius[len(centerh1):len(LogP)]
            log_rl_radius = log_rl_radius[len(centerh1):len(LogP)]
            age = age[len(centerh1):len(LogP)]
            Thermal_diff = Thermal_diff[len(centerh1):len(LogP)]
            LogP = LogP[len(centerh1):len(LogP)]
        ######################################################

        RLOF = radius/log_rl_radius
        Thermal_eq = np.abs(LogL-LogL_nuc)
        return LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP

    # =========================================================================

    def eeps(self):
        ''' Picks out the main evolutionary points that take place during the main phase and returns them as 6 seperate arrays'''
        LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP = self.clean(0)
        ######################################################
        # The TAMS and Zams are given as the data is already cleaned for the ms
        x = np.array(range(0,len(LogL),1))
        mass_loss = np.stack((x,m_dot1,LogT,LogL,star1_mass,m_dot1),axis=1)
        Thermal_diff_0 = np.stack((x,Thermal_diff,LogT,LogL,star1_mass,m_dot1),axis=1)
        TAMS = np.array([x[-1],LogT[-1],LogL[-1],star1_mass[-1],m_dot1[-1]])
        ZAMS = np.array([x[0],LogT[0],LogL[0],star1_mass[0],m_dot1[0]])
        ######################################################
        # Roche lobe overflow defined by radius and the roche lobe radius being equal
        RLOF = np.stack((x,RLOF,LogT,LogL,star1_mass,m_dot1),axis=1)
        RLOF_df = pd.DataFrame(RLOF,columns=["num","RLOF","LogT","LogL","star1_mass","m_dot1"])
        RLOF_df.drop(RLOF_df[RLOF_df["RLOF"] <= 0.99].index, inplace=True)
        RLOF_99 = RLOF_df.to_numpy()
        RLOF_99 = RLOF_99[-1,:]
        RLOF_df.drop(RLOF_df[RLOF_df["RLOF"] < 1].index, inplace=True)
        RLOF_1 = RLOF_df.to_numpy()
        RLOF_1 = RLOF_1[0,:]
        ######################################################
        # Not currently used, hardcoded values to be used instead of Th eq
        mass_loss_df = pd.DataFrame(mass_loss,columns=["num","mass_loss","LogT","LogL","star1_mass","m_dot1"])
        mass_loss_df.drop(mass_loss_df[mass_loss_df["mass_loss"] < -5].index, inplace=True)
        mass_loss = mass_loss_df.to_numpy()
        mass_loss_0= mass_loss[0,:]
        mass_loss_1= mass_loss[-1,:]
        ######################################################
        # Not my problem anymore
        Thermal_diff_df = pd.DataFrame(Thermal_diff_0,columns=["num","Thermal_diff","LogT","LogL","star1_mass","m_dot1"])
        Thermal_diff_df.drop(Thermal_diff_df[Thermal_diff_df["Thermal_diff"] < 0.2].index, inplace=True)
        Thermal_diff_non = Thermal_diff_df.to_numpy()
        
        non_th = [Thermal_diff_non[0,:]]
        Thermal_diff_df = pd.DataFrame(Thermal_diff_0,columns=["num","Thermal_diff","LogT","LogL","star1_mass","m_dot1"])
        Thermal_diff_df.drop(Thermal_diff_df[Thermal_diff_df["Thermal_diff"] > 0.2].index, inplace=True)
        Thermal_diff_non = Thermal_diff_df.to_numpy()
        non_th.append(Thermal_diff_non[-1,:])
        ######################################################

        Thermal_diff_df = pd.DataFrame(Thermal_diff_0,columns=["num","Thermal_diff","LogT","LogL","star1_mass","m_dot1"])
        Thermal_diff_df.drop(Thermal_diff_df[Thermal_diff_df["Thermal_diff"] > 0.1].index, inplace=True)
        Thermal_diff_on = Thermal_diff_df.to_numpy()

        on_th= []
        for x in range(len(Thermal_diff_on)-2):
            test = Thermal_diff_on[x+1,0]- Thermal_diff_on[x,0]
            #print(test)
            if test > 5:
                on_th.append(Thermal_diff_on[x+1,:])
                break
            else:
                pass
        ######################################################
        non_th = np.array(non_th)
        on_th = np.array(on_th)
        
        return TAMS,ZAMS, RLOF_1, RLOF_99,on_th,non_th
        

    def eeps_post(self):
        ''' Picks out the main evolutionary points that take place after the main sequence and returns them as 6 seperate arrays'''
        LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP = self.clean(1)
        # Same as the eeps but for post ms
        x = np.array(range(0,len(LogL),1))
        TAMS = np.array([x[0],LogT[0],LogL[0],star1_mass[0],m_dot1[0]])
        TACHeB = np.array([x[-1],LogT[-1],LogL[-1],star1_mass[-1],m_dot1[-1]])

        Thermal_diff_0 = np.stack((x,Thermal_diff,LogT,LogL,star1_mass,m_dot1),axis=1)
        RLOF = np.stack((x,RLOF,LogT,LogL,star1_mass,m_dot1),axis=1)
        ######################################################
        RLOF_df = pd.DataFrame(RLOF,columns=["num","RLOF","LogT","LogL","star1_mass","m_dot1"])
        RLOF_df.drop(RLOF_df[RLOF_df["RLOF"] <= 0.99].index, inplace=True)
        RLOF_99 = RLOF_df.to_numpy()
        RLOF_99 = RLOF_99[-1,:]
        RLOF_df.drop(RLOF_df[RLOF_df["RLOF"] < 1].index, inplace=True)
        RLOF_1 = RLOF_df.to_numpy()
        RLOF_1 = RLOF_1[0,:]

        ######################################################
        non_th = []
        Thermal_diff_df = pd.DataFrame(Thermal_diff_0,columns=["num","Thermal_diff","LogT","LogL","star1_mass","m_dot1"])
        Thermal_diff_df.drop(Thermal_diff_df[Thermal_diff_df["Thermal_diff"] > 0.2].index, inplace=True)
        Thermal_diff_non = Thermal_diff_df.to_numpy()
        non_th.append(Thermal_diff_non[-1,:])
        ######################################################

        Thermal_diff_df = pd.DataFrame(Thermal_diff_0,columns=["num","Thermal_diff","LogT","LogL","star1_mass","m_dot1"])
        Thermal_diff_df.drop(Thermal_diff_df[Thermal_diff_df["Thermal_diff"] > 0.1].index, inplace=True)
        Thermal_diff_on = Thermal_diff_df.to_numpy()

        on_th= [Thermal_diff_on[0,:]]
        ######################################################
        non_th = np.array(non_th)
        on_th = np.array(on_th)

        #print(non_th,on_th)
        #plt.plot(age,Thermal_diff)
        #plt.yscale("log")
        #plt.plot(age,np.linspace(0.25,0.25,len(age)))
        #plt.plot(age,np.linspace(0.15,0.15,len(age)))
        return TAMS,TACHeB, RLOF_1, RLOF_99,on_th,non_th
    # =========================================================================
    def extract(self,num):
        # simple function to where each eeps takes place, num is for deciding if one is working with ms or post ms, in case a eeps has more then two relevant points, a small fix is needed to append them
        if num == 0:
            TAMS,ZAMS, RLOF_1, RLOF_99,on_th,non_th = self.eeps() 
            ls = [TAMS,ZAMS, RLOF_1, RLOF_99,on_th,non_th]
        else : 
            TAMS,TACHeB, RLOF_1, RLOF_99,on_th,non_th = self.eeps_post() 
            ls = [TAMS,TACHeB, RLOF_1, RLOF_99,on_th,non_th]
        arr = []
        for x in ls:
            if x.ndim != 1:
                y= x[:,0].flatten()
                y = list(y)
                
                if len(y) == 1:
                    arr.append(y)
                else :
                    arr.append(y[0])
                    arr.append(y[1])
            else:
                arr.append(x[0])
        return np.array([arr],dtype=object)

    # =========================================================================

    def normalize(self,arr, t_min, t_max):
        # Normalize function to normalize the metric between each main eeps
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)   
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    # =========================================================================

    def metric(self,ls,arr):
        # Calculates and normalizes the metric value of every singel data point
        list = []
        eeps_D = np.linspace(0,len(arr)-1,len(arr))
        ###################################################### 
        for i in range(0,len(ls)-1):
            X = np.array(ls[i:i+1])
            x = X[:,:,1].flatten()
            y = X[:,:,2].flatten()
            z = X[:,:,3].flatten()
            D_i = i
            for n in range(0,len(x)-1):
                sum_1 = (x[n+1]-x[n])**2 
                sum_2 = (y[n+1]-y[n])**2
                D_i = D_i+ np.sqrt((sum_1+sum_2))
                list.append(D_i)
                if n==1:
                    D_first = D_i
                    D_last = (i+1)-(D_i-i)
                if n== len(x)-2:
                    arr_0 = np.array(list)
                    arr_0 = self.normalize(arr_0,D_first,D_last)
                    list = []
                    if i ==0:
                        D_points = (np.append(arr_0,eeps_D))
                    else :
                        D_points = np.append(D_points,arr_0)
        ###################################################### 
        D_points = np.sort(D_points)
        return D_points


    # =========================================================================

    def second_eeps(self,num):
        ''' Constructs a metric distance between the main EEPS, creating secondary eeps for each value of the data, !!returned as a panda dataframe!!'''
        if num == 0:
            TAMS,ZAMS, RLOF_1, RLOF_99,on_th,non_th = self.eeps() 
        else : 
            TAMS,TACHeB, RLOF_1, RLOF_99,on_th,non_th = self.eeps_post() 
        LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP  = self.clean(num)
        x1 = np.array(range(0,len(LogL),1))
        data = np.stack((x1,LogT,LogL,age,star1_mass,m_dot1,LogP),axis=1)
        sec_eeps = pd.DataFrame(data, columns=["num","LogT","LogL","age","Mass","M_dot","LogP"])
        arr = self.extract(num)
        arr = arr.flatten()
        #print(arr)
        arr = np.sort(arr)
        for x in range(0,len(arr)):
            sec_eeps["num"] = sec_eeps["num"].replace([arr[x]],0)
        sec_eeps_1 = sec_eeps.to_numpy()
        prufa = np.split(sec_eeps_1,np.where(sec_eeps_1[:,0]== 0)[0][1:])
        prufa_2 = self.metric(prufa,arr)
        sec_eeps["Metric"]= prufa_2
        #display(sec_eeps)
        return sec_eeps

        
    # =========================================================================

    def plot_HR(self):
        '''Plots the HR diagram for the given datasets and maps out the main EEPS that take place'''
        LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP  = self.clean(0)
        TAMS,ZAMS, RLOF_1, RLOF_99,on_th,non_th = self.eeps()
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(LogT,LogL, label =self.file)
        plt.xlim(max(LogT)+0.05,min(LogT)-0.1)
        plt.plot(TAMS[1],TAMS[2], marker="*",color="red", label= "TAMS")
        plt.plot(ZAMS[1],ZAMS[2], marker="*",color="blue", label= "ZAMS")
        plt.plot(RLOF_1[2],RLOF_1[3], marker="*",color="black", label= "RLOF")
        plt.plot(RLOF_99[2],RLOF_99[3], marker="*",color="orange", label= "NON_RLOF")
        plt.scatter(non_th[:,2],non_th[:,3], marker="*",color="purple", label= "NON_TH_EQ")
        #plt.scatter(on_th[:,2],on_th[:,3], marker="*",color="green", label= "TH_EQ")
        #plt.plot(mass_loss_0[2],mass_loss_0[3], marker="*",color="yellow", label= "mass_loss_0")
        #plt.plot(mass_loss_1[2],mass_loss_1[3], marker="*",color="grey", label= "mass_loss_1")
        plt.legend()
        plt.show()

    # =========================================================================

    def plot_mdot(self):
        '''Plots the mass loss of the system that takes place of the main phase against the mass of the main star'''
        LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP  = self.clean(0)
        TAMS,ZAMS, RLOF_1, RLOF_99,on_th,non_th = self.eeps()
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(star1_mass,m_dot1, label =self.file)
        plt.xlim(max(star1_mass)+0.05,min(star1_mass)-0.1)
        plt.plot(TAMS[3],TAMS[4], marker="*",color="red", label= "TAMS")
        plt.plot(ZAMS[3],ZAMS[4], marker="*",color="blue", label= "ZAMS")
        plt.plot(RLOF_1[4],RLOF_1[5], marker="*",color="black", label= "RLOF")
        plt.plot(RLOF_99[4],RLOF_99[5], marker="*",color="orange", label= "NON_RLOF")
        plt.scatter(non_th[:,4],non_th[:,5], marker="*",color="purple", label= "NON_TH_EQ")
        plt.scatter(on_th[:,4],on_th[:,5], marker="*",color="green", label= "TH_EQ")
        #plt.plot(mass_loss_0[4],mass_loss_0[5], marker="*",color="yellow", label= "mass_loss_0")
        #plt.plot(mass_loss_1[4],mass_loss_1[5], marker="*",color="grey", label= "mass_loss_1")
        plt.legend()
        plt.show()


    def plot_HR_post(self):
        '''Plots the HR diagram for the given datasets and maps out the main EEPS that take place'''
        LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP  = self.clean(1)
        TAMS,TACHeB, RLOF_1, RLOF_99,on_th,non_th = self.eeps_post()
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(LogT,LogL, label =self.file)
        plt.xlim(max(LogT)+0.05,min(LogT)-0.1)
        plt.plot(TAMS[1],TAMS[2], marker="*",color="red", label= "TAMS")
        plt.plot(TACHeB[1],TACHeB[2], marker="*",color="blue", label= "TACHeB")
        plt.plot(RLOF_1[2],RLOF_1[3], marker="*",color="black", label= "RLOF")
        plt.plot(RLOF_99[2],RLOF_99[3], marker="*",color="orange", label= "NON_RLOF")
        plt.scatter(non_th[:,2],non_th[:,3], marker="*",color="purple", label= "NON_TH_EQ")
        plt.scatter(on_th[:,2],on_th[:,3], marker="*",color="green", label= "TH_EQ")
        #plt.plot(mass_loss_0[2],mass_loss_0[3], marker="*",color="yellow", label= "mass_loss_0")
        #plt.plot(mass_loss_1[2],mass_loss_1[3], marker="*",color="grey", label= "mass_loss_1")
        plt.legend()
        plt.show()
# =========================================================================
# =========================================================================

def interpolate(x,y_11,y_12,x_11,x_12,y_21,y_22,x_21,x_22,z,z_1,z_2):
    # z is the initial value of log(P) and z_1,z_2 the initial log(P) of the two datasets being interpolatet
    # x is the point beig interpolated and x_1 and x_2 the closest points to the x
    # y_1 and y_2 is the value of the of the y axis at the points x_1 etc..
    # x-axis is the METRIC and the y axis is any given value we are interpolating
    # !!The x and y values that start with one is the datasets set with LogP > LogP of the system that is being interpolated and vice versa
    y_1 = y_11 + (y_12 - y_11)*(x-x_11)/(x_12-x_11) 
    y_2 = y_21 + (y_22 - y_21)*(x-x_21)/(x_22-x_21)
    y = y_1 +(y_2 - y_1)*(z-z_1)/(z_2-z_1)
    return y

# =========================================================================

def int_data(dict1,dict2,val,P_1,P_2,P_3,s):
    # val = 0 is LogT, 1 is LogL, 2 is Mass, 3 is M_dot, s is the size of the array returned, should be the same size as the one being interpolated to for comparisons
    # P_1 is the initial logP of the first data set (< P3), P_2 is the initial LogP of the second dataset (> P3) and P_3 is logP of the value we want to interpolate
    ''' Interpolates two dataset and returns the dataset between them as a array along with its metric'''
    x_1 = dict1["Metric"].to_numpy()
    x_2 = dict2["Metric"].to_numpy()
     # Does not work if one datasets has more main eeps than the other as the metric is unusable in that case
    if max(x_1) != max(x_2):
        raise ValueError("Main EEPS are not equal")
    if val == 0:
        y_1 = dict1["LogT"].to_numpy()
        y_2 = dict2["LogT"].to_numpy()
    elif val== 1:
        y_1 = dict1["LogL"].to_numpy()
        y_2 = dict2["LogL"].to_numpy()
    elif val== 2:
        y_1 = dict1["Mass"].to_numpy()
        y_2 = dict2["Mass"].to_numpy()
    else:
        y_1 = dict1["M_dot"].to_numpy()
        y_2 = dict2["M_dot"].to_numpy()
    x = np.linspace(0,max(x_1),s)
    ls_x1 = []
    ls_x2 = []
    # some very slow nested loops to find the two closest points in the metric corresponding to the point being interpolated
    for n in x:
        i = np.searchsorted(x_1,n,side="right")
        if n==max(x):
            ls_x1.append([x_1[i-2],x_1[i-1],y_1[i-2],y_1[i-1],i-1])
        else:
            if n < x_1[i]:
                ls_x1.append([x_1[i-1],x_1[i],y_1[i-1],y_1[i],i])
            else : 
                ls_x1.append([x_1[i],x_1[i+1],y_1[i],y_1[i+1],i])
    for n in x:
        i = np.searchsorted(x_2,n,side="right")
        if n==max(x):
            ls_x2.append([x_2[i-2],x_2[i-1],y_2[i-2],y_2[i-1],i-1])
        else:
            if n < x_2[i]:
                ls_x2.append([x_2[i-1],x_2[i],y_2[i-1],y_2[i],i])
            else : 
                ls_x2.append([x_2[i],x_2[i+1],y_2[i],y_2[i+1],i])
    ls_x1 = np.array(ls_x1)
    ls_x2 = np.array(ls_x2)
    ls_y = []
    for n in range(len(x)):
        y = interpolate(x[n],ls_x2[n,2],ls_x2[n,3],ls_x2[n,0],ls_x2[n,1],ls_x1[n,2],ls_x1[n,3],ls_x1[n,0],ls_x1[n,1],P_3,P_2,P_1)
        ls_y.append(y)
    # resulting size of the interpolated data is the same as the larger dataset of the two datasets being interpolated
    return np.array(ls_y),x


# =========================================================================
# =========================================================================
 #Work in progress, 4 point interpolation


def interpolate4p(x,y_11,y_12,x_11,x_12,y_21,y_22,x_21,x_22,y_31,y_32,x_31,x_32,y_41,y_42,x_41,x_42,z,z_1,z_2,z_3,z_4):

    y_1 = y_11 + (y_12 - y_11)*(x-x_11)/(x_12-x_11) # middle up
    y_2 = y_21 + (y_22 - y_21)*(x-x_21)/(x_22-x_21) #middle down
    y_3 = y_31 + (y_32 - y_31)*(x-x_31)/(x_32-x_31) #top
    y_4 = y_41 + (y_42 - y_41)*(x-x_41)/(x_42-x_41) #bot
    w_1 = 1/(z+z_1)
    w_2 = 1/(z+z_3)
    # Tried working out of the calculation of the point would go, without succes
    y =(y_1 + (y_2 - y_1)*(z-z_1)/(z_2-z_1) + (y_4 - y_3)*(z-z_3)/(z_4-z_3)) +( y_3 + (y_2 - y_3)*(z-z_3)/(z_2-z_3) + (y_4 - y_3)*(z-z_3)/(z_4-z_3))
    #y = y_1 + ((y_2+y_4)/2 - y_1)*(z-z_1)/(z_2-z_1)*(z-z_1)/(z_4-z_1) + y_3 + ((y_2+y_4)/2 - y_3)*(z-z_3)/(z_2-z_3)*(z-z_1)/(z_4-z_1)
    #y = (w_1*y_1+w_2*y_3)+ (y_2+y_4 - y_1-y_3)*(z-z_1)/(z_2-z_1)*(z-z_3)/(z_4-z_3)#+ (y_4 - y_1)*(z-z_1)/(z_4-z_1) +( y_3 + (y_2 - y_3)*(z-z_3)/(z_2-z_3) + (y_4 - y_3)*(z-z_3)/(z_4-z_3))
    #y = w_1*(y_1 + (y_2 - y_1)*(z-z_1)/(z_2-z_1)+(y_4 - y_1)*(z-z_1)/(z_4-z_1)) +w_2*( y_3 + (y_2 - y_3)*(z-z_3)/(z_2-z_3) + (y_4 - y_3)*(z-z_3)/(z_4-z_3))
    return y


# =========================================================================

def int_data4p(dict1,dict2,dict3,dict4,val,P_1,P_2,P_3,P_4,P_5,s):
    #Work in progress
    ''' Interpolates two dataset and returns the dataset between them as a array along with its metric'''
    x_1 = dict1["Metric"].to_numpy()
    x_2 = dict2["Metric"].to_numpy()
    x_3 = dict3["Metric"].to_numpy()
    x_4 = dict4["Metric"].to_numpy()
     # Does not work if one datasets has more main eeps than the other as the metric is unusable in that case
    if max(x_1) != max(x_2):
        raise ValueError("Main EEPS are not equal")
    if val == 0:
        y_1 = dict1["LogT"].to_numpy()
        y_2 = dict2["LogT"].to_numpy()
        y_3 = dict3["LogT"].to_numpy()
        y_4 = dict4["LogT"].to_numpy()
    elif val== 1:
        y_1 = dict1["LogL"].to_numpy()
        y_2 = dict2["LogL"].to_numpy()
        y_3 = dict3["LogL"].to_numpy()
        y_4 = dict4["LogL"].to_numpy()
    elif val== 2:
        y_1 = dict1["Mass"].to_numpy()
        y_2 = dict2["Mass"].to_numpy()
        y_3 = dict3["Mass"].to_numpy()
        y_4 = dict4["Mass"].to_numpy()
    else:
        y_1 = dict1["M_dot"].to_numpy()
        y_2 = dict2["M_dot"].to_numpy()
        y_3 = dict3["M_dot"].to_numpy()
        y_4 = dict4["M_dot"].to_numpy()
    x = np.linspace(0,max(x_1),s)
    ls_x1 = []
    ls_x2 = []
    ls_x3 = []
    ls_x4 = []
    # some very slow nested loops to find the two closest points in the metric corresponding to the point being interpolated
    for n in x:
        i = np.searchsorted(x_1,n,side="right")
        if n==max(x):
            ls_x1.append([x_1[i-2],x_1[i-1],y_1[i-2],y_1[i-1],i-1])
        else:
            if n < x_1[i]:
                ls_x1.append([x_1[i-1],x_1[i],y_1[i-1],y_1[i],i])
            else : 
                ls_x1.append([x_1[i],x_1[i+1],y_1[i],y_1[i+1],i])
    for n in x:
        i = np.searchsorted(x_2,n,side="right")
        if n==max(x):
            ls_x2.append([x_2[i-2],x_2[i-1],y_2[i-2],y_2[i-1],i-1])
        else:
            if n < x_2[i]:
                ls_x2.append([x_2[i-1],x_2[i],y_2[i-1],y_2[i],i])
            else : 
                ls_x2.append([x_2[i],x_2[i+1],y_2[i],y_2[i+1],i])

    for n in x:
        i = np.searchsorted(x_3,n,side="right")
        if n==max(x):
            ls_x3.append([x_3[i-2],x_3[i-1],y_3[i-2],y_3[i-1],i-1])
        else:
            if n < x_3[i]:
                ls_x3.append([x_3[i-1],x_3[i],y_3[i-1],y_3[i],i])
            else : 
                ls_x3.append([x_3[i],x_3[i+1],y_3[i],y_3[i+1],i])

    for n in x:
        i = np.searchsorted(x_4,n,side="right")
        if n==max(x):
            ls_x4.append([x_4[i-2],x_4[i-1],y_4[i-2],y_4[i-1],i-1])
        else:
            if n < x_4[i]:
                ls_x4.append([x_4[i-1],x_4[i],y_4[i-1],y_4[i],i])
            else : 
                ls_x2.append([x_4[i],x_4[i+1],y_4[i],y_4[i+1],i])

    ls_x1 = np.array(ls_x1)
    ls_x2 = np.array(ls_x2)
    ls_x3 = np.array(ls_x3)
    ls_x4 = np.array(ls_x4)
    ls_y = []
    for n in range(len(x)):
        y = interpolate4p(x[n],ls_x2[n,2],ls_x2[n,3],ls_x2[n,0],ls_x2[n,1],ls_x1[n,2],ls_x1[n,3],ls_x1[n,0],ls_x1[n,1]
                          ,ls_x3[n,2],ls_x3[n,3],ls_x3[n,0],ls_x3[n,1],ls_x4[n,2],ls_x4[n,3],ls_x4[n,0],ls_x4[n,1]
                          ,P_3,P_2,P_1,P_4,P_5)
        ls_y.append(y)

    return np.array(ls_y),x

# =========================================================================
# =========================================================================

def plot_Hr(dict1,dict2,P1,P2,P3,dict3,points):
    '''Plots the HR diagram of the newly interpolated data against the HR of the orignal data'''
    test,x = int_data(dict1,dict2,0,P1,P2,P3,len(dict3))
    test1,x = int_data(dict1,dict2,1,P1,P2,P3,len(dict3))
    TAMS,ZAMS, RLOF_1, RLOF_99,on_th,non_th = points()
    plt.figure(figsize=(9, 7), dpi=200)
    plt.xlim(max(dict3["LogT"])+0.05,min(dict3["LogT"])-0.1)
    plt.plot(TAMS[1],TAMS[2], marker="*",color="red", label= "TAMS")
    plt.plot(ZAMS[1],ZAMS[2], marker="*",color="blue", label= "ZAMS/TACHeB")
    plt.plot(RLOF_1[2],RLOF_1[3], marker="*",color="black", label= "RLOF")
    plt.plot(RLOF_99[2],RLOF_99[3], marker="*",color="orange", label= "NON_RLOF")
    plt.scatter(non_th[:,2],non_th[:,3], marker="*",color="purple", label= "NON_TH_EQ")
    plt.scatter(on_th[:,2],on_th[:,3], marker="*",color="green", label= "TH_EQ")
    plt.plot(test,test1,linestyle='--',label=f"Interpolated LogP {P1} and {P2}")
    plt.plot(dict3["LogT"],dict3["LogL"],label=f'LogP={P3}')
    plt.xlabel("Log(T)")
    plt.ylabel("Log(L)")
    plt.legend()
    plt.show()
    MSE = np.square(np.subtract(dict3["LogT"],test)).mean()
    MSE1 = np.square(np.subtract(dict3["LogL"],test1)).mean()
    print(f"Mean squeared error of LogT {MSE}, LogL {MSE1}")

# =========================================================================

def close():
    '''Plots the evolution of each relevant final value against the inital LogP value of each run'''
    x_1 = np.linspace(0.1,1.1,10)
    list = [dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7,dict_8,dict_8,dict_9,dict_10,dict_11]
    y_1 = []
    list_1 = ["LogT","LogL","age","Mass","M_dot","LogP"]
    for l in list_1: 
        for n in range(len(x_1)):
            y_1.append(list[n][l].iloc[-1])
    y_1 = np.array(y_1)
    y_1 = np.reshape(y_1,(6,10))
    fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax1.set_title(f"Final values of parameters for each simulation")
    ax2.plot(x_1,y_1[1,:])
    ax2.set_ylabel(list_1[1])
    ax3.plot(x_1,y_1[2,:])
    ax3.set_ylabel(list_1[2])
    ax4.plot(x_1,y_1[3,:])
    ax4.set_ylabel(list_1[3])
    ax5.plot(x_1,y_1[4,:])
    ax5.set_ylabel(list_1[4])
    ax6.plot(x_1,y_1[5,:])
    ax6.set_ylabel(list_1[5])
    ax1.plot(x_1,y_1[0,:])
    ax1.set_ylabel(list_1[0])
    plt.xlabel("LogP initial")
    plt.show()

# =========================================================================

def comaprisons(dict1,dict2,P1,P2,P3,dict3):
    +'''Plots the comparison between the simulation and interpolated values agains the metric'''
    test,x = int_data(dict1,dict2,0,P1,P2,P3,len(dict3))
    test1,x = int_data(dict1,dict2,1,P1,P2,P3,len(dict3))
    test2,x = int_data(dict1,dict2,2,P1,P2,P3,len(dict3))
    test3,x = int_data(dict1,dict2,3,P1,P2,P3,len(dict3))
    metric = dict3["Metric"].to_numpy()
    fig,(ax1,ax2,ax3,ax4) = plt.subplots(4)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax1.set_title(f"Interpolated values compared to simulation values")
    ax1.plot(metric,dict3["LogT"],label="Simulation")
    ax1.plot(x,test,linestyle='--',label="Interpolated value")
    ax1.set_ylabel("LogT")
    ax1.legend()
    ax2.plot(metric,dict3["LogL"],label="Simulation")
    ax2.plot(x,test1,linestyle='--',label="Interpolated value")
    ax2.set_ylabel("LogL")
    ax2.legend()
    ax3.plot(metric,dict3["Mass"],label="Simulation")
    ax3.plot(x,test2,linestyle='--',label="Interpolated value")
    ax3.set_ylabel("Mass")
    ax3.legend()
    ax4.plot(metric,dict3["M_dot"],label="Simulation")
    ax4.plot(x,test3,linestyle='--',label="Interpolated value")
    ax4.set_ylabel("M_dot")
    ax4.legend()
    ax4.set_xlabel("Metric")
    plt.show()

# =========================================================================

def data(num):
    """Loads in the data needed for interpolation, num = 0 for ms, any other int for post ms"""
    dict_1 = test_1.second_eeps(num)
    dict_2 = test_2.second_eeps(num)
    dict_3 = test_3.second_eeps(num)
    dict_4 = test_4.second_eeps(num)
    dict_5 = test_5.second_eeps(num)
    dict_6 = test_6.second_eeps(num)
    dict_8 = test_8.second_eeps(num)
    dict_7 = test_7.second_eeps(num)
    dict_9 = test_9.second_eeps(num)
    dict_10 = test_10.second_eeps(num)
    dict_11 = test_11.second_eeps(num)
    return dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7,dict_8,dict_9,dict_10,dict_11

def main():
    """Takes in four arguments, first, either 0 for ms, 1 for post ms, argument 2,3,4 are the runs being interpolated from and two, from 2,3 and to 4, numbers from 1-11"""
    num = sys.argv[1]
    dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7,dict_8,dict_9,dict_10,dict_11 = data(num)
    dict = {1:["dict_1",0.1,"test_1.eeps","test_1.eeps_post"],2:["dict_2",0.2,"test_2.eeps","test_2.eeps_post"],
            3:["dict_3",0.3,"test_3.eeps","test_3.eeps_post"],4:["dict_4",0.4,"test_4.eeps","test_4.eeps_post"],
            5:["dict_5",0.5,"test_5.eeps","test_5.eeps_post"],6:["dict_6",0.6,"test_6.eeps","test_6.eeps_post"],
            7:["dict_7",0.7,"test_7.eeps","test_7.eeps_post"],8:["dict_8",0.8,"test_8.eeps","test_8.eeps_post"],
            9:["dict_9",0.9,"test_9.eeps","test_9.eeps_post"],10:["dict_10",1,"test_10.eeps","test_10.eeps_post"],
            11:["dict_11",1.1,"test_11.eeps","test_11.eeps_post"]}
    inter1= sys.argv[2]
    inter2= sys.argv[3]
    inter3= sys.argv[4]
    if num != 0:
        plot_Hr(dict[inter1][0],dict[inter2][0],dict[inter1][1],dict[inter2][1],dict[inter3][1],dict[inter3][0],dict[inter3][3])
    else:
        plot_Hr(dict[inter1][0],dict[inter2][0],dict[inter1][1],dict[inter2][1],dict[inter3][1],dict[inter3][0],dict[inter3][2])
    close()
# ========================================================================= 
# ========================================================================= 

#  Main cause why not, everybody else does it...   
if __name__ == '__main__':
    # If one wants to add other history data files, keep the name the same and start by running the trough the class.
    test_1 = mp_overflow("mesa_bin\logP0.10\LOGS1\history.data")
    test_2 = mp_overflow("mesa_bin\logP0.20\LOGS1\history.data")
    test_3 = mp_overflow("mesa_bin\logP0.30\LOGS1\history.data")
    test_4 = mp_overflow("mesa_bin\logP0.40\LOGS1\history.data")
    test_5 = mp_overflow("mesa_bin\logP0.50\LOGS1\history.data")
    test_6 = mp_overflow("mesa_bin\logP0.60\LOGS1\history.data")
    test_7 = mp_overflow("mesa_bin\logP0.70\LOGS1\history.data")
    test_8 = mp_overflow("mesa_bin\logP0.80\LOGS1\history.data")
    test_9 = mp_overflow("mesa_bin\logP0.90\LOGS1\history.data")
    test_10 = mp_overflow("mesa_bin\logP1.00\LOGS1\history.data")
    test_11 = mp_overflow("mesa_bin\logP1.10\LOGS1\history.data")



    test_21 = mp_overflow("mesa_bin\\twobin\LOGS1_P0.6\history.data")
    test_22 = mp_overflow("mesa_bin\\twobin\LOGS2_P0.6\history.data")
    test_21.plot_HR()
    #LogT,LogL,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP = test_21.clean(0)
    #LogT1,LogL1,star1_mass,m_dot1,RLOF,Thermal_eq,age,Thermal_diff,LogP = test_22.clean(0)
    #plt.xlim(max(LogT)+0.05,min(LogT)-0.1)
    #plt.plot(LogT,LogL)
    #plt.plot(LogT1,LogL1,linestyle="--")
    #main()
    
    #Running not from the terminal
    
    #dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7,dict_8,dict_9,dict_10,dict_11 = data(0)
    #dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7,dict_8,dict_9,dict_10,dict_11 = data(1)
    #plot_Hr(dict_1,dict_3,0.1,0.3,0.2,dict_2,test_2.eeps_post)
    #plot_Hr(dict_2,dict_4,0.2,0.4,0.3,dict_3,test_3.eeps_post)
    #plot_Hr(dict_3,dict_5,0.3,0.5,0.4,dict_4,test_4.eeps_post)
    #plot_Hr(dict_4,dict_6,0.4,0.6,0.5,dict_5,test_5.eeps_post)
    #plot_Hr(dict_3,dict_9,0.3,0.9,0.6,dict_6,test_6.eeps)
    #plot_Hr(dict_6,dict_8,0.6,0.8,0.7,dict_7,test_7.eeps)
    #plot_Hr(dict_7,dict_9,0.7,0.9,0.8,dict_8,test_8.eeps_post)
    #plot_Hr(dict_7,dict_11,0.7,1.1,0.9,dict_9,test_9.eeps)
    #plot_Hr(dict_9,dict_11,0.9,1.1,1,dict_10,test_10.eeps_post)
    #close()
    #comaprisons(dict_5,dict_7,0.5,0.7,0.6,dict_6)
    
    # In progress
    
    #T,x = int_data4p(dict_5,dict_7,dict_8,dict_4,0,0.5,0.7,0.6,0.8,0.4,len(dict_6))
    #L,x = int_data4p(dict_5,dict_7,dict_8,dict_4,1,0.5,0.7,0.6,0.8,0.4,len(dict_6))
    #T,x = int_data4p(dict_8,dict_10,dict_11,dict_7,0,0.8,1,0.9,1.1,0.7,len(dict_9))
    #L,x = int_data4p(dict_8,dict_10,dict_11,dict_7,1,0.8,1,0.9,1.1,0.7,len(dict_9))
    #test,x = int_data(dict_8,dict_10,0,0.8,1,0.9,len(dict_9))
    #test1,x = int_data(dict_8,dict_10,1,0.8,1,0.9,len(dict_9))
    #plt.xlim(max(T)+0.05,min(T)-0.1)
    #plt.plot(T,L)
    #plt.plot(test,test1,linestyle="--")
    

    

    

