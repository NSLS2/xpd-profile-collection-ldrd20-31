import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate  
#import scipy.signal as scipy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import importlib
de = importlib.import_module("_data_export")

""" This module provides functions for data analysis.
    Usually imported as da.
"""



def _read_meta_csv(fn, header=14):
    """Read meta data (fitting result) in csv

    Args:
        fn (str): full file name with path
        header (int, optional): numbers of line for metadata in header. Defaults to 14.

    Returns:
        dict: metadata
    """
    meta = {}
    with open (fn, 'r') as f:
        temp = f.readlines()[:header]
        for i in range(len(temp)):
            t = temp[i].strip('\n').split(',')
            meta[t[0]] = t[1:]
    return meta



## Fit a peak by 1 Gaussian or Lorentz distribution
## http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn2.html
## https://en.wikipedia.org/wiki/Cauchy_distribution
def _1gauss(x, A, x0, sigma):
    """One-peak gaussian distribution

    Args:
        x (array_like): input value
        A (float): amplitude of distribution
        x0 (float): mean value of distribution
        sigma (float): variance of distribution (sigma**2)

    Returns:
        array_like: distribution
    """
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def _1Lorentz(x, A, x0, sigma):
    """One peak Lorentz distribution

    Args:
        x (array_like): input value
        A (float): amplitude of distribution
        x0 (float): mean value of distribution
        sigma (float): variance of distribution (sigma**2)

    Returns:
        array_like: distribution
    """
    return A*sigma**2/((x-x0)**2+sigma**2)


## Fit a peak by Multi Gaussian or Lorentz distributions
## http://hyperphysics.phy-astr.gsu.edu/hbase/Math/gaufcn2.html

def _2gauss(x, A1, x1, s1, A2, x2, s2):
    """Two-peak gaussian distribution

    Args:
        x (array_like): input value
        A1 (float): amplitude of 1st distribution
        x1 (float): mean value of 1st distribution
        s1 (float): variance of 1st distribution (sigma**2)
        A2 (float): amplitude of 2nd distribution
        x2 (float): mean value of 2nd distribution
        s2 (float): variance of 2nd distribution (sigma**2)

    Returns:
        array_like: distribution
    """
    return (_1gauss(x, A1, x1, s1) +
            _1gauss(x, A2, x2, s2))


def _3gauss(x, A1, x1, s1, A2, x2, s2, A3, x3, s3):
    """Three-peak gaussian distribution

    Args:
        x (array_like): input value
        A1 (float): amplitude of 1st distribution
        x1 (float): mean value of 1st distribution
        s1 (float): variance of 1st distribution (sigma**2)
        A2 (float): amplitude of 2nd distribution
        x2 (float): mean value of 2nd distribution
        s2 (float): variance of 2nd distribution (sigma**2)
        A3 (float): amplitude of 3rd distribution
        x3 (float): mean value of 3rd distribution
        s3 (float): variance of 3rd distribution (sigma**2)

    Returns:
        array_like: distribution
    """
    return (_1gauss(x, A1, x1, s1) +
            _1gauss(x, A2, x2, s2) + 
            _1gauss(x, A3, x3, s3))


def _2Lorentz(x, A1, x1, s1, A2, x2, s2):
    """Two-peak Lorentz distribution

    Args:
        x (array_like): input value
        A1 (float): amplitude of 1st distribution
        x1 (float): mean value of 1st distribution
        s1 (float): variance of 1st distribution (sigma**2)
        A2 (float): amplitude of 2nd distribution
        x2 (float): mean value of 2nd distribution
        s2 (float): variance of 2nd distribution (sigma**2)

    Returns:
        array_like: distribution
    """
    return (_1Lorentz(x, A1, x1, s1) +
            _1Lorentz(x, A2, x2, s2))


def _3Lorentz(x, A1, x1, s1, A2, x2, s2, A3, x3, s3):
    """Three-peak Lorentz distribution

    Args:
        x (array_like): input value
        A1 (float): amplitude of 1st distribution
        x1 (float): mean value of 1st distribution
        s1 (float): variance of 1st distribution (sigma**2)
        A2 (float): amplitude of 2nd distribution
        x2 (float): mean value of 2nd distribution
        s2 (float): variance of 2nd distribution (sigma**2)
        A3 (float): amplitude of 3rd distribution
        x3 (float): mean value of 3rd distribution
        s3 (float): variance of 3rd distribution (sigma**2)

    Returns:
        array_like: distribution
    """
    return (_1Lorentz(x, A1, x1, s1) +
            _1Lorentz(x, A2, x2, s2) +
            _1Lorentz(x, A3, x3, s3))



def find_nearest(array, value):
    """find the nearest value in a given array

    Args:
        array (array_like): input array
        value (float): target value

    Returns:
        int: index of the nearest value in the array
        float: the nearest value in the array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]



def r_square(x, y, fitted_y, y_low_limit=200):
    """R square value for a 

    Args:
        x (array_like): input values of x-axis
        y (array_like): input values of y-axis
        fitted_y (array_like): fitting data or values of input y-axis
        y_low_limit (float, optional): y smaller than this value will be excluded. Defaults to 200.

    Returns:
        float: R square, which corresponds to the coefficient of determination
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    fitted_y = np.asarray(fitted_y)
    
    y1 = y[y>=y_low_limit]
    x1 = x[y>=y_low_limit]
    fitted_y1 = fitted_y[y>=y_low_limit]
    
    residulas = y1 - fitted_y1
    ss_res = np.sum(residulas**2)
    ss_tot = np.sum((y1-np.mean(y1))**2)
    r_sq = 1 - (ss_res / ss_tot)
    return r_sq



def vol_unit_converter(v0 = 'ul', v1 = 'ml'):
    """Convert volume between 'pl', 'nl', 'ul', 'ml'

    Args:
        v0 (str, optional): initial volume unit. Defaults to 'ul'.
        v1 (str, optional): targeted volume unit. Defaults to 'ml'.

    Returns:
        float: conversion constant (ex: vol_unit_converter(v0 = 'ul', v1 = 'ml')=0.001)
    """
    vol_unit = ['pl', 'nl', 'ul', 'ml']
    vol_frame = pd.DataFrame(data={'pl': np.geomspace(1, 1E9, num=4), 'nl': np.geomspace(1E-3, 1E6, num=4),
                                   'ul': np.geomspace(1E-6, 1E3, num=4), 'ml': np.geomspace(1E-9, 1, num=4)}, index=vol_unit)
    return vol_frame.loc[v0, v1]


def t_unit_converter(t0 = 'min', t1 = 'min'):
    """Convert time between 'sec', 'min', 'hr'

    Args:
        t0 (str, optional): initial time unit. Defaults to 'min'.
        t1 (str, optional): targeted time unit. Defaults to 'min'.

    Returns:
        float: conversion constant (ex: t_unit_converter(v0 = 'min', v1 = 'sec')=60)
    """
    t_unit = ['sec', 'min', 'hr']
    t_frame = pd.DataFrame(data={'sec': np.geomspace(1, 3600, num=3), 
                                 'min': np.geomspace(1/60, 60, num=3), 
                                 'hr' : np.geomspace(1/3600, 1, num=3)}, index=t_unit)
    return t_frame.loc[t0, t1]


def rate_unit_converter(r0 = 'ul/min', r1 = 'ul/min'):
    """Conver rate unit 

    Args:
        r0 (str, optional): initial rate unit. Defaults to 'ul/min'.
        r1 (str, optional): targeted rate unit. Defaults to 'ul/min'.

    Returns:
        float: conversion constant
    """
    
    v0 = r0.split('/')[0]
    t0 = r0.split('/')[1]
    v1 = r1.split('/')[0]
    t1 = r1.split('/')[1]

    ## ruc = rate_unit_converter
    ruc = vol_unit_converter(v0=v0, v1=v1) / t_unit_converter(t0=t0, t1=t1)
    return ruc



### Filter out bad absorption data due to PF oil ###
def percentile_abs(wavelength, absorbance, w_range=[210, 700], percent_range=[15, 85]):
    absorbance = np.nan_to_num(absorbance, nan=0)
    try:
        idx1, _ = find_nearest(wavelength[0], w_range[0])
        idx2, _ = find_nearest(wavelength[0], w_range[1])
    except (IndexError):
        idx1, _ = find_nearest(wavelength, w_range[0])
        idx2, _ = find_nearest(wavelength, w_range[1]) 

    abs_array2 = absorbance[:, idx1:idx2]
    
    try:
        wavelength2 = wavelength[:, idx1:idx2]
    except (IndexError):
        wavelength2 = wavelength[idx1:idx2]

    iqr = np.multiply(abs_array2, wavelength2).mean(axis=1)
    q0, q1 = np.percentile(iqr, percent_range)
    idx_iqr = np.argwhere(np.logical_and(iqr>=q0, iqr<=q1))


    abs_percentile = np.zeros((idx_iqr.shape[0], absorbance.shape[1]))
    j = 0
    for i in idx_iqr.flatten():
        abs_percentile[j] = absorbance[i]
        j += 1

    return abs_percentile



### Filter out bad fluorescence PL data due to PF oil ###
def percentile_PL(wavelength, fluorescence, w_range=[400, 800], percent_range=[30, 100]):
    fluorescence = np.nan_to_num(fluorescence, nan=0)
    try:
        idx1, _ = find_nearest(wavelength[0], w_range[0])
        idx2, _ = find_nearest(wavelength[0], w_range[1])
    except (IndexError):
        idx1, _ = find_nearest(wavelength, w_range[0])
        idx2, _ = find_nearest(wavelength, w_range[1])        

    PL_array2 = fluorescence[:, idx1:idx2]
    
    try:
        wavelength2 = wavelength[:, idx1:idx2]
    except (IndexError):
        wavelength2 = wavelength[idx1:idx2]

    # iqr = np.multiply(abs_array2, wavelength2).mean(axis=1)
    iqr = np.max(fluorescence, axis=1)
    q0, q1 = np.percentile(iqr, percent_range)
    idx_iqr = np.argwhere(np.logical_and(iqr>=q0, iqr<=q1))


    PL_percentile = np.zeros((idx_iqr.shape[0], fluorescence.shape[1]))
    j = 0
    for i in idx_iqr.flatten():
        PL_percentile[j] = fluorescence[i]
        j += 1

    return PL_percentile




#### Criteria to identify a bad fluorescence peak ####
# c1. PL peak height < 2000 --> execute in scipy.find_peaks
# c2. PL peak wavelength < 560 nm, and the diference of peak integrstion (PL minus LED) < 100000
# c3. PL peak wavelength >= 560 nm, and the diference of peak integrstion (PL minus LED) < 200000
def good_bad_data(x, y, key_height = 2000, data_id = 'test', distance=30, height=30, 
                  c2_c3 = False, threshold=[560, 100000, 200000], int_boundary = [340, 400, 800], 
                  dummy_test = False):
    
    # Identify all peaks, including LED source, of UVVIS spectrum
    peak, prop = find_peaks(y, height=height, distance=distance)

    # Divide the spectrum into 2 regions for integration:
    # 1. 340 nm - 400 nm : LED source
    # 2. 400 nm - 800 nm : fluorescence
    if (len(int_boundary)>=3 and c2_c3==True):
        w1, _ = find_nearest(x, int_boundary[0])
        w2, _ = find_nearest(x, int_boundary[1])
        w3, _ = find_nearest(x, int_boundary[2])

        LED_integration = integrate.simpson(y[w1:w2])
        PL_integration = integrate.simpson(y[w2:w3])
        peak_diff = PL_integration - LED_integration

    peak_heights_2 = []
    peak2 = []
    prop2 = {'peak_heights':[]}

    # Take LED peak intensity off and only consider peaks > 400 nm for good/bad data because
    # 1. sometines LED peak has very high intensity which is larger than key_height
    # 2. fitting function depends on the # of peaks excluding the LED peak

    for i in peak:
        if dummy_test:
            peak_heights_2.append(y[i])
            peak2.append(i)
            prop2['peak_heights'].append(y[i])
        else:
            if x[i] < 400:
                peak_heights_2.append(0.0)
            else:
                peak_heights_2.append(y[i])
                peak2.append(i)
                prop2['peak_heights'].append(y[i])

    max_idx = peak_heights_2.index(max(peak_heights_2))

    # c1 equls True which indicates bad data
    c1 = (peak_heights_2[max_idx] < key_height)

    c2, c3 = False, False
    if c2_c3 == True:
        try:
            # c2 equls True and c3 equls None which indicates bad data
            if x[peak[max_idx]]<threshold[0]:
                c3 = None
                c2 = (x[peak[max_idx]]<threshold[0] and peak_diff < threshold[1])
            
            # c2 equls None and c3 equls Turn which indicates bad data
            else:
                c2 = None
                c3 = (x[peak[max_idx]]>threshold[0] and peak_diff < threshold[2])
        except (IndexError, TypeError):
            pass
       
    # when c1, c2, or c3 is True which indicates bad data, return peak2 and prop2 as []
    if c1:
        print(f'{data_id} is bad due to a low peak height (c1).')
        peak2, prop2 = [], []
        return peak2, prop2
    elif c2:
        print(f'{data_id} is bad due to a low peak integartion (c2).')
        peak2, prop2 = [], []
        return peak2, prop2
    elif c3:
        print(f'{data_id} is bad due to a low peak integartion (c3).')
        peak2, prop2 = [], []
        return peak2, prop2
    
    
    # Otherwise return peak2 and prop2 for good data
    else:
        if c2_c3 == False:
            print(f'{data_id} passes c1 so is good.')
        else:
            if c3 == None:
                print(f'{data_id} passes c1, c2, so is good.')
            if c2 == None:
                print(f'{data_id} passes c1, c3 so is good.')
    
        # therefore if data is good, peak2 will be an array and prop2 will be {}.
        return np.asarray(peak2), prop2





def _1peak_fit_good_PL(x0, y0, fit_function, peak=False, maxfev=100000, fit_boundary=[340, 400, 800], raw_data=False,
                       plot=False, plot_title=None, dummy_test=False):    
    try:
        w1, _ = find_nearest(x0, fit_boundary[0])
        w2, _ = find_nearest(x0, fit_boundary[1])
        w3, _ = find_nearest(x0, fit_boundary[2])
    except IndexError:
        w1, _ = find_nearest(x0, 340)
        w2, _ = find_nearest(x0, 400)
        w3, _ = find_nearest(x0, 800)
    
    if dummy_test:
        x = x0[w1:w2]
        y = y0[w1:w2]
        x_low_bnd = x0[w1]
    else:         
        x = x0[w2:w3]
        y = y0[w2:w3]
        x_low_bnd = x0[w2]
    
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(abs(y) * (x - mean) ** 2) / sum(y))
    
    
    try:
        initial_guess = [y0[peak[-1]], x0[peak[-1]], sigma]
    except (TypeError, IndexError):
        initial_guess = [max(y), mean, sigma]
    

    try:
        bnd = ((0, x_low_bnd, 0),(y.max()*1.15, 1000, np.inf))
        popt, pcov = curve_fit(fit_function, x, y, p0=initial_guess, bounds=bnd, maxfev=maxfev)
    except (RuntimeError, ValueError):
        bnd = (-np.inf, np.inf)
        maxfev=1000000
        popt, pcov = curve_fit(fit_function, x, y, p0=initial_guess, bounds=bnd, maxfev=maxfev)
    
    # A = popt[0]
    # x0 = popt[1]
    # sigma = popt[2]

    
    if plot == True:
        fitted_result = fit_function(x, *popt)
        r_2 = r_square(x, y, fitted_result)
        r2 = f'R\u00b2={r_2:.2f}'
        plt.figure()
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,fitted_result,'r--',label='Total fit\n'+r2)
        plt.legend()
        plt.title(f'{fit_function.__name__} : {plot_title}')
        plt.show()
    else: pass
    
    if raw_data:
        return popt, pcov, x, y
    else:
        return popt, pcov
    



def _2peak_fit_good_PL(x0, y0, fit_function, peak=False, maxfev=100000, fit_boundary=[340, 400, 800], raw_data=False, 
                       second_peak=None, plot=False, plot_title=None):    
    try:
        w1, _ = find_nearest(x0, fit_boundary[0])
        w2, _ = find_nearest(x0, fit_boundary[1])
        w3, _ = find_nearest(x0, fit_boundary[2])
    except IndexError:
        w1, _ = find_nearest(x0, 340)
        w2, _ = find_nearest(x0, 400)
        w3, _ = find_nearest(x0, 800)
    
    
    x = x0[w2:w3]
    y = y0[w2:w3]
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(abs(y) * (x - mean) ** 2) / sum(y))
    
    
    try:
        initial_guess = [y.max(), x[y.argmax()], sigma, y[find_nearest(x, second_peak)[0]], second_peak, sigma]
    except (TypeError, IndexError):
        initial_guess = [y0[peak[0]], x0[peak[0]], sigma, y0[peak[-1]], x0[peak[-1]], sigma]


    try:
        bnd = ((0,200,0,0,200,0),(y.max()*1.15,1000, np.inf, y.max()*1.15,1000, np.inf))
        popt, pcov = curve_fit(fit_function, x, y, p0=initial_guess, bounds=bnd, maxfev=maxfev)
    except (RuntimeError, ValueError):
        bnd = (-np.inf, np.inf)
        maxfev=1000000
        popt, pcov = curve_fit(fit_function, x, y, p0=initial_guess, bounds=bnd, maxfev=maxfev)
   
    # print(popt, len(popt))
    # A = popt[0]
    # x0 = popt[1]
    # sigma = popt[2]
   
    if plot == True:
        fitted_result = fit_function(x, *popt)
        r_2 = r_square(x, y, fitted_result)
        r2 = f'R\u00b2={r_2:.2f}'
        plt.figure()
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,fitted_result,'r--',label='Total fit\n'+r2)
        
        if 'gauss' in fit_function.__name__:
            f1 = _1gauss
        else:
            f1 = _1Lorentz
        
        pars_1 = popt[0:3]
        pars_2 = popt[3:6]
        peak_1 = f1(x, *pars_1)
        peak_2 = f1(x, *pars_2)
        
        # peak 1
        plt.plot(x, peak_1, "g", label='peak 1')
        plt.fill_between(x, peak_1.min(), peak_1, facecolor="green", alpha=0.5)
  
        # peak 2
        plt.plot(x, peak_2, "y", label='peak 2')
        plt.fill_between(x, peak_2.min(), peak_2, facecolor="yellow", alpha=0.5)  
        
        plt.legend()
        plt.title(f'{fit_function.__name__} : {plot_title}')
        plt.show()
    else: pass
    
    if raw_data:
        return popt, pcov, x, y
    else:
        return popt, pcov




def _identify_one_in_kafka(qepro_dic, metadata_dic, key_height=200, distance=100, height=50, dummy_test=False):
    t0 = de._readable_time(metadata_dic['time'])
    data_id = f'{t0[0]}_{t0[1]}_{metadata_dic["uid"][:8]}'
    x0 = qepro_dic['QEPro_x_axis'][0]
    y0 = qepro_dic['QEPro_output'][0]
    peak, prop = good_bad_data(x0, y0, key_height=key_height, data_id = f'{data_id}', distance=distance, height=height, dummy_test=dummy_test)
    return x0, y0, data_id, peak, prop




def _identify_multi_in_kafka(qepro_dic, metadata_dic, key_height=200, distance=100, height=50, dummy_test=False, w_range=[400, 800], percent_range=[30, 100]):
    t0 = de._readable_time(metadata_dic['time'])
    data_id = f'{t0[0]}_{t0[1]}_{metadata_dic["uid"][:8]}'
    _for_average = pd.DataFrame()
    for i in range(1, qepro_dic['QEPro_spectrum_type'].shape[0]):
        x_i = qepro_dic['QEPro_x_axis'][i]
        y_i = qepro_dic['QEPro_output'][i]
        p1, p2 = good_bad_data(x_i, y_i, key_height=key_height, data_id = f'{data_id}_{i:03d}', distance=distance, height=height, dummy_test=dummy_test)
        if (type(p1) is np.ndarray) and (type(p2) is dict):
            _for_average[f'{data_id}_{i:03d}'] = y_i
    
    
    ## Add try/except Error for pass bad PL to save df in sandbox on 2025/06/06
    try:
        _for_average[f'{data_id}_mean'] = _for_average.mean(axis=1)

        x0 = x_i
        PL_per = percentile_PL(x0, _for_average.to_numpy().T, w_range=w_range, percent_range=percent_range)
        # y0 = _for_average[f'{data_id}_mean'].values
        y0 = PL_per.mean(axis=0)
        
        peak, prop = good_bad_data(x0, y0, key_height=key_height, data_id = f'{data_id}_average', distance=distance, height=height, dummy_test=dummy_test)                            
        return x0, y0, data_id, peak, prop
    
    except ValueError:
        x0 = x_i
        y0 = y_i
        return x0, y0, data_id, [], []
    
    
def _fitting_in_kafka(x0, y0, data_id, peak, prop, is_one_peak=True, dummy_test=False):
    print(f'\n** Average of {data_id} has peaks at {peak}**\n')
    
    print(f'\n** start to do peak fitting by Gaussian**\n')
    
    if is_one_peak:
        f = _1gauss
        M = max(prop['peak_heights'])
        M_idx, _ = find_nearest(prop['peak_heights'], M)
        peak = np.asarray([peak[M_idx]])
        popt, _, x, y = _1peak_fit_good_PL(x0, y0, f, peak=peak, raw_data=True, dummy_test=dummy_test)

    else:    
        if len(peak) == 1:
            f = _1gauss
            popt, _, x, y = _1peak_fit_good_PL(x0, y0, f, peak=peak, raw_data=True, dummy_test=dummy_test)
        elif len(peak) == 2:
            try:
                f = _2gauss
                popt, _, x, y = _2peak_fit_good_PL(x0, y0, f, peak=peak, raw_data=True)
            except RuntimeError:
                f = _1gauss
                M = max(prop['peak_heights'])
                M_idx, _ = find_nearest(prop['peak_heights'], M)
                peak = np.asarray([peak[M_idx]])
                popt, _, x, y = _1peak_fit_good_PL(x0, y0, f, peak=peak, raw_data=True, dummy_test=dummy_test)
        else:
            f = _1gauss
            M = max(prop['peak_heights'])
            M_idx, _ = find_nearest(prop['peak_heights'], M)
            peak = np.asarray([peak[M_idx]])
            popt, _, x, y = _1peak_fit_good_PL(x0, y0, f, peak=peak, raw_data=True, dummy_test=dummy_test)

    shift, _ = find_nearest(x0, x[0])

    return x, y, peak-shift, f, popt



### Calculate photoluminescence quantum yield (plqy) ###
## reference 1: Fluorescein, SI: https://onlinelibrary.wiley.com/doi/full/10.1002/adfm.201900712
## reference 2: Quinine, SI: https://pubs.rsc.org/en/content/articlelanding/2020/re/d0re00129e
'''
https://github.com/cheng-hung/Data_process/blob/main/20230726_CsPbBr_ZnI/20230726_CsPb_ZnI_PL.ipynb
abs_365 = np.asarray([0.45788178937234225, 0.906788585562671, 1.3468533683956367, 1.8042517715092394, 2.0145695678124844])
abs_365_r = 0.376390
plqy_r = 0.546
ref_idx_toluene = 1.506
ref_idx_H2SO4 = 1.337
integral_r = 468573.0
integral_pqds = np.asarray(simpson_int)
plqy = plqy_r*abs_365_r*(ref_idx_toluene**2)*integral_pqds / (integral_r*(ref_idx_H2SO4**2)*abs_365)
'''

def plqy_fluorescein(absorbance_sample, PL_integral_sample, refractive_index_solvent, 
                     absorbance_reference, PL_integral_reference, refractive_index_reference, plqy_reference):
    
    integral_ratio = PL_integral_sample / PL_integral_reference
    absorbance_ratio = (1-10**(np.negative(absorbance_reference))) / (1-10**(np.negative(absorbance_sample)))
    refractive_index_ratio = (refractive_index_solvent / refractive_index_reference)**2

    plqy = plqy_reference * integral_ratio * absorbance_ratio * refractive_index_ratio
    return plqy
    

def plqy_quinine(absorbance_sample, PL_integral_sample, refractive_index_solvent, 
                     absorbance_reference, PL_integral_reference, refractive_index_reference, plqy_reference):
    
    integral_ratio = PL_integral_sample / PL_integral_reference
    absorbance_ratio = absorbance_reference / absorbance_sample
    refractive_index_ratio = (refractive_index_solvent / refractive_index_reference)**2

    plqy = plqy_reference * integral_ratio * absorbance_ratio * refractive_index_ratio
    return plqy



### Functions doing line fitting for baseline correction / offset of absorption spectra
def line_2D(x, slope, y_intercept):
    y = x*slope + y_intercept
    return y

def fit_line_2D(x, y, fit_function, x_range=[600, 900], maxfev=10000, plot=False):
    x = np.asarray(x)
    y = np.asarray(y)
    y = np.nan_to_num(y, nan=0)
    
    try:        
        idx0, _ = find_nearest(x, x_range[0])
        idx1, _ = find_nearest(x, x_range[1])
    except (TypeError, IndexError):
        idx0 = 0
        idx1 = -1
    
    slope = (y[idx1]-y[idx0]) / (x[idx1]-x[idx0])
    y_intercept = np.mean(y[idx0:idx1])
    
    try:
        initial_guess = [slope, y_intercept]
    except (TypeError, IndexError):
        initial_guess = [0.01, 0]
    
    try:
        popt, pcov = curve_fit(fit_function, x[idx0:idx1], y[idx0:idx1], p0=initial_guess, maxfev=maxfev)
    except RuntimeError:
        maxfev=1000000
        popt, pcov = curve_fit(fit_function, x[idx0:idx1], y[idx0:idx1], p0=initial_guess, maxfev=maxfev)
        
    if plot:
        plt.figure()
        plt.plot(x, y, label='data')
        plt.plot(x, fit_function(x, popt[0], popt[1]), label=f'y={popt[0]:.4f}x+{popt[1]:.4f}')
        plt.legend()
    
    return popt, pcov




# def l_unit_converter(l0 = 'm', l1 = 'm'):
#     l_unit = ['mm', 'cm', 'm']
#     l_frame = pd.DataFrame(data={'mm': np.array([1, 10, 1000]), 
#                                  'cm': np.array([0.1, 1, 100]), 
#                                  'm' : np.array([0.001, 0.01, 1])}, index=l_unit)
#     return l_frame.loc[l0, l1]




def cal_equili_from_list(rate_list, mixer, rate_unit = 'ul/min', ratio=1, tubing_ID_mm=1.016):

    if type(mixer) != list:
        raise TypeError('Type of mixer must be a list.')

    total_rate = 0
    for i in range(len(rate_list)):
        rate = rate_list[i]
        unit = rate_unit
        unit_const = vol_unit_converter(v0=unit[:2], v1='ul')/t_unit_converter(t0=unit[3:], t1='min')
        total_rate += rate*unit_const

    l = []
    for i in mixer:
        ll = float(i.split(' ')[0])
        l.append(ll)
    mixer_length = sum(l)
    
    mixer_unit = mixer[-1].split(' ')[1]
    l_unit_const = l_unit_converter(l0=mixer_unit, l1='m')
    mixer_meter = mixer_length * l_unit_const
    mixer_vol_mm3 = np.pi*((tubing_ID_mm/2)**2)*mixer_meter*1000
    res_time_sec = 60*mixer_vol_mm3/total_rate
    
    print(f'Reaction resident time is {res_time_sec:.2f} seconds.')
    print(f'Wait for {ratio} times of resident time, in total of {res_time_sec*ratio:.2f} seconds.')
    # yield from bps.sleep(res_time_sec*ratio)



######## Old versions of function #########    

def _2peak_fit_PL3(x, y, distr='G', distance=30, height=930, plot=False, plot_title=None, second_peak=None, maxfev=100000):
    # 'G': Guassian
    # 'L': Lorentz  
    peak, _ = find_peaks(y, distance=distance, height=height)
    # peaks=[peak[0]]
    # for i in range(1, len(peak)):
    #     if peak[i]-peak[i-1]>20:
    #         peaks.append(peak[i])
        
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    
    if len(peaks) == 2:
        if distr == 'G':
            popt, pcov = curve_fit(_2gauss, x, y, p0=[y[peaks[0]], x[peaks[0]], sigma, y[peaks[-1]], x[peaks[-1]], sigma], maxfev=maxfev)
        else:
            popt, pcov = curve_fit(_2Lorentz, x, y, p0=[y[peaks[0]], x[peaks[0]], sigma, y[peaks[-1]], x[peaks[-1]], sigma], maxfev=maxfev)
    else:
        if abs(second_peak)<20:
            if distr == 'G' and abs(second_peak)<20:
                popt, pcov = curve_fit(_2gauss, x, y, p0=[y[peaks[0]], x[peaks[0]], sigma, y[peaks[0]]/abs(second_peak), x[peaks[0]]-second_peak*sigma, sigma], maxfev=maxfev)
            else:
                popt, pcov = curve_fit(_2Lorentz, x, y, p0=[y[peaks[0]], x[peaks[0]], sigma, y[peaks[0]]/abs(second_peak*sigma), x[peaks[0]]-second_peak*sigma, sigma], maxfev=maxfev)
            
        else:
            if distr == 'G' and abs(second_peak)>=20:
                popt, pcov = curve_fit(_2gauss, x, y, p0=[y[peaks[0]], x[peaks[0]], sigma, y[find_nearest(x, second_peak)[0]], second_peak, sigma], maxfev=maxfev)
            else:
                popt, pcov = curve_fit(_2Lorentz, x, y, p0=[y[peaks[0]], x[peaks[0]], sigma, y[peaks[0]]/abs(second_peak*sigma), x[peaks[0]]-second_peak*sigma, sigma], maxfev=maxfev)
    #A = popt[0]
    #x0 = popt[1]
    #sigma = popt[2]
    
    pars_1 = popt[0:3]
    pars_2 = popt[3:6]
    
    if distr == 'G':
        peak_1 = _1gauss(x, *pars_1)
        peak_2 = _1gauss(x, *pars_2)
        fit_model = 'Gaussian'
    else:
        peak_1 = _1Lorentz(x, *pars_1)
        peak_2 = _1Lorentz(x, *pars_2)
        fit_model = 'Lorentz'
    
    fitted_result = _2gauss(x, *popt)
    residulas = y - fitted_result
    ss_res = np.sum(residulas**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_2 = 1 - (ss_res / ss_tot)
    r2 = f'R\u00b2={r_2:.2f}'
    
    if plot == True:
        plt.figure()
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,fitted_result,'ro:',label='Total fit\n'+r2)
        
        # peak 1
        plt.plot(x, peak_1, "g", label='peak 1')
        plt.fill_between(x, peak_1.min(), peak_1, facecolor="green", alpha=0.5)
  
        # peak 2
        plt.plot(x, peak_2, "y", label='peak 2')
        plt.fill_between(x, peak_2.min(), peak_2, facecolor="yellow", alpha=0.5)  
        
        plt.title(f'{fit_model} : {plot_title}')
        plt.legend()
        plt.show()
    else: pass
    
    return popt, r_2



def _3peak_fit_PL2(x, y, distr='G', height=930, plot=False, plot_title=None, second_peak=None, third_peak=None,maxfev=100000):
    # 'G': Guassian
    # 'L': Lorentz  
    peak, _ = find_peaks(y, height=height)
    peaks=[peak[0]]
    for i in range(1, len(peak)):
        if peak[i]-peak[i-1]>20:
            peaks.append(peak[i])
    
    if len(peaks) >3:
        raise IndexError('Number of peaks should be less than 2.')
        
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

    
    if len(peaks) == 3:
        if distr == 'G':
            popt, pcov = curve_fit(_3gauss, x, y, 
                                   p0=[y[peaks[0]], x[peaks[0]], sigma, y[peaks[1]], x[peaks[1]], sigma, y[peaks[-1]], x[peaks[-1]], sigma], 
                                   maxfev=maxfev)
        else:
            popt, pcov = curve_fit(_3Lorentz, x, y, 
                                   p0=[y[peaks[0]], x[peaks[0]], sigma, y[peaks[1]], x[peaks[1]], sigma, y[peaks[-1]], x[peaks[-1]], sigma], 
                                   maxfev=maxfev)
    else:            
        if distr == 'G':
            popt, pcov = curve_fit(_3gauss, x, y, 
                                   p0=[y[peaks[0]], x[peaks[0]], sigma, y[find_nearest(x, second_peak)[0]], second_peak, sigma, y[find_nearest(x, third_peak)[0]], third_peak, sigma], 
                                   maxfev=maxfev)
        else:
            popt, pcov = curve_fit(_3Lorentz, x, y, 
                                   p0=[y[peaks[0]], x[peaks[0]], sigma, y[find_nearest(x, second_peak)[0]], second_peak, sigma, y[find_nearest(x, third_peak)[0]], third_peak, sigma], 
                                   maxfev=maxfev)
    #A = popt[0]
    #x0 = popt[1]
    #sigma = popt[2]
    
    pars_1 = popt[0:3]
    pars_2 = popt[3:6]
    pars_3 = popt[6:9]
    
    if distr == 'G':
        peak_1 = _1gauss(x, *pars_1)
        peak_2 = _1gauss(x, *pars_2)
        peak_3 = _1gauss(x, *pars_3)
        fit_model = 'Gaussian'
    else:
        peak_1 = _1Lorentz(x, *pars_1)
        peak_2 = _1Lorentz(x, *pars_2)
        peak_3 = _1gauss(x, *pars_3)
        fit_model = 'Lorentz'
    
    fitted_result = _3gauss(x, *popt)
    residulas = y - fitted_result
    ss_res = np.sum(residulas**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_2 = 1 - (ss_res / ss_tot)
    r2 = f'R\u00b2={r_2:.2f}'
    
    if plot == True:
        plt.figure()
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,fitted_result,'ro:',label='Total fit\n'+r2)
        
        # peak 1
        plt.plot(x, peak_1, "g", label='peak 1')
        plt.fill_between(x, peak_1.min(), peak_1, facecolor="green", alpha=0.5)
  
        # peak 2
        plt.plot(x, peak_2, "y", label='peak 2')
        plt.fill_between(x, peak_2.min(), peak_2, facecolor="yellow", alpha=0.5)
        
        # peak 3
        plt.plot(x, peak_3, "b", label='peak 3')
        plt.fill_between(x, peak_3.min(), peak_3, facecolor="blue", alpha=0.5)
        
        plt.title(f'{fit_model} : {plot_title}')
        plt.legend()
        plt.show()
    else: pass
    
    return popt, r_2




def _1peak_fit_PL(x, y, distr='G', plot=False, plot_title=None, maxfev=100000):    
    # 'G': Guassian
    # 'L': Lorentz        
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    if distr == 'G':
        popt, pcov = curve_fit(_1gauss, x, y, p0=[max(y), mean, sigma], maxfev=maxfev)
    else:
        popt, pcov = curve_fit(_1Lorentz, x, y, p0=[max(y), mean, sigma], maxfev=maxfev)
    
    A = popt[0]
    x0 = popt[1]
    sigma = popt[2]
    
    if distr == 'G':
        fitted_result = _1gauss(x, *popt)
        fit_model = 'Gaussian'
    else:
        fitted_result = _1Lorentz(x, *popt)
        fit_model = 'Lorentz'
    
    #fitted_result = _1gauss(x, *popt)
    residulas = y - fitted_result
    ss_res = np.sum(residulas**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_2 = 1 - (ss_res / ss_tot)
    r2 = f'R\u00b2={r_2:.2f}'
    
    if plot == True:
        plt.figure()
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,fitted_result,'ro:',label='Total fit\n'+r2)
        plt.legend()
        plt.title(f'{fit_model} : {plot_title}')
        plt.show()
    else: pass
    
    return popt, r_2


if __name__ == "__main__":
    pass