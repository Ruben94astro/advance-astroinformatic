
import math
import os
import glob
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import gc
from astropy.timeseries import LombScargle
from scipy import stats
from statsmodels.tsa import stattools
from scipy.interpolate import interp1d
from scipy.stats import chi2

def remove_first_data_row(csv_path="all_features.csv", output_path=None):
    """
    function that takse a csv file and delete the first row, if not exits pass
    """
    # Leer el CSV completo (encabezados incluidos)
    if not os.path.isfile(csv_path):
        print(f"file '{csv_path}' does not exist")
    return
    df = pd.read_csv(csv_path)
    
    # delete the first row
    df = df.iloc[1:]
    
    # save
    if output_path is None:
        output_path = csv_path
    df.to_csv(output_path, index=False)

def ls_periodogram(time_no_outliers, flux_no_outliers, ferr_no_outliers):
    minP, maxP = 0.1, 50.0
    minF, maxF = 1/maxP, 1/minP
    
    # --- Lomb–Scargle ---
    Nf = 20000
    freq = np.linspace(minF, maxF, Nf)
    ls = LombScargle(time_no_outliers, flux_no_outliers, ferr_no_outliers)
    power = ls.power(freq)
    
    best_idx = np.argmax(power)
    best_freq = freq[best_idx]
    best_period = 1.0 / best_freq
    

    # --- Significance ---
    fap = ls.false_alarm_probability(power[best_idx])
 
    # Calculate FAP levels
    fap_levels = [0.01, 0.05, 0.1]  # 1%, 5%, 10%
    power_levels = [ls.false_alarm_level(fap_level) for fap_level in fap_levels]
    return best_period, fap  




def features(name_lc):
    """
function that takes lc and gives, magnitude, mgnitud error,
time, name, type
    """
    # reading data
    lc = pd.read_csv(name_lc, sep=',', names=['JD', 'mag', 'err'])
    # nan 
    lc = lc.apply(pd.to_numeric, errors='coerce').dropna()
    if lc.empty:
        #gives a warning about the data
        print(f"warning: {name_lc} no valid data")
        return
    folder_name = os.path.basename(os.path.dirname(name_lc))# ------>takes the type of the star e.g AVC, ROT, RR-L    
    base = os.path.basename(name_lc).replace('.lc', '')#------>take the base name 
    without_ext = os.path.splitext(base)[0]
    parts = without_ext.split('_')
    name_obj = parts[0]

    

    # calculate features 
    mag = lc['mag'].values
    err = lc['err'].values
    time = lc['JD'].values

    return time, mag, err, name_obj, folder_name


def compute_all_features(name, star_type, time, mag, magerr, period, fap):
    ''' funtcion that print features of a single lc and add to all_features.csv
    '''
    features_dict = {
        'id_star': name,
        'type': star_type,
        'Period': period,      # Si tienes periodograma, llámalo aquí
        'FAP': fap,
        'Amplitud': Amplitude(mag),
        'Rcs': Rcs(mag),
        'Stetsonk': StetsonK(mag, magerr),
        'Mean_variance': Meanvariance(mag),
        'Autocor_length': Autocor_length(mag),
        'Con': Con(mag),
        'Beyond1Std': Beyond1Std(mag, magerr),
        'SmallKurtosis': SmallKurtosis(mag),
        'Std': Std(mag),
        'Skew': Skew(mag),
        'MaxSlope': MaxSlope(mag, time),
        'MedianAbsDev': MedianAbsDev(mag),
        'MedianBRP': MedianBRP(mag),
        'PairSlopeTrend': PairSlopeTrend(mag),
        'FluxPercentileRatioMid20': FluxPercentileRatioMid20(mag),
        'FluxPercentileRatioMid35': FluxPercentileRatioMid35(mag),
        'FluxPercentileRatioMid50': FluxPercentileRatioMid50(mag),
        'FluxPercentileRatioMid65': FluxPercentileRatioMid65(mag),
        'FluxPercentileRatioMid80': FluxPercentileRatioMid80(mag),
        'PercentDifferenceFluxPercentile': PercentDifferenceFluxPercentile(mag),
        'PercentAmplitude': PercentAmplitude(mag),
        'LinearTrend': LinearTrend(mag, time),
        'Eta_e': Eta_e(mag, time),
        'Mean': Mean(mag),
        'Q31': Q31(mag),
        'AndersonDarling': AndersonDarling(mag),
        'Gskew': Gskew(mag),
        'StructureFunction_index_21': StructureFunction_index_21(mag, time),
        'StructureFunction_index_31': StructureFunction_index_31(mag, time),
        'StructureFunction_index_32': StructureFunction_index_32(mag, time),
        'Pvar': Pvar(mag, magerr),
        'ExcessVar': ExcessVar(mag, magerr)}
    
    return features_dict


    
def append_features_to_csv(features_dict, csv_path="all_features.csv"):
    """
    function that put information to the csv file
   """
    # check if the file exist
    if not os.path.isfile(csv_path):
        print(f"warning '{csv_path}' does not exist.")
        return
    
    # create a df with one row
    df_new = pd.DataFrame([features_dict])
    
    # add to csv
    df_new.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"Data updated '{csv_path}'")




def Amplitude(mag):
    
        n = len(mag)
        sorted_mag = np.sort(mag)

        return (np.median(sorted_mag[int(-math.ceil(0.05 * n)):]) -
                np.median(sorted_mag[0:int(math.ceil(0.05 * n))])) / 2.0


def Rcs(mag):
        """Range of cumulative sum"""

        sigma = np.std(mag)
        N = len(mag)
        m = np.mean(mag)
        s = np.cumsum(mag - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return R


def StetsonK(mag,magerr):
        n = len(mag)

        mean_mag = (np.sum(mag/(magerr*magerr))/np.sum(1.0 / (magerr * magerr)))
        sigmap = (np.sqrt(n * 1.0 / (n - 1)) * (mag - mean_mag) / magerr)

        k = (1 / np.sqrt(n * 1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))
        return k


def Meanvariance(mag):
  
        return np.std(mag) / np.mean(mag)


def Autocor_length(mag):
    
        nlags=100
        ac = stattools.acf(mag, nlags=nlags, fft=False)

        k = next((index for index, value in
                enumerate(ac) if value < np.exp(-1)), None)

        while k is None:
                if nlags > len(mag):
                        warnings.warn('Setting autocorrelation length as light curve length')
                return len(mag)
                nlags = nlags + 100
                ac = stattools.acf(mag, nlags=nlags, fft=False)
                k = next((index for index, value in enumerate(ac) if value < np.exp(-1)), None)

        return k



def Con(mag):
        """Index introduced for selection of variable starts from OGLE database.
        To calculate Con, we counted the number of three consecutive measurements
        that are out of 2sigma range, and normalized by N-2
        """
     
        consecutiveStar=3
        N = len(mag)
        if N < consecutiveStar:
                return 0
        sigma = np.std(mag)
        m = np.mean(mag)
        count = 0

        for i in range(N - consecutiveStar + 1):
                flag = 0
                for j in range(consecutiveStar):
                        if (mag[i + j] > m + 2 * sigma or mag[i + j] < m - 2 * sigma):
                                flag = 1
                        else:
                                flag = 0
                        break
                        if flag:
                                count = count + 1
        return count * 1.0 / (N - consecutiveStar + 1)



def Beyond1Std(mag,magerr):
        """Percentage of points beyond one st. dev. from the weighted
        (by photometric errors) mean
        """

        n = len(mag)

        weighted_mean = np.average(mag, weights=1 / magerr ** 2)

        # Standard deviation with respect to the weighted mean
        var = sum((mag - weighted_mean) ** 2)
        std = np.sqrt((1.0 / (n - 1)) * var)

        count = np.sum(np.logical_or(mag > weighted_mean + std,
                                     mag < weighted_mean - std))
        return float(count) / n


def SmallKurtosis(mag):
        """Small sample kurtosis of the magnitudes.

        See http://www.xycoon.com/peakedness_small_sample_test_1.htm
        """
    
        n = len(mag)
        mean = np.mean(mag)
        std = np.std(mag)

        S = sum(((mag - mean) / std) ** 4)

        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return c1 * S - c2


def Std(mag):
        """Standard deviation of the magnitudes"""

        return np.std(mag)


def Skew(mag):
        """Skewness of the magnitudes"""
        return stats.skew(mag)


def MaxSlope(mag,time):
        """
        Examining successive (time-sorted) magnitudes, the maximal first difference
        (value of delta magnitude over delta time)
        """
  
        slope = np.abs(mag[1:] - mag[:-1]) / (time[1:] - time[:-1])
        np.max(slope)

        return np.max(slope)


def MedianAbsDev(mag):
   
        median = np.median(mag)

        devs = (abs(mag - median))

        return np.median(devs)


def MedianBRP(mag):
        """Median buffer range percentage
        Fraction (<= 1) of photometric points within amplitude/10
        of the median magnitude
        """

        median = np.median(mag)
        amplitude = (np.max(mag) - np.min(mag)) / 10
        n = len(mag)

        count = np.sum(np.logical_and(mag < median + amplitude, mag > median - amplitude))

        return float(count) / n


def PairSlopeTrend(mag):
        """
        Considering the last 30 (time-sorted) measurements of source magnitude,
        the fraction of increasing first differences minus the fraction of
        decreasing first differences.
        """
  
        data_last = mag[-30:]

        return (float(len(np.where(np.diff(data_last) > 0)[0]) -
                      len(np.where(np.diff(data_last) <= 0)[0])) / 30)


def FluxPercentileRatioMid20(mag):
  
        sorted_data = np.sort(mag)
        lc_length = len(sorted_data)

        F_60_index = math.ceil(0.60 * lc_length)
        F_40_index = math.ceil(0.40 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid20 = F_40_60 / F_5_95

        return F_mid20


def FluxPercentileRatioMid35(mag):
   
        sorted_data = np.sort(mag)
        lc_length = len(sorted_data)

        F_325_index = math.ceil(0.325 * lc_length)
        F_675_index = math.ceil(0.675 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid35 = F_325_675 / F_5_95

        return F_mid35


def FluxPercentileRatioMid50(mag):
   
        sorted_data = np.sort(mag)
        lc_length = len(sorted_data)

        F_25_index = math.ceil(0.25 * lc_length)
        F_75_index = math.ceil(0.75 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid50 = F_25_75 / F_5_95

        return F_mid50


def FluxPercentileRatioMid65(mag):
 
        sorted_data = np.sort(mag)
        lc_length = len(sorted_data)

        F_175_index = math.ceil(0.175 * lc_length)
        F_825_index = math.ceil(0.825 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid65 = F_175_825 / F_5_95

        return F_mid65


def FluxPercentileRatioMid80(mag):
  
        sorted_data = np.sort(mag)
        lc_length = len(sorted_data)

        F_10_index = math.ceil(0.10 * lc_length)
        F_90_index = math.ceil(0.90 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.floor(0.95 * lc_length)

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid80 = F_10_90 / F_5_95

        return F_mid80


def PercentDifferenceFluxPercentile(mag):
   
        median_data = np.median(mag)

        sorted_data = np.sort(mag)
        lc_length = len(sorted_data)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

        percent_difference = F_5_95 / median_data

        return percent_difference


def PercentAmplitude(mag):
   
        median_data = np.median(mag)
        distance_median = np.abs(mag - median_data)
        max_distance = np.max(distance_median)

        percent_amplitude = max_distance / median_data

        return percent_amplitude


def LinearTrend(mag,time):
 
        regression_slope = stats.linregress(time, mag)[0]

        return regression_slope


def Eta_e(mag,time):
    
        w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
        sigma2 = np.var(mag)

        S1 = np.sum(w * np.power(np.subtract(mag[1:], mag[:-1]), 2))
        S2 = np.sum(w)

        eta_e = (1 / sigma2) * (S1 / S2)
        return eta_e


def Mean(mag):
  
        B_mean = np.mean(mag)
        return B_mean


def Q31(mag):
   
        percentiles = np.percentile(mag, (25, 75))
        return percentiles[1] - percentiles[0]




def AndersonDarling(mag):
   
        ander = stats.anderson(mag)[0]
        return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))




def Gskew(mag):

        median_mag = np.median(mag)
        F_3_value, F_97_value = np.percentile(mag, (3, 97))

        return (np.median(mag[mag <= F_3_value]) +
                np.median(mag[mag >= F_97_value])
                - 2*median_mag)


def StructureFunction_index_21(mag,time):

        Nsf = 100
        Np = 100
        sf1 = np.zeros(Nsf)
        sf2 = np.zeros(Nsf)
        sf3 = np.zeros(Nsf)
        f = interp1d(time, mag)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
                sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 1.0))
                sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 2.0)))
                sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)

        return m_21


def StructureFunction_index_31(mag,time):

        Nsf = 100
        Np = 100
        sf1 = np.zeros(Nsf)
        sf2 = np.zeros(Nsf)
        sf3 = np.zeros(Nsf)
        f = interp1d(time, mag)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
                sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 1.0))
                sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 2.0)))
                sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)

        return m_31


def StructureFunction_index_32(mag,time):


        Nsf = 100
        Np = 100
        sf1 = np.zeros(Nsf)
        sf2 = np.zeros(Nsf)
        sf3 = np.zeros(Nsf)
        f = interp1d(time, mag)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
                sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 1.0))
                sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 2.0)))
                sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]), 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)

        return m_32
    
    

def Pvar(mag,magerr):
        """
        Calculate the probability of a light curve to be variable.
        """
    
        mean_mag = np.mean(mag)
        nepochs = float(len(mag))

        chi = np.sum((mag - mean_mag)**2. / magerr**2.)
        p_chi = chi2.cdf(chi, (nepochs-1))

        return p_chi


def ExcessVar(mag,magerr):
        """
        Calculate the excess variance,which is a measure of the intrinsic variability amplitude.
        """
   

        mean_mag = np.mean(mag)
        nepochs = float(len(mag))

        a = (mag-mean_mag)**2
        ex_var = (np.sum(a-magerr**2) / (nepochs * (mean_mag ** 2)))

        return ex_var

import os

def process_all_folder(root_folder, csv_path="all_features.csv", suffix=""):
    """
    Recursively walks through root_folder, finds .lc files (optionally containing 'suffix'),
    extracts time, magnitude, error, computes the Lomb–Scargle periodogram,
    calculates all features, and appends them to a CSV file.
    
    Parameters:
        root_folder (str): directory to search for .lc files
        csv_path (str): output CSV file path
        suffix (str): optional substring to filter file names (e.g., '_cleaned')
    """


    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if not file.endswith('.lc'):
                continue
            if suffix and suffix not in file:
                continue

            lc_path = os.path.join(root, file)
            print(f"Processing: {lc_path}")

            # 1. Extract basic series (time, mag, err, object name, base name)
            result = features(lc_path)   # should return (time, mag, magerr, name_obj, base)
            if result is None:
                continue
            time, mag, magerr, name_obj, base = result

            # 2. Compute Lomb–Scargle periodogram
            period, fap = ls_periodogram(time, mag, magerr)

            # 3. Compute all features (including period and FAP)
            features_dict = compute_all_features(name_obj, base, time, mag, magerr, period, fap)

            # 4. Append to CSV
            append_features_to_csv(features_dict, csv_path)

    print(f"Processing completed. CSV updated: {csv_path}")








#example with one file
if __name__ == "__main__":
        # creating a dictionary to put it on the header 

    features_dict = {
        'id_star': '',
        'type': '',
        'Period': '',
        'FAP': '',
        'Amplitud': '',
        'Rcs': '',
        'Stetsonk': '',
        'Mean_variance': '',
        'Autocor_length': '',
        'Con': '',
        'Beyond1Std': '',
        'SmallKurtosis': '',
        'Std': '',
        'Skew': '',
        'MaxSlope': '',
        'MedianAbsDev': '',
        'MedianBRP': '',
        'PairSlopeTrend': '',
        'FluxPercentileRatioMid20': '',
        'FluxPercentileRatioMid35': '',
        'FluxPercentileRatioMid50': '',
        'FluxPercentileRatioMid65': '',
        'FluxPercentileRatioMid80': '',
        'PercentDifferenceFluxPercentile': '',
        'PercentAmplitude': '',
        'LinearTrend': '',
        'Eta_e': '',
        'Mean': '',
        'Q31': '',
        'AndersonDarling': '',
        'Gskew': '',
        'StructureFunction_index_21': '',
        'StructureFunction_index_31': '',
        'StructureFunction_index_32': '',
        'Pvar': '',
        'ExcessVar': ''
    }
    
    # creates a data frame
    df_empty = pd.DataFrame([features_dict])
    # delete empty row
    df_empty = df_empty.iloc[0:0]
    
    if not os.path.isfile("all_features.csv"):
        df_empty.to_csv("all_features.csv", index=False)
        print("file created")
    else:
        print("file already exist.")

    process_all_folder("_TESS_lightcurves_outliercleaned", csv_path="all_features.csv", suffix="")  

    # #extractig main values like magnitude, name, period, etc
    # time, mag, magerr, name_obj, base = features("_TESS_lightcurves_outliercleaned/ACV/41259805_sector01_4_2_cleaned.lc")
    # period, fap = ls_periodogram(time, mag, magerr)
    # # #calculate features
    # features_dict = compute_all_features(name_obj, base, time, mag, magerr, period, fap)
    # append_features_to_csv(features_dict, csv_path="all_features.csv")
    
    