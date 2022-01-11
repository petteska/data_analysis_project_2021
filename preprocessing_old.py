import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
import statistics
import copy


from sklearn.linear_model import LinearRegression

from PyEMD import EMD


import sys

sys.path.append('./utils')
# My files
import utils

def trim_signal(signal, sampling_freq, trim_from, trim_to):
    """ Trim signal and keep everything between {trim_from} to {trim_to}
    Args:
        Signal:     (np.array)  - signal to trim
        timestep:   (float)     - timestep between each sample (1/freq)
        trim_from:  (float)     - start time in seconds
        trim_to:    (float)     - End time in seconds
    
    Return:
        trimed signal (np.array)
    """
    
    from_step = int(np.floor(trim_from*sampling_freq))
    to_step = int(np.floor(trim_to*sampling_freq))-1

    return signal[from_step:to_step]


def normalize_wrt_baseline(signal, baseline):
    """ Normalize the signal with respect to baseline
    Args:
        Signal:     (np.array)
        baseline:   (np.array)
    
    Return:
        normalized signal (np.array)
    """
    return (signal - np.mean(baseline))/np.std(baseline)


def min_max_scale(signal, min, max):
    return (signal - min)/(max - min)



def mean(signal):
    """ find mean of signal
    Args:
        Signal:     (np.array)

    Return:
        mean of signal (float)
    """
    return np.mean(signal)


def remove_mean(signal):
    """ Remove the mean from the signal
    Args:
        Signal:     (np.array)

    Return:
        signal without mean (np.array)
    """
    return signal - mean(signal)


def normalize(signal):
    """ Normalize the signal
    Args:
        Signal:     (np.array)
    
    Return:
        normalized signal (np.array)
    """
    return (signal - mean(signal))/np.std(signal)


def savitsky_golay(signal, filter_length):
    """ Filter signal with hard-coded SG-filter
    Args:
        Signal:     (np.array)
    
    Return:
        filtered signal (np.array)
    """
    return savgol_filter(signal, filter_length, 2, mode="nearest")


def psd(signal, N):
    """ Calculate Power Spectrum Denisty of signal (NOT IN USE!)
    Args:
        Signal:     (np.array)
    
    Return:
        normalized signal (np.array)
    """
    result = (signal * np.conj(signal))/N
    return result


def display_psd_fourier(signal, sampling_freq):
    """ Plot the power spectrum density of fourier transformed signal
    Args:
        Signal:     (np.array)
        frequency:  (np.array) - Sampling frequency of the signal
    
    Return:
        
    """
    N = signal.size
    fourier = np.fft.fft(signal,N)
    PSD = (fourier * np.conj(fourier))/N

    x = sampling_freq/N*np.arange(N)
    
    plt.figure()
    plt.plot(x,PSD)
    plt.xlabel("Frequency")
    plt.ylabel("PSD")
    plt.show()


def reduce_signal_fft(signal, psd_limit):
    """ Remove all parts of the signal with PSD lower than psd_limit.
        Might be smart to examine the signal with {display_psd_fourier()} first
    Args:
        Signal:     (np.array)
        psd_limit:  (np.array)
    
    Return:
        filtered signal (np.array)  
    """
    N = signal.size
    fourier = np.fft.fft(signal,N)
    PSD = (fourier * np.conj(fourier))/N

    indices = PSD>psd_limit
    fourier = indices*fourier
    ffilt = np.fft.ifft(fourier)

    return ffilt.real

def lowpass_fft(signal, sampling_freq, freq_lim):
    """ Remove all parts of the signal with frequency higher than freq_lim.
    Args:
        Signal:         (np.array)
        sampling_freq:  (float)     -   Sampling frequency of the signal
        freq_lim:       (float)     -   Limit frequency
    
    Return:
        filtered signal (np.array)  
    """
    N = signal.size
    fourier = np.fft.fft(signal,N)
    x = sampling_freq/N*np.arange(N)

    indices = x<freq_lim
    fourier = indices*fourier
    ffilt = np.fft.ifft(fourier)

    return ffilt.real

def highpass_fft(signal, sampling_freq, frequency_lim):
    """ Remove all parts of the signal with frequency lower than freq_lim.
    Args:
        Signal:         (np.array)
        sampling_freq:  (float)     -   Sampling frequency of the signal
        freq_lim:       (float)     -   Limit frequency
    
    Return:
        filtered signal (np.array)  
    """
    N = signal.size
    fourier = np.fft.fft(signal,N)
    x = sampling_freq/N*np.arange(N)

    indices = x>frequency_lim
    fourier = indices*fourier
    ffilt = np.fft.ifft(fourier)

    return ffilt.real

def bandpass_fft(signal, sampling_freq, lower_lim, upper_lim):
    """ Remove all parts of the signal with frequency lower than lower_lim and higher than upper_lim.
    Args:
        Signal:         (np.array)
        sampling_freq:  (float)     -   Sampling frequency of the signal
        lower_lim:      (float)     -   Lower limit frequency
        upper_lim:      (float)     -   Upper limit frequency
    
    Return:
        filtered signal (np.array)  
    """
    N = signal.size
    fourier = np.fft.fft(signal,N)
    x = sampling_freq/N*np.arange(N)

    indices = np.logical_and(x > lower_lim, x < upper_lim)
    fourier = indices * fourier
    ffilt = np.fft.ifft(fourier)

    return ffilt.real

def bandstopp_fft(signal, sampling_freq, lower_lim, upper_lim):
    """ Remove all parts of the signal with frequency between  lower_lim and upper_lim.
    Args:
        Signal:         (np.array)
        sampling_freq:  (float)     -   Sampling frequency of the signal
        lower_lim:      (float)     -   Lower limit frequency
        upper_lim:      (float)     -   Upper limit frequency
    
    Return:
        filtered signal (np.array)  
    """
    N = signal.size
    fourier = np.fft.fft(signal,N)
    x = sampling_freq/N*np.arange(N)

    indices = np.logical_or(x < lower_lim, x > upper_lim)
    fourier = indices * fourier
    ffilt = np.fft.ifft(fourier)

    return ffilt.real


def remove_trend(time, signal, order):
    """ Remove the slowly varying trend from a signal.
        The trend is found by a polynomial regression of order {order}
    Args:
        time:           (np.array)  -   Timesteps for the signal
        Signal:         (np.array)
        order:          (int)       -   Order of trend to be removed
    
    Return:
        filtered signal (np.array)  
    """
    trend_coef = np.polyfit(time, signal, deg=order)

    trend_signal = np.polyval(trend_coef,time)

    return signal - trend_signal


def plot_emd_imfs(signal, time):
    """ Plot the imfs of a signal obtained by Empirical Module Decomposition
    Args:
        Signal:         (np.array)
        time:           (np.array)  -   Timesteps for the signal
    
    Return:

    """
    emd = EMD()
    imfs = emd(signal)

    fig,axs = plt.subplots(imfs.shape[0], figsize=(10,20))
    for i in range(imfs.shape[0]):
        axs[i].plot(time,imfs[i,:])


def remove_emd_imfs(signal, imfs_to_remove):
    """ Remove the specified imfs from the signal
        Might be smart to examine the imfs with {plot_emd_imfs()} first
    Args:
        Signal:         (np.array)
        imf_to_remove:  (array)      -   array of trends to me removed. Specified as integer values of their index.
    
    Return:
        filtered signal (np.array)  
    """
    emd = EMD()
    imfs = emd(signal)
    for i in imfs_to_remove:
        if i > imfs.shape[0]:
            print("IMF index out of range")
            return
        signal = signal - imfs[i]
    
    return signal

def get_preprocess_pipeline(data_type):
    if data_type.upper() == "ACC":
        return []
    elif data_type.upper() == "BVP":
        return ["CROP", "FILTER", "RM_BASELINE"]
    elif data_type.upper() == "EDA":
        return ["CROP","CONVERT_TO_GSR", "RM_BASELINE"]
    elif data_type.upper() == "HR":
        return ["CROP","RM_BASELINE"]
    elif data_type.upper() == "IBI":
        return []
    elif data_type.upper() == "TEMP":
        return ["CROP", "RM_BASELINE"]
    else:
        return np.NaN

def crop_signal(signal, sampling_freq, crop_settings): # This should replace the trim_signal() function
    """
    Args:
        crop_settings    -   (list)  -   [signal_length, crop_before, crop_after]
    """

    # Crop signal according to crop_settings
    try:
        crop_length = crop_settings[0]
        crop_before = crop_settings[1]
        crop_after = crop_settings[2]
    except:
        print("The crop settings were not entered correctly!")
        return False

    signal_length = utils.get_signal_length(signal, sampling_freq)

    if crop_before and crop_after:
        trim_from = int(np.floor((signal_length - crop_length)/2 * sampling_freq))
        trim_to = int(np.floor(signal_length - (signal_length - crop_length)/2 * sampling_freq))

    elif crop_before:
        trim_from = int(np.floor((signal_length - crop_length) * sampling_freq))
        trim_to = int(signal_length*sampling_freq)
    elif crop_after:
        trim_from = 0
        trim_to = int(np.floor(crop_length*sampling_freq))
    else:
        return False
   
    return signal[trim_from:trim_to]


def filter_signal(signal, signal_type):
    """
    Args:
        filter_settings -   (list)  -   [filter_type]
    """

    # filtered_signal = savgol_filter(signal, window_length= 21, polyorder= 2, delta=1, mode="interp")
    # filtered_signal = lowpass_fft(signal, sampling_freq, sampling_freq*2)
    s = signal.shape

    if signal_type == "BVP":
        window_length = 351
    else:
        window_length = 3
    filtered_signal = medfilt(np.ravel(signal), window_length)
    # filtered_signal = np.ravel(signal)
    return filtered_signal.reshape(s)

def get_min_max(e4_data_object):
    
    min_max = {}
    for data_type, emotion_list in e4_data_object.items():
        min = 9999999
        max = -9999999
        min_max[data_type] = {}
        for emotion, data in emotion_list.items():
            if emotion == "header":
                continue
            # print(f"Data: type - {type(np.array(data))} shape - {data.shape}, min: {type(min)}")
            if np.min(data) < min:
                min = np.min(data)
            if np.max(data)> max:
                max = np.max(data)
        
        min_max[data_type]["min"] = min
        min_max[data_type]["max"] = max
    
    return min_max


# def preprocess_signal(data, baseline_data, data_type, crop_settings = [], filter_settings = [], remove_baseline = True):
#     """
#     Args:
#         crop_settings    -   (list)  -   [signal_length, crop_before, crop_after]
#     """
    # preprocessed_signal = copy.deepcopy(data)
    # pipeline = get_preprocess_pipeline(data_type)
    # sampling_freq = utils.get_sampling_freq(data_type)
    # for i in pipeline:
    #     if i == "CROP":
    #         preprocessed_signal = crop_signal(signal=preprocessed_signal, sampling_freq=sampling_freq, crop_settings=crop_settings)
        
    #     if i == "FILTER":
    #         preprocessed_signal = filter_signal(signal = preprocessed_signal, sampling_freq=sampling_freq, filter_settings= filter_settings)
        
    #     if remove_baseline and i == "RM_BASELINE":
    #         preprocessed_signal = normalize_wrt_baseline(data, baseline_data)
    #     else:
    #         min = min_max[data_type]["min"]
    #         max = min_max[data_type]["max"]
    #         preprocessed_signal = min_max_scale(data, min, max)
    

#     return preprocessed_signal






def preprocess_e4(e4_data_object, remove_baseline = True):
    """ Preprocesses the E4 data of a single experiment.
    args:
        e4_data_object: struct  -   See file organize_data.py function get_e4_data_object()

    returns:
        struct  -   On the same form as the e4_data_object, but with preprocessed data.
    """
    print("Preprocessing data")

    # State crop settings
    base_length = 3*60 #seconds
    crop_before = True
    crop_after = False

    crop_settings = [base_length, crop_before, crop_after]

    min_max = get_min_max(e4_data_object)
    
    # preprocessed_data_object = e4_data_object
    preprocessed_data_object = {}
    for data_type, emotion_list in e4_data_object.items():
        preprocessed_data_object[data_type] = {}
        baseline_data = emotion_list["baseline"]
        # print(f"data_type: {data_type}, shape: {baseline_data.shape}")
        
        for emotion, data in emotion_list.items():
            if emotion == "header":
                continue
            # print(emotion)
            preprocessed_signal = copy.deepcopy(data)
            pipeline = get_preprocess_pipeline(data_type)
            sampling_freq = utils.get_sampling_freq(data_type)
            for i in pipeline:
                # print(i)
                if i == "CROP":
                    preprocessed_signal = crop_signal(signal=preprocessed_signal, sampling_freq=sampling_freq, crop_settings=crop_settings)
                
                if i == "FILTER":
                    # print("Filter signal")
                    preprocessed_signal = filter_signal(signal = preprocessed_signal, signal_type= data_type)
                
                if remove_baseline and i == "RM_BASELINE":
                    preprocessed_signal = normalize_wrt_baseline(data, baseline_data)
                else:
                    min = min_max[data_type]["min"]
                    max = min_max[data_type]["max"]
                    if(max <= min):
                        print("ERROR!!!")
                    preprocessed_signal = min_max_scale(data, min, max)
            # preprocessed_data = preprocess_signal(data= data, baseline_data= baseline_data, data_type= data_type, crop_settings= crop_settings, remove_baseline= remove_baseline)
            preprocessed_data_object[data_type][emotion] = preprocessed_signal
            
    return preprocessed_data_object


def test():
    print("Test")
    dt = 0.001
    f = 1/dt
    t = np.arange(5.0,step=dt)
    signal = 2*np.sin(2*np.pi*10*t) + np.sin(2*np.pi*120*t) + 10*t + np.random.normal(0, 0.5, t.size)

    print(utils.get_signal_length(signal, f))
