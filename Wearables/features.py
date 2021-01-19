""" Feature Extraction for BEAT-PD
Ethan Lew
(elew@pdx.edu)
4/05/2020
"""
import typing
import abc

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage.filters import maximum_filter1d

SignalLike = typing.Union[list, tuple, np.array]


def assign_activity(times, df):
    """
    Helper function to assign the activity of a subject based on
    a list times using the original data
    :param times: list of times like from a downsampled data set
    :param df: orignal data
    :return:
    """
    activities = []
    for i in range(len(times) - 1):
        min_ = times[i]
        max_ = times[i + 1]
        set_activity = list(set(df[(df.index >= min_) & (df.index <= max_)]['activity_code']))
        if len(set_activity) > 1:
            return
        elif len(set_activity) == 0:
            return
        else:
            activities.append(set_activity[0])

    set_activity = list(set(df[(df.index >= times[-1])]['activity_code']))
    if len(set_activity) > 1:
        print(set_activity)
        return
    elif len(set_activity) == 0:
        return
    activities.append(set_activity[0])
    return activities

def _downsample_mean(data: pd.DataFrame,ds_rate: str = "5S") -> pd.DataFrame:
    """
    :param data: (N,d) data frame of raw accel / gyro data
    :param ds_rate: downsample rate
    :return: dataframe of (M,d) where M < N
    """
    data_down = data.resample(ds_rate).mean()
    data_down = data_down.dropna()
    data_down['activity'] = assign_activity(data_down.index, data)
    del data
    return data_down

def _magnitude(data: np.array) -> np.array:
    """ get magnitude of BEAT-PD measurement array
    :param data: (N, 3), where columns are x, y and z respectively
    :return: (N) magnitude
    """
    return np.linalg.norm(data, axis=1)

def _apply_bandpass(data: SignalLike, fs: float, freqs: typing.List[float], order=10) -> np.array:
    """ given a stream of data, apply a bandpass filter to each column
    :param data: (N, M)
    :param fs: sampling frequency
    :param freqs: pass band interval
    :return: (N-W-1, M) filter output
    """
    sos = signal.butter(order, freqs, 'bandpass', analog=False, fs=fs, output='sos')
    return signal.sosfilt(sos, data)

def _median_spectral_power(psd):
    """ Braybrook's Mentioned median spectral power
    :param psd: array of values -- power spectral density
    :return: median of psd
    """
    return np.median(psd,axis=0)

class Feature(abc.ABC):
    """ Temporal Feature Extraction for BEAT-PD """
    def __init__(self, tspan=None, fs=50, **user_data):
        """ use same schema in user_data as that seen in BeatPDDataset.load_user_data
        :param tspan: time span [tstart, tstop] to extract features in
        :param fs: sampling frequency
        :param user_data: from BeatPDDataset.load_user_data
        """
        self._fs = fs
        self._tspan = tspan
        self._subject_measurements = user_data['measurements'][1]

    def block_process(self, *args, **kwargs) -> typing.List[typing.Tuple[SignalLike]]:
        """ feature extract everything
        :return: list of processed measurements
        """
        return [self.process(idx, *args, **kwargs) for idx in range(self.num_measurements)]

    @property
    def num_measurements(self) -> int:
        """ number of measurements from 'measurements' """
        return len(self._subject_measurements)

    def get_measurement_chunk(self, idx: int) -> np.array:
        """ given a measurement index, get the time truncated measurement """
        meas = self._subject_measurements[idx]
        meas = np.array(meas)
        if self._tspan is not None:
            t = meas[:, 0]
            tspanidx = [(np.abs(t-self._tspan[0])).argmin(), (np.abs(t-self._tspan[1])).argmin()]
            return meas[tspanidx[0]:tspanidx[1], :]
        else:
            return meas

    def process(self, idx: int, *args, **kwargs) -> typing.Tuple[SignalLike, SignalLike]:
        """ transform input, assert state and process data """
        meas = self.get_measurement_chunk(idx)

        if len(meas.shape) == 1:
            meas = meas[0]

        if meas is None:
            return ([],[])

        assert idx < self.num_measurements, \
               f"index {idx} needs to be less than total number of measurements {self.num_measurements}"
        out = self._process(meas[:, 0], meas[:, 1:], *args, **kwargs)
        return out

    def process_signal(self, t: np.array, meas: np.ndarray, **kwargs):
        """ process a signal as input """
        assert len(t) == meas.shape[0], f"time vector t must be same length as measurement array"
        return self._process(t, meas, **kwargs)

    @abc.abstractmethod
    def _process(self, t: SignalLike, measure: SignalLike, *args, **kwargs) -> typing.List[SignalLike]:
        """ process implementation """
        raise NotImplementedError

class FeatureSpectrogram(Feature):
    """ Easy Spectrogram Method """

    def _spectrogram(self, t: SignalLike, measure: SignalLike, window_size=5, window_overlap=None, spectral_resolution=0.2,
                     freqs=(.2, 4)) -> typing.Tuple[SignalLike]:
        sig = _apply_bandpass(_magnitude(measure), self._fs, freqs=freqs)

        if window_overlap is None:
            window_overlap = window_size * 0.8
        fxx, txx, zxx = signal.spectrogram(sig, self._fs, window="hann", nperseg=window_size * self._fs,
                                           nfft=max(window_size * self._fs, (self._fs // 2) // (spectral_resolution)),
                                           noverlap=window_overlap * self._fs, mode='psd')
        return txx, fxx, zxx

    def _process(self, *args, **kwargs):
        return self._spectrogram(*args, **kwargs)

class FeatureRest(FeatureSpectrogram):
    """ Immobile Detection Filter described in
    Braybrook, M., O’Connor, S., Churchward, P., Perera, T., Farzanehfar, P., & Horne, M. (2016). An ambulatory tremor
    score for Parkinson’s disease. Journal of Parkinson's disease, 6(4), 723-731.
    """
    def _process(self, t: SignalLike, measure: SignalLike, freqs=(.2, 8), window_size=5, **kwargs) -> typing.Tuple[SignalLike]:
        sig = _magnitude(measure)
        sig = _apply_bandpass(sig, self._fs, freqs)
        window_digital = window_size * self._fs
        f, t, Zxx = signal.spectrogram(sig, self._fs, window=("hann"), nperseg=window_digital, nfft=window_digital,
                                       noverlap=200, mode='psd')
        Zabs = np.abs(Zxx)
        Zxx[np.where(f < 0.4)] = 0
        tmax, smax = t, np.max(Zabs, axis=0) / 9.81e-3
        return tmax, smax <= 12.0

class FeatureMSP(FeatureSpectrogram):
    """ Moving Mean Spectral Power """
    def _process(self, t: SignalLike, measure: SignalLike, freqs=[0.4, 4] ,**kwargs) -> typing.List[SignalLike]:
        txx, _ , zxx = self._spectrogram(t, measure, **kwargs)
        return txx, np.average(np.abs(zxx), axis=0)/9.81e-3

class FeatureTremorBraybrook(FeatureSpectrogram):
    """ Implement Tremor Detection Described in
    Braybrook, M., O’Connor, S., Churchward, P., Perera, T., Farzanehfar, P., & Horne, M. (2016). An ambulatory tremor
    score for Parkinson’s disease. Journal of Parkinson's disease, 6(4), 723-731.
    """
    def _process(self, t: SignalLike, measure: SignalLike, **kwargs) -> typing.List[SignalLike]:
        sig = _apply_bandpass(_magnitude(measure), self._fs, freqs=[0.2, 8])
        f, t, zxx = signal.spectrogram(sig, self._fs, nfft=250, noverlap=200, nperseg=250, mode="complex",
                                      detrend="linear")  # psd after taking 20log10()
        zxx[np.where(f < .4)] = 0
        idxa = np.argmin(np.abs(f[:] - 1))
        idxb = np.argmin(np.abs(f[:] - 10))
        median_zxx = zxx[idxa:idxb, :]
        tremor_detections = np.ones(t.shape[0])
        # Braybrook - bullet 1 -stft
        tremor_detections[np.where(20 * np.log10(np.max(median_zxx, 0)) - 20 * np.log10(
            _median_spectral_power(median_zxx)) < 6 / 9.8)] = 0
        peak_spectral_power = f[np.argmax(median_zxx, 0)]
        # Braybrook - bullet 2
        tremor_detections[peak_spectral_power < 2.8] = 0
        # Braybrook - bullet 2
        tremor_detections[peak_spectral_power > 10] = 0
        # Braybrook - bullet 3
        tremor_detections[np.where(np.abs(peak_spectral_power[0:-5] - peak_spectral_power[5:]) > .4)] = 0
        # Braybrook - bullet 3
        tremor_detections[np.where(np.abs(peak_spectral_power[5:] - peak_spectral_power[0:-5]) > .4)[0] + 1] = 0
        return t, tremor_detections

class FeatureMax(Feature):
    """ Moving Max Filter """
    def _process(self, t: SignalLike, measure: SignalLike, window_size=5) -> typing.Tuple[SignalLike]:
        sig = _magnitude(measure)
        w = window_size*self._fs
        hw = (w-1)//2
        filt_out = maximum_filter1d(sig, size=w)[hw:-hw]
        return t[w//2-1: -w//2+1], filt_out

class FeatureMovingAverage(Feature):
    """ Moving Average Filter """
    def _process(self, t: SignalLike, measure: SignalLike, window_size=5) -> typing.Tuple[SignalLike]:
        sig = _magnitude(measure)
        w = window_size*self._fs
        return t[w // 2 - 1: -w // 2], np.convolve(sig, np.ones((w,))/w, mode='valid')

class Transform:
    """
    A transform is an object that takes in a config
    and applys a series of tranformations
    """
    def __init__(self,config):
        pass