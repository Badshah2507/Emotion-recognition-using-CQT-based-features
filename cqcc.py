import numpy as np
import librosa
import scipy

def cqtspec(audio,sr,min_freq=30,octave_resolution=14):
    max_frequency= sr/2
    num_freq = round(octave_resolution * np.log2(max_frequency/min_freq))
    step_length = int(pow(2, int(np.ceil(np.log2(0.04 * sr)))) / 2)
    cqt_spectrogram = np.abs(librosa.cqt(audio,sr=sr,fmin=min_freq,bins_per_octave=octave_resolution,n_bins=num_freq))
    return cqt_spectrogram

def cqcc(audio,sr,min_freq=30,octave_resolution=14,num_coeff=20):
    cqt_spectrogram=np.abs(np.power(cqtspec(audio,sr,min_freq,octave_resolution),2))
    num_freq=np.shape(cqt_spectrogram)[0] 
    cqcc=scipy.fft.dct(np.log(cqt_spectrogram+0.001))
    ftcqt_spectrogram=np.fft.fft(cqt_spectrogram,2*num_freq-1,axis=0)
    absftcqt_spectrogram=abs(ftcqt_spectrogram)
    spectral_component=(np.real(np.fft.ifft(absftcqt_spectrogram,axis=0)[0:num_freq,:]))
    coeff_indices=np.round(octave_resolution*np.log2(np.arange(1,num_coeff+1))).astype(int)
    audio_cqcc=cqcc[coeff_indices,:]
    return audio_cqcc