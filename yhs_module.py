import numpy as np
import librosa
import os
from glob import glob
import soundfile as sf
# import tensorflow as tf
import torch


def make_mask(wave_mag, th_hi=-0.9, th_lo=-0.7, time_cut=15):
    mask = np.zeros_like(wave_mag)
    for i in range(wave_mag.shape[0]):
        for j in range(wave_mag.shape[1]):
            if (wave_mag[i][j] > th_hi) and (i > 128):
                mask[i][j] = 1
            elif (wave_mag[i][j] > th_lo) and (i <= 128):
                mask[i][j] = 1

    time_masking = np.sum(mask, axis=0)

    for i in range(time_masking.shape[0]):
        if time_masking[i] < time_cut:
            mask[:, i] -= mask[:, i]  # time masking
            time_masking[i] = 0
        else:
            time_masking[i] = 1

    mask[0:4, :] = 0  # remove DC

    mask = np.transpose(mask, (1, 0))

    return time_masking, mask


def adjust_snr(target, noise, db):  # Because of abs, it didn't return good scale value. We need bug fix
    sum_original = np.power(np.abs(target), 2)
    sum_noise = np.power(np.abs(noise), 2)
    sum_original = np.sum(sum_original)
    sum_noise = np.sum(sum_noise)
    sum_original = np.log10(sum_original)
    sum_noise = np.log10(sum_noise)
    scale = np.power(10, (sum_original-sum_noise)/2-(db/20))
    # SNR = 10 * log(power of signal(S)/power of noise(N))
    # SNR = 10 * (log(S) - log(N) - 2 log(noise scale))
    # log(noise scale) = (log(S) - log(N))/2 - SNR/20

    return scale


def save_wav(path_time, wave, scale, file_name, sr=16000):
    # path = '..\\results\\wav_file\\spec_only_g\\' + path_time
    path = '.'
    # os.makedirs(path, exist_ok=True)
    sf.write(path + '\\test_file_' + file_name + '.wav', wave * scale, sr)  # We use 16k sampling datasets


def path_to_txt(basic_path, min_size, file_name):
    segment_name = []
    if file_name[:5] == 'libri':
        speaker_dir = [f.path for f in os.scandir(basic_path) if f.is_dir()]

        chapter_dir = []
        for one_path in speaker_dir:
            chapter_dir += [f.path for f in os.scandir(one_path) if f.is_dir()]

        for one_path in chapter_dir:
            segment_name += glob(one_path + '\\*.flac')

    elif file_name[:6] == 'demand':
        noise_dir = [f.path for f in os.scandir(basic_path) if f.is_dir()]

        for one_path in noise_dir:
            segment_name += glob(one_path + '\\*.wav')

    else:
        print("Method for this dataset is not ready")
        return -1

    delete_file = []
    for one_path in segment_name:
        if os.stat(one_path).st_size < min_size:
            delete_file.append(one_path)

    for one_path in delete_file:
        segment_name.remove(one_path)  # Delete too small segment

    f = open(file_name + '.txt', 'w')
    for path in segment_name:
        f.write(path + "\n")
    f.close()


def snr(original, denoise):
    sum_noise = original - denoise
    sum_original = np.power(np.abs(original), 2)
    sum_noise = np.power(np.abs(sum_noise), 2)
    sum_original = np.sum(sum_original, axis=-1)
    sum_noise = np.sum(sum_noise, axis=-1)
    res = np.log10(sum_original/sum_noise) * 10

    return res


def cut_dc(spec):
    return spec[:, :, 1:]


class BasicWave:
    def __init__(self, number_sample=16000, sr=16000):
        self.number_sample = number_sample
        # ******************************************number sample is must less than min sample(not yet: padding method)
        self.sr = sr
        self.dataset_name = [None, None, None]  # memory name
        self.wave = [np.zeros(1), np.zeros(1), np.zeros(1)]  # wave data(numpy)

    def load_data(self, memory_number, path_set, start_index, number_file, dataset_name='_'):
        offset = 32000
        if memory_number > 3 or memory_number < 1:
            print("For the memory number, please enter a number between 1 and 3")
            return -1

        self.dataset_name[memory_number - 1] = dataset_name

        f = open(path_set, 'r')
        for i in range(start_index):
            f.readline()

        wave = None
        for i in range(number_file):
            path = f.readline()[:-1]
            print(path)
            if wave is None:
                wave, _ = librosa.load(path, sr=self.sr)
                wave = np.expand_dims(wave[offset:offset + self.number_sample], axis=0)
            else:
                wave_temp, _ = librosa.load(path, sr=self.sr)
                wave_temp = np.expand_dims(wave_temp[offset:offset + self.number_sample], axis=0)
                wave = np.concatenate((wave, wave_temp), axis=0)

        f.close()
        print("End loading")
        self.wave[memory_number - 1] = np.copy(wave)

    def info(self):
        for _ in range(10):
            print('*', end='')
        print()
        for i in range(3):
            if self.dataset_name[i] is not None:
                print(
                    f'Wave{i + 1} info\n'
                    f'Name: {self.dataset_name[i]}\n'
                    f'Shape: {self.wave[i].shape}\n'
                    f'Sampling rate: {self.sr}Hz'
                )
                for _ in range(10):
                    print('*', end='')
                print()


class TimeDomainData(BasicWave):
    def __init__(self, frame_size):
        super(TimeDomainData, self).__init__()
        self.frame_size = frame_size

    def add_noise(self, noise_snr):  # wave[0] + wave[1]
        noisy_wave = None
        for i in range(self.wave[1].shape[0]):
            for j in range(self.wave[0].shape[0]):
                scale = adjust_snr(self.wave[0][j], self.wave[1][i], noise_snr)
                if noisy_wave is None:
                    noisy_wave = self.wave[0][j] + (self.wave[1][i] * scale)
                    noisy_wave = np.expand_dims(noisy_wave, axis=0)
                else:
                    noisy_wave_temp = self.wave[0][j] + (self.wave[1][i] * scale)
                    noisy_wave_temp = np.expand_dims(noisy_wave_temp, axis=0)
                    noisy_wave = np.concatenate((noisy_wave, noisy_wave_temp), axis=0)

        return noisy_wave

    def pre_function(self, x_y_snr):
        y_data = np.copy(self.wave[0])
        y_data_temp = np.copy(self.wave[0])

        for i in range(self.wave[1].shape[0] - 1):
            y_data = np.concatenate((y_data, y_data_temp), axis=0)

        x_data = self.add_noise(x_y_snr)

        return x_data, y_data

    def dataset(self, x_y_snr, form='numpy'):
        x_data, y_data = self.pre_function(x_y_snr)

        shape = y_data.shape
        y_data = np.reshape(y_data, (shape[0], shape[1] // self.frame_size, self.frame_size)).astype(np.float32)
        x_data = np.reshape(x_data, (shape[0], shape[1] // self.frame_size, self.frame_size)).astype(np.float32)

        return x_data, y_data


class SpectralDomainData(TimeDomainData):
    def __init__(self, win_size, dc: bool, y_form):
        super(SpectralDomainData, self).__init__(win_size)
        self.dc = dc  # Retain DC bin
        self.y_form = y_form

    def np_spec(self, data):
        spec_data = np.concatenate((np.zeros(data.shape[0], self.frame_size//2), data), axis=-1)  # padding
        spec_data = librosa.stft(spec_data, n_fft=self.frame_size, hop_length=self.frame_size // 2,
                            win_length=self.frame_size, window='hann', center=False)
        spec_data = np.transpose(spec_data, (0, 2, 1))
        if self.dc is False:
            spec_data = cut_dc(spec_data)

        return spec_data

    # def tf_spec(self, data):
    #     spec_data = tf.signal.stft(data, self.frame_size, self.frame_size // 2, self.frame_size)
    #     if self.dc is False:
    #         spec_data = cut_dc(spec_data)
    #
    #     return spec_data

    def torch_spec(self, data):
        sine_window = (torch.Tensor(range(self.frame_size + 2))
                       * (4.0 * torch.atan(torch.Tensor([1.0]))) / (self.frame_size + 2))
        sine_window = torch.sin(sine_window[1:-1])
        spec_data = np.concatenate((data, np.zeros(data.shape[0], self.frame_size // 2)), axis=-1)  # padding
        spec_data = torch.Tensor(spec_data)
        spec_data = torch.stft(spec_data, window=sine_window, n_fft=self.frame_size, hop_length=self.frame_size // 2,
                                 win_length=self.frame_size, center=False, return_complex=False)
        spec_data = torch.transpose(spec_data, 1, 2)
        if self.dc is False:
            spec_data[:, :, :, 0] = cut_dc(spec_data[:, :, :, 0])
            spec_data[:, :, :, 1] = cut_dc(spec_data[:, :, :, 1])

        return spec_data

    def dataset(self, x_y_snr, form='numpy'):  # @overrides
        x_data, y_data = self.pre_function(x_y_snr)
        y_data_real = None
        y_data_imag = None

        if form == 'numpy':
            x_data = self.np_spec(x_data)
            x_data_real = x_data.real.astype(np.float32)
            x_data_imag = x_data.imag.astype(np.float32)

            if self.y_form != 'time':
                y_data = self.np_spec(y_data)
                y_data_real = y_data.real.astype(np.float32)
                y_data_imag = y_data.imag.astype(np.float32)

        # elif form == 'tensorflow':
        #     x_data = self.tf_spec(x_data)
        #     x_data_real = tf.math.real(x_data)
        #     x_data_imag = tf.math.imag(x_data)
        #
        #     if self.y_form != 'time':
        #         y_data = self.tf_spec(y_data)
        #
        #         y_data_real = tf.math.real(y_data)
        #         y_data_imag = tf.math.imag(y_data)

        elif form == 'torch':
            x_data = self.torch_spec(x_data)
            x_data_real = x_data[:, :, :, 0]
            x_data_imag = x_data[:, :, :, 0]

        else:
            print("Method for this dataset is not ready")
            return -1

        if self.y_form == 'time':
            y_data.astype(np.float32)
            return x_data_real, x_data_imag, y_data
        elif self.y_form == 'spectrum':
            return x_data_real, x_data_imag, y_data_real, y_data_imag
        else:
            print("Method for this dataset is not ready")
            return -1


class SpectralMagnitudeData(SpectralDomainData):
    def __init__(self, win_size, dc: bool, y_form):
        super(SpectralMagnitudeData, self).__init__(win_size, dc, y_form)

    def dataset(self, x_y_snr, form='log'):  # @overrides
        x_data, y_data = self.pre_function(x_y_snr)
        y_phase = None

        x_data = self.np_spec(x_data)
        x_data, x_phase = librosa.magphase(x_data)
        if form == 'log':
            x_data = np.log10(x_data + 1.0e-7)

        if self.y_form != 'time':
            y_data = self.np_spec(y_data)
            y_data, y_phase = librosa.magphase(y_data)
            if form == 'log':
                y_data = np.log10(y_data + 1.0e-7)

        x_data.astype(np.float32)
        y_data.astype(np.float32)

        if self.y_form == 'time':
            return x_data, x_phase, y_data
        if self.y_form == 'spectrum':
            return x_data, x_phase, y_data, y_phase
