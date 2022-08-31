#!/user/bin/env python
# -*- coding:utf-8 -*-
"""
@Time       : 2022/5/20 20:27
@Author     : zhushuli
@File       : fft_extract.py
@DevTool    : PyCharm
@Refer      : https://github.com/balzer82/FFT-Python
              https://github.com/cingtiye/Basic-time-domain-and-frequency-domain-features-of-signals
"""
import numpy as np


class FtrFFT(object):
    def __init__(self, dt=0.02):
        self.epsn = 1e-8
        self.dt = dt  # duration between neighboring frames
        self.fa = 1.0 / self.dt  # sacn frequency
        self.seq_data = None

    def __fft(self, seq_data):
        if self.seq_data is not None and self.seq_data == seq_data:
            return self.freq_spectrum, self.freq_sum
        Y = np.fft.fft(seq_data)
        N = len(Y) // 2 + 1
        X = np.linspace(0, self.fa / 2, N, endpoint=True)

        hann = np.hanning(len(seq_data))
        hamm = np.hamming(len(seq_data))
        black = np.blackman(len(seq_data))

        Yhann = np.fft.fft(hann * seq_data)
        self.freq_spectrum = 2.0 * np.abs(Yhann[:N] / N)[1:]
        self.freq_sum = np.sum(self.freq_spectrum)
        return self.freq_spectrum, self.freq_sum

    def fft_mean(self, seq_data):
        """ 频谱均值 """
        freq_spectrum, _ = self.__fft(seq_data)
        return np.mean(freq_spectrum)

    def fft_std(self, seq_data):
        """ 频谱标准差 """
        freq_spectrum, _ = self.__fft(seq_data)
        return np.std(freq_spectrum)

    def fft_entropy(self, seq_data):
        freq_spectrum, freq_sum = self.__fft(seq_data)
        pr_freq = freq_spectrum * 1.0 / freq_sum
        entropy = -1 * np.sum([np.log2(p) * p for p in pr_freq])
        return entropy

    def fft_energy(self, seq_data):
        freq_spectrum, freq_sum = self.__fft(seq_data)
        return np.sum(freq_spectrum ** 2) / len(freq_spectrum)

    def fft_skew(self, seq_data):
        freq_spectrum, freq_sum = self.__fft(seq_data)
        _fft_mean, _fft_std = self.fft_mean(seq_data), self.fft_std(seq_data)
        return np.mean([0 if _fft_std < self.epsn else np.power((x - _fft_mean) / _fft_std, 3)
                        for x in freq_spectrum])

    def fft_kurt(self, seq_data):
        freq_spectrum, freq_sum = self.__fft(seq_data)
        _fft_mean, _fft_std = self.fft_mean(seq_data), self.fft_std(seq_data)
        return np.mean([0 if _fft_std < self.epsn else np.power((x - _fft_mean) / _fft_std, 4)
                        for x in freq_spectrum])

    @property
    def fft_shape_mean(self):
        return None

    @property
    def fft_shape_std(self):
        return None

    @property
    def fft_shape_skew(self):
        return None

    @property
    def fft_shape_kurt(self):
        return None

    def plot_spectrogram(self, seq_data, imu='acc', axis='x'):
        """ 绘制频谱图 """
        Y = np.fft.fft(seq_data)
        N = len(Y) // 2 + 1
        X = np.linspace(0, self.fa / 2, N, endpoint=True)

        hann = np.hanning(len(seq_data))
        hamm = np.hamming(len(seq_data))
        black = np.blackman(len(seq_data))

        Yhann = np.fft.fft(hann * seq_data)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
        axs[1].plot(seq_data, color='IndianRed')
        axs[1].set_xlabel('Time ($Hz$)', fontsize=16)
        axs[1].set_ylabel('Value', fontsize=16)
        axs[0].plot(X, 2.0 * np.abs(Yhann[:N]) / N, color='Goldenrod')
        axs[0].set_xlabel('Frequency ($Hz$)', fontsize=16)
        axs[0].set_ylabel('Amplitude ($Unit$)', fontsize=16)

        fig.suptitle('Amplitude Spectrum by FFT', fontsize=24)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./images/FFT_extract_%s_%s.png' % (imu, axis), dpi=400)
        plt.show()


fft_ftr_gen = FtrFFT(dt=1.0 / 50)

__all__ = ['fft_ftr_gen']

if __name__ == '__main__':
    # 读取数据
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_struct import *

    file_path = '/Users/zhushuli/Desktop/PDR_dataset/220407_Wground_recovery/phone1/user8/2022-04-08 15-47-51.csv'
    ts_sen_map = {}
    fd = open(file_path, 'r')
    for i, line in enumerate(fd):
        if i == 0:
            continue
        sen_unit = SenUnitV2(line)
        if sen_unit.ts not in ts_sen_map:
            ts_sen_map[sen_unit.ts] = [sen_unit]
        else:
            ts_sen_map[sen_unit.ts].append(sen_unit)
    fd.close()

    ts_list = sorted(ts_sen_map.keys())
    picked_imu_list = []
    for ts in ts_list[:10]:
        picked_imu_list.extend([u.get_by_imu_axis(imu='acc', axis='x') for u in ts_sen_map[ts]])
    fft_ftr_gen.plot_spectrogram(picked_imu_list, imu='acc', axis='x')
