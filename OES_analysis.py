# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 10:22:36 2024

@author: Lukas
"""

import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import pandas as pd


def ptest_lorentzian(n_peaks, length=200, mean_I=1, mean_fwhm=2, mean_off=1):
    # mean_fwhm?
    x = np.arange(length)
    off = np.random.normal(mean_off, mean_off / 5)
    x0 = np.random.randint(0, length + 1, n_peaks)
    I = np.random.normal(mean_I, mean_I / 10, n_peaks)
    fwhm = np.random.normal(mean_fwhm, mean_fwhm / 5, n_peaks)
    params = np.concatenate(([off], x0, I, fwhm))
    y_data = SpectralProperties.multi_peak_func(x, params) + np.random.normal(0, 0.1, length)
    testing = FindPeaks(y_data)
    assert testing.i_x0 - x0 < fwhm


class SpectralProperties:
    def lorentzian(self, x, x0, a, gam):
        """
        Parameters
        ----------
        x : np array.
        x0 : peak position.
        a : amplitude.
        gam : width.

        Returns
        -------
        TYPE
            Lorentzian.
        """
        return a * gam ** 2 / (gam ** 2 + (x - x0) ** 2)

    def multi_peak_func(self, x, params):
        off = params[0]
        params = params[1:]
        assert not len(params) % 3
        n_peaks = int(len(params) / 3)
        params_x0 = params[:n_peaks]
        params_I = params[n_peaks:2 * n_peaks]
        params_fwhm = params[2 * n_peaks:]
        return off + sum([self.lorentzian(x, x0_i, I_i, fwhm_i) for x0_i, I_i, fwhm_i
                          in zip(params_x0, params_I, params_fwhm)])


class FindPeaks(SpectralProperties):
    def __init__(self, y_data, full_fit=False, height_factor=10, width_factor=1, prominence_factor=0):
        """

        :param y_data: np array of spectral amplitude data
        :param height_factor: conditions for recognising a peak in initial guess phase
        """
        self.y0 = y_data
        self.x_data = np.arange(len(y_data))
        self.y_ground = min(y_data)
        self.y_data = y_data - self.y_ground
        self.y_amp = max(self.y_data)
        self.y_data = self.y_data / self.y_amp
        height_req = np.average(np.abs(np.diff(self.y_data))) * height_factor
        width_req = len(self.y_data) / 1000 * width_factor
        prominence_req = width_req * prominence_factor
        self.i_guess = self._initial_guess(height_req, width_req, prominence_req)
        block_len = int((len(self.i_guess) - 1) / 3)
        self.i_off = self.i_guess[0]
        self.i_x0 = self.i_guess[1:block_len + 1]
        self.i_I = self.i_guess[block_len + 1:2 * block_len + 1]
        self.i_fwhm = self.i_guess[2 * block_len + 1:]
        if full_fit:
            i_popt = self.fit_peaks(self.i_guess)
            # warum hat self.i_popt nen self?
            self.i_off = i_popt[0]
            self.i_x0 = i_popt[1:block_len + 1]
            self.i_I = i_popt[block_len + 1:2 * block_len + 1]
            self.i_fwhm = i_popt[2 * block_len + 1:]

    def fit_peaks(self, guess):
        result = least_squares(self.res_multi_peak_func, x0=guess)
        popt = result.x  # order: off, n*x0,n*I, n*fwhm
        return popt

    def res_multi_peak_func(self, params):
        """
        Function to minimize

        Parameters
        ----------
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        diff : TYPE
            DESCRIPTION.

        """
        diff = [self.multi_peak_func(x, params) - y
                for x, y in zip(self.x_data, self.y_data)]
        return diff

    def _initial_guess(self, height_req, width_req, prominence_req):
        """
        Guesses initial peak positions etc. The number of peaks is determined
        here. Makes or breaks the fit

        Parameters
        ----------
        height_req : TYPE
            DESCRIPTION.
        width_req : TYPE
            DESCRIPTION.
        prominence_req : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        guess : TYPE
            DESCRIPTION.

        """
        # initial properties of peaks
        pk, properties = find_peaks(self.y_data, height=height_req,
                                    width=width_req, prominence=prominence_req)
        # extract peak heights and fwhm
        _I = properties['peak_heights']
        fwhm = properties['widths']

        # pack initial guesses together
        guess = np.concatenate(([self.y_ground], pk, _I, fwhm))
        return guess

    # def return_results(self, popt):
    #    """
    #    get the results into an easily accessible order. Note that off is
    #    returned last.
    #    """
    #    n_peaks = np.round((len(popt) - 1) / 3)
    #    off = popt[0]
    #    popt = popt[1:]
    #    x0 = popt[:n_peaks]
    #    I = popt[n_peaks:2 * n_peaks]
    #    fwhm = popt[n_peaks:2 * n_peaks]
    #    return x0, I, fwhm, off

    def plot_results(self, popt):
        resolution = 2000
        x_plot = np.linspace(0, max(self.x_data), resolution)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(2, 1, 1)
        bx = fig.add_subplot(2, 1, 2)
        if len(popt) > 1:
            x0 = popt[1:int(len(popt) / 3) + 1]
            ax.plot(x0, self.multi_peak_func(x0, popt), 'o', ms=5)
            test_data = self.multi_peak_func(x_plot, popt)
            fit_data = self.y_ground + self.y_amp * self.multi_peak_func(x_plot, popt)
            ax.plot(x_plot, test_data, 'r--', lw=0.5)
            bx.plot(x_plot, fit_data)
        ax.plot(self.x_data, self.y_data, 'ok', ms=1)
        bx.plot(self.x_data, self.y0, ls='', marker='o', markersize=1)

        plt.show()


class Spectrum:
    def __init__(self, _y_data, _spectral_range):
        self.data = _y_data
        self.cleaned_data, self.uniformity = self.clean_data()
        self.cycles = self.extract_cycles()

    def clean_data(self):
        """
        Removes times at which the spectrum doesn't show sufficient variation and removes ground signal strength
        :return:
        """
        absolute_differences = np.abs(np.diff(self.data, axis=0))
        # check whether the variation of the data is sufficiently large compared to the total max. Should remove flat
        # periods
        threshold = np.max(self.data) * 10
        uniformity_check = np.sum(absolute_differences, axis=0) > threshold
        _data = self.data[:, uniformity_check]
        ground = np.min(_data, axis=0)
        _data = _data - ground
        return _data, uniformity_check

    def extract_cycles(self):
        """
        Splits up the data into cycles separated by dark periods and returns the average value of the spectrum in these
        cycles.
        :return:
        """
        # Find the indices separating activity from darkness by using the previous definition of uniformity
        indices = np.nonzero(self.uniformity[1:] != self.uniformity[:-1])[0] + 1
        cycle_data = np.split(self.cleaned_data, indices, axis=1)
        cycle_data = cycle_data[0::2] if self.uniformity[0] else cycle_data[1::2]
        # As periods may be of different length, they must be saved as a list and iterated over
        cycle_average = np.zeros((np.shape(data)[0], len(cycle_data)))
        for _i, cycle in enumerate(cycle_data):
            cycle_average[:, _i] = np.average(cycle, axis=1)
        return cycle_average

    def return_results(self, *params):
        return self.cleaned_data, self.cycles

    def peak_analysis(self, height_factor, width_factor):
        peak_table = []
        for i in range(np.shape(self.cycles)[1]):
            y_data = self.cycles[:, i]
            analysis = FindPeaks(y_data, height_factor=height_factor, width_factor=width_factor)
            df = pd.DataFrame(data={f'x0_{i}': analysis.i_x0, f'I_{i}': analysis.i_I})
            peak_table.append(df)
        return peak_table


names = glob.glob("Data/*.csv")
fig = plt.figure(figsize=(15, 15))

for image, name in enumerate(names):
    data = np.loadtxt(name, skiprows=24, delimiter=',', )
    extent_data = np.loadtxt(name, skiprows=24, usecols=1, delimiter=',')
    extent = 0, 800, np.min(extent_data), np.max(extent_data)
    data = data[4:]
    ax = fig.subplots()
    # ax = fig.add_subplot(3, 4, image+1)
    ax.set_ylabel("nm")
    ax.set_xlabel("t")
    spec = Spectrum(data)
    cleaned_data, cycle_averages = spec.return_results()
    peaks = spec.peak_analysis()
    ax.imshow(cleaned_data, norm='linear', origin='lower', aspect='auto')
    break

fig.tight_layout()
plt.show()
