import pandas as pd
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import signal

def loadfiles(files):
    """
    load all files in the designated path into a very big df, store that for easier access later
    """
    for file in files:
        df = pd.read_csv(file, skiprows=skiprows, engine="c", header=0, na_values=[' '], dtype=np.float64)
        return df

def process_store(files):
    """
    files: list of filepaths to read
    If store is still the full dataset as loaded, this will split it into a part dedicated to the spectrum (an in the process halfing the size of the file,
    because that is terribly memory inefficient and stores the utterly redundant information about pixel-wavelength assignment at every timestep) and a much
    smaller part dedicated to all other variables, which are also grouped according to their observation interval. 
    should do nothing, if store was already processed
    """
    df = loadfiles(files)
    # see if the df was processed already
    try:
        df.columns.get_loc("aWavelengthIntensity ")
    except KeyError:
        return
    # Move the Spectrum to its own df
    spec_intensities_start = df.columns.get_loc("aWavelengthIntensity ")
    spec_wavelengths_start = df.columns.get_loc("aWavelengthRange ")
    spec_wavelengths_end = df.columns.get_loc("O2Intnsty") - 1
    spec_intensities = df.iloc[:, spec_intensities_start:spec_wavelengths_start - 1]
    spec_wavelengths = df.iloc[0, spec_wavelengths_start:spec_wavelengths_end]
    col_labels = spec_wavelengths.values.tolist()
    index_labels = df.iloc[:,spec_intensities_start-1].tolist()
    spec_intensities.columns = col_labels
    spec_intensities.index = index_labels
    store["df"] = spec_intensities
    df.drop(df.columns[spec_intensities_start - 1:spec_wavelengths_end], axis=1, inplace=True)
    # group observables according to measurement intervals
    datasets = {} # list of all different lengths of datasets
    # structure: {7000: n*7000 df, 70000: n*70000 df, ...}
    for i in range(0, len(df.columns), 2):
        df_col = df.iloc[:,i+1]
        df_col_timestamps = df.iloc[:,i]
        df_col.index = df_col_timestamps    # set timestamps as indices instead
        df_col.dropna(inplace=True)
        if df_col.size in datasets:
            datasets[df_col.size] = pd.concat([datasets[df_col.size], df_col], axis=1) # add the column to the dataframe of that length
        else:
            datasets[df_col.size] = df_col # Initialize the dictionary entry
    # save data to files so the small part can be easily accessed
    with open('non_spectra.pkl', 'wb') as pickle_file:
        pickle.dump(datasets, pickle_file)



def find_seasons(series:pd.Series):
    """
    Find indices in the data at which a pattern repeats using autocorrelation
    """
    min_time = 200 # minimum accepted cycle length in ms
    autocorr = np.zeros(len(series))
    # TODO find a way to avoid loops here
    for i in range(len(series)):
        autocorr[i] = series.autocorr(lag=i)
    delta_t = series.index[1] - series.index[0]
    min_distance = min_time/delta_t  
    seasonal_indices = signal.find_peaks(autocorr, height=0.95, distance = min_distance, width=5)[0] # find the indices at the peaks of the autocorrelation
    return series.index[seasonal_indices]

def plot_seasons(series:pd.Series, season_times, ax):
    for i in range(len(season_times)-1):
        df_plot = series.loc[season_times[i]:season_times[i+1]]
        ax.plot(df_plot.index - season_times[i], df_plot.values)

def main():
    file_location = "\Dokumente\Privat\Plasway\Al2O3 Process Data"
    files = glob.glob(file_location+"\*.csv")
    skiprows = np.concatenate([np.arange(0,5), np.arange(7,24)])
    store = pd.HDFStore("good_process_spectra.h5")
    with open('non_spectra.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    SetPoint14 = loaded_dict[7110]["reaPositionSetPoint (14)"]
    pressure_series = loaded_dict[71084]["reaActualPressure_mbar"]
    season_times = find_seasons(SetPoint14)
    fig = plt.figure()
    ax = fig.subplots()
    plot_seasons(pressure_series, season_times, ax)
    plt.show()

main()
