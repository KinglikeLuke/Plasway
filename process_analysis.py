import pandas as pd
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from scipy import signal
import os
import warnings

def loadfiles(file, skiprows):
    """
    load all files in the designated path into a very big df
    """
    df = pd.read_csv(file, skiprows=skiprows, engine="c", header=0, na_values=[' ', 'EOF'], dtype=np.float32)
    return df

def process_store(file, skiprows, store):
    """
    files: list of filepaths to read
    If store is still the full dataset as loaded, this will split it into a part dedicated to the spectrum (an in the process halfing the size of the file,
    because that is terribly memory inefficient and stores the utterly redundant information about pixel-wavelength assignment at every timestep) and a much
    smaller part dedicated to all other variables, which are also grouped according to their observation interval. 
    should do nothing, if store was already processed
    """
    df = loadfiles(file, skiprows)  # TODO allow for reading of multiple files?
    # see if the df was processed already
    spectrum = True
    try:
        df.columns.get_loc("aWavelengthIntensity ")
    except KeyError:
        spectrum = False
    if spectrum:
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
        if np.all(df_col.values < 0.001):    # If all values are 0, skip adding that column to the data
            continue
        if df_col.size in datasets:
            datasets[df_col.size] = pd.concat([datasets[df_col.size], df_col], axis=1) # add the column to the dataframe of that length
        else:
            datasets[df_col.size] = pd.DataFrame(df_col) # Initialize the dictionary entry
    # save data to files so the small part can be easily accessed
    with open('non_spectra.pkl', 'wb') as pickle_file:
        pickle.dump(datasets, pickle_file)


def find_seasons(df:pd.DataFrame):
    """
    Find indices in the data at which a pattern repeats using autocorrelation. Beginning and end are *not* included! TODO get the ends tucked in
    """
    min_time = 500 # minimum accepted cycle length in ms
    autocorr = np.zeros_like(df)
    for i, col in enumerate(df.columns):
        autocorr[:,i] = signal.correlate(df[col], df[col])[:df.shape[0]]
    delta_t = df.index[1] - df.index[0]
    min_distance = min_time/delta_t
    autocorr = autocorr/np.max(autocorr, axis=0)
    seasonal_indices_old = []
    for column in autocorr.T:
        seasonal_indices = signal.find_peaks(column, distance = min_distance, width=min_distance/2)[0] # find the indices at the peaks of the autocorrelation
        if len(seasonal_indices) > len(seasonal_indices_old):   # The most seasons are likely the most accurate
            seasonal_indices_old = seasonal_indices
    return df.index[seasonal_indices_old[:-2]] # last index is not really a cycle due to the tail of the recording

def plot_seasons(seasons, season_statistics, ax: Axes):
    """Takes in a pd Series and some cyclicity determined elsewhere and plots it so that the seasons overlap

    Args:
        series (pd.Series): time-series of type Number
        season_times (ndarray): Indices at which the seasons repeat
        ax (Axes): plot area
    """
    for s in seasons:
        ax.plot(s)
    ax.plot(season_statistics, c='navy', label="Mean of seasons")

def analyse_seasons(series:pd.Series, season_times):
    shortest_season = np.argmin(np.diff(season_times))
    shortest_season_indices = series.loc[season_times[shortest_season]:season_times[shortest_season+1]].index - season_times[shortest_season]
    # Include the non-seasoned data as pre-and postlude
    pre_run = series.loc[series.index[0]:season_times[0]]
    pre_run.index -= season_times[0]
    post_run = series.loc[season_times[-1]:series.index[-1]]
    post_run.index -= season_times[-1]
    seasons = [pre_run]
    season_number = len(season_times)-1
    truncated_seasons = np.zeros((season_number, len(shortest_season_indices))) # to store truncated seasons to do quick statistical analysis
    for i in range(season_number):
        df_season = series.loc[season_times[i]:season_times[i+1]]
        df_season.index -= season_times[i]
        seasons.append(df_season)
        truncated_seasons[i,:] = df_season.values[:len(shortest_season_indices)]
    mean = np.mean(truncated_seasons, axis=0)
    seasons.append(post_run)
    analysis_df = pd.DataFrame(data=mean, index=shortest_season_indices)
    deviation_from_mean = truncated_seasons - mean
    var = np.sqrt(np.sum(deviation_from_mean**2, axis=1)/season_number) # not quite the variance, quantifies the difference of the whole season from the mean
    return seasons, analysis_df
        
def display_data(loaded_dict:dict, dict_key:int, table_key:str, season_times):
    """Somewhat prototypical action on interaction: plot and (as yet unreturned) data-analysis

    Args:
        loaded_dict (dict): The data in form of a dictionary
        dict_key (int): which entry of the dictionary one wants to look at
        table_key (_type_): which column of the data frame is interesting
        season_times (_type_): the times at which cycles in the data repeat
    """
    series = loaded_dict[dict_key][table_key]
    seasons, mean_season = analyse_seasons(series, season_times)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.cool(np.linspace(0.01,1,len(seasons))))
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xlabel("t/ms")
    ax.set_title(table_key)
    ax.set_xlim(mean_season.index[0] - 20, mean_season.index[-1] + 20)
    plot_seasons(seasons, mean_season, ax)
    ax.legend()
    plt.show()
    
def enter_file(file_location=None):
    #D:\Local\Analysis\202402 Al2O3
    if not file_location:
        file_location = input("Specify folder path: ")
        try:
            glob.glob(file_location+"\*.csv")[0]
        except:
            print("No valid files at this path!")
            return 0
    files = glob.glob(file_location+"\*.csv")
    print("Available files:")
    for i in sorted(files):
        print(os.path.basename(i))
    filepath = os.path.join(file_location, input("Which file do you want to load? "))
    if not filepath in files:
        print("Please choose a file in the path")
        filepath = enter_file(file_location)       # Go back to selection from folder
    return filepath
    

def main():
    """
    Interesting comparisons: Input MFC needle ("reaPosSetpoint (34)"), Output MFC needle("reaPosSetpoint (30)"), O2Flow, Pressure, oxygen/argon "reaActFlo(35)" flow ratio
    """
    # TODO interactive file loading. Might make problem with opening multiple files moot, as you would just select multiple
    loading = input("Do you want to load a new file?[y/n]" )
    if loading == 'y':
        while True:
            filepath = enter_file()
            if filepath:
                break
        skiprows = np.concatenate([np.arange(0,5), np.arange(7,24)])
        store = pd.HDFStore("good_process_spectra.h5")
        process_store(filepath, skiprows, store) 

    with open('non_spectra.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    #print("Opened file: {}".format(os.path.basename(files[0])))
    order_df_key = min(list(loaded_dict.keys()))       
    season_gauge = loaded_dict[order_df_key]
    with warnings.catch_warnings(action="ignore"):
        season_times = find_seasons(season_gauge)
    while True:
        print("available measurement series sizes: {}".format(list(loaded_dict.keys())))
        dict_key = (input("Which series do you want to consider? "))
        try: 
            loaded_dict[int(dict_key)]
        except KeyError:
            print("Please choose a valid key.")
            continue
        except ValueError:
            print("Please choose a numerical key.")
            continue
        dict_key = int(dict_key)
        print("Measurements in this series:")
        for i in sorted(list(loaded_dict[dict_key])):
            print(i)
        table_key = input("Which measurement interests you? (or 'return' to go back to series selection) ")
        if table_key=='return':
            continue
        while True:
            try:
                loaded_dict[dict_key][table_key]
            except KeyError:
                table_key = input("Please input a valid measuremnt from the list (without quotation marks): ")
                continue
            break
        display_data(loaded_dict, dict_key, table_key, season_times)
        end = input("End program? [y/n] ")
        if end == 'y':
            break

main()
