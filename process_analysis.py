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
    df = pd.read_csv(file, skiprows=skiprows, engine="c", header=0, na_values=[' '], dtype=np.float32)
    return df

def process_store(files, skiprows, store):
    """
    files: list of filepaths to read
    If store is still the full dataset as loaded, this will split it into a part dedicated to the spectrum (an in the process halfing the size of the file,
    because that is terribly memory inefficient and stores the utterly redundant information about pixel-wavelength assignment at every timestep) and a much
    smaller part dedicated to all other variables, which are also grouped according to their observation interval. 
    should do nothing, if store was already processed
    """
    df = loadfiles(files[0], skiprows)  # TODO allow for reading of multiple files?
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
        if np.all(df_col.values==0):    # If all values are 0, skip adding that column to the data
            continue
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

def plot_seasons(seasons, season_statistics, ax: Axes):
    """Takes in a pd Series and some cyclicity determined elsewhere and plots it so that the seasons overlap

    Args:
        series (pd.Series): time-series of type Number
        season_times (ndarray): Indices at which the seasons repeat
        ax (Axes): plot area
    """
    # get the shortest season for initialising array for calculating mean value over seasons
    
    for i in seasons:
        ax.plot(i, c='lightsteelblue', alpha = 0.7)
    ax.plot(season_statistics, c='navy')

def analyse_seasons(series:pd.Series, season_times):
    shortest_season = np.argmin(np.diff(season_times))
    shortest_season_indices = series.loc[season_times[shortest_season]:season_times[shortest_season+1]].index - season_times[shortest_season]
    sum = np.zeros(len(shortest_season_indices))
    seasons = []
    season_number = len(season_times)-1
    truncated_seasons = np.zeros((season_number, len(shortest_season_indices))) # to store truncated seasons to do quick statistical analysis
    for i in range(season_number):
        df_season = series.loc[season_times[i]:season_times[i+1]]
        sum += df_season.values[:len(sum)]
        df_season.index -= season_times[i]
        seasons.append(df_season)
        truncated_seasons[i,:] = df_season.values
    mean = sum/season_number
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
    fig = plt.figure()
    ax = fig.subplots()
    series = loaded_dict[dict_key][table_key]
    seasons, mean_season = analyse_seasons(series, season_times)
    plot_seasons(seasons, mean_season, ax)
    plt.show()
    
def main():
    # TODO interactive file loading. Might make problem with opening multiple files moot, as you would just select multiple
    file_location = "\Dokumente\Privat\Plasway\Al2O3 Process Data"
    files = glob.glob(file_location+"\*.csv")
    skiprows = np.concatenate([np.arange(0,5), np.arange(7,24)])
    store = pd.HDFStore("good_process_spectra.h5")
    # process_store(files, skiprows, store) # only for loading a new file!
    with open('non_spectra.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    print("Opened file: {}".format(os.path.basename(files[0])))        
    SetPoint14 = loaded_dict[7110]["reaPositionSetPoint (14)"]
    with warnings.catch_warnings(action="ignore"):
        season_times = find_seasons(SetPoint14)
    while True:
        print("available measurement series sizes: {}".format(list(loaded_dict.keys())))
        dict_key = int(input("Which series do you want to consider? "))
        try: 
            loaded_dict[dict_key]
        except KeyError:
            print("Please choose a valid key.")
            continue
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
