import glob
import math
import os
from pathlib import Path
import warnings
import pickle
import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
import pandas as pd

plt.rcParams.update({'figure.autolayout': True})
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linewidth"] = 0.75
def loadfiles(file, skiprows):
    """
    load all files in the designated path into a very big df
    """
    df = pd.read_csv(file, skiprows=skiprows, engine="c", header=0, na_values=[' ', 'EOF'], dtype=np.float32)
    return df

def process_csv(filepath, skiprows, target_folder):
    """
    file: absolute path to file to read
    If store is still the full dataset as loaded, this will split it into a part dedicated to the spectrum and a much
    smaller part dedicated to all other variables, which are also grouped according to their observation interval. 
    should do nothing, if store was already processed
    """
    target_filename = os.path.join(target_folder,f"{Path(filepath).stem}.pkl")
    # Check if theres a folder for the analysis exists already, if not make one
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    # Check if this file was already read in, if yes, skip this whole mess
    if os.path.isfile(target_filename):
        print("File already loaded")
        return target_filename
    df = loadfiles(filepath, skiprows)  # TODO allow for reading of multiple files?
    # see if the df has a spectrum part
    spectrum = True
    try:
        df.columns.get_loc("aWavelengthIntensity ")
    except KeyError:
        spectrum = False
    if spectrum:
        # Move the Spectrum to its own df/store
        df = process_spectrum(df, filepath, target_folder)
    # group observables according to measurement intervals
    datasets = {} # list of all different lengths of datasets
    # structure: {7000: n*7000 df, 70000: n*70000 df, ...}
    for i in range(0, len(df.columns), 2):
        df_col = df.iloc[:,i+1]
        df_col_timestamps = df.iloc[:,i]
        df_col.index = df_col_timestamps - df_col_timestamps[0]   # set timestamps as indices instead
        df_col.dropna(inplace=True)
        if np.all(df_col.values < 0.001):    # If all values are 0, skip adding that column to the data
            continue
        if df_col.size in datasets:
            datasets[df_col.size] = pd.concat([datasets[df_col.size], df_col], axis=1) # add the column to the dataframe of that length
        else:
            datasets[df_col.size] = pd.DataFrame(df_col) # Initialize the dictionary entry
    # save data to files so the small part can be easily accessed
    
    with open(target_filename, 'wb') as pickle_file:
        pickle.dump(datasets, pickle_file)
    return target_filename

def process_spectrum(df:pd.DataFrame, file, target_folder):
    """Cleans up the spectrum file somewhat (in the process halfing the size of the file,
    because that is terribly memory inefficient and stores the utterly redundant information about 
    pixel-wavelength assignment at every timestep) and moves it into its own store. Drops those files from the spectrum

    Args:
        df (_type_): _description_
        store (_type_): _description_
    """
    store = pd.HDFStore(os.path.join(target_folder, f"{Path(file).stem}.h5"))
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
    store.close()
    df.drop(df.columns[spec_intensities_start - 1:spec_wavelengths_end], axis=1, inplace=True)
    return df

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
    seasonal_indices = []
    for i, autocorr_column in enumerate(autocorr.T):
        seasonal_indices_new = signal.find_peaks(autocorr_column, distance = min_distance, width=min_distance/3)[0] # find the indices at the peaks of the autocorrelation
        if len(seasonal_indices_new) > len(seasonal_indices):   # The most seasons are likely the most accurate
            x = signal.find_peaks(df.iloc[:, i], distance=min_distance, width=min_distance/5)[0]
            seasonal_indices = (x[1] - seasonal_indices_new[1]) + seasonal_indices_new      # Hopefully the second peak is less influcenced by startup weirdness
    return df.index[seasonal_indices[2:-2]] # get into the bulk data


class Process:
    """Very prototype class for managing Process data
    """
    def __init__(self, loaded_dict:typing.Dict[int, pd.DataFrame], season_times, name, filename) -> None:
        self.loaded_dict = loaded_dict
        self.name = name
        self.filename = filename
        self._season_times = season_times
    
    @classmethod
    def from_pkl(cls, filename:str):
        """If Process data is given as a pkl file (that must have been created using this program), load that 

        Args:
            filename (str): the name of the pkl file with suffix!

        Returns:
            _type_: class Process with dict and season times
        """
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
        
        order_df_key = min(list(loaded_dict.keys()))       
        season_gauge = loaded_dict[order_df_key]
        with warnings.catch_warnings(action="ignore"):
            season_times = find_seasons(season_gauge)
        process1_dict = cls(loaded_dict, season_times, Path(filename).stem, filename)
        return process1_dict   

    def save_to_pkl(self):
        """Saves changes made to dictonary into the associated pickle file
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self.loaded_dict, f)
    
    def add_ratio_entry(self, table_key1:str, table_key2:str, name:str):
        """
        computes table_key1/table_key2 of two columns of identical length and adds that as a new column
        TODO maybe add support for different length series
        """
        dict_key1 = self.find_dict_key(table_key1)
        dict_key2 = self.find_dict_key(table_key2)
        if dict_key1 != dict_key2 or not dict_key1:
            print("Must be two columns with the same lengths in the dataframe!")
            return
        series1 = self.loaded_dict[dict_key1][table_key1]
        series2 = self.loaded_dict[dict_key2][table_key2]
        self.loaded_dict[min(dict_key1, dict_key2)].loc[:, name] = series1/series2
        print(len(self.loaded_dict[min(dict_key1, dict_key2)].loc[:, name]))


    def print_entries(self):
        """Prints a dict with all the names of observations in that process file

        Returns:
            dict: Dict of available measurements
        """
        entry_dict = {}
        for key in self.loaded_dict:
            entry_dict[key] = list(self.loaded_dict[key].columns)
        for key in entry_dict:
            observations = entry_dict[key]
            observations_str = ', '.join(map(str, sorted(observations)))
            print("Observations of length {}: {}".format(key, observations_str))
               
    def find_dict_key(self, table_key):
        """Looks for a table key in the entirety of the data of process, returns first found instance or none if not in class

        Args:
        process (Process): _description_
        table_key (_type_): _description_

        Returns:
        _type_: _description_
        """
        for key in self.loaded_dict:
            if table_key in self.loaded_dict[key]:
                return key
        return None

    def analyse_seasons(self, dict_key, table_key):
        """Use the season time-stamps to structure the data into seasons and do some statistical analysis regarding seasons

        Args:
            series (pd.Series): a specific series of measurement values to be analyzed
            season_times (_type_): a series of time indices (that have to also appear in series' index) that should ideally sort the data into similar chunks

        Returns:
            _type_: a list of seasons and a df with statistical information
        """
        series = self.loaded_dict[dict_key][table_key]
        shortest_season = np.argmin(np.diff(self._season_times))
        # shortest_season_indices_new = series.iloc[0:np.min(np.diff(self._season_times))].index TODO make this readable
        shortest_season_indices = series.loc[self._season_times[shortest_season]:self._season_times[shortest_season+1]].index - self._season_times[shortest_season]
        # print(shortest_season_indices_new-shortest_season_indices)
        # Include the non-seasoned data as pre-and postlude
        pre_run = series.loc[series.index[0]:self._season_times[0]]
        pre_run.index -= self._season_times[0]
        post_run = series.loc[self._season_times[-1]:series.index[-1]]
        post_run.index -= self._season_times[-1]
        seasons = [pre_run]
        season_number = len(self._season_times)-1
        truncated_seasons = np.zeros((season_number, len(shortest_season_indices))) # to store truncated seasons to do quick statistical analysis
        for i in range(season_number):
            df_season = series.loc[self._season_times[i]:self._season_times[i+1]]
            df_season.index -= self._season_times[i]
            seasons.append(df_season)
            truncated_seasons[i,:] = df_season.values[:len(shortest_season_indices)]
        mean = np.mean(truncated_seasons, axis=0)
        seasons.append(post_run)
        analysis_df = pd.DataFrame(data=mean, index=shortest_season_indices)
        deviation_from_mean = truncated_seasons - mean
        var = np.sqrt(np.sum(deviation_from_mean**2, axis=1)/season_number) # not quite the variance, quantifies the difference of the whole season from the mean
        return seasons, analysis_df
        
    def display_data(self, table_key:str, ax = None):
        """Somewhat prototypical action on interaction: plot and (as yet unreturned) data-analysis

        Args:
            dict_key (int): which entry of the dictionary one wants to look at
            table_key (_type_): which column of the data frame is to be plotted
        """
        dict_key = self.find_dict_key(table_key)
        seasons, mean_season = self.analyse_seasons(dict_key, table_key)
        if not ax:
            fig = plt.figure()
            ax = fig.subplots()
            ax.set_xlabel("t/ms")
            ax.set_title(table_key)
            ax.set_xlim(mean_season.index[0] - 20, mean_season.index[-1] + 20)
        ax.set_prop_cycle(plt.cycler("color", plt.cm.cool(np.linspace(0.01,1,len(seasons)))))  
        for s in seasons:
            ax.plot(s, alpha=0.5)
        ax.plot(mean_season, c='navy', label="Mean of seasons")
        ax.legend()

def pkl_from_csv(filepath:str, target_folder:str):
    """If Process data is not yet read in, parse the csv containing the data, create a pkl file

    Args:
        filepath (str): the path to the folder with the csv file

    Returns:
        _type_: class Process with dict and season times
    """
    skiprows = np.concatenate([np.arange(0,5), np.arange(7,24)])
    process_csv(filepath, skiprows, target_folder)

def compare(table_keys, processes:typing.List[Process]):
    """Compares the measurements in different processes. 

    Args:
        table_keys: list of strings
        processes: list of process
    """
    n = len(table_keys)
    plotcount = int(n*(n-1)/2) + 1 # vestigial from when all comparisons were plotted into the same figure
    fig = plt.figure(figsize = (7,7))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("Full data for {}".format(*table_keys))
    means = []
    color_maps = [plt.cm.cool, plt.cm.magma, plt.cm.summer, plt.cm.bone]
    minimum = np.infty
    maximum = - np.infty
    for i in range(len(table_keys)):
        dict_key = processes[i].find_dict_key(table_keys[i])
        if not dict_key:
            print("Measurement not in dataset!")
            return
        _seasons, _mean= processes[i].analyse_seasons(dict_key, table_keys[i])
        means.append(_mean)
        # Reset color-cycling to fresh cool scale
        gradient = np.linspace(0.01,1,len(_seasons))
        ax1.set_prop_cycle(plt.cycler("color", color_maps[i](gradient)))
        ax_colormap = inset_axes(ax1, width="30%", height="2%", loc=("lower center"), borderpad=1.5+i*1.1)   # Inset that shows the cmaps for the cycle data. Must be located in a center coordinate so that the border hack works
        ax_colormap.imshow(np.vstack((gradient, gradient)), aspect="auto", cmap=color_maps[i])
        ax_colormap.text(-0.01, 0, processes[i].name, va='bottom', ha='right', fontsize=10, transform=ax_colormap.transAxes)
        if i == 0: 
            ax_colormap.axes.get_yaxis().set_visible(False) # only the lowest bar gets the bottom axis
            ax_colormap.grid(False)
        else: ax_colormap.set_axis_off()
        for s in _seasons[1:-1]:
            ax1.plot(s, alpha=0.2)
            if min(s) < minimum: minimum = min(s)
            if max(s) > maximum: maximum = max(s)
    # set color cycling to greens for plotting the means
    ax1.set_prop_cycle(plt.cycler("color", plt.cm.Greens(np.linspace(0.2,0.8,len(means)))))
    for process, mean in zip(processes, means):
        ax1.plot(mean, label=f"Mean of {process.name}")
        
    ax1.legend(loc=("upper center"))
    ax1.set_xlim(min(_mean.index) - 100, max(_mean.index) + 100)
    ax1.set_xlabel("t/ms")
    ax1.set_ylim(minimum - 0.2*(maximum-minimum), maximum + 0.1*(maximum-minimum))
    fig.tight_layout()
    for i in range(len(means)):
        for j in range(len(means)):
            if j > i:
                mean_i = means[i]
                mean_j = means[j]
                # Synchornise the means so that the peaks overlap? 
                mean_i_max_at = mean_i.idxmax(axis=0).values[0]
                mean_j_max_at = mean_j.idxmax(axis=0).values[0]
                mean_j.reindex(index = np.roll(mean_j.index, int((mean_i_max_at-mean_j_max_at)/(mean_j.index[1]- mean_j.index[0]))))
                _fig = plt.figure()
                _ax = _fig.add_subplot(1, 1, 1)
                comparison_plot = mean_i - mean_j
                comparison_plot.dropna()
                _ax.plot(comparison_plot, label=f"Mean of {processes[i].name} - mean of {processes[j].name}")
                if table_keys[i]!=table_keys[j]:
                    _ax.set_title(f"Difference in {table_keys[i]} and {table_keys[j]}")
                else:
                    _ax.set_title(f"Difference in {table_keys[i]}")
                _ax.set_xlabel("t/ms")
                _ax.legend()
                _fig.tight_layout()
    plt.show()

def enter_file(extension:str, file_location=None):
    """Gets user to provide a filepath of chosen type in chosen folder

    Args:
        suffix (str): file extension, with dot
        file_location (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    #D:\Local\Analysis\202402 Al2O3
    if not file_location:
        file_location = input("Specify folder path: ")
    try:
        glob.glob(file_location+f"\\*{extension}")[0]
    except IndexError:
        print("No valid files at this path!")
        return enter_file(extension, None)
    files = glob.glob(file_location+f"\\*{extension}")
    print("Available files:")
    for i in sorted(files):
        print(os.path.basename(i))
    filepath = os.path.join(file_location, input("Which file do you want to load? "))
    while not filepath in files:
        print("Please choose a file in the path")
        filepath = os.path.join(file_location, input("Which file do you want to load? "))
    return filepath

def enter_table_key(process:Process):
    """Helps user choose the observation they want to open in process

    Args:
        process (Process): _description_

    Returns:
        _type_: _description_
    """
    while True:
        print(f"available measurements in {process.name}: ")
        process.print_entries()
        table_key = input("Which observation do you want to consider? ")
        if not process.find_dict_key(table_key):
            print("Please input a valid measuremnt from the list (without quotation marks): ")
            continue
        break
    return table_key

def main():
    """
    Interesting comparisons: Input MFC needle ("reaPosSetpoint (34)"), Output MFC needle("reaPosSetpoint (30)"), O2Flow, Pressure, oxygen/argon "reaActFlo(35)" flow ratio
    """
    loadpath = None # path to pkl file folder
    loadnames = [] # path to pkl file
    filepath = None # path to csv file
    processes: typing.List[Process]  = []
    while input("Do you want to read a new file?[y/n]" ) == 'y':
        if filepath: folder = os.path.dirname(filepath) 
        else: folder = None
        filepath = enter_file(".csv", file_location=folder)
        pkl_from_csv(filepath, target_folder=os.path.join(os.path.dirname(filepath), "processed_data"))
    
    # loading a pkl file into the program
    loadpath=r"D:\Dokumente\Privat\Plasway\Al2O3 Process Data\processed_data"   # use standard path
    loading = "y"
    if filepath:
        loadpath = os.path.join(os.path.dirname(filepath), "processed_data")
        print(f"Loading files from {loadpath}")
    while loading == "y":
        loadname = enter_file(".pkl", file_location=loadpath)
        loadnames.append(loadname)
        print(f"Loaded files: {[os.path.basename(name) for name in loadnames]}")
        loadpath = os.path.dirname(loadname)
        loading = input("Load another file?[y/n]")
    for i in loadnames:
        processes.append(Process.from_pkl(i))
    
    # Main control loop: what to do with the files
    
    # Adding ratios
    for process in processes:
        while input(f"Add any ratios to {process.name}? [y/n]") == "y":
            print("Numerator")
            table_key1 = enter_table_key(process)
            print("Denominator (must be of identical length)")
            table_key2 = enter_table_key(process)
            name = input("Name for new dataframe: ")
            process.add_ratio_entry(table_key1, table_key2, name)
    while True:
        # Creating comparison plots
        table_keys = []
        print("Comparing Measurements")
        for process in processes:
            table_keys.append(enter_table_key(process))
        compare(table_keys, processes)
        if input("End program? [y/n] ") == 'y':
            break

def testing_compare():
    processes = []
    path = r"D:\Local\Analysis\202402 Al2O3\processed_data"
    names = [os.path.join(path, "20240305_2_Al2O3+ExtraBias.pkl"), os.path.join(path, "20240305_1_Al2O3_REP.pkl"), os.path.join(path, "20240305_1_Al2O3_REP.pkl")]
    for name in names:
        processes.append(Process.from_pkl(name))
    for process in processes:    
        process.print_entries()
    compare(["reaPositionSetPoint (30)", "reaPositionSetPoint (30)", "reaPositionSetPoint.1"], processes)
    
def testing_add_col():
    path = r"D:\Local\Analysis\202402 Al2O3\processed_data"
    name = os.path.join(path, "20240305_2_Al2O3+ExtraBias.pkl")
    process = Process.from_pkl(name)
    process.print_entries()
    process.add_ratio_entry("reaSetFlow (13)", "reaSetFlow (19)", "13 to 19")
    process.print_entries()
    process.display_data("13 to 19")

testing_compare()
