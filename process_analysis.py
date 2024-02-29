import pandas as pd
import glob
import numpy as np
import pickle

file_location = "\Dokumente\Privat\Plasway\Al2O3 Process Data"
files = glob.glob(file_location+"\*.csv")
skiprows = np.concatenate([np.arange(0,5), np.arange(7,24)])
store = pd.HDFStore("good_process.h5")

def loadfiles(files):
    for file in files:
        df = pd.read_csv(file, skiprows=skiprows, engine="c", header=0, na_values=[' '], dtype=np.float64)
        #df.dropna(inplace=True)
        store["df"] = df

def process_store():
    """
    If store is still the full dataset as loaded, this will split it into a part dedicated to the spectrum (an in the process halfing the size of the file,
    because that is terribly memory inefficient and stores the utterly redundant information about pixel-wavelength assignment at every timestep) and a much
    smaller part dedicated to all other variables, which are also grouped according to their observation interval. 
    should do nothing, if store was already processed
    """
    df = store["df"]
    # see if the df was processed already
    try:
        df.columns.get_loc("aWavelengthIntensity ")
    except KeyError:
        return
    # Move the Spectrum to its own df's
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
        df_col.dropna(inplace=True)
        df_col_timestamps = df.iloc[:,i]
        df_col_timestamps.dropna(inplace=True)
        if df_col.size in datasets:
            datasets[df_col.size] = pd.concat([datasets[df_col.size], df_col], axis=1) # add the column to the dataframe of that length
        else:
            datasets[df_col.size] = pd.concat([df_col_timestamps, df_col], axis=1) # Initialize the dictionary entry with the timescale specific to that length
    # save data to files so the small part can be easily accessed
    with open('non_spectra.pkl', 'wb') as pickle_file:
        pickle.dump(datasets, pickle_file)

process_store()