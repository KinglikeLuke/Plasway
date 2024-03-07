using DataFrames
using Glob
using Serialization
using Plots
using StatsBase
using Peaks


function loadfiles(file, skiprows)
    df = CSV.File(file, header=true, allowmissing=true, normalizenames=true, silencewarnings=true, typemap=Dict(:aWavelengthIntensity => Float32)).Data
    return df
end

function process_store(files, skiprows, store)
    df = loadfiles(files[1], skiprows)
    try
        findfirst(x -> x == "aWavelengthIntensity ", names(df))
    catch
        return
    end
    
    spec_intensities_start = findfirst(x -> x == "aWavelengthIntensity ", names(df))
    spec_wavelengths_start = findfirst(x -> x == "aWavelengthRange ", names(df))
    spec_wavelengths_end = findfirst(x -> x == "O2Intnsty", names(df)) - 1
    
    spec_intensities = df[:, (spec_intensities_start + 1):(spec_wavelengths_start - 1)]
    spec_wavelengths = df[!, spec_wavelengths_start:spec_wavelengths_end]
    
    col_labels = Symbol.(names(spec_wavelengths))
    index_labels = df[!, spec_intensities_start].data
    
    spec_intensities = DataFrame(spec_intensities, Symbol.(col_labels), index_labels)
    store["df"] = spec_intensities
    
    select!(df, Not(names(df))[spec_intensities_start:(spec_wavelengths_end + 1)])
    
    datasets = Dict{Int, DataFrame}() # list of all different lengths of datasets
    
    for i in 1:2:size(df, 2)
        df_col = df[!, i + 1]
        if all(iszero, df_col)    # If all values are 0, skip adding that column to the data
            continue
        end
        
        df_col_timestamps = df[!, i]
        df_col_timestamps = df_col_timestamps.data
        
        df_col = DataFrame(df_col, :auto)
        df_col[!, 1] = df_col_timestamps    # set timestamps as indices instead
        dropmissing!(df_col)
        
        if size(df_col, 2) in keys(datasets)
            datasets[size(df_col, 2)] = vcat(datasets[size(df_col, 2)], df_col) # add the column to the dataframe of that length
        else
            datasets[size(df_col, 2)] = df_col # Initialize the dictionary entry
        end
    end
    
    # save data to files so the small part can be easily accessed
    Serialization.serialize("non_spectra.jls", datasets)
end

function find_seasons(series)
    min_time = 200 # minimum accepted cycle length in ms
    autocorr = autocor(series, 1:length(series))
    delta_t = series[2] - series[1]
    min_distance = min_time / delta_t
    pk, vals = findmaxima(autocorr)
    pk = pk[vals>0.95]
    pk, proms = peakproms!(pk, autocorr; minprom=0)
    pk, widths, _, _ = peakwidths!(pk, ydata, proms; minwidth=min_distance)
    seasonal_indices = 
    return seasonal_indices
end

function plot_seasons(seasons, season_statistics, ax)
    for i in seasons
        plot!(ax, i, c=:lightsteelblue, alpha=0.7)
    end
    plot!(ax, season_statistics, c=:navy)
end

function analyse_seasons(series, season_times)
    shortest_season = argmin(diff(season_times))
    shortest_season_indices = series[season_times[shortest_season]:season_times[shortest_season + 1]].index .- season_times[shortest_season]
    sum = zeros(length(shortest_season_indices))
    seasons = []
    season_number = length(season_times) - 1
    truncated_seasons = zeros(Float32, season_number, length(shortest_season_indices)) # to store truncated seasons to do quick statistical analysis
    
    for i in 1:season_number
        df_season = series[season_times[i]:season_times[i + 1]]
        sum .+= df_season.data[1:length(sum)]
        df_season.index .-= season_times[i]
        push!(seasons, df_season)
        truncated_seasons[i, :] .= df_season.data
    end
    
    mean = sum / season_number
    analysis_df = DataFrame(Mean=mean, Index=shortest_season_indices)
    
    deviation_from_mean = truncated_seasons .- mean'
    var = sqrt(sum(deviation_from_mean .^ 2, dims=2) / season_number) # not quite the variance, quantifies the difference of the whole season from the mean
    return seasons, analysis_df
end

function display_data(loaded_dict, dict_key, table_key, season_times)
    series = loaded_dict[dict_key][table_key]
    seasons, mean_season = analyse_seasons(series, season_times)
    
    plot_seasons(seasons, mean_season, plot())
    display(plot!())
end

function main()
    file_location = "/Dokumente/Privat/Plasway/Al2O3 Process Data"
    files = glob.glob(file_location * "/*.csv")
    skiprows = vcat(1:5, 8:24)
    store = Dict{Any, Any}()
    
    if !isfile("non_spectra.jls")
        process_store(files, skiprows, store)
    else
        store = Serialization.deserialize("non_spectra.jls")
    end
    
    println("Opened file: ", basename(files[1]))
    
    SetPoint14 = store[7110]["reaPositionSetPoint (14)"]
    season_times = find_seasons(SetPoint14)
    
    while true
        println("available measurement series sizes: ", keys(store))
        dict_key = parse(Int, readline("Which series do you want to consider? "))
        
        if !haskey(store, dict_key)
            println("Please choose a valid key.")
            continue
        end
        
        println("Measurements in this series:")
        for i in sort(keys(store[dict_key]))
            println(i)
        end
        
        table_key = readline("Which measurement interests you? (or 'return' to go back to series selection) ")
        
        if table_key == "return"
            continue
        end
        
        while true
            if !haskey(store[dict_key], table_key)
                table_key = readline("Please input a valid measurement from the list (without quotation marks): ")
                continue
            end
            break
        end
        
        display_data(store, dict_key, table_key, season_times)
        
        end_program = lowercase(readline("End program? [y/n] "))
        if end_program == "y"
            break
        end
    end
end

main()