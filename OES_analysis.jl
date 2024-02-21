using Plots
using LsqFit
using Random
using CalculusWithJulia
using Peaks
using DelimitedFiles
using DataFrames
using Profile
using CSV

function testdata_lorentzian(n_peaks, length=200, mean_I=1, mean_fwhm=2, mean_off=1)
    """
    Generate a random lorentzian spectrum modified by random noise with the given Parameters
    mean_fwhm: how broad the peaks are supposed to be
    """
    x_max = 100
    x = Vector{Float64}(range(0, x_max, length))
    off = (randn().+mean_off).*mean_off/5
    x0 = rand(1:x_max, n_peaks)
    I = (randn(n_peaks).+ 1).*mean_I .+ mean_I .+ 1
    fwhm = randn(n_peaks).*mean_fwhm/5 .+mean_fwhm
    params = vcat(off, x0, I, fwhm)
    ydata = multi_peak_func(lorentzian, x, params) + randn(length).*0.1
    return Spectrum(ydata, x)
    # testing = FindPeaks(y_data)
    # assert testing.i_x0 - x0 < fwhm
end

function lorentzian(x, x0, a, gam)
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
        return ((a * gam ^ 2) ./ ((x .- x0) .^ 2 .+ gam^2))
end

function multi_peak_func(shape::Function, x, params)
    """
    Sums over multiple peak shapes to get spectrum
    """
    off = params[1]
    # need to unpack or find prettier way to unpack list in for loop
    block_len = floor(Int, (length(params)-1)/3)
    params_x0 = params[2:block_len+1]
    params_I = params[block_len+2:2*block_len+1]
    params_fwhm = params[2*block_len+2:end]
    return off .+ sum([shape(x, i_params_x0, i_params_I, i_params_fwhm) for 
    (i_params_x0, i_params_I, i_params_fwhm) = zip(params_x0, params_I, params_fwhm)])
    
end

struct Spectrum{T}
    ydata::Vector{T}
    xdata::Vector{T}
end

function fit_peaks(func::Function,  data::Spectrum, guess)
    _multi_peak_func(x, params) = multi_peak_func(func, x, params)
    fit = curve_fit(_multi_peak_func, data.xdata, data.ydata, guess)
    return fit.param
end

function initial_guess(data::Spectrum, heightfactor, widthfactor)
    """
    achieve initial guess of peak positions and parameters for fit, using local maxima of the data 
    """
    min_prom = heightfactor * sum(abs.(diff(data.ydata)))/length(data.ydata)
    min_width = widthfactor
    pk, vals = findmaxima(data.ydata)
    pk, proms = peakproms!(pk, data.ydata; minprom=min_prom)
    pk, widths, _, _ = peakwidths!(pk, data.ydata, proms; minwidth=min_width)
    heights = data.ydata[pk]
    pk = data.xdata[pk]
    guess = vcat(minimum(data.ydata), pk, heights, widths)
    return guess
end

function plot_results(func, data::Spectrum, popt, yground, yamp)
    npixel = 2000
    x_plot = range(data.xdata[1], data.xdata[end], npixel)
    p1 = scatter(data.xdata, data.ydata, ms=0.8)
    p2 = scatter(data.xdata, yground .+ yamp .* data.ydata, ms=0.8)
    x0 = popt[2:Int((length(popt)-1)/3)+1]
    print(x0)
    if length(popt) > 1    
        x0_fit = zeros(0)
        plot_fit = zeros(0)
        for x_i in x0
            push!(x0_fit, multi_peak_func(func, x_i, popt))
        end
        for x_i in x_plot
            push!(plot_fit, yground + yamp * multi_peak_func(func, x_i, popt))
        end
        scatter!(p1, x0, x0_fit)
        plot!(p2, x_plot, plot_fit, label="fit")
        display(plot(p1, p2, layout=(2,1), legend=false))
        return
    end
    display(plot(p1, p2, layout=(2,1)))
    return
end

function cleandata(data::Matrix)
    """
    Data has uniteresting down time, which gets removed due to its lower intensity
    Ground is also removed for better contrast in plot
    """

    ground = minimum(data, dims=1)
    _data = data .- ground
    n = 10  # how many interesting points are necessary for spectrum to be interesting.
    average = sum(_data, dims=1)/size(_data)[1]
    n_maxima_average = zeros(1, size(_data)[2]) # same dim as average
    for (i, col) in enumerate(eachcol(_data))
        n_maxima = sort(col,rev=true)[1:n]
        n_maxima_average[i] = sum(n_maxima)/n
    end
    print(size(n_maxima_average))
    threshold = 10  # find way to distinguish random noise from spectrum!!
    uniformitycheck = vec(n_maxima_average./average .> threshold) # check which columns exceed the threshold
    _data = _data[:,uniformitycheck]
    
    return _data, uniformitycheck
end

function extractcycles(data, uniformitycheck)
    """
    extracts averages of the active cycles. 
    data: the uncleaned data!
    """
    indices = findall(!iszero, uniformitycheck[2:end] .!= uniformitycheck[1:end-1]) .+ 1 # extract periods of falses and trues
    indices = vcat(1, indices, length(uniformitycheck))
    # indices = remove_close_integers(indices)
    cycle_averages = zeros(size(data)[1],0)
    for i in range(1, length(indices)-1)
        if uniformitycheck[1]==1 && i%2!=0 # check whether the first or second block is the active one
            cycle_averages = average_cycles(data, cycle_averages, (indices[i], indices[i+1]))
        elseif  uniformitycheck[1]==0 && i%2==0
            cycle_averages = average_cycles(data, cycle_averages, (indices[i], indices[i+1]))
        end
    end
    return cycle_averages
end

function average_cycles(data, cycle_averages, index::Tuple{Int64, Int64})
    cycle = data[:,index[1]:index[2]]
    if size(cycle)[2]>2
        return hcat(cycle_averages, sum(cycle, dims=2)./size(cycle)[2]) # compute the rowwise average within one cycle
    end
    return cycle_averages
end

function remove_close_integers(arr)
    result = Int[]
    
    for i in eachindex(arr)
        push!(result, arr[i])
        
        if i > 1 && abs(arr[i] - arr[i-1]) < 2
            pop!(result)
        end
    end
    
    return result
end

function peak_analysis(func::Function, cycles, xdata, heightfactor, widthfactor)
    peaktable=[]
    spec1 = Spectrum(cycles[:, 1], xdata)
    _multi_peak_func(x, params) = multi_peak_func(func, x, params)
    jacobian = (outjacin,p) -> ForwardDiff.jacobian!(outjacin, _multi_peak_func!, out, p)
    guess = initial_guess(spec1, heightfactor, widthfactor) # inital peak positions that get passed thorugh every fit. conserves peak number and hopefully assignment
    for i in range(1, size(cycles)[2])
        analysis = curve_fit(_multi_peak_func, jacobian, xdata, cycles[:, i], guess)
        analysis = analysis.param
        block_len = Int((length(analysis)-1)/3)
        plot_results(lorentzian, spec, analysis, 0, 1)
        df = DataFrame(x0 = analysis[2:block_len+1], I = analysis[block_len+2:2*block_len+1])
        push!(peaktable, df)
        break
    end
    return peaktable
end

# replace this some time
function peak_analysis_sparse(cycles, xdata, heightfactor, widthfactor)
    peaktable=[]
    spec1 = Spectrum(cycles[:, 1], xdata)
    # jacobian = (outjacin,p) -> ForwardDiff.jacobian!(outjacin, f!, out, p)
    guess = initial_guess(spec1, heightfactor, widthfactor) # inital peak positions that get passed thorugh every fit. conserves peak number and hopefully assignment
    for i in (1, size(cycles)[2])
        print(i)
        spec = Spectrum(cycles[:, i], xdata)
        analysis = fit_peaks(lorentzian, spec, guess)
        block_len = Int((length(analysis)-1)/3)
        df = DataFrame(x0 = analysis[2:block_len+1], I = analysis[block_len+2:2*block_len+1])
        push!(peaktable, df)
    end
    return peaktable
end

function visualize_data(data; limit::Int64, clean::Bool, normalise::Bool)
    """
    Make data more legible for heatmap plot
    """
    display_data = data .- minimum(data, dims=1)
    if clean
        display_data, uniform = cleandata(display_data)
    end
    if normalise
        bin, wavelength_uniformity = cleandata(permutedims(data, (2,1)))
        display_data = display_data ./ maximum(display_data, dims=2)
        display_data[.!wavelength_uniformity,:].=0
    end
    if limit != 0
        display_data[data > limit] = limit
    end

    return display_data
end

function analyse_folder()
    names = readdir("Data", join = true)
    filenames = readdir("Data", join=false)
    output_folder = "results"
    try readdir(output_folder) catch e mkdir("results") end
    for name in names[5:end]
        data = readdlm(name, ',', Float64, skipstart=25) # some oes data
        x_data = data[:,2]
        data = data[:,5:end]
        cleaned_data, uniformity = cleandata(data)
        display_data = visualize_data(data, limit=0, clean=true, normalise=true)
        display(heatmap(display_data, c=:heat))
        #cycles = extractcycles(data, uniformity)
        #peaks_data = peak_analysis_sparse(cycles, x_data, 3, 3)
        #refined_peaks = DataFrame(x0 = vec(peaks_data[1].x0), I_ratio=vec(peaks_data[1].I./peaks_data[2].I), 
        #δx0 = vec(peaks_data[1].x0.-peaks_data[2].x0))
        
        #savefig(output_folder*"/"*filename[1:end-4]*"_heatmap.png")
        #CSV.write(output_folder*"/"*filename[1:end-4]*"peaks.csv", refined_peaks)
        break
    end
end

analyse_folder()