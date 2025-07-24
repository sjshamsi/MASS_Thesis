import os
import math
import shutil
import pickle
import random
import warnings
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations

from pathlib import Path
from minisom import MiniSom

from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from astropy.cosmology import FlatLambdaCDM

from tslearn.barycenters import dtw_barycenter_averaging
import plotly.io as pio
import plotly.graph_objects as go
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



#Function to plot light curve adapted from Damir's notebook
def plot_lc(lightcurve,
            band=None,
            title='',
            x_label='MJD [days]',
            y_label='Magnitude',
            save_fig=False,
            figure_save_path=None,
            return_fig=False,
        ):
    '''
    Plots light curves interactively. Adapted from https://github.com/DamirBogdan39/time-series-analysis/tree/main
    
    Parameters
    ----------
    Light_Curve: Dataframe 
    The light curve to plot. Should be in a dataframe with mjd (or any x axis), mag and magerr
    
    header: str
    The header of the file
    
    save_fig: bool
    Whether to save the figure
    
    filename: str
    What to name the saved html file
    
    x_axis: str
    What to label the x axis 
    
    return_fig: bool
    Whether the figure is returned
    
    Returns
    --------
    Figure:
    The interactive plot of the light curve
    '''
    fig = go.Figure()
    error_bars = go.Scatter(
        x=lightcurve[:, 0],
        y=lightcurve[:, 1],
        y_err=lightcurve[:, 2],
        mode='markers',
        marker=dict(size=4),
        name='mag with error bars'
     )
    
    fig.add_trace(error_bars)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(yaxis = dict(autorange="reversed"))
    fig.update_layout(title_text=title, showlegend=True)

    fig.show()
    if return_fig:
        return fig



def pad_lightcurves(lightcurve_arrays,
                    time_bins=None,
                    timespan_max=None,
                    bin_width=1, # in days
                    use_shortest_span=False):
    '''
    Pads the light curves with the mean value at the end of the curve. Make sure to use when x is MJDs
    
    Parameters
    ----------
    light_curves: lists of dataframes 
    The light curves stored in a list
    
    minimum_length: int
    The minimum length to pad to
    
    save_padded_lcs: bool
    If True, will save the light curves into a folder known as Padded_Lc in the specified directory
    
    padded_lcs_save_path: str
    The directory to save the light curves in
    
    ids: list of str
    A list of the ids. Must provided in order to save the light curves
    
    Returns
    --------
    light_curves: list of lists
    The new padded light curves
    '''
    time_arrays = [LC[:, 0] - LC[0, 0] for LC in lightcurve_arrays]
    timespan_array = np.array([time_array[-1] - time_array[0] for time_array in time_arrays])
    if timespan_max is None:
        timespan_max = np.min(timespan_array) if use_shortest_span else np.max(timespan_array)
    
    if time_bins is None:
        time_bins = np.arange(0, timespan_max + bin_width, bin_width)

        all_mjds = np.concatenate(time_arrays)
        bin_counts, _ = np.histogram(all_mjds, bins=time_bins)
        i = 0
        while i < len(bin_counts):
            if bin_counts[i] == 0:
                if i + 1 < len(bin_counts):
                    time_bins = np.delete(time_bins, i + 1)
                    bin_counts, _ = np.histogram(all_mjds, bins=time_bins)
                    continue
                else:
                    time_bins = np.delete(time_bins, i)
                    bin_counts, _ = np.histogram(all_mjds, bins=time_bins)
                    i -= 1
                    continue
            i += 1


    common_x = np.arange(0, time_bins.size, 1)
    stacked_magnitudes = np.zeros(shape=(len(lightcurve_arrays), common_x.size))

    for k, (time_array, LC) in enumerate(zip(time_arrays, lightcurve_arrays)):

        bin_indices = np.digitize(time_array, time_bins)
        binned_x = []
        binned_magnitudes = []
        for j in range(1, len(time_bins)):
            mask = bin_indices == j
            if np.any(mask):
                binned_x.append(j - 1)
                mean_magnitude = np.average(LC[:, 1][mask])
                binned_magnitudes.append(mean_magnitude)

        interpolated_mags = np.interp(common_x,
                                      binned_x,
                                      binned_magnitudes,
                                      right=np.mean(binned_magnitudes))
        stacked_magnitudes[k] = interpolated_mags
    return stacked_magnitudes



def scale_curve(lightcurve, scaler=None, scale_times=False):
    '''
    Scaling the curves (from a single filter) from the choice of minmax, standard and robust. By default, it scales to a range of [-1,1]
    Parameters
    ----------
    light_curves: list of dataframes 
    The light curves stored in a list.
    
    what_scaler: string | Scipy scaler Class NOT initialised object
    The type of scaler to use. There are default (see above), standard scaler, min-max scaler and robust scalers available
    
    scale_times: bool
    Whether to scale the time axis as well (These are always scaled to the default scaler)
    
    Returns
    --------
    scaled_curves: np.ndarray 
    The scaled light curves
    
    scaled_times:np.ndarray
    The scaled time steps. It is an empty list if the keyword scale_times is False
    '''
    scaler = MinMaxScaler(feature_range=(-1, 1)) if scaler is None else scaler()
    
    scaled_mags = scaler.fit_transform(lightcurve[:, 1].reshape(-1, 1))
    scaled_mag_errs = lightcurve[:, 2] / (scaler.data_max_ - scaler.data_min_)
    
    if scale_times:
        time_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_times = time_scaler.fit_transform(lightcurve[:, 0])
    else:
        scaled_times = lightcurve[:, 0]

    return np.column_stack((scaled_times, scaled_mags, scaled_mag_errs))



def SOM_1D(data,
           model_save_path,
           n_random_selections=None,
           som_x=None,
           som_y=None,
           use_epochs=False,
           learning_rate=0.1,
           sigma=1.0,
           topology='rectangular',
           neighborhood_function='gaussian',
           pca_init=True,
           random_seed=21,
           num_iterations_per_batch=10_000,
           batch_size=100,
           stat=True,
           verbose=False,
           log_stat_every_n_batch=1,
           early_stopping=True,
           early_stopping_patience=5,
           early_stopping_min_delta=0.05,
        ):
    model_save_dir = os.path.dirname(model_save_path)
    model_filename = os.path.basename(model_save_path).split('.')[0]

    q_error_path = model_save_dir + f'/{model_filename}_q_error.npy'
    t_error_path = model_save_dir + f'/{model_filename}_t_error.npy'
    indices_to_plot_path = model_save_dir + f'/{model_filename}_indices_to_plot.npy'

    if os.path.exists(model_save_path):
        with open(model_save_path, 'rb') as infile:
            som_model = pickle.load(infile)
        q_error = np.load(q_error_path)
        t_error = np.load(t_error_path)
        indices_to_plot = np.load(indices_to_plot_path)
        return som_model, q_error, t_error, indices_to_plot


    n_random_selections = data.shape[0] if n_random_selections is None else n_random_selections
    data = data[np.random.choice(len(data), size=n_random_selections, replace=False)]
    data_len, n_features = len(data), data.shape[1]

    if (som_x is None) or (som_y is None):
        default_som_grid_length = int(np.ceil(np.cbrt(data_len)))
        warnings.warn(f"Both som_x and som_y are not provided. Using recommended grid shape w.r.t to {data_len} datapoints ({default_som_grid_length}x{default_som_grid_length}).")
        som_x, som_y = default_som_grid_length, default_som_grid_length
        
    som_model = MiniSom(som_x, som_y, n_features, learning_rate=learning_rate, sigma=sigma,
                        topology=topology, neighborhood_function=neighborhood_function, random_seed=random_seed)
    som_model.pca_weights_init(data) if pca_init else som_model.random_weights_init(data)


    q_error, t_error, indices_to_plot = [], [], []
    early_stopping_count = 0
    current_error = np.inf

    data_chunks = np.array_split(data, batch_size, axis=0)
    for i, chunk in tqdm(enumerate(data_chunks), desc='Training Batch'):
        som_model.train(chunk, num_iterations_per_batch, verbose=verbose, use_epochs=use_epochs)

        if (stat==True) and ((i % log_stat_every_n_batch) == 0):
            q_error.append(som_model.quantization_error(data))
            t_error.append(som_model.topographic_error(data))
            indices_to_plot.append(i)

            if early_stopping:
                previous_error = current_error
                current_error = q_error[-1]
                early_stopping_count = 0 if ((previous_error - current_error) > early_stopping_min_delta) else early_stopping_count + 1
                
                if early_stopping_count == early_stopping_patience:
                    print('Early Stopping...')
                    break

    with open(model_save_path, 'wb') as f:
        pickle.dump(som_model, f)
    np.save(q_error_path, q_error)
    np.save(t_error_path, t_error)
    np.save(indices_to_plot_path, indices_to_plot)

    with open(os.path.join(model_save_dir, f"{model_filename}_paramas.txt"), "w") as f:
        for key, value in locals().items():
            f.write(f"{key} = {value}\n")

    return som_model, q_error, t_error, indices_to_plot



def plot_training(ax, metric_name, metric_array, indices_to_plot, fig_save_path):
    ax.plot(indices_to_plot, metric_array)
    plt.ylabel(metric_name)
    plt.xlabel('Batch Number')
    
    plt.savefig(fig_save_path)  
    plt.show()



def _plot_neuron_average(ax, neuron_num, series_array=None,
                         plot_background=True, num_background_LCs=500,
                         plot_avg=True, dba=True,
                         plot_weights=True, neuron_weights=True):
    ax.set_title(f"Neuron {neuron_num}: {len(series_array)} Curves")
    ax.legend()
    
    if series_array is None:
        return ax
    
    series_array_subset = random.sample(series_array, num_background_LCs)
    if plot_background:
        for series in series_array_subset:
            ax.plot(series, c="gray", alpha=0.5, label='Neuron Lightcurves')

    if plot_avg and len(series_array) > 0:
        cluster_average_series = dtw_barycenter_averaging(np.vstack(series_array)) if dba else np.mean(np.vstack(series), axis=0)
        average_curve_label = 'Barycentric Average Curve' if dba else 'Mean Curve'
        ax.plot(cluster_average_series, c="blue", label=average_curve_label)
    
    if plot_weights:
        ax.plot(neuron_weights, c='red', label='SOM Representation')

    return ax



def Plot_Neurons_Average_LC(som_model,
                            scaled_curves,
                            figs_save_path,
                            plot_background=True,
                            num_background_LCs=500,
                            plot_avg=True,
                            dba=True,
                            plot_weights=True,
                            show_fig=True):
    '''
    Plotting the SOM Clusters with the average light curve and the SOM weights of each cluster. The average can be either simple mean
    or using a dba averaging method (https://github.com/fpetitjean/DBA)
    
    Parameters
    ----------
    som_model:  
    The trained SOM
    
    scaled_curves: np.ndarray
    The scaled curves that were the input for training
    
    dba: bool
    Whether to use Dynamic Barymetric Time Averaging
    
    figsize: tuple
    The size of the figure
    
    save_figs: bool
    Whether to save the figure or not
    
    fig_save_path: str
    Where to save the figure. Note that it creates a directory called Plots in the location given.
    
    plot_avg: bool
    Whether to plot the mean light curve of the cluster
    
    plot_weights: bool
    Whether to plot the SOM weight of the cluster
    
    plot_background: bool
    Whether to plot the light curves that make up the cluster
    
    one_fig: bool
    Whether to plot all the clusters into one figure or seperate figures
    
    show_fig: bool
    Whether to show each of the plots in the seperate figures case
    
    Returns
    --------
    Plot:
    The plots of the clusters
    '''
    som_x, som_y = som_model.get_weights().shape[:2]
    win_map = som_model.win_map(scaled_curves)

    fig, axs = plt.subplots(nrows=som_y,
                            ncols=som_x,
                            figsize=(som_x * 3, som_y * 3),
                            layout='constrained'
                        )
    fig.suptitle('Lightcurve Association by Neuron')
    axs_flattened = axs.flatten()
    
    neuron_coordinates = [(x, y) for x in range(som_x) for y in range(som_y)]
    for i, (coord, ax) in tqdm(enumerate(zip(neuron_coordinates, axs_flattened)), desc='Clustering Neuron'):
        if coord in win_map.keys():
            series_array = win_map[coord]
            neuron_weights = som_model.get_weights()[coord[0]][coord[1]]
        else:
            series_array, neuron_weights = None, None

        _plot_neuron_average(ax, neuron_num=i + 1, series_array=series_array,
                             plot_background=plot_background, num_background_LCs=num_background_LCs,
                             plot_avg=plot_avg, dba=dba,
                             plot_weights=plot_weights, neuron_weights=neuron_weights)

    plt.savefig(figs_save_path)
    if show_fig: plt.show()
    return fig, axs



def SOM_Distance_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
    '''
    Plots a heatmap of the SOM Nodes. The brighter, the further away they are from their neighbors
    
    Parameters
    ----------
    som_model:  
    The trained SOM
    
    cmap: str
    The matplotlib based color scale to use for the plots
    
    figsize: tuple
    The size of the figure
    
    save_figs: bool
    Whether to save the figure or not
    
    fig_save_path: str
    Where to save the figure. Note that it creates a directory called Plots in the location given.
    
    Returns
    --------
    Plot:
    The heatmap plot
    '''
    plt.figure(figsize = figsize)
    distance_map = som_model.distance_map()
    plt.pcolormesh(distance_map, cmap=cmap,edgecolors='k')
    for i in range(len(distance_map)):
        for j in range(len(distance_map)):
            plt.text(j+0.1,i+0.5,'Clus. {}'.format(i*len(distance_map[i])+j+1))
    cbar = plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/SOM_Distance_Map.png', dpi=600)
    plt.show()
    


def SOM_Activation_Map(som_model,figsize = (5,5),cmap = 'YlOrRd',save_figs = False,figs_save_path = './'):
    '''
    Plots a heatmap of the SOM Nodes. The brighter, the more light curves activate the SOM
    
    Parameters
    ----------
    som_model:  
    The trained SOM
    
    cmap: str
    The matplotlib based color scale to use for the plots
    
    figsize: tuple
    The size of the figure
    
    save_figs: bool
    Whether to save the figure or not
    
    fig_save_path: str
    Where to save the figure. Note that it creates a directory called Plots in the location given.
    
    Returns
    --------
    Plot:
    The heatmap plot
    '''
    plt.figure(figsize = figsize)
    activation_response = som_model.activation_response()
    plt.pcolormesh(activation_response, cmap=cmap,edgecolors='k')
    for i in range(len(activation_response)):
        for j in range(len(activation_response)):
            plt.text(j+0.1,i+0.5,'Clus. {}'.format(i*len(activation_response[i])+j+1))
    cbar = plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/SOM_Activation_Map.png', dpi=600)
    plt.show()
    


def Assign_Cluster_Labels(som_model, scaled_curves, ids):
    '''
    Assigns Cluster labels to each of the curves, making a dataframe with their ids
    
    Parameters
    ----------
    som_model:  
    The trained SOM
    
    scaled_curves: np.ndarray
    The scaled curves that were used to train the SOM
    
    ids: list
    The ids of the curves
    
    Returns
    --------
    cluster_df: Dataframe
    A map matching each of the cluster ids with the cluster they belong to
    '''
    cluster_map = []
    som_y = som_model.distance_map().shape[1]
    for i in tqdm(range(len(scaled_curves)), desc='Creating Dataframe'):
        winner_node = som_model.winner(scaled_curves[i])
        cluster_map.append((ids[i], winner_node[0] * som_y + winner_node[1] + 1))
    clusters_df = pd.DataFrame(cluster_map, columns=["ID", "Cluster"])
    return clusters_df



def SOM_Clusters_Histogram(cluster_map,color = 'tab:blue',save_figs = True,figs_save_path = './',figsize = (5,5)):
    '''
    Plots a heatmap of the SOM Nodes. The brighter, the further away they are from their neighbors
    
    Parameters
    ----------
    cluster_map:  
    The dataframe with each id and the cluster that it belongs to
    
    color: str
    The color to plot the histogram
    
    save_figs: bool
    Whether to save the figure or not
    
    fig_save_path: str
    Where to save the figure. Note that it creates a directory called Plots in the location given.
    
    figsize: tuple
    The size of the figure
    
    Returns
    --------
    Plot:
    The Histogram of how many curves are in each cluster
    '''
    cluster_map.value_counts('Cluster').plot(kind = 'bar',color = color,figsize = figsize)
    plt.ylabel('No of quasars')
    if save_figs:
        if 'Plots' not in os.listdir():
            os.makedirs(figs_save_path+'Plots')
        plt.savefig(figs_save_path+'Plots/Clusters_Histogram.png', dpi=600)
        


def findMin(x, y, umat):
    '''
    Finds the minimum node in the unified matrix when given the x and y coordinate
    
    Parameters
    ----------
    x: int  
    The x position of the given input node
    
    y: int  
    The y position of the given input node
    
    umat: np.ndarry
    The unified distance matrix of the nodes of the SOM
    
    Returns
    --------
    minx, miny:
    The minumum x node and minimum y node
    '''
    #Finds minimum node
    newxmin=max(0,x-1)
    newxmax=min(umat.shape[0],x+2)
    newymin=max(0,y-1)
    newymax=min(umat.shape[1],y+2)
    minx, miny = np.where(umat[newxmin:newxmax,newymin:newymax] == umat[newxmin:newxmax,newymin:newymax].min())
    return newxmin+minx[0], newymin+miny[0]



def findInternalNode(x, y, umat):
    '''
    Finds the minimum node in the unified matrix when given the x and y coordinate, taking into account if the current node is min
    
    Parameters
    ----------
    x: int  
    The x position of the given input node
    
    y: int  
    The y position of the given input node
    
    umat: np.ndarry
    The unified distance matrix of the nodes of the SOM
    
    Returns
    --------
    minx, miny:
    The minumum x node and minimum y node
    '''
    minx, miny = findMin(x,y,umat)
    if (minx == x and miny == y):
        cx = minx
        cy = miny
    else:
        cx,cy = findInternalNode(minx,miny,umat)
    return cx, cy



def Get_Gradient_Cluster(som):
    '''
    Finds the center of the gradient for each node of the SOM
    
    Parameters
    ----------
    som: int  
    The trained SOM
    
    Returns
    --------
    cluster_centers:
    The center nodes that become the new gradient clusters
    
    cluster_pos:
    The original SOM cluster centers
    '''
    cluster_centers = []
    cluster_pos  = []
    for row in np.arange(som.distance_map().shape[0]):
        for col in np.arange(som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, som.distance_map().T)
            cluster_centers.append(np.array([cx,cy]))
            cluster_pos.append(np.array([row,col]))
    return np.array(cluster_centers),np.array(cluster_pos)



def Normal_Cluster_to_Grad(cluster_map, gradient_cluster_map):
    '''
    Maps the normal cluster map to the gradient clusters
    
    Parameters
    ----------
    cluster_map: pd.DataFrame  
    The map of the ids to the original SOM node clusters
    
    gradient_cluster_map: pd.DataFrame  
    The map of the ids to the gradient SOM node clusters
    
    Returns
    --------
    joint_map:
    Mapping of each SOM node cluster to the gradient cluster
    '''
    joint_map = cluster_map.join(gradient_cluster_map,rsuffix='_grad')[['Cluster','Cluster_grad']].groupby('Cluster').mean()
    joint_map.columns = ['Gradient Clusters']
    joint_map.index.name = 'Normal Clusters'
    return joint_map


def Gradient_Cluster_Map(som, scaled_curves, ids,
                         dimension='1D', fill='mean', interpolation_kind='cubic',
                         clusters=None, som_x=None, som_y=None):
    '''
    Translates the SOM nodes into larger clusters based on their gradients. Implementation of 
    https://homepage.cs.uri.edu/faculty/hamel/pubs/improved-umat-dmin11.pdf
    
    Parameters
    ----------
    som_model:  
    The trained SOM
    
    scaled_curves: np.ndarray
    The scaled curves used to train the SOM
    
    ids: list
    The ids of the curves
    
    dimension: str
    If 1D, does 1D clustering, else multivariate
    
    fill: str
    'mean' or 'interpolate'. Either the empty values are filled with the mean or they are interpolated with a function
    
    interpolation_kind:
    Any of the scipy.interp1d interpolation kinds. Recommended to use cubic
    
    clusters:
    The clusters that the ids are in (only for multi-variate)
 
    som_x: int
    The x-dimensions of the SOM
    
    som_y: int
    The y-dimensions of the SOM
 
    Returns
    --------
    cluster_map:
    The new clusters that the ids are in
    '''
    if dimension == '1D':
        cluster_centers, cluster_pos = Get_Gradient_Cluster(som)
    else:
        cluster_centers, cluster_pos = Get_Gradient_Cluster_2D(som, fill, interpolation_kind)
    cluster_numbers = np.arange(len(np.unique(cluster_centers, axis=0)))
    unique_cluster_centers = np.unique(cluster_centers,axis = 0)
    cluster_numbers_map = []
    for i in range(len(scaled_curves)):
        if dimension == '1D':
            winner_node = som.winner(scaled_curves[i])
            winner_node = np.array(winner_node)
        else:
            winner_x = (clusters[i]-1)//som_y
            winner_y = (clusters[i]-1)%som_y
            winner_node = np.array([winner_x,winner_y])
        #Gets the central node where the winning cluster is in
        central_cluster = cluster_centers[np.where(np.isclose(cluster_pos,winner_node).sum(axis = 1) == 2)][0]
        cluster_number = cluster_numbers[np.where(np.isclose(unique_cluster_centers,central_cluster).sum(axis = 1) == 2)]
        cluster_numbers_map.append(cluster_number[0]+1)
    return pd.DataFrame({'ID':ids,'Cluster':cluster_numbers_map})



def matplotlib_cmap_to_plotly(cmap, entries):
    '''
    Creates a colorscale used to create an interactive plot
    
    Parameters
    ----------
    cmap:  
    The colormap
    
    entries: 
    The colormap entries

    Returns
    --------
    colorscale:
    The colorscale for the interactive plot
    '''
    #Used for creating interactive plot
    h = 1.0/(entries-1)
    colorscale = []

    for k in range(entries):
        C = (np.array(cmap(k*h)[:3])*255)
        colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return colorscale



def plotStarburstMap(som):
    '''
    Interactive plot of the distance map and gradients of the SOM
    
    Parameters
    ----------
    som:  
    The trained SOM

    Returns
    --------
    Plot of the distance map and gradients
    '''
    boner_rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    bone_r_cmap = matplotlib.colormaps.get_cmap('bone_r')

    bone_r = matplotlib_cmap_to_plotly(bone_r_cmap, 255)

    layout = go.Layout(title='Gradient Based Clustering')
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Heatmap(z=som.distance_map().T, colorscale=bone_r))
    shapes=[]
    for row in np.arange(som.distance_map().shape[0]):
        for col in np.arange(som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, som.distance_map().T)
            shape=go.layout.Shape(
                    type="line",
                    x0=row,
                    y0=col,
                    x1=cx,
                    y1=cy,
                    line=dict(
                        color="Black",
                        width=1
                    )
                )
            shapes=np.append(shapes, shape)

    fig.update_layout(shapes=shapes.tolist(), 
        width=500,
        height=500) 
    
    fig.show()



def outliers_detection(clusters_df,som,scaled_curves,ids,outlier_percentage = 0.2):
    '''
    Gives the percentage of the clusters that have high quanitization errors (defined by percentile) for each cluster
    
    Parameters
    ----------
    clusters_df:
    A map of each of the ids to the clusters
    
    som:  
    The trained SOM
    
    scaled_curves: np.ndarray
    The scaled curves used to train the SOM
    
    ids: list
    The ids of the curves
    
    outlier_percentage: float
    This top percentile that defines an outlier
 
    Returns
    --------
    Plots:
    Distribution of Outliers per cluster and distribution of quantization error
    '''
    #Detects outliers that aren't quantized well as a percentage of the clusters
    quantization_errors = np.linalg.norm(som.quantization(scaled_curves) - scaled_curves, axis=1)
    error_treshold = np.percentile(quantization_errors, 
                               100*(1-outliers_percentage)+5)
    outlier_ids = np.array(ids)[quantization_errors>error_treshold]
    outlier_cluster = []
    for i in range(len(clus.ID)):
        if str(clus.ID[i]) in outlier_ids:
            outlier_cluster.append(clus.Cluster[i])
    #Plot the number of outliers per cluster
    plt.figure()
    plt.hist(clus['Cluster'],bins = len(np.unique(clus.Cluster))-1,alpha = 0.35,label = 'Total number of clusters',edgecolor = 'k')
    plt.hist(outlier_cluster,bins = len(np.unique(clus.Cluster))-1,alpha = 0.35,label = 'outliers',edgecolor = 'k')
    plt.xlabel('Cluster')
    plt.ylabel('No of Quasars')
    plt.legend()
    #Plot the treshold for quantization error
    plt.figure()
    plt.hist(quantization_errors,edgecolor = 'k',label = f'Threshold = {outlier_percentage}')
    plt.axvline(error_treshold, color='k', linestyle='--')
    plt.legend()
    plt.xlabel('Quantization Error')
    plt.ylabel('No of Quasars')



def Cluster_Metrics(scaled_curves,cluster_map,metric = 'Silhoutte'):
    '''
    Measures metrics related to the clustering
    
    Parameters
    ----------
    scaled_curves: np.ndarray
    The scaled curves used to train the SOM
    
    cluster_map: pd.Dataframe
    A map of each of the ids to the clusters
    
    metric: str
    The metric to be measured. It can be Silhoutte, DBI or CH. This is for silhoutte score, Davies-Bouldin index and calinski-harabasz score
 
    Returns
    --------
    score:
    The metric that is calculated
    '''
    if metric == 'Silhoutte':
        score = silhouette_score(scaled_curves,cluster_map['Cluster'])
    elif metric == 'DBI':
        score = davies_bouldin_score(scaled_curves,cluster_map['Cluster'])
    elif metric == 'CH':
        score = calinski_harabasz_score(scaled_curves,cluster_map['Cluster'])
    else:
        score = 0
        print('Please use Silhoutte, DBI or CH')
    return score



def save_chosen_cluster(chosen_cluster,cluster_map,one_filter = True,filters = 'a',overwrite = True,save_path = './',source_path = './Light_Curves'):
    '''
    Saves the chosen cluster into a folder
    
    Parameters
    ----------
    chosen_cluster: int
    The cluster to save
    
    cluster_map: pd.Dataframe
    A map of each of the ids to the clusters
    
    one_filter: bool
    Whether to save just one filter or all the filters
    
    filters: str
    The filters to save
    
    overwrite: bool
    Whether to overwrite the current folder
    
    save_path: str
    The path to save to. This creates a folder for the cluster in that directory
    
    source_path: str
    The path that the light curves are saved in. If multifilter, provide the entire larger folder.
    '''
    #Save the light curves in the chosen cluster in a way they can be processed by QNPy
    #If folder isn't made, will create this folder to save the curves
    if overwrite:
        if f'Cluster_{chosen_cluster}' in os.listdir(save_path):
            shutil.rmtree(save_path+f'Cluster_{chosen_cluster}')
    os.makedirs(save_path+f'Cluster_{chosen_cluster}',exist_ok = True)
    chosen_ids = cluster_map['ID'][cluster_map['Cluster'] == chosen_cluster].to_numpy()
    if one_filter:
        for ID in chosen_ids:
            shutil.copyfile(source_path+f'/{ID}.csv', save_path+f'Cluster_{chosen_cluster}/{ID}.csv')
    else:
        for Filter in tqdm(filters,desc = 'Saving Filter for Cluster'):
            os.makedirs(save_path+f'Cluster_{chosen_cluster}/'+Filter,exist_ok = True)
            for ID in chosen_ids:
                shutil.copyfile(source_path+'/'+Filter+f'/{ID}.csv', save_path+f'Cluster_{chosen_cluster}/{Filter}/{ID}.csv')
    print('Cluster Saved')



### 2D Clustering Functions

def scale_to_range(series, min_val=-2, max_val=2):
    '''
    Scales a series to a range
    
    Parameters
    ----------
    series: pd.Series 
    The series to scale
    
    min_val: int
    The minimum value to scale to
    
    max_val: int
    The maximum value to scale to

    Returns
    --------
    scaled_series:
    The scaled series between the max and min values
    '''
    min_series = series.min()
    max_series = series.max()
    return min_val + (max_val - min_val) * (series - min_series) / (max_series - min_series)



def masked_euclidean_distance(data1, data2, mask):
    '''
    Calculates the masked euclidean distance between two arrays using a common mask
    
    Parameters
    ----------
    data1: np.ndarray
    The first array
    
    data2: np.ndarray
    The second array
    
    mask: np.ma.mask
    The mask used to get the distance

    Returns
    --------
    masked_distance:
    The masked distance measure
    '''
    return np.sqrt(np.ma.sum((np.ma.array(data1, mask=mask) - np.ma.array(data2, mask=mask)) ** 2))



def multi_band_processing(light_curves,ids,filter_names = 'ugriz',return_wide = False):
    '''
    Processes the light curves into a wide table
    
    Parameters
    ----------
    light_curves: 
    The light curves to be used
    
    ids: list,array
    The ids of the quasars
    
    filter_names: list or str(if the filters are one letter)
    The filters that are used
    
    return_wide: bool
    Whether to return the wide table or a flat table
    
    Returns
    --------
    light_curves_wide:
    The pivot table of the light curves with time steps
    
    light_curves_wide.isna():
    The mask used with the wide light curves
    
    OR 
    
    light_curves_flat:
    The flattened pivot table of the light curves
   
    mask_flat:
    The mask used
    '''
    #The preprocessing of tne light curves
    data = deepcopy(light_curves)
    #Adding ID and Filter Columns to the data
    for filter_name in range(len(filter_names)):
        for j in range(len(data[filter_name])):
            data[filter_name][j]['Filter'] = filter_name
            data[filter_name][j]['ID'] = ids[j]
    #Concatenating all the light curves together
    concatenated_by_filter = []
    for i in tqdm(range(len(filter_names)),desc = 'concat'):
        concat_filter = pd.concat(data[i])
        concatenated_by_filter.append(concat_filter)
    big_df = pd.concat(concatenated_by_filter)
    #Scaling the time and magnitude
    big_df['time_scaled'] = big_df.groupby(['ID', 'Filter'])['mjd'].transform(scale_to_range)
    big_df['mag_scaled'] = big_df.groupby(['ID', 'Filter'])['mag'].transform(scale_to_range)
    #Creating the mask from the densest bin (This isn't being used currently, but can be changed)
    densest_band = big_df.groupby('Filter').count().idxmax()['mjd']
    big_df['mask'] = big_df['Filter'] != densest_band
    # Pivot the DataFrame to wide format
    light_curves_wide = big_df.pivot_table(index='ID', columns=['time_scaled', 'Filter'], values='mag_scaled')
    # Create mask for missing data (This mask fills in all the missing data)
    mask = light_curves_wide.isna()
    #Flatten the pivot table
    light_curves_flat = light_curves_wide.values
    mask_flat = mask.values
    if return_wide:
        return light_curves_wide,light_curves_wide.isna()
    else:
        return light_curves_flat,mask_flat




def multi_band_clustering(light_curves,ids,filter_names = 'ugriz',som_x = None,som_y = None,sigma = 1.0,learning_rate = 0.5,\
                          num_iterations = 2,batch_size = 5,early_stopping_no = None):
    '''
    Multiband light curve clustering
    
    Parameters
    ----------
    light_curves: 
    The light curves to be used
    
    ids: list,array
    The ids of the quasars
    
    filter_names: list or str(if the filters are one letter)
    The filters that are used
    
    som_x: int
    The x size of the SOM. If None is given, make sure the som_y is None as well. Then, it chooses the recommended SOM 
    size of sqrt(sqrt(length))
    
    som_y: int
    The y size of the SOM. If None is given, make sure the som_x is None as well. Then, it chooses the recommended SOM 
    size of sqrt(sqrt(length))
    
    sigma: float
    The effect each node has on its neighboring nodes
    
    learning_rate: float
    How much the SOM learns from the new data that it sees
    
    num_iterations: int
    The number of iterations that the som is trained on each batch
    
    batch_size: int
    The size of each batch
    
    early_stopping_no: int or None
    The number of batches to process before stopping. Use None if you should train on all
    
    Returns
    --------
    som:
    The trained SOM
    
    processed_light_curve:
    The flat light curves used for the SOM
    
    processed_mask:
    The mask used for the SOM
    '''
    #First, process the input data for clustering
    processed_light_curves, processed_mask = multi_band_processing(light_curves,ids,filter_names)
    print('Processed')
    default_som_grid_length = math.ceil(math.sqrt(math.sqrt(len(processed_light_curves))))
    print(default_som_grid_length)
    #Now, initialize the SOM
    if som_x is None and som_y is None:
        som_x = som_y = default_som_grid_length
    elif som_x is None or som_y is None:
        print('Please Provide both som_x and som_y or neither, going with the default values of the sqrt')
        som_x = som_y = default_som_grid_length
    som = MiniSom(som_x, som_y, processed_light_curves.shape[1], sigma, learning_rate)
    som.random_weights_init(processed_light_curves)
    failed = 0
    #Training the SOM
    for i in tqdm(range(0, len(processed_light_curves), batch_size),desc = 'Batch Training'):
        batch_data = processed_light_curves[i:i + batch_size]
        batch_mask = processed_mask[i:i + batch_size]
        if early_stopping_no is None:
            early_stopping_no = len(processed_light_curves)+batch_size
        for t in range(num_iterations):
            for idx, data_vector in enumerate(batch_data):
                data_mask = batch_mask[idx]
                bmu_index = None
                min_distance = float('inf')
                iteration_weights = som.get_weights()
                # Find BMU considering masked data
                for x in range(som_x):
                    for y in range(som_y):
                        w = iteration_weights[x, y]
                        distance = masked_euclidean_distance(data_vector, w, data_mask)
                        if distance < min_distance:
                            min_distance = distance
                            bmu_index = (x, y)
                # Update SOM weights
                try:
                    som.update(data_vector, bmu_index, t, num_iterations)
                except:
                    failed += 1
        if i == early_stopping_no:
                break
    return som,processed_light_curves,processed_mask

def find_cluster_and_quantization_errors(som,data,masks):
    '''
    Finding the clusters and the quantization errors from the trained 2D SOM
    
    Parameters
    ----------
    som: 
    The trained SOM
    
    data: 
    The processed light curves from the trained SOM
    
    masks: 
    The masks used from the trained SOM
    
    Returns
    --------
    min_clusters:
    The clusters for each of the data points
    
    quantization_error:
    The quantization error of each of the data points
    '''
    #Finding the BMU and the quantization error of each data point
    flat_weights = som.get_weights().reshape(-1,som.get_weights().shape[2])
    min_clusters = []
    quantization_error = []
    for i in tqdm(range(len(data))):
        data_point = data[i]
        mask = masks[i]
        distances = []
        for weight in flat_weights:
            distances.append(masked_euclidean_distance(data_point,weight,mask))
        min_clusters.append(np.argmin(distances)+1)
        quantization_error.append(np.min(distances))
    return min_clusters,quantization_error

def Get_Gradient_Cluster_2D(som,fill = 'mean',interpolation_kind = 'cubic'):
    '''
    Finding the gradient clusters from the 2D SOM
    
    Parameters
    ----------
    som: 
    The trained SOM
    
    fill: str
    'mean' or 'interpolate'. Either the empty values are filled with the mean or they are interpolated with a function
    
    interpolation_kind: 
    Any of the scipy.interp1d interpolation kinds. Recommended to use cubic
    
    Returns
    --------
    cluster_centers:
    The cluster centers
    
    cluster_pos:
    The cluster positions
    '''
    new_som = deepcopy(som)
    for i in tqdm(range(len(new_som._weights))):
        for j in range(len(new_som._weights[i])):
            if fill == 'mean':
                new_som._weights[i][j] = np.nan_to_num(new_som._weights[i][j],nan = np.nanmean(new_som._weights[i][j]))
            elif fill == 'interpolate':
                array = new_som._weights[i][j]
                # Find indices of non-NaN values
                non_nan_indices = np.where(~np.isnan(array))[0]
                interpolator = interp1d(non_nan_indices, array[non_nan_indices], kind=interpolation_kind, fill_value='extrapolate')
                array_interpolated = interpolator(np.arange(len(array)))
                new_som._weights[i][j] = array_interpolated
    cluster_centers = []
    cluster_pos  = []
    for row in np.arange(new_som.distance_map().shape[0]):
        for col in np.arange(new_som.distance_map().shape[1]):
            cx,cy = findInternalNode(row, col, new_som.distance_map().T)
            cluster_centers.append(np.array([cx,cy]))
            cluster_pos.append(np.array([row,col]))
    return np.array(cluster_centers),np.array(cluster_pos)


##Visualization Functions
def create_dir_save_plot(path,plot_name):
    '''
    If there is no folder named plots in the path, it creates one and saves an figure
    
    Parameters
    ----------
    path: str  
    The path to create the Plots folder in
    
    plot_name: 
    The name to save the plot under
    ''' 
    if 'Plots' not in os.listdir(path):
            os.makedirs(path+'Plots')
    plt.savefig(path+'Plots/'+plot_name+'.png')

def tolerant_mean(arrs):
    '''
    Calculates the mean of arrays without them having to be the same length
    
    Parameters
    ----------
    arrs:  
    The arrays to calculate the mean for 

    Returns
    --------
    mean:
    The tolerant mean of the arrays
    
    std:
    The tolerant std of the arrays
    '''
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def get_best_grid(number_of_objects):
    '''
    Creates a grid that is optimal for the number of objects
    
    Parameters
    ----------
    number_of_objects: int  
    The number of objects to make the grid for

    Returns
    --------
    rows:
    The number of rows for the grid
    
    cols:
    The number of cols for the grid
    '''
    cols = int(np.sqrt(number_of_objects))
    rows = number_of_objects//cols
    if number_of_objects % cols != 0:
        rows += 1
    return rows,cols

def Averaging_Clusters(chosen_cluster,cluster_map,lcs,plot = True,dba = True):
    '''
    Creating a representation of the chosen cluster with the light curves and the average light curve
    
    Parameters
    ----------
    chosen_cluster: int 
    The cluster of interest
    
    cluster_map: pd.Dataframe
    A map of each of the ids to the clusters
    
    lcs: list of list of pd.Dataframes
    The light curves (provide the input from just one filter)
    
    plot: bool
    Whether to plot or just return the average value 
    
    dba: bool
    Whether to use Dynamic Barymetric Time Averaging or to use a simple mean of the light curves
    
    Returns
    --------
    average_x:
    The x_axis (i.e timesteps) of the average light curve
    
    average_y:
    The y_axis (i.e magnitudes) of the average light curve
    
    x:
    The timesteps of all the light curves concatenated into one array
    
    y: 
    The magnitudes of all the light curves concatenated into one array
    
    len(x):
    The length of all the light curves
    '''
  #Either plots each light curve with its errors and the average or just returns the average
    x = []
    x_arrs = []
    y = []
    y_arrs = []
    err = []
    ids = []
    all_curves = []
    #Getting the data to plot for the cluster
    for quasar_id,cluster,i in zip(cluster_map.ID,cluster_map.Cluster,range(len(cluster_map))):
        if cluster == chosen_cluster:
            light_curve = lcs[i]
            light_curve = light_curve.dropna(subset = ['mag'])
            light_curve = light_curve.sort_values(by = 'mjd')
            all_curves.append(light_curve)
            times = light_curve.mjd - light_curve.mjd.min()
            x.append(times)
            x_arrs.append(times.to_numpy()) 
            y.append(light_curve.mag)
            y_arrs.append(light_curve.mag.to_numpy())
            err.append(light_curve.magerr)
    cmap = plt.cm.prism
    norm = Normalize(vmin=1, vmax=len(x))
    if dba: #Need to build support for mean (even though the light curves are not same length)
        average_x = dtw_barycenter_averaging(x)
        average_y = dtw_barycenter_averaging(y)
    else:
        average_x = tolerant_mean(x_arrs)[0]
        average_y = tolerant_mean(y_arrs)[0]
    #Plotting a scatter plot and line plot
    if plot is True:
        fig,(ax1,ax2) = plt.subplots(1,2,sharey= True,figsize = (6,7))
        for i in range(len(x)):
            ax1.errorbar(x[i], y[i]*(i+1), err[i], fmt='.',color = cmap(int(i)),alpha = 0.1)
            ax2.plot(x[i],y[i]*(i+1),alpha = 0.5,c = cmap(int(i)))
        ax1.invert_yaxis()
        ax1.set_ylabel('Magnitude')
        ax2.set_ylabel('Magnitude')
        ax1.set_xlabel('Days')
        ax2.set_xlabel('Days')
        print('Length of Cluster: '+str(len(x)))
        plt.figure()
        for i in range(len(x)):
            plt.plot(x[i],y[i],alpha = 0.5,c = 'grey')
        plt.plot(average_x,average_y,label = 'Averaged Curves')
        plt.ylabel('Magnitude')
        plt.xlabel('Days')
        plt.gca().invert_yaxis()
    else:
        return average_x,average_y,x,y,len(x)

def Plot_All_Clusters(cluster_map,lcs,color = 'tab:blue',dba = True,figsize = (10,10),save_figs = True,figs_save_path = './'):
    '''
    Plots all of the clusters on a magnitude plot with the average representation included
    
    Parameters
    ----------
    cluster_map: pd.Dataframe
    A map of each of the ids to the clusters
    
    lcs: list of list of pd.Dataframes
    The light curves (provide the input from just one filter)
    
    color: str
    The color to plot the averaged curve in
    
    dba: bool
    Whether to use Dynamic Barymetric Time Averaging or to use a simple mean of the light curves
    
    figsize: tuple
    The figure size
    
    save_figs: bool
    Whether to save the figure or not
    
    figs_save_path: str
    Where to save the figure. Note that it is saved under a directory called Plots in that directory.
    '''
    #Getting the shape of the subplots
    clusters = cluster_map.value_counts('Cluster').index.to_numpy()
    total = len(clusters)
    cols = int(np.sqrt(len(clusters)))
    rows = total//cols
    if total % cols != 0:
        rows += 1
    fig, axs = plt.subplots(rows,cols,figsize=figsize,layout="constrained",sharey = True)
    #fig.suptitle('Clusters')
    #Getting the values to plot
    x_axis = []
    y_axis = []
    for i in tqdm(range(len(clusters)),desc = 'Plotting Averaged Clusters'):
        x,y,back_x,back_y,no = Averaging_Clusters(clusters[i],cluster_map,lcs,plot = False,dba = dba)
        for j in range(no):
            axs.flat[i].plot(back_x[j],back_y[j],color = 'gray',alpha = 0.5)
            axs.flat[i].plot(x,y,color = color)
            axs.flat[i].set_title(f'Cluster {clusters[i]}, {no} curves')
            axs.flat[i].set_xlabel('Days')
            axs.flat[i].set_ylabel('Magnitude')
            axs.flat[i].invert_yaxis()
    axs[0,0].invert_yaxis()
    if save_figs:
        create_dir_save_plot(figs_save_path,'SOM_Nodes_Map')
    plt.show()

def get_redshifts(redshifts_map):
    '''
    Gets all the redshifts from a redshifts map  
    
    Parameters
    ----------
    redshifts_map:  pd.Dataframe
    The mapping of ids to redshifts 

    Returns
    --------
    redshifts:
    The list of redshifts
    '''
    redshifts = redshifts_map.z.to_list()
    return redshifts

def get_fvars(lcs):
    '''
    Calculates the variability function of the light curves
    
    Parameters
    ----------
    lcs:  List of pd.Dataframes
    The light curves of interest

    Returns
    --------
    fvars:
    The list of variability functions
    '''
    fvars = []
    for quasar in lcs:
        N = len(quasar)
        meanmag = quasar['mag'].mean()
        s2 = (1/(N-1))*np.sum((quasar['mag']-meanmag)**2)
        erm = np.mean(quasar['magerr']**2)
        f_var = np.sqrt((s2-erm)/(np.mean(quasar["mag"])**2))
        if np.isnan(f_var):
            f_var = np.nan
        fvars.append(f_var)
    return fvars

def get_luminosities_and_masses(lcs, redshifts_map, H0 = 67.4, Om0 = 0.315):
    '''
    Randomly samples the luminosity and masses of the quasar black holes assuming a given Hubble Constant, Omega_0, and redshift
    
    Parameters
    ----------
    lcs:  List of pd.Dataframes
    The light curves of interest
    
    redshifts_map: pd.DataFrame
    The map from the ids to their redshifts
    
    H0: float
    The hubble constant at z=0
    
    Om0: float
    Omega matter: density of non-relativistic matter in units of the critical density at z=0. 

    Returns
    --------
    Log_lum:
    The logarithm of the luminosities
    
    Log_Mass:
    The logarithm of the masses
    '''

    #Calculating luminosities
    c = 299792.458
    Tcmb0 = 2.725
    #Cosmological Model Chosen
    cosmo = FlatLambdaCDM(H0,Om0,Tcmb0)
    redshifts = get_redshifts(redshifts_map)

    distances=[]
    #Getting the distances from the redshifts
    for i in range(len(redshifts)):
        distances.append(cosmo.comoving_distance(redshifts[i]).value)
    #Converting distance to luminosities and correcting the magnitudes
    F0=3.75079e-09
    lambeff=3608.4
    Mpc_to_cm = 3.086e+24
    luminosity=[]
    absolmag=[]
    for i,quasar in enumerate(lcs):
        meanmag= quasar['mag'].mean()
        const=4*np.pi*((distances[i]*Mpc_to_cm)**2)*lambeff*F0
        luminosity.append(const*np.power(10, -meanmag/2.5))
        Kcorr=-2.5*(1-0.5)*np.log(1+redshifts[i])
        absolmag.append(meanmag-5*np.log10(distances[i])-25-Kcorr)
    #Using the corrected magnitudes to calculate black hole masses
    mu=2.0-0.27*np.asarray(absolmag)
    sigma=np.abs(0.580+0.11*np.asarray(absolmag))
    #Random Sampling the Masses from Distribution
    Log_Mass=np.zeros(len(mu))
    for i in range(len(mu)):
        k1=np.random.normal(mu[i],sigma[i],10).mean()
        Log_Mass[i] = k1

    return np.log10(luminosity),Log_Mass

def Cluster_Properties(cluster_map,selected_cluster,lcs,redshifts_map = None,plot = True,return_values = False,\
                       the_property = 'all',save_figs = True,figs_save_path = './'):
    '''
    Getting the selected property of a chosen cluster
    
    Parameters
    ----------
    cluster_map: pd.Dataframe
    A map of each of the ids to the clusters
    
    chosen_cluster: int 
    The cluster of interest
    
    lcs: list of list of pd.Dataframes
    The light curves (provide the input from just one filter)
    
    redshifts_map: pd.Dataframe
    The redshift associated with each source id
    
    plot: bool
    Whether to plot or just return the average value 
    
    return_values: bool
    Whether to return the values for the property
    
    the_property: str
    The property to plot. Choice from z (redshift), Fvar (the variability function), Lum (luminosity), Mass, or all
    
    save_figs: bool
    Whether to save the figure or not
    
    figs_save_path: str
    Where to save the figure. Note that it is saved under a directory called Plots in that directory.
    
    Returns
    --------
    return_list: 
    The list of the property of interest
    '''
    if redshifts_map is None and the_property != 'Fvar':
        print('Need Redshifts to plot selected property')
        return
    all_quasar_ids = cluster_map.ID.astype('int')
    if selected_cluster == 'all':
        selected_quasar_ids = all_quasar_ids
    else:
        selected_quasar_ids = cluster_map.ID[cluster_map.Cluster == selected_cluster].to_list()
    quasar_light_curves = []
    new_selected_quasar_ids = [int(ID) for ID in selected_quasar_ids]
    for i in range(len(lcs)):
        if all_quasar_ids[i] in new_selected_quasar_ids:
              quasar_light_curves.append(lcs[i])
    return_list = [[np.nan]*len(selected_quasar_ids)]*4
    if the_property == 'all':
        the_property = 'zFvarLumMass'
    new_selected_quasar_ids = [int(ID) for ID in selected_quasar_ids]
    if 'z' in the_property:
        redshifts_map_selected = redshifts_map[redshifts_map.ID.isin(new_selected_quasar_ids)]
        redshifts = get_redshifts(redshifts_map_selected)
        return_list[0] = redshifts
        if 'Lum' not in the_property or 'Mass' not in the_property:
            log_luminosities = [None]*len(redshifts)
            log_masses = [None]*len(redshifts)
        if 'Fvar' not in the_property:
            fvars = [None]*len(redshifts)
    if 'Fvar' in the_property:
        fvars = get_fvars(quasar_light_curves)
        return_list[1] = fvars
        if 'z' not in the_property:
            redshifts = [None]*len(fvars)
        if 'Lum' not in the_property or 'Mass' not in the_property:
            log_luminosities = [None]*len(fvars)
            log_masses = [None]*len(fvars)
    if 'Lum' in the_property or 'Mass' in the_property:
        redshifts_map_selected = redshifts_map[redshifts_map.ID.isin(new_selected_quasar_ids)]
        log_luminosities,log_masses = get_luminosities_and_masses(quasar_light_curves,redshifts_map_selected)
        return_list[2] = log_luminosities
        return_list[3] = log_masses
        if 'z' not in the_property:
            redshifts = [None]*len(log_masses)
        if 'Fvar' not in the_property:
            fvars = [None]*len(log_masses)
    if plot is True:
        new_dataframe = pd.DataFrame({'z':redshifts,r'$F_{var}$':fvars,r'$log_{10}{L[erg s^{-1}]}$':log_luminosities,r'$log_{10}{M[M_{\odot}]}$':log_masses})
        plt.figure()
        sns.pairplot(new_dataframe,corner = True)
        plt.minorticks_off()
        if save_figs:
            create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_Properties')
    if return_values is True:
        return return_list

def Cluster_Properties_Comparison(cluster_map,lcs,redshifts_map,the_property = 'Fvar',color = '#1f77b4',\
                                  figsize = (15,15),save_figs = True,figs_save_path = './'):
    '''
    Plotting the property of interest for all the clusters onto one figure
    
    Parameters
    ----------
    cluster_map: pd.Dataframe
    A map of each of the ids to the clusters
    
    lcs: list of list of pd.Dataframes
    The light curves (provide the input from just one filter)
    
    redshifts_map: pd.Dataframe
    The redshift associated with each source id
    
    the_property: str
    The property to plot. Choice from z (redshift), Fvar (the variability function), Lum (luminosity), Mass, or all
    
    color: str
    The color to make the histogram
    
    figsize: tuple
    The figure size
    
    save_figs: bool
    Whether to save the figure or not
    
    figs_save_path: str
    Where to save the figure. Note that it is saved under a directory called Plots in that directory.
    
    Returns
    --------
    return_list: 
    The list of the property of interest
    '''
    #Plots a histogram for one of the 4 different properties that we see in the clusters
    property_to_num_dict = {'z':0,'Fvar':1,'Lum':2,'Mass':3}
    property_to_label_dict = {'z':'z','Fvar':'$F_{var}$','Lum':r'$log_{10}{L[erg s^{-1}]}$','Mass':r'$log_{10}{M[M_{\odot}]}$'}
    rows,cols = get_best_grid(len(cluster_map.value_counts('Cluster')))
    fig,axs = plt.subplots(rows,cols,figsize = figsize,layout = 'constrained')
    count = 0
    #Setting the x scale for all the clusters
    properties = Cluster_Properties(cluster_map,'all',lcs,redshifts_map,plot = False,return_values = True,the_property = the_property)[property_to_num_dict[the_property]]
    min_x = min(properties)
    max_x = max(properties)
    bins = np.linspace(min_x,max_x,10)
    plt.setp(axs, xlim=(min_x,max_x))
  #Plotting the different subclusters
    for i in tqdm(range(rows),desc = 'Plotting '+the_property+' Distribution'):
        for j in range(cols):
            plotting_values = np.array(Cluster_Properties(cluster_map,count+1,lcs,redshifts_map,the_property = the_property,plot = False,return_values = True)[property_to_num_dict[the_property]])
            plotting_values = plotting_values[np.isfinite(plotting_values)]
            #Splitting it so we can plot the log dist of the luminosities
            counts,bins = np.histogram(plotting_values,bins = bins)
            axs[i][j].hist(plotting_values,bins = bins,color = color,edgecolor='black',linewidth=1.2)
            axs[i][j].set_title(f'Cluster {count+1}, {len(plotting_values)} curves')
            axs[i][j].set_ylabel('Number of Curves')
            axs[i][j].set_xlabel(property_to_label_dict[the_property])
            count += 1
    if save_figs:
        create_dir_save_plot(figs_save_path,the_property+'_Distribution_Plot')

def SFplus(magnitudes):
    '''
    Calculates the S+ function of given light curves. S+ is the variance of magnitudes where the brightness increases
    
    Parameters
    ----------
    lcs:  List of pd.Dataframes
    The light curves of interest

    Returns
    --------
    sfplus:
    The list of S+
    '''
    #Calculate the S+ Function
    combs=combinations(magnitudes,2)
    sf_vals=[]
    for x,y in combs:
        if x-y>0:
            sf_vals.append((x-y)**2)
    #sfplus=np.sqrt(np.mean(sf_vals))
    sfplus = np.mean(sf_vals)
    return sfplus

def SFminus(magnitudes):
    '''
    Calculates the S- function of given light curves. S- is the variance of magnitudes where the brightness decreases
    
    Parameters
    ----------
    lcs:  List of pd.Dataframes
    The light curves of interest

    Returns
    --------
    sfminus:
    The list of S-
    '''
    combs=combinations(magnitudes,2)
    sf_vals=[]
    for x,y in combs:
        if x-y<0:
            sf_vals.append((x-y)**2)
    #sfmin=np.sqrt(np.mean(sf_vals))
    sfmin = np.mean(sf_vals)
    return sfmin

def SF(magnitudes):
    '''
    Calculates the S function of given light curves. S is the variance of all magnitudes
    
    Parameters
    ----------
    lcs:  List of pd.Dataframes
    The light curves of interest

    Returns
    --------
    sf:
    The list of SFs
    '''
    combs=combinations(magnitudes,2)
    sf_vals=[]
    for x,y in combs:
        if x-y>0:
            sf_vals.append((x-y)**2)
    #sf=np.sqrt(np.mean(sf_vals))
    sf = np.mean(sf_vals)
    return sf

def Structure_Function(cluster_map, selected_cluster,lcs,bins,save_figs = True,figs_save_path = './'):
    '''
    Create the structure function for a given cluster
    
    Parameters
    ----------
    cluster_map: pd.Dataframe
    A map of each of the ids to the clusters
    
    selected_cluster: int
    The cluster of interest
    
    lcs: list of list of pd.Dataframes
    The light curves (provide the input from just one filter)
    
    bins:int or list
    The bins to use for the structure function
    
    save_figs: bool
    Whether to save the figure or not
    
    figs_save_path: str
    Where to save the figure. Note that it is saved under a directory called Plots in that directory.
    
    Returns
    --------
    S+ and S- Plot: 
    A plot of the S+ and S- functions for the cluster
    
    Difference Plot:
    The evolution of the normalized S+ - S- throughout the observation time of the cluster
    
    S Plot:
    The evolution of the (regular) structure function through the observation time of the cluster
    '''
    all_quasar_ids = cluster_map.ID
    if selected_cluster == 'all':
        selected_quasar_ids = all_quasar_ids.to_list()
    else:
        selected_quasar_ids = cluster_map.ID[cluster_map.Cluster == selected_cluster].to_list()
    quasar_light_curves = []
    for i in range(len(all_quasar_ids)):
        if all_quasar_ids[i] in selected_quasar_ids:
            quasar_light_curves.append(lcs[i])
  #Putting all the magnitudes and times together
    mag_composite=[]
    time_composite=[]
    for light_curve in quasar_light_curves:
        mag_composite=mag_composite+light_curve["mag"].to_list()
        time_composite=time_composite+ light_curve["mjd"].to_list()
    time_composite_zeroed = time_composite - np.min(time_composite)
    #Calculating the S+, S- and S for all bins
    betaplus, timeplus,xx=stats.binned_statistic(time_composite_zeroed, mag_composite, statistic=SFplus, bins=bins, range=None)
    betamin, timemin,xx=stats.binned_statistic(time_composite_zeroed, mag_composite, statistic=SFminus, bins=bins, range=None)
    beta, time,xx=stats.binned_statistic(time_composite_zeroed, mag_composite, statistic=SF, bins=bins, range=None)

    #Calculating the normalizing function
    bbeta=((betaplus)-(betamin))/(beta+0.01)

    #Plotting S+ and S-
    plt.figure()
    plt.scatter(timeplus[:-1],betaplus,marker = 'v',label = r'$S_+$ observed QSO',c = 'orange')
    plt.plot(timeplus[:-1],betaplus,c = 'orange',linestyle = '--')
    plt.scatter(timemin[:-1],betamin,marker = '^',label = r'$S_-$ observed QSO',c = 'b')
    plt.plot(timemin[:-1],betamin,c = 'b',linestyle = '--')
    plt.xlabel(r'$\tau[day]$')
    plt.ylabel(r'$S_+\cup S_-$')
    if save_figs:
        create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_S+&S-_Plot')
    plt.legend()
    #Plotting the normalized difference
    plt.figure()
    plt.scatter(time[:-1],bbeta,label = r'$\beta$ Observed QSO',c = 'black',s = 4)
    plt.plot(time[:-1],bbeta,linestyle = '--')
    plt.xlabel(r'$\tau[day]$')
    plt.ylabel(r'$\beta = \frac{S_+ - S_-}{S}$')
    if save_figs:
        create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_S+S-_Difference_Plot')
    #plt.legend()
    #Plotting the structure function evolution
    plt.figure()
    plt.scatter(time[:-1],beta,label = r'SF Observed QSO',c = 'black',s = 4)
    plt.plot(time[:-1],beta,linestyle = '--')
    plt.xlabel(r'$\tau[day]$')
    plt.ylabel(r'Structure Function')
    if save_figs:
        create_dir_save_plot(figs_save_path,f'Cluster_{selected_cluster}_Structure_Function_Plot')
