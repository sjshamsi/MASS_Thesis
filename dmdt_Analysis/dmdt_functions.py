import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# The following functions are used to calculate the Mahabal dquantities themselves
#------------------------------------------------------------------------------
def return_dquantities(quantity_array, quantity_err_array=None):
    dquantity_matrix = np.subtract.outer(quantity_array, quantity_array)
    upper_triangle_indices = np.triu_indices_from(dquantity_matrix, k=1)
    dquantity = dquantity_matrix[upper_triangle_indices] * -1 # for example turning t_j - t_i to t_i - t_j
    if quantity_err_array is not None:
        dquantity_err_matrix = np.sqrt(np.add.outer(quantity_err_array**2, quantity_err_array**2))
        dquantity_err = dquantity_err_matrix[upper_triangle_indices]
        return dquantity, dquantity_err
    return dquantity


def return_dms_and_dts(lightcurve):
    time_array, magnitude_array, magnitude_err_array = lightcurve[:, 0], lightcurve[:, 1], lightcurve[:, 2]
    dm, dm_err = return_dquantities(quantity_array=magnitude_array, quantity_err_array=magnitude_err_array)
    dt = return_dquantities(quantity_array=time_array)
    return dm, dm_err, dt


def return_d2ms_and_dt2s(dmdt_curve):
    time_array, dmdt_array, dmdt_err_array = dmdt_curve[:, 0], dmdt_curve[:, 1], dmdt_curve[:, 2]
    d2m, d2m_err = return_dquantities(quantity_array=dmdt_array, quantity_err_array=dmdt_err_array)
    dt2 = return_dquantities(quantity_array=time_array)
    return d2m, d2m_err, dt2


def return_1D_dm_histogram(lightcurve, dm_bins, weights_by_errs=True):
    magnitude_array, magnitude_err_array = lightcurve[:, 1], lightcurve[:, 2]
    if weights_by_errs:
        dm, dm_err = return_dquantities(quantity_array=magnitude_array, quantity_err_array=magnitude_err_array)
        return np.histogram(dm, bins=dm_bins, weights=1/dm_err**2)[0]
    else:
        dm = return_dquantities(quantity_array=magnitude_array)
        return np.histogram(dm, bins=dm_bins)[0]


def return_1D_dt_histogram(lightcurve, dt_bins):
    time_array = lightcurve[:, 0]
    dt = return_dquantities(quantity_array=time_array)
    return np.histogram(dt, bins=dt_bins)[0]


# The following functions are used to make 2D histograms from dquantities
#------------------------------------------------------------------------------
def density(array):
    # if not isinstance(array, np.ndarray): return np.nan
    sum = array.sum()
    return array / sum if sum != 0 else array


def density_rgb(array_3D):
    # if not isinstance(array_3D, np.ndarray): return np.nan
    assert array_3D.ndim == 3 and array_3D.shape[2] == 3
    for i in range(3):
        array_3D[:, :, i] = density(array_3D[:, :, i])
    return array_3D


def norm_0_1(array):
    # if not isinstance(array, np.ndarray): return np.nan
    array_min, array_max = array.min(), array.max()
    if array_min == array_max: return array

    return (array - array.min()) / (array.max() - array.min())


def norm_0_1_rgb(array_3D):
    # if not isinstance(array, np.ndarray): return np.nan
    assert array_3D.ndim == 3 and array_3D.shape[2] == 3
    for i in range(3):
        array_3D[:, :, i] = norm_0_1(array_3D[:, :, i])
    return array_3D


def int_scale(array):
    '''Scale the array to 0-255 integers. Assumes input is 0-1 or density normalised.
    Parameters
    ----------
    array : numpy.ndarray
        2D array representing the histogram.
    Returns
    -------
    numpy.ndarray
        2D array with scaled values.
    '''
    array = ((255 * array) + 0.99999).astype(int)
    return array


def int_scale_rgb(array_3D):
    '''Scale the RGB array to 0-255 integers. Assumes input is 0-1 or density normalised.
    Parameters
    ----------
    array_3D : numpy.ndarray
        3D array with shape (x, y, 3) representing RGB values.
    Returns
    -------
    numpy.ndarray
        3D array with scaled RGB values.
    '''
    assert array_3D.ndim == 3 and array_3D.shape[2] == 3
    for i in range(3):
        array_3D[:, :, i] = int_scale(array_3D[:, :, i])
    return array_3D


def get_2Dhistogram(quantity1, quantity1_bins, quantity2, quantity2_bins, quantity1_errs=None, quantity2_errs=None):
    if (quantity1_errs is not None) and (quantity2_errs is not None):
        weigths = 1 / (quantity1_errs**2 + quantity2_errs**2)
    elif quantity1_errs is not None:
        weigths = 1 / quantity1_errs**2
    elif quantity2_errs is not None:
        weigths = 1 / quantity2_errs**2
    else:
        weigths = np.ones_like(quantity1)
        
    hist, _quantity1_edges, _quantity2_edges = np.histogram2d(quantity1, quantity2, bins=[quantity1_bins, quantity2_bins], weights=weigths)
    return hist, _quantity1_edges, _quantity2_edges


def get_dmdt_histogram(lightcurve, dmagnitude_bins, dtime_bins, dmagnitude_weights=True):
    dm_array, dm_err_array, dt_array = return_dms_and_dts(lightcurve)
    dt_err_array = None
    dm_err_array = dm_err_array if dmagnitude_weights else None

    hist, _dm_edges, _dt_edges = get_2Dhistogram(dm_array, dmagnitude_bins, dt_array, dtime_bins, quantity1_errs=dm_err_array, quantity2_errs=dt_err_array)
    return hist, _dm_edges, _dt_edges


def get_d2mdt2_histogram(dmdt_curve, d2m_bins, dt2_bins, dm2_weights=True):
    d2ms, d2m_errs, dt2s = return_d2ms_and_dt2s(dmdt_curve)
    dt2_errs = None
    d2m_errs = d2m_errs if dm2_weights else None

    hist, _d2m_edges, _dt2_edges = get_2Dhistogram(d2ms, d2m_bins, dt2s, dt2_bins, quantity1_errs=d2m_errs, quantity2_errs=dt2_errs)
    return hist, _d2m_edges, _dt2_edges


def return_processing_histogram(r_histogram, g_histogram):
    ### Make the RBG image
    rbg_image = np.zeros((r_histogram.shape[0], r_histogram.shape[1], 3), dtype=np.float32)
    rbg_image[:, :, 0] = r_histogram
    rbg_image[:, :, 1] = g_histogram
    rbg_image = norm_0_1_rgb(rbg_image) ### norm the r and g histograms if needed
    return rbg_image


def return_showcase_histogram(r_histogram, g_histogram):
    ### norm the r and g histograms if needed
    r_histogram = norm_0_1(r_histogram)
    g_histogram = norm_0_1(g_histogram)

    ### Make the RBG image
    rbg_image = np.ones((r_histogram.shape[0], r_histogram.shape[1], 3), dtype=np.float32)
    rbg_image[:, :, 1] -= r_histogram
    rbg_image[:, :, 2] -= r_histogram
    rbg_image[:, :, 0] -= g_histogram
    rbg_image[:, :, 2] -= g_histogram

    if rbg_image[:, :, 2].max() > 1 or rbg_image[:, :, 2].min() < 0:
        rbg_image[:, :, 2] = norm_0_1(rbg_image[:, :, 2])
    return rbg_image


# The following functions are used to plot 2D histograms from dquantities
#------------------------------------------------------------------------------
def draw_rgb_histogram(ax, histogram):
    ax.imshow(histogram, origin='lower', aspect='auto', extent=[0, histogram.shape[1], 0, histogram.shape[0]])
    ax.grid(False)
    return ax


def add_histogram_ticks(ax, quantity1_bins, quantity2_bins, quantity1_nticks=10, quantity2_nticks=5):
    quantity1_indices = np.linspace(0, len(quantity1_bins) - 1, quantity1_nticks, dtype=int)
    quantity2_indices = np.linspace(0, len(quantity2_bins) - 1, quantity2_nticks, dtype=int)
    
    quantity1_tick_labels = []
    for index in quantity1_indices:
        formatted_string = f"{quantity1_bins[index]:.2f}"
        quantity1_tick_labels.append(formatted_string)

    quantity2_tick_labels = []
    for index in quantity2_indices:
        formatted_string = f"{quantity2_bins[index]:.1e}"
        base, exponent = formatted_string.split('e')
        quantity2_tick_labels.append(f"${base} \\times 10^{{{int(exponent)}}}$")

    ax.set_yticks(ticks=quantity1_indices, labels=quantity1_tick_labels)
    ax.set_xticks(ticks=quantity2_indices, labels=quantity2_tick_labels, size=10)
    return ax


def draw_dmdt_rgb_histogram(ax, histogram, dm_bins, dt_bins, dm_nticks=10, dt_nticks=5):
    ax = draw_rgb_histogram(ax=ax, histogram=histogram)
    ax = add_histogram_ticks(ax=ax, quantity1_bins=dm_bins, quantity2_bins=dt_bins, quantity1_nticks=dm_nticks, quantity2_nticks=dt_nticks)
    ax.set_xlabel(f'{dt_bins.min():.2f} $< dt <$ {dt_bins.max():.2f} (days) {len(dt_bins) - 1} bins', size=12)
    ax.set_ylabel(f'{dm_bins.min():.2f} $< dm <$ {dm_bins.max():.2f} (magnitude) {len(dm_bins) - 1} bins', size=12)
    return ax


def draw_d2mdt2_rgb_histogram(ax, histogram, d2m_bins, dt2_bins, d2m_nticks=10, dt2_nticks=5):
    ax = draw_rgb_histogram(ax=ax, histogram=histogram)
    ax = add_histogram_ticks(ax=ax, quantity1_bins=d2m_bins, quantity2_bins=dt2_bins, quantity1_nticks=d2m_nticks, quantity2_nticks=dt2_nticks)
    ax.set_xlabel(f'{dt2_bins.min():.2f} $< dt^2 <$ {dt2_bins.max():.2f} (days) {len(dt2_bins) - 1} bins', size=12)
    ax.set_ylabel(f'{d2m_bins.min():.2f} $< d^2m <$ {d2m_bins.max():.2f} ($d$ magnitude) {len(d2m_bins) - 1} bins', size=12)
    return ax


def draw_single_channel_histogram(ax, histogram, band, cbar=False):
    cmap = 'Reds' if band == 'r' else 'Greens'
    imshow_object = ax.imshow(histogram, origin='lower', aspect='auto', cmap=cmap)
    if cbar:
        colorbar = plt.colorbar(imshow_object, ax=ax, shrink=0.8, label=f"${band}$ band", location='right')
    ax.grid(False)
    return ax


def draw_single_channel_dmdt_histogram(ax, histogram, band, dm_bins, dt_bins, cbar=False, dm_nticks=10, dt_nticks=5):
    ax = draw_single_channel_histogram(ax=ax, histogram=histogram, band=band, cbar=cbar)
    ax = add_histogram_ticks(ax=ax, quantity1_bins=dm_bins, quantity2_bins=dt_bins, quantity1_nticks=dm_nticks, quantity2_nticks=dt_nticks)
    ax.set_xlabel(f'{dt_bins.min():.2f} $< dt <$ {dt_bins.max():.2f} (days) {len(dt_bins) - 1} bins', size=12)
    ax.set_ylabel(f'{dm_bins.min():.2f} $< dm <$ {dm_bins.max():.2f} (magnitude) {len(dm_bins) - 1} bins', size=12)
    return ax


# The following functions are used to plot lightcurves and thier derivatives
#------------------------------------------------------------------------------
def draw_timeseries(ax, band, time_array, y_array, y_err_array=None, **kwargs):
    if y_err_array is None:
        if 'color' not in kwargs:
            kwargs['color'] = band
        ax.scatter(time_array, y_array, '.', alpha=0.5, label=f'ZTF ${band}$ Band', **kwargs)
    else:
        if 'color' not in kwargs:
            kwargs['color'] = band
            kwargs['ecolor'] = band
        ax.errorbar(time_array, y_array, y_err_array, fmt='.', elinewidth=0.5, markersize=5, alpha=0.5, label=f'ZTF ${band}$ Band', **kwargs)
    return ax

def draw_lightcurve(ax, band, lightcurve, **kwargs):
    ax = draw_timeseries(ax=ax, band=band, time_array=lightcurve[:, 0], y_array=lightcurve[:, 1], y_err_array=lightcurve[:, 2], **kwargs)
    ax.set_xlabel('MJD', size=14)
    ax.set_ylabel('Magnitude', size=14)
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=14)
    return ax

def draw_dmdt_curve(ax, band, timeseries, **kwargs):
    ax = draw_timeseries(ax=ax, band=band, time_array=timeseries[:, 0], y_array=timeseries[:, 1], y_err_array=timeseries[:, 2], **kwargs)
    ax.set_xlabel('MJD', size=14)   
    ax.set_ylabel('$\\frac{dM}{dt}$ [Rate of Change in Magnitude]', size=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=14)
    return ax

# The following functions are used to differentiate time series data
#------------------------------------------------------------------------------
def differentiate_1D(x_arr, y_arr, yerr_arr):
    dy_arr = np.gradient(y_arr, x_arr)
    
    dyerr_arr = np.empty_like(dy_arr)
    for i in range(1, len(dy_arr) - 1):
        hd, hs = x_arr[i + 1] - x_arr[i], x_arr[i] - x_arr[i - 1]
        
        dy_err_squared = ((hs / (hd * (hd + hs))) * yerr_arr[i + 1])**2 + (((hd - hs) / (hd * hs)) * yerr_arr[i])**2 + ((hd / (hs * (hd + hs))) * yerr_arr[i - 1])**2
        dyerr_arr[i] = np.sqrt(dy_err_squared)
    
    dyerr_arr[0] = np.sqrt((yerr_arr[1]**2 + yerr_arr[0]**2) / (x_arr[1] - x_arr[0])**2)
    dyerr_arr[-1] = np.sqrt((yerr_arr[-1]**2 + yerr_arr[-2]**2) / (x_arr[-1] - x_arr[-2])**2)
    return x_arr, dy_arr, dyerr_arr


def differentiate_1D_manual(x_arr, y_arr, yerr_arr):
    dy_arr, dyerr_arr = np.zeros_like(y_arr), np.zeros_like(yerr_arr)

    for i in range(1, len(x_arr) - 1):
        hd, hs = x_arr[i + 1] - x_arr[i], x_arr[i] - x_arr[i - 1]

        dy = (y_arr[i] * (hd - hs) / (2 * hd * hs)) + (y_arr[i + 1] / (2 * hd)) - (y_arr[i - 1] / (2 * hs))
        dy_variance = (yerr_arr[i] * (hd - hs) / (2 * hd * hs))**2 + (yerr_arr[i + 1] / (2 * hd))**2 + (yerr_arr[i - 1] / (2 * hs))**2

        dy_arr[i] = dy
        dyerr_arr[i] = np.sqrt(dy_variance)

    dy_arr[0] = (y_arr[1] - y_arr[0]) / (x_arr[1] - x_arr[0])
    dy_arr[-1] = (y_arr[-1] - y_arr[-2]) / (x_arr[-1] - x_arr[-2])

    dyerr_arr[0] = np.sqrt((yerr_arr[1]**2 + yerr_arr[0]**2) / (x_arr[1] - x_arr[0])**2)
    dyerr_arr[-1] = np.sqrt((yerr_arr[-1]**2 + yerr_arr[-2]**2) / (x_arr[-1] - x_arr[-2])**2)
    return x_arr, dy_arr, dyerr_arr


def compute_derivatives_with_errors(lightcurve):
    t, m, sigma_m = lightcurve[:, 0], lightcurve[:, 1], lightcurve[:, 2]

    dm_dt = np.full_like(m, np.nan)
    sigma_dm_dt = np.full_like(m, np.nan)
    d2m_dt2 = np.full_like(m, np.nan)
    sigma_d2m_dt2 = np.full_like(m, np.nan)

    for i in range(1, len(t) - 1):
        t1, t2 = t[i - 1], t[i + 1]
        m1, m2 = m[i - 1], m[i + 1]
        dt = t2 - t1
        dm = m2 - m1

        dm_dt[i] = dm / dt
        sigma_m_term = (sigma_m[i - 1] ** 2 + sigma_m[i + 1] ** 2) / dt**2
        sigma_dm_dt[i] = np.sqrt(sigma_m_term)
    return t, dm_dt, sigma_dm_dt

    for i in range(1, len(t) - 1):
        t0, t1, t2 = t[i - 1], t[i], t[i + 1]
        m0, m1, m2 = m[i - 1], m[i], m[i + 1]
        
        dt1 = t1 - t0
        dt2 = t2 - t1
        if dt1 <= 0 or dt2 <= 0:
            continue
        fwd = (m2 - m1) / dt2
        bwd = (m1 - m0) / dt1
        d2m_dt2[i] = 2 * (fwd - bwd) / (dt1 + dt2)
        # Error propagation for second derivative
        s_m = sigma_m
        # s_t = sigma_t
        # Variance of fwd and bwd components
        var_fwd = (sigma_m[i+1]**2 + sigma_m[i]**2) / dt2**2
        var_bwd = (sigma_m[i]**2 + sigma_m[i-1]**2) / dt1**2
        sigma_d2m_dt2[i] = np.sqrt(4 * (var_fwd + var_bwd) / (dt1 + dt2)**2)
    # return dm_dt, d2m_dt2, sigma_dm_dt, sigma_d2m_dt2
    return t, dm_dt, sigma_dm_dt


def differentiate_lightcurve(lightcurve):
    time_array, mag_array, mag_err_array = lightcurve[:, 0], lightcurve[:, 1], lightcurve[:, 2]
    time_array, dmdt_array, dmdt_err_array = differentiate_1D(x_arr=time_array, y_arr=mag_array, yerr_arr=mag_err_array)
    
    filter = ~np.isnan(dmdt_array) & ~np.isnan(dmdt_err_array) & ~np.isinf(dmdt_array) & ~np.isinf(dmdt_err_array) & ~(dmdt_err_array==0)
    time_array, dmdt_array, dmdt_err_array = time_array[filter], dmdt_array[filter], dmdt_err_array[filter]
    if time_array.size < 2: return np.nan
    dmdt_curve = np.column_stack((time_array, dmdt_array, dmdt_err_array))
    return dmdt_curve


def differentiate_lightcurve_manual(lightcurve):
    time_array, mag_array, mag_err_array = lightcurve[:, 0], lightcurve[:, 1], lightcurve[:, 2]
    time_array, dmdt_array, dmdt_err_array = differentiate_1D_manual(x_arr=time_array, y_arr=mag_array, yerr_arr=mag_err_array)
    
    filter = ~np.isnan(dmdt_array) & ~np.isnan(dmdt_err_array) & ~np.isinf(dmdt_array) & ~np.isinf(dmdt_err_array) & ~(dmdt_err_array==0)
    time_array, dmdt_array, dmdt_err_array = time_array[filter], dmdt_array[filter], dmdt_err_array[filter]
    if time_array.size < 2: return np.nan
    dmdt_curve = np.column_stack((time_array, dmdt_array, dmdt_err_array))
    return dmdt_curve


def differentiate_lightcurve_PAnew(lightcurve):
    time_array, dmdt_array, dmdt_err_array = compute_derivatives_with_errors(lightcurve)
    
    filter = ~np.isnan(dmdt_array) & ~np.isnan(dmdt_err_array) & ~np.isinf(dmdt_array) & ~np.isinf(dmdt_err_array) & ~(dmdt_err_array==0)
    time_array, dmdt_array, dmdt_err_array = time_array[filter], dmdt_array[filter], dmdt_err_array[filter]
    if time_array.size < 2: return np.nan
    dmdt_curve = np.column_stack((time_array, dmdt_array, dmdt_err_array))
    return dmdt_curve


def differentiate_lightcurve_PA(lightcurve):
    mjds = lightcurve[:, 0]
    mags = lightcurve[:, 1]
    dm_dt = [((2 / (mjds[i + 1] - mjds[i - 1])) * (((mags[i + 1] - mags[i]) / (mjds[i + 1] - mjds[i])) - ((mags[i] - mags[i - 1]) / (mjds[i] - mjds[i - 1])))) for i in range(1, mjds.size - 1)]
    return np.array(dm_dt)

# Misc.
#------------------------------------------------------------------------------
def save_histogram_image(histogram, filepath, cmap=cm.viridis):
    """Saves a histogram image as it is shown in plt.imshow()

    Parameters
    ----------
    histogram : numpy.ndarray
        The 2D histogram.
    filepath : str
        Name and path for the saved image
    cmap : matplotlib.colors.ListedColormap, optional
        Colourmap to be used, by default cm.viridis
    """
    image = cmap(histogram)
    plt.imsave(f'{filepath}', image)