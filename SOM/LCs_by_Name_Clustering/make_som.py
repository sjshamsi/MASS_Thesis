# %%
base_directory = '/home/shoaib/ZTFDataChallenge/'

import sys
sys.path.insert(0, base_directory + 'SOM/')

from QNPY_SOM_LC_Clustering import *

# %%
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas(desc="Lightcurves Processed")

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
plt.rcParams['axes.grid'] = False
# plt.style.use('seaborn-v0_8-colorblind')

# %%
### Loading the lightcurves by name
lightcurves_by_name = pd.read_pickle(base_directory + 'lightcurves_by_name_1day_binned.pkl')[['name', 'r_lightcurve','r_n_good_det','r_timespan_good','g_lightcurve','g_n_good_det','g_timespan_good']]
lightcurves_by_name = lightcurves_by_name.dropna(axis=0)
# lightcurves_by_name = lightcurves_by_name.query("type in @qso_types")
### Sampling to save memory
# lightcurves_by_name = lightcurves_by_name.sample(1000)

### Loading the lightcurves by OID
# lightcurves_by_oid = pd.read_pickle(base_directory + 'lightcurves_by_oid_1day_binned.pkl')[['oid_alerce', 'lightcurve','n_good_det','timespan_good']]
# lightcurves_by_oid = lightcurves_by_oid.dropna(axis=0)
# lightcurves_by_oid = lightcurves_by_oid.query("type in @qso_types")
### Sampling to save memory
# lightcurves_by_oid = lightcurves_by_oid.sample(frac=0.1)

# %%
lcs = lightcurves_by_name['r_lightcurve'].values

# %%
scaled_padded_lcs = np.array([scale_curve(LC) for LC in tqdm(lcs, desc='Scaling')], dtype='object')

# %%
time_bin_width = 1
timespan_max = 1150

scaled_padded_lcs, time_bins = pad_lightcurves(scaled_padded_lcs, timespan_max=timespan_max, bin_width=time_bin_width)

# %%
### Let's see some of these scaled and padded lightcurves

n_curves = 5
subset_idxs = np.random.choice(len(scaled_padded_lcs), n_curves, replace=False)
lcs_subset = scaled_padded_lcs[subset_idxs]

for mag_array in lcs_subset:
    plt.scatter(time_bins, mag_array, s=4)

plt.ylabel('Scaled Magnitude', size=14)
plt.xlabel('MJD', size=14)

plt.show()


# %% [markdown]
# ## Making SOMs

# %% [markdown]
# ### $(1 \times 3)$

# %%
som_x = 1
som_y = 3
use_epochs = False
num_iterations_per_batch = 10_000
batch_size = 5
log_stat_every_n_batch = 1
model_save_dir = base_directory + 'SOM/LCs_by_Name_Clustering/'
model_name = f'som_{som_x}x{som_y}_{num_iterations_per_batch}Iters_{use_epochs}Epochs_{batch_size}BS_{log_stat_every_n_batch}Stat_Freq'
model_save_path = model_save_dir + model_name

som_model, q_error, t_error, indices_to_plot = SOM_1D(scaled_padded_lcs,
                                                      model_save_path=model_save_path,
                                                      n_random_selections=None,
                                                      som_x=som_x,
                                                      som_y=som_y,
                                                      use_epochs=use_epochs,
                                                      pca_init=False,
                                                      num_iterations_per_batch=num_iterations_per_batch,
                                                      batch_size=batch_size,
                                                      log_stat_every_n_batch=log_stat_every_n_batch,
                                                    )

# %%
som_x = 6
som_y = 6
use_epochs = False
num_iterations_per_batch = 10_000
batch_size = 5
log_stat_every_n_batch = 1
model_save_dir = base_directory + 'SOM/LCs_by_Name_Clustering/'
model_name = f'som_{som_x}x{som_y}_{num_iterations_per_batch}Iters_{use_epochs}Epochs_{batch_size}BS_{log_stat_every_n_batch}Stat_Freq'
model_save_path = model_save_dir + model_name

som_model, q_error, t_error, indices_to_plot = SOM_1D(scaled_padded_lcs,
                                                      model_save_path=model_save_path,
                                                      n_random_selections=None,
                                                      som_x=som_x,
                                                      som_y=som_y,
                                                      use_epochs=use_epochs,
                                                      pca_init=False,
                                                      num_iterations_per_batch=num_iterations_per_batch,
                                                      batch_size=batch_size,
                                                      log_stat_every_n_batch=log_stat_every_n_batch,
                                                    )

# %%
som_x = int(np.ceil(np.cbrt(scaled_padded_lcs.shape[0])))
som_y = int(np.ceil(np.cbrt(scaled_padded_lcs.shape[0])))
use_epochs = False
num_iterations_per_batch = 10_000
batch_size = 5
log_stat_every_n_batch = 1
model_save_dir = base_directory + 'SOM/LCs_by_Name_Clustering/'
model_name = f'som_{som_x}x{som_y}_{num_iterations_per_batch}Iters_{use_epochs}Epochs_{batch_size}BS_{log_stat_every_n_batch}Stat_Freq'
model_save_path = model_save_dir + model_name

som_model, q_error, t_error, indices_to_plot = SOM_1D(scaled_padded_lcs,
                                                      model_save_path=model_save_path,
                                                      n_random_selections=None,
                                                      som_x=som_x,
                                                      som_y=som_y,
                                                      use_epochs=use_epochs,
                                                      pca_init=False,
                                                      num_iterations_per_batch=num_iterations_per_batch,
                                                      batch_size=batch_size,
                                                      log_stat_every_n_batch=log_stat_every_n_batch,
                                                    )


