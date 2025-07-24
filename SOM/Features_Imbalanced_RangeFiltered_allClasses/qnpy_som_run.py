# %%
base_directory = '/home/shoaib/ZTFDataChallenge/'

import sys
sys.path.insert(0, base_directory + 'dmdt_Analysis/')
sys.path.insert(0, base_directory + 'SOM/')

from dmdt_functions import *
from SOM.QNPY_SOM_LC_Clustering import *

# %%
import numpy as np
import pandas as pd
from os import listdir
from PIL import Image
from minisom import MiniSom

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
### Loading the features by name
features_by_name = pd.read_parquet(base_directory + 'original_features_by_name.parquet')
# features_by_name = features_by_name.dropna(axis=0)
# features_by_name = features_by_name.query("type in @qso_types")
### Sampling to save memory
# features_by_name = features_by_name.sample(frac=0.1)

### Loading the features by OID
# features_by_oid = pd.read_parquet(base_directory + 'original_features_by_oid.parquet')
# features_by_oid = features_by_oid.dropna(axis=0)
# features_by_oid = features_by_oid.query("type in @qso_types")
### Sampling to save memory
# features_by_oid = features_by_oid.sample(frac=0.1)

# %%
### Loading the lightcurves by name
lightcurves_by_name = pd.read_pickle(base_directory + 'lightcurves_by_name_1day_binned.pkl')[['name', 'r_lightcurve','r_n_good_det','r_timespan_good','g_lightcurve','g_n_good_det','g_timespan_good']]
# lightcurves_by_name = lightcurves_by_name.dropna(axis=0)
# lightcurves_by_name = lightcurves_by_name.query("type in @qso_types")
### Sampling to save memory
# lightcurves_by_name = lightcurves_by_name.sample(frac=0.1)

### Loading the lightcurves by OID
# lightcurves_by_oid = pd.read_pickle(base_directory + 'lightcurves_by_oid_1day_binned.pkl')[['oid_alerce', 'lightcurve','n_good_det','timespan_good']]
# lightcurves_by_oid = lightcurves_by_oid.dropna(axis=0)
# lightcurves_by_oid = lightcurves_by_oid.query("type in @qso_types")
### Sampling to save memory
# lightcurves_by_oid = lightcurves_by_oid.sample(frac=0.1)

# %%
features_by_name = features_by_name.merge(right=lightcurves_by_name, on='name', how='inner', suffixes=('feat', None))
# features_by_oid = features_by_oid.merge(right=lightcurves_by_oid, on='oid_alerce', how='inner', suffixes=('feat', None))

# %%
feature_names = ['T2020_sigma2', 'mhps_ratio', 'mhps_low', 'mhps_high', 'Amplitude', 'AndersonDarling', 'Autocor_length',
                 'Beyond1Std', 'Con', 'Eta_e', 'Gskew', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev', 'MedianBRP', 'PairSlopeTrend',
                 'PercentAmplitude', 'Q31', 'PeriodLS_v2', 'Period_fit_v2', 'Psi_CS_v2', 'Psi_eta_v2', 'Rcs', 'Skew', 'SmallKurtosis',
                 'Std', 'StetsonK', 'Pvar', 'ExcessVar', 'GP_DRW_sigma', 'GP_DRW_tau', 'SF_ML_amplitude', 'SF_ML_gamma', 'IAR_phi',
                 'LinearTrend', 'Harmonics_mag_1', 'Harmonics_mag_2', 'Harmonics_mag_3', 'Harmonics_mag_4', 'Harmonics_mag_5',
                 'Harmonics_mag_6', 'Harmonics_mag_7', 'Harmonics_phase_2', 'Harmonics_phase_3', 'Harmonics_phase_4', 'Harmonics_phase_5',
                 'Harmonics_phase_6', 'Harmonics_phase_7', 'Harmonics_mse', 'mhps_non_zero', 'mhps_PN_flag']

band_feature_names = ['r_' + feature_name for feature_name in feature_names] + ['g_' + feature_name for feature_name in feature_names]
band_label_names = {i: feature_name for i, feature_name in enumerate(band_feature_names)}

# %%
### How big is data?
data = features_by_name[['type'] + band_feature_names]

print(f'There are {len(data)} rows in data right now.\n')
data['type'].value_counts()

# %%
feature_ranges = {
    'T2020_sigma2': (0, 0.2),
    'mhps_ratio': (0, 1000),
    'mhps_low': (0, 2),
    'mhps_high': (0, 0.2),
    'SmallKurtosis': (0, 20),
    'ExcessVar': (0, 0.0004),
    'GP_DRW_sigma': (0, 0.3),
    'GP_DRW_tau': (0, 4000),
    'SF_ML_amplitude': (0, 2),
    'LinearTrend': (-0.003, 0.003),
    'Harmonics_mag_1': (0, 400),
    'Harmonics_mag_2': (0, 400),
    'Harmonics_mag_3': (0, 400),
    'Harmonics_mag_4': (0, 400),
    'Harmonics_mag_5': (0, 400),
    'Harmonics_mag_6': (0, 400),
    'Harmonics_mag_7': (0, 400),
    'Harmonics_mse': (0, 0.1)
}

# %%
for feature in feature_names:
    if feature in feature_ranges:
        range_min, range_max = feature_ranges[feature]
        data = data[(data['r_' + feature] >= range_min) & (data['g_' + feature] <= range_max)]

print(f'There are {len(data)} rows in data right now.\n')
data['type'].value_counts()

# %%
data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0).reset_index(drop=True)

print(f'There are {len(data)} rows in data right now.\n')
data['type'].value_counts()

# %%
data_array = (data[band_feature_names] - np.mean(data[band_feature_names], axis=0)) / np.std(data[band_feature_names], axis=0)
data_array = data_array.values

# %% [markdown]
# ### Now train

# %%
learning_rate=0.1
sigma=1.0
train_mode='all'
batch_size=5
epochs=500

model_save_path=base_directory + f'SOM/Features_Imbalanced_RangeFiltered_allClasses/som_{epochs}epochs_{batch_size}bs_train{train_mode}_{learning_rate}lr_{sigma}sigma.p'

# %%
som_model, q_error, t_error, indices_to_plot = SOM_1D(data_array,
                                                      learning_rate=learning_rate,
                                                      sigma=sigma,
                                                      train_mode=train_mode,
                                                      batch_size=batch_size,
                                                      epochs=epochs,
                                                      model_save_path=model_save_path,
                                                      random_seed=21,
                                                      stat='qt',
                                                      plot_frequency=100,
                                                      early_stopping_no=None,
                                                 )

# %%



