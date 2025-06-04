import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
from sklearn.metrics import classification_report


fit_params = {'save_directory': None,
              'model': None,
              'datablock': None,
              'dataloader': None,
              'learner': None,
              
              'validation_percentage': None,
              'histogram_array': None,
              'label_array': None,
              'image'
              }


class MyLogger:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path

    def log_write(self, string):
        with open(self.logfile_path, 'a') as f:
            f.write(string + '\n')


class FastaiRun:
    def __init__(self, save_directory, longitude, latitude, object_name, sky_model_name, observation_hours, observation_filepath, original_spectral_model, original_spectral_energy_bounds):
        self.save_directory = save_directory
        self.logger = MyLogger(save_directory + "log.txt")
        self.longitude = longitude
        self.latitude = latitude
        self.object_name = object_name
        self.sky_model_name = sky_model_name
        self.observation_hours = observation_hours
        self.observation_filepath = observation_filepath
        self.original_spectral_model = original_spectral_model
        self.original_spectral_energy_bounds = original_spectral_energy_bounds

        self._prepare_empty_directory(self.save_directory)
        self._make_point_model()
        self._save_original_spectral_model()


    def _prepare_empty_directory(self, path):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise NotADirectoryError(f"Path exists but is not a directory: {path}")
            if os.listdir(path):  # directory is not empty
                raise FileExistsError(f"Directory '{path}' is not empty.")
        else:
            os.makedirs(path)


    def _make_point_model(self):
        self.point_model = PointSpatialModel(lon_0=f"{self.longitude:.3f} deg", lat_0=f"{self.latitude:.3f} deg", frame='icrs')
        self.point_model.plot()
        plt.savefig(self.save_directory + "point_model.png")
        plt.close()


    def _save_original_spectral_model(self):
        self.original_spectral_model.plot(self.original_spectral_energy_bounds, sed_type="e2dnde")
        plt.grid(which="both")
        plt.savefig(self.save_directory + "original_spectral_model.png")
        plt.close()        


    def _make_observation(self):
        #IRFS = load_cta_irfs(filename)
        IRFS =  load_irf_dict_from_file(self.observation_filepath)
        livetime = self.observation_hours * u.hr
        obs_id = "0001"
        fixed_icrs = SkyCoord(ra=f"{self.longitude:.3f} deg", dec=f"{self.latitude:.3f} deg",frame="icrs")
        pointing = FixedPointingInfo(fixed_icrs=fixed_icrs)
        self.observation = Observation.create(obs_id=obs_id, pointing=pointing, livetime=livetime, irfs=IRFS)
        self.logger.log_write('\n' + str(self.observation) + '\n')


    def make_original_dataset(self):
        sky_model = SkyModel(spectral_model=self.original_spectral_model, spatial_model=self.point_model,
                             # temporal_model=expdecay_model,
                             name=self.object_name)
        self.bkg_model = FoVBackgroundModel(dataset_name="my_dataset")
        models = Models([sky_model, self.bkg_model])

        models[0].spatial_model.parameters['lon_0'].frozen=True
        models[0].spatial_model.parameters['lat_0'].frozen=True
        self.logger.log_write('\n' + str(models) + '\n')
        
        self._make_observation()

        energy_axis = MapAxis.from_energy_bounds("0.01 TeV", "100 TeV", nbin=10, per_decade=True)
        energy_axis_true = MapAxis.from_energy_bounds("0.01 TeV", "200 TeV", nbin=20, per_decade=True, name="energy_true")
        migra_axis = MapAxis.from_bounds(0.5, 2, nbin=150, node_type="edges", name="migra")
        pointing = SkyCoord(ra=f"{self.longitude:.3f} deg", dec=f"{self.latitude:.3f} deg", frame="icrs")

        self.geom = WcsGeom.create(skydir=pointing, width=(4, 4), binsz=0.02, frame="icrs", axes=[energy_axis])
        self.logger.log_write('\n' + str(self.geom) + '\n')

        empty = MapDataset.create(self.geom, energy_axis_true=energy_axis_true, migra_axis=migra_axis, name="my_dataset")
        maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
        self.dataset = maker.run(empty, self.observation)
        self.dataset.models = models
        self.dataset_original = models.copy()
        self.logger.log_write('\nORIGINAL DATASET\n' + str(self.dataset) + '\n')


    def _generate_TS_map(self, model):
        dataset_TS = self.dataset.to_image()
        estimator = TSMapEstimator(model=model)

        images = estimator.run(dataset_TS)
        self.logger.log_write('\n' + str(images) + '\n')

        images["ts"].plot(add_cbar=True, vmin=0, vmax=100)
        plt.savefig(self.save_directory + "ts_map.png")
        plt.close()
        # images["ts"].write('NGC1068_50hr_TSmap.fits', hdu='IMAGE', overwrite=True)  # write a file
        # filename = 'NGC1068_50hr_TSmap.png'   # Save png
        #save_figure(filename)
        return images

    
    def _generate_source_map(self, images):
        self.sources = find_peaks(images["sqrt_ts"], threshold=5, min_distance="0.25 deg")
        nsou = len(self.sources)
        self.logger.log_write(f'\nNUMBER OF SOURCES: {nsou}\n')
        self.sources.write(self.save_directory + 'sources.csv', format='csv', overwrite=True)

        if nsou > 0:
            plt.figure(figsize=(9, 5))
            ax = images["sqrt_ts"].plot(add_cbar=True)
            ax.scatter(self.sources["ra"], self.sources["dec"], transform=ax.get_transform("icrs"), color="none", edgecolor="w", marker="o", s=600, lw=1.5)
            plt.savefig(self.save_directory + 'source_map.png')
            plt.close()

    
    def run_simulation(self,):
        seed=0
        sampler = MapDatasetEventSampler(random_state=seed)
        events = sampler.run(self.dataset, self.observation)
        # events.table.write(self.save_directory + 'events_table.csv', format='csv', overwrite=True)
        self.logger.log_write("\n" + f"Source events: {(events.table['MC_ID'] == 1).sum()}" + "\n")
        self.logger.log_write("\n" + f"Background events: {(events.table['MC_ID'] == 0).sum()}" + "\n")
        events.peek()
        plt.savefig(self.save_directory + 'events_peek.png')
        plt.close()
        src_pos = SkyCoord(ra=f"{self.longitude:.3f} deg", dec=f"{self.latitude:.3f} deg", frame="icrs")
        region_sky = CircleSkyRegion(center=src_pos, radius=1.0 * u.deg)
        evt = events.select_region(region_sky)
        evt.peek()
        plt.savefig(self.save_directory + 'zoomed_events_peek.png')
        plt.close()

        counts = Map.from_geom(self.geom)
        counts.fill_events(events)
        counts.sum_over_axes().plot(add_cbar=True)
        plt.savefig(self.save_directory + 'sky_map.png')
        plt.close()

        spectral_model = PowerLawSpectralModel(amplitude=2.5e-12 * u.Unit("cm-2 s-1 TeV-1"), index=2.2, reference=1 * u.TeV)
        model = SkyModel(spectral_model=spectral_model, spatial_model=self.point_model,
                         # temporal_model=expdecay_model,
                         name=self.sky_model_name)
        models = Models([model, self.bkg_model])
        self.logger.log_write('\nmodels to YAML\n' + str(models.to_yaml()))
        file_model = self.save_directory + f"{self.sky_model_name}.yaml"
        models.write(file_model, overwrite=True)
        models_fit = Models.read(self.save_directory + f"{self.sky_model_name}.yaml")
        self.dataset.counts = counts
        self.dataset.models = models_fit
        self.logger.log_write('\nFITTED DATASET' + str(self.dataset) + '\n')

        fit = Fit()
        result = fit.run(self.dataset)
        self.logger.log_write('\nRESULTS' + str(result) + '\n')
        result.parameters.to_table().write(self.save_directory + 'result_parameters.csv', format='csv', overwrite=True)
        models.to_parameters_table().write(self.save_directory + 'models_parameters.csv', format='csv', overwrite=True)

        images = self._generate_TS_map(model=model)
        self._generate_source_map(images)


    def spectral_analysis(self):
        e_min, e_max, e_bins = 0.01, 50, 8
        e_edges = np.logspace(np.log10(e_min), np.log10(e_max), e_bins) * u.TeV
        self.logger.log_write('\nE EDGES\n' + str(e_edges) + '\n')

        fpe = FluxPointsEstimator(energy_edges=e_edges, source=self.sky_model_name, selection_optional="all")
        flux_points = fpe.run(datasets=self.dataset)
        flux_points.sqrt_ts_threshold_ul = 3
        flux_point_table = flux_points.to_table(sed_type="dnde", formatted=True)

        self.logger.log_write(str(flux_point_table["ts"]))
        self.logger.log_write('Emin= ' + str(flux_points.energy_min))
        self.logger.log_write('Emax= ' + str(flux_points.energy_max))
        self.logger.log_write(str(flux_point_table["counts"]))
        self.logger.log_write('##################################')
        self.logger.log_write(str(flux_point_table))

        # flux_point_table.write(self.save_directory + 'flux_point_table.ecsvecsv', format='ecsv', overwrite=True)

        plt.figure(figsize=(8, 5))
        ax = flux_points.plot(sed_type="e2dnde", color="darkorange")
        flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde")
        plt.savefig(self.save_directory + 'ts_profiles.png')
        plt.close()

        ax = self.dataset_original[0].spectral_model.plot(energy_bounds=[e_min, e_max] * u.TeV, sed_type="e2dnde", label="Sim. model", color='black')
        px = {"color": "red", "marker":"o"}
        mx = {"color": "blue"}
        flux_points_dataset = FluxPointsDataset(data=flux_points, models=self.dataset.models[0])
        flux_points_dataset.plot_spectrum(ax=ax, kwargs_fp=px, kwargs_model=mx)
        ax.legend()
        plt.savefig(self.save_directory + f'{self.object_name}_{self.observation_hours}')

        #ax.set_xlim(0.67,200)
        #ax.set_ylim(2e-12,1e-10)

        #filename = "NGC1068_50hr_spectrum.png"
        #save_figure(filename)
        print(flux_points_dataset)



MODEL_SAVE_PATH = Path('/content/drive/MyDrive/Thesis_Files/Thesis/dmdt_Analysis/Models/')
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


def numpy_to_pil(numpy_array):
    numpy_array = (numpy_array - numpy_array[:, :, :-1].min()) / (numpy_array[:, :, :-1].max() - numpy_array[:, :, :-1].min())
    numpy_array[:, :, 2] = np.zeros_like(numpy_array[:, :, 2])
    return PILImage.create(Image.fromarray((numpy_array * 255).astype(np.uint8)))

def density(numpy_array):
    numpy_array = (numpy_array - numpy_array[:, :, :-1].min()) / (numpy_array[:, :, :-1].max() - numpy_array[:, :, :-1].min())
    numpy_array[:, :, 2] = np.zeros_like(numpy_array[:, :, 2])
    return numpy_array

class FastAI_Fit:
    def __init__(self, df: pd.DataFrame, data_column_name: str, label_column_name: str,
                 model, batch_size: int, validation_percentage: float, model_save_name: str):
        self.df = df
        self.data_column_name = data_column_name
        self.label_column_name = label_column_name
        self.model = model
        self.model_save_name = model_save_name
        self.batch_size = batch_size
        self.validation_percentage = validation_percentage
        self.dls = self.create_dataloaders()
        
    def create_dataloaders(self):
        """Creates FastAI DataLoaders from the given DataFrame."""
        dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=lambda r: numpy_to_pil(r[self.data_column_name]),  # Convert NumPy array to image
            get_y=lambda r: r[self.label_column_name],     
            splitter=RandomSplitter(valid_pct=self.validation_percentage),
            item_tfms=Resize(224)
        ).dataloaders(self.df, bs=self.batch_size)
        return dls

    def initialise_learner(self):
        self.learn = vision_learner(self.dls, self.model, metrics=[accuracy, error_rate])

    def train(self, epochs: int = 20, lr: float = None, show_lr_plot=False):
        self.initialise_learner()
        if not lr:
            self.lr_min = self.learn.lr_find(show_plot=show_lr_plot).valley
        else:
            self.lr_min = lr
        self.learn.fine_tune(epochs, base_lr=self.lr_min,
                             cbs=[SaveModelCallback(monitor='valid_loss', comp=np.less, fname=MODEL_SAVE_PATH/self.model_save_name),
                                  EarlyStoppingCallback(monitor='valid_loss', patience=3)])
    
    def plot_confusion_matrix(self, ax, title):
        self.interp = ClassificationInterpretation.from_learner(self.learn)
        cm = self.interp.confusion_matrix()
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        im = ax.imshow(cm_percent, cmap='Blues', interpolation='nearest')

        for i in range(len(cm_percent)):
            for j in range(len(cm_percent[i])):
                text = f"{cm_percent[i, j]:.1f}%\n({int(cm[i, j])})"
                textcolour = color = "white" if cm_percent[i, j] > 50 else "black"
                ax.text(j, i, text, ha="center", va="center", color=textcolour)

        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks(range(len(self.interp.vocab)))
        ax.set_yticks(range(len(self.interp.vocab)))
        ax.set_xticklabels(self.interp.vocab)
        ax.set_yticklabels(self.interp.vocab)

    def plot_losses(self, ax):
        self.learn.recorder.plot_losses(ax=ax)

    def get_classitifation_report(self):
        preds, targets = self.learn.get_preds()
        pred_classes = preds.argmax(dim=1)
        
        report = classification_report(targets, pred_classes, target_names=self.dls.vocab)
        return report

    def load_model(self, model_path):
        self.initialise_learner()
        self.learn.load(model_path)
