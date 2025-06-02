import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback
from sklearn.metrics import classification_report


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
