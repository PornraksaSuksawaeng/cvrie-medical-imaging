import os
from matplotlib import image as mpimg
import numpy as np
from pathlib import Path


def load_images(base_path: str | Path) -> tuple[list, list]:
    """Load all the .jpg images in base_path and use the folder name as label.

    There is one folder per class, so the folder name is the label. The images
    are not all the same: they have different sizes, and some are in grey
    (h, w) while others are in color (h, w, 3).
    """
    images = []
    labels = []

    for foldername in os.listdir(base_path):
        folder_path = os.path.join(base_path, foldername)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img = mpimg.imread(os.path.join(folder_path, filename))
                images.append(img)
                labels.append(foldername)
    return images, np.array(labels)


def load_dataset(base_path: str | Path) -> tuple[list, list, list, list]:
    """Load the train and test sets from the Training and Testing folders.

    The dataset is already split in two folders, so we keep that split instead
    of making a new one.
    """
    X_train, y_train = load_images(os.path.join(base_path, "Training"))
    X_test, y_test = load_images(os.path.join(base_path, "Testing"))
    return X_train, y_train, X_test, y_test
