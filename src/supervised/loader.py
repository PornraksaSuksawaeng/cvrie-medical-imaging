import os
from matplotlib import image as mpimg
from pathlib import Path


def load_images(base_path: str | Path) -> tuple[list, list]:
    """
    Load all .jpg images from a folder structured by class subfolders,
    and assign each image the label corresponding to its parent folder name.
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
    return images, labels


def load_dataset(base_path: str | Path) -> tuple[list, list, list, list]:
    """
    Load the full supervised dataset by combining the Training and
    Testing subfolders into separate train/test sets.
    """
    X_train, y_train = load_images(os.path.join(base_path, "Training"))
    X_test, y_test = load_images(os.path.join(base_path, "Testing"))
    return X_train, y_train, X_test, y_test
