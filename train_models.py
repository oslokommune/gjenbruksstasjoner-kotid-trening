import os
import pickle
import datetime

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

# Located in gjenbruksstasjoner-kotid-merking
from common import (
    crop_image,
    paint_everything_outside_ROI,
    ROI,
    XData,
    normalize_image,
    save_data,
    get_X_data,
    get_y_data,
    train_model,
    get_sample_weight,
)

np.random.seed(1)
assert tf.__version__[0] == "2"

# HARDCODED PARAMETERS
IMAGE_SHAPE = (110, 1227, 3)
LABELS_FILENAME = "../gjenbruksstasjoner-kotid-merking/labels_data.csv"
PICTURES_DIR = "../gjenbruksstasjoner-kotid-merking/actual_images"
CROPPED_DIR = "./cropped_images"
PRECONV_FILENAME = "VGG16_preconv.bin"


def crop_and_save_images(
    src_folder: str, dst_folder: str, roi: np.ndarray, force_all=False
):

    """
    Run the imported convolutional base on the cropped images.

    Parameters
    ----------
    src_folder:
        The folder of the images to be cropped.
    dst_folder:
        The destination folder of the images which have been cropped.
    roi:
        The Region of Interest which should be kept in the image.
    force_all, default False:
        If false, process only images which are present in the src_folder, but not the dst_folder.
        If True, delete everything in the dst_folder and process everything in the src_folder.

    Returns
    -------
        None
    """

    if not os.path.isdir(dst_folder):
        os.makedirs(dst_folder)

    if force_all:  # ...remove all files in this folder
        dst_images = os.listdir(dst_folder)
        for filename in dst_images:
            fullpath = os.path.join(dst_folder, filename)
            os.remove(fullpath)

    # Which images are in src, but not in dst?
    dst_images = os.listdir(dst_folder)
    src_images = os.listdir(src_folder)
    missing_in_dst = set(src_images).difference(set(dst_images))

    # Process and copy those missing in dst_folder
    # Normalization done at a later stage.
    i = 1
    for i, filename in enumerate(missing_in_dst):
        print(f"{i + 1}/{len(missing_in_dst)}: Cropping {filename}")
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        im = cv2.imread(src_path)
        im = paint_everything_outside_ROI(im, roi)
        im = crop_image(im, roi)
        cv2.imwrite(dst_path, im)


def run_convbase_on_images(src_folder, dst_file, image_shape, force_all=False):

    """
    Run the imported convolutional base on the cropped images.
    Save the result in a file.

    Parameters
    ----------
    src_folder (str):
        The folder of the images which have been cropped.
    dst_file (str):
        The destination file of images which have been processes in the convolutional base.
    force_all (bool), default False.
        If false, process only images which are present in the src_folder, but not the dst_folder.
        If True, delete everything in the dst_folder and process everything in the src_folder.
    Returns
    -------
        None
    """

    if force_all:
        xd = XData()
    else:
        try:
            with open(dst_file, "rb") as f:
                xd = pickle.load(f)
        except FileNotFoundError:
            xd = XData()

    src_images = os.listdir(src_folder)
    dst_images = xd.get_stored_filenames()
    missing_in_dst = list(set(src_images).difference(set(dst_images)))
    convbase = get_VGG16_convbase(image_shape)

    for i, filename in enumerate(missing_in_dst):
        print(f"{i + 1}/{len(missing_in_dst)}: Processing {filename}")
        src_path = os.path.join(src_folder, filename)
        im = cv2.imread(src_path)
        im = normalize_image(im)

        single_image_batch = np.array([im])
        after_convbase_single = convbase.predict(single_image_batch)
        after_convbase_single = after_convbase_single.flatten()

        xd.add_image(filename, after_convbase_single, assert_dimensions=1)

        if i % 100 == 0:
            print("Saving to avoid losing work if interrupted...")
            save_data(xd, dst_file)

    save_data(xd, dst_file)

    return xd


def get_VGG16_convbase(image_shape):

    """Get the convolution , pre-trained base."""

    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=image_shape)

    return conv_base


def define_dense_layers(input_shape, target_variable, hyperparameters):
    """Define the end layer(s) which will be trained based on the data processed through the convolutional base."""

    assert target_variable in ["queue_full", "queue_end_pos", "lanes"]
    assert type(input_shape) == tuple

    model = models.Sequential()
    model.add(
        layers.Dense(
            hyperparameters["layer1"], activation="relu", input_dim=input_shape[0]
        )
    )
    model.add(layers.Dropout(hyperparameters["dropout"]))
    model.add(layers.Dense(hyperparameters["layer2"], activation="relu"))
    model.add(layers.Dense(1, activation=None))

    # Final layer - binary, queue full?
    if target_variable == "queue_full":
        model.add(layers.Dense(1, activation="sigmoid"))  # Binary
    elif target_variable == "queue_end_pos":
        model.add(layers.Dense(1, activation=None))  # Regularization
    elif target_variable == "lanes":
        model.add(
            layers.Dense(1, activation="sigmoid")
        )  # Binary, 1=False, 2=True, only used when queue_empty==True

    return model


def plot_history(h, target_variable):
    """
    Save the progress of training and validation metrics to a file fig.png.
    The purpose is to choose a suitable number of training epochs.
    """

    assert target_variable in ["queue_full", "queue_end_pos", "lanes"]

    if target_variable == "queue_full":
        metric = h.history["accuracy"]
        val_metric = h.history["val_accuracy"]
    elif target_variable == "queue_end_pos":
        metric = h.history["mse"]
        val_metric = h.history["val_mse"]
    elif target_variable == "lanes":
        metric = h.history["accuracy"]
        val_metric = h.history["val_accuracy"]

    epochs = range(1, len(metric) + 1)
    ax = plt.subplot(111)
    ax.plot(epochs, metric, "b", label="Training acc")
    ax.plot(epochs, val_metric, "y", label="Validation acc")
    ax.set_ylim([0, int(np.min(val_metric) * 3.0)])
    ax.hlines(0, 0, len(epochs))
    ax.set_title("Training and validation accuracy")

    ax.legend()
    plt.savefig("fig.png")
    print("Check fig.png")


def print_data_shape(X_train=None, y_train=None, X_valid=None, y_valid=None):

    print("TRAINING DATA")
    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print("VALIDATION DATA")
    print(f"X_train.shape: {X_valid.shape}")
    print(f"y_train.shape: {y_valid.shape}")


def model_size_in_MB(model):
    """Simple function to get the size of the model in MB."""

    TMP_FILE_NAME = "tmp.h5"
    model.save(TMP_FILE_NAME)
    size_in_MB = round(os.stat(TMP_FILE_NAME).st_size / (1024 * 1024), 1)
    os.remove(TMP_FILE_NAME)
    return size_in_MB


def get_set(set_name: str, target_variable: str, xd: XData) -> tuple:
    """
    Return the a set of data (Train, Valid or Test) with X, y and
    filenames of the source files.
    """

    if set_name not in ["Train", "Valid", "Test"]:
        raise ValueError(f"set_name '{set_name}' is not 'Train', 'Valid' or 'Test'.")

    y = get_y_data(
        LABELS_FILENAME, set_name, target_variable, required_dtype=np.float64
    )
    filenames = list(y.index)
    X = get_X_data(filenames, xd)

    return X, y, filenames


def main(
    epochs, model_filename, target_variable, pictures_dir, cropped_dir, image_shape, roi
):

    # Some notes:
    # The approach here uses the VGG16 convolutional base to create a set of static pre-convoluted images.
    # A more robust (but computationally far more expensive) approach would be to use image augmentation
    # (i.e. doing small modifications to the images on the fly during the training), then run the convolutional
    # base on them on the fly.

    # Paint and crop all downloaded images
    crop_and_save_images(pictures_dir, cropped_dir, roi, force_all=False)

    # Run convbase on painted and cropped images to create static set of pre-convoluted images.
    xd = run_convbase_on_images(
        cropped_dir, PRECONV_FILENAME, image_shape, force_all=False
    )

    # Then train a model on top of this.
    X_train, y_train, train_filenames = get_set("Train", target_variable, xd)
    X_valid, y_valid, valid_filenames = get_set("Valid", target_variable, xd)

    print_data_shape(X_train, y_train, X_valid, y_valid)

    hyperparam_results = {}

    # Experiment with combinations of hyperparameters
    # The current selection of (125, 25, 0.2) is chosen because of reasonably good results and
    # being within the AWS storage space constraints (<= 100 MB) for the current setup.
    for layer1 in [125]:  # [100, 125]:
        for layer2 in [25]:  # , 50, 100]:
            for dropout in [0.2]:  # [0.1, 0.25, 0.5]:

                hp_str = "{0},{1},{2}".format(layer1, layer2, dropout)
                hyper_parameters = {
                    "layer1": layer1,
                    "layer2": layer2,
                    "dropout": dropout,
                }

                print(f"Hyper parameters: {hp_str}")

                # Define the architecture
                arch = define_dense_layers(
                    X_train.shape[1:], target_variable, hyper_parameters
                )

                # Set sample weights
                if target_variable == "queue_end_pos":
                    sample_weight = get_sample_weight(
                        train_filenames, LABELS_FILENAME, "queue_end_pos_2"
                    )
                elif target_variable == "lanes":
                    # Weigh 2 lanes higher - since they are underrepresented
                    sample_weight = get_sample_weight(
                        train_filenames, LABELS_FILENAME, "lanes_1"
                    )
                else:
                    sample_weight = None

                if target_variable in ["queue_end_pos"]:
                    target_type = "continuous"
                    metric = "val_mse"
                elif target_variable in ["queue_full", "lanes"]:
                    target_type = "binary"
                    metric = "val_accuracy"
                else:
                    raise Exception(f"{target_variable} not covered!")

                print(f"Samples: {len(y_train)}")

                model, history = train_model(
                    arch,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    target_type,
                    epochs=epochs,
                    sample_weight=sample_weight,
                )

                hyperparam_results[hp_str] = (
                    f"Model size: {model_size_in_MB(model)} MB",
                    history.history[metric][-1],
                )

    model.save(model_filename)
    load_model(model_filename)
    print("Successfully saved and test re-loaded model.")

    plot_history(history, target_variable)
    for key, value in sorted(hyperparam_results.items(), key=lambda item: item[1][1]):
        print(key, value)


if __name__ == "__main__":

    # Currently manually choosing which model to train and which number of epochs to use during the model training.
    # Typically this is done by using the standard number of epochs for each model or alternatively checking the
    # graphs with metrics for training and validation data, then setting the number of epochs based on when the
    # error metric starts to rise on the validation set.
    # A better approach would be to do automated hyperparameter optimization and keeping the model from the
    # lowest point of validation set error.

    dt = datetime.date.today().strftime("%Y%m%d")

    if False:  # Queue full
        epochs = 20
        cfg = {
            "model_file_name": "CNN_Queue_full_{0}.h5".format(dt),
            "target_variable": "queue_full",
        }  # Binary

    if True:  # End of Queue
        epochs = 30  # 50 is better
        cfg = {
            "model_file_name": "CNN_EoQ_VGG16_noaug_weighted_{0}.h5".format(dt),
            "target_variable": "queue_end_pos",
        }  # Continuous

    if False:  # Lanes
        epochs = 20
        cfg = {
            "model_file_name": "CNN_Lanes_VGG16_weighted_{0}.h5".format(dt),
            "target_variable": "lanes",
        }  # Binary

    main(
        epochs,
        cfg["model_file_name"],
        cfg["target_variable"],
        PICTURES_DIR,
        CROPPED_DIR,
        IMAGE_SHAPE,
        ROI,
    )

    print("That was fun!")
