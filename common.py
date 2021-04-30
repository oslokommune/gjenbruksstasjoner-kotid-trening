import sys
import pandas as pd
import numpy as np
import cv2
import pickle

pd.set_option("display.max_columns", None)

ROI = np.array([[0, 132], [0, 211], [1227, 125], [1075, 101]], dtype=np.int32)


def assert_target_variable(target_variable):

    """
    Assert that one or more target_variables are valid.
    The argument may be a list or str.
    """

    assert isinstance(target_variable, (list, str, tuple))

    valid = ["queue_full", "queue_end_pos", "lanes"]

    if isinstance(target_variable, str):
        target_variables = [target_variable]
    else:
        target_variables = target_variable

    for tv in target_variables:
        assert tv in valid


def open_labels_data(fullpath):

    """Open this file and returns it as suitable formatted pd.DataFrame."""

    assert isinstance(fullpath, str)
    assert fullpath[-4:].lower() == ".csv"

    df = pd.read_csv(fullpath, index_col=0)

    # Set dtype releflecting the underlying nature of the data.
    # May be cast depending on useage.
    df["open"] = df["open"].astype("bool")
    df["set_type"] = df["set_type"].astype("object")
    df["queue_full"] = df["queue_full"].astype("bool")
    df["queue_empty"] = df["queue_empty"].astype("bool")
    df["lanes"] = df["lanes"].astype("Int64")

    return df


def paint_everything_outside_ROI(im, roi):
    """Paint everything outside the ROI white to remove noise (useless information)."""

    try:
        assert type(im) == np.ndarray
    except AssertionError:
        print("Problem in common.paint_everything_outside_ROI")
        print(type(im))
        sys.exit(1)

    assert type(roi) == np.ndarray

    mask = np.ones_like(im) * 255
    mask = cv2.drawContours(mask, [roi], -1, 0, -1)
    im = np.where(mask == 255, mask, im)

    return im


def crop_image(im, roi):
    """Keep a rectangle minimized around the ROI (the reduce network size and useless processing)."""

    assert type(im) == np.ndarray
    assert type(roi) == np.ndarray

    x_min = roi[:, 0].min()
    x_max = roi[:, 0].max()
    y_min = roi[:, 1].min()
    y_max = roi[:, 1].max()

    im = im[y_min:y_max, x_min:x_max].copy()

    return im


def normalize_image(im):
    """Normalize the image, since ANNs works best with small values."""

    im = im.astype(np.float64)
    im = im * (1.0 / 255)

    assert im.min() >= 0
    assert im.max() <= 1

    return im


def save_data(object, full_path):

    with open(full_path, "wb") as f:
        pickle.dump(object, f)


def show_im(im):

    """Mostly meant for debugging"""

    while True:

        cv2.imshow("Image", im)
        k = cv2.waitKey(1)
        if k % 256 == 32:
            break

    cv2.destroyAllWindows()


class XData(object):
    """A class which can contain images or other numpy arrays for retrieval."""

    def __init__(self):
        self.content = {}  # Key: file_name, Value: numpy-array

    def add_image(self, file_name, im, assert_dimensions=3):
        """Add a numpy array, with a reference string (key), most likely the filename."""
        assert len(im.shape) == assert_dimensions
        self.content[file_name] = im.copy()

    def get_image(self, file_name):
        """Return a single image (or data processed from it)."""
        im = self.content[file_name].copy()
        return im

    def get_stored_filenames(self):
        """Return a list of the filenames which are currently stored in this object."""
        return list(self.content.keys())

    def get_collection_array(self, list_of_filenames):
        """
        Returns a 4d array of the images (or data processed from it)
        in the same order as the file names.
        """
        images = [
            self.content[fn] for fn in list_of_filenames if fn in self.content.keys()
        ]
        data = np.array(images)
        return data


def get_sample_weight(filenames, label_filename, rule_set, numpy=True):
    """
    Given the relevant rule_set, returns a suitable set of weights.

    Parameters
    ----------
    filenames (list):
        A list of filenames for the relevant images.
    label_filename (str):
        The full path of the label file.
    rule_set (str):
        A reference to the rules will assigns sample_weights.

    Return (if numpy==True)
    -----------------------
    sample_weights (np.ndarray):
        The weight of each sample.

    Return (if numpy==False)
    -----------------------
    sample_weight (pd.Series):
        The weight of each sample. filenames as index.
    """

    assert rule_set in ["queue_end_pos_1", "queue_end_pos_2", "lanes_1"]

    df = open_labels_data(label_filename)

    sample_weight = pd.Series(
        data=np.repeat(1, df.shape[0]).astype(np.float64), index=df.index
    )

    sample_weight = sample_weight.loc[filenames]

    if rule_set == "queue_end_pos_1":
        sample_weight.loc[df["queue_end_pos"].notnull()] = 1.0

    if rule_set == "queue_end_pos_2":
        sample_weight.loc[df["queue_end_pos"].notnull()] = 1.0
        sample_weight.loc[df["queue_full"]] = 0.5
        sample_weight.loc[df["queue_empty"]] = 0.5

    if rule_set == "lanes_1":
        sample_weight.loc[df["lanes"].notnull()] = 1.0
        sample_weight.loc[df["lanes"] == 1] = 1.0
        sample_weight.loc[
            df["lanes"] == 2
        ] = 3.0  # There are roughly 4x as many images with 1 lane as with 2.
        sample_weight.loc[df["lanes"].isnull()] = 0.0

    # print(sample_weight.value_counts())

    assert isinstance(sample_weight, pd.Series)

    if numpy:
        sample_weight = sample_weight.values
        assert isinstance(sample_weight, np.ndarray)

    return sample_weight


def convert_float_lanes_to_boolean(y, input_is_12):
    """
    For Haraldrud, convert the float number of actual observations or predictions to boolean values
    (~2 lanes = True, ~1 lane = False)

    To be used during training and evaluation.

    Parameters
    ----------
    y (pd.Series):
        A series with
    input_is_12 (bool):
        If False, the input floats range from 0-1, which is equivalent of 1-2 lanes.
        If True, the input floats range from 1-2, which is equivalent of 1-2 lanes.

    Return
    ------
    y_bool (pd.Series):
        A boolean series. The index is the same as in the parameter y.
    """

    assert isinstance(y, pd.Series)
    try:
        assert y.dtype in [np.int32, np.int64, np.float32, np.float64]
    except AssertionError:
        print(y.dtype)
        raise AssertionError

    if input_is_12:
        THRESHOLD = 1.5
    else:
        THRESHOLD = 0.5

    y_bool = y.map(lambda x: True if x > THRESHOLD else False)

    return y_bool


def get_X_data(filenames, XData, XData_filename=None):
    """
    By either using the XData object directly or reading it from a file, this function returns the X-data for the
    mentioned files (probably taken from the index of the y-data).

    NOTE: This approach is not scalable as the amount of data in the training set increases. Switch to generators.

    Parameters
    ----------
    filenames (list):
        A list of the filenames to be retrieved.
    XData (XData):
        The XData object or alternatively None if XData_filename is given.
    XData_filename (str, default=None):
        Must be provided if XData is not provided.

    Return
    ------
    X (np.ndarray):
        The first dimension should have the same length as the list of filenames.
    """

    assert isinstance(filenames, list)

    if (XData is None) and (XData_filename is not None):
        with open(XData_filename, "rb") as f:
            XData = pickle.load(f)

    X = XData.get_collection_array(filenames)

    assert isinstance(X, np.ndarray)
    assert len(filenames) == X.shape[0]

    return X


def get_filenames(labels_filename, target_variable, set_type="All"):
    """
    Opens the (csv) file with the labelled data and returns a list of
    filenames which are:
    a) Labelled
    b) Relevant
    c) Has a value (notnull) for the target_variable
       TBD: UNLESS QUEUE_END_POS!
    d) Are in the specified set_type, unless "All" is given, then
       Train, Valid and Test are all included.

    Parameters
    ----------
    labels_filename (str):
        The full path to the file where the labels are stored.
    target_variable (str):
        The variable which is the y-value to be predicted.
    set_type (str):
        Specifies if data from Train, Valid, Test or alternatively all
        should be included.

    Return
    ------
    filenames (list):
        A list taken from the label file, after filtering on the
        criteria explained above.
    """

    assert_target_variable(target_variable)
    assert set_type in ["All", "Train", "Valid", "Test"]

    df = open_labels_data(labels_filename)

    df = df.loc[df["labelled"], :].copy()
    df = df.loc[df["relevant"], :].copy()

    df = df.loc[df[target_variable].notnull(), :].copy()

    if set_type in ["Train", "Valid", "Test"]:
        df = df.loc[df["set_type"] == set_type, :].copy()

    filenames = list(df.index)
    assert isinstance(filenames, list)

    return filenames


def get_y_data(labels_filename, set_type, target_variable, required_dtype=None):
    """
    Opens the (csv) file with the labelled data and returns as a
    pd.Series.

    Parameters
    ----------
    labels_filename (str):
        The filename (or full path) of the csv file.
    set_type (str):
        "Train", "Valid" or "Test".
    target_variable (str):
        Which variable should get extract and train for?
    required_dtype (type):
        The function will try to cast the values into this dtype.

    Returns
    -------
    y (pd.Series):
        The series of y-data, with the filename of the images in the
         index.
    """

    assert set_type in ["Train", "Valid", "Test"]
    assert target_variable in ["queue_full", "queue_end_pos", "lanes"]

    # Read the relevant data of the correct set_type. Do some minor house keeping.
    df = open_labels_data(labels_filename)

    # Fill in data (move to separate function)
    df.loc[df["queue_empty"], "queue_end_pos"] = 0
    df.loc[df["queue_full"], "queue_end_pos"] = 1138
    df["relevant"] = df["relevant"].mask(df["relevant"].isna(), False)

    # Filter away generally useless data
    df = df.loc[df["relevant"], :]
    df = df.loc[df["set_type"] == set_type, :]

    # Filter away specifically useless data
    df = df.loc[df[target_variable].notnull(), :].copy()
    y = df.loc[:, target_variable]

    ### COUPLED LOGIC _ TAKE OUT
    if target_variable == "lanes":
        assert set(df[target_variable]) == {1, 2}  # Assert no None / np.nan values
        y = y.map(lambda x: 1.0 if (x == 2) else 0.0)

    if required_dtype is not None:
        y = y.astype(required_dtype)

    return y


def train_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    target_type,
    epochs=30,
    sample_weight=None,
):

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, (np.ndarray, pd.Series))
    assert isinstance(X_valid, np.ndarray)
    assert isinstance(y_valid, (np.ndarray, pd.Series))
    assert target_type in ["binary", "continuous"]

    if target_type == "binary":

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    elif target_type == "continuous":

        model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        sample_weight=sample_weight,
        shuffle=True,
    )

    return model, history
