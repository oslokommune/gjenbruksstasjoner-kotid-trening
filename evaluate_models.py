import os
import functools
import cv2
import datetime
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from sklearn.metrics import confusion_matrix

from common import XData
from common import convert_float_lanes_to_boolean
from common import get_filenames
from common import assert_target_variable

from train_models import (
    PRECONV_FILENAME,
    LABELS_FILENAME,
    PICTURES_DIR,
)

LABELS_FILENAME = "../gjenbruksstasjoner-kotid-merking/labels_data.csv"
CV2_NP_FOLDER = "OpenCV_Numpy"
END_OF_QUEUE_POS = 1138  # Warning - This number is defined also in the estimering repo.


def assert_evaluable_on(method):
    """These are the valid entries for evaluable_on."""

    if method not in [
        "with_confusion_matrix",
        "with_residuals",
        "show_on_images",
        "dump_to_file",
    ]:
        raise ValueError(f"{method} is not a valid method for evaluation.")


class Series_y(object):
    """
    Predictions from a model or actual true values.
    """

    def __init__(self, name, actual, color, target_variable):

        assert isinstance(name, str)
        self.name = name

        assert isinstance(actual, bool)
        self.actual = actual  # If True, an actual true result. If false, these results are predictions.

        assert isinstance(color, tuple)
        assert len(color) == 3
        self.color = color

        assert isinstance(target_variable, str)
        self.target_variable = target_variable

        self.results = None

    def set_results(self, results):

        assert type(results) == pd.Series
        self.results = results

    def get_y(self, item):

        return self.results[item]

    def get_filenames(self):

        return list(self.results.index)

    def get_number_of_entries(self):

        if self.results is not None:
            return len(self.results)
        else:
            raise Exception("Results not set yet.")


def get_imdata(imdata_filename):

    with open(imdata_filename, "rb") as f:
        imdata = pickle.load(f)

    return imdata


def get_y_data(labels_filename, filenames, target_variable):
    """Get the actual results for a target variable."""

    assert_target_variable(target_variable)

    df = pd.read_csv(labels_filename, index_col=0)

    # Make sure the data is in the right data type
    df["queue_full"] = df["queue_full"].astype("bool")
    df["queue_empty"] = df["queue_empty"].astype("bool")

    # Set queue_end_pos when the queue is not half-full.
    df.loc[
        df["queue_full"], "queue_end_pos"
    ] = END_OF_QUEUE_POS  # When the queue is full, set a predefined queue_end_pos
    df.loc[
        df["queue_empty"], "queue_end_pos"
    ] = 0  # When the queue is empty, set a predefined queue_end_pos

    # The results to be returned
    ser = df.loc[filenames, target_variable].copy()
    assert len(filenames) == len(ser)

    return ser


def convert_predictions_to_evaluable(predictions, target_variable, bool_threshold=None):
    """Convert raw predictions in numpy to the format needed for evaluation, still in numpy arrays."""

    assert isinstance(predictions, np.ndarray)
    assert_target_variable(target_variable)

    if target_variable == "queue_end_pos":
        predictions = predictions.astype(np.int32)
        predictions = np.where(predictions < 0, 0, predictions)
        predictions = np.where(
            predictions > END_OF_QUEUE_POS, END_OF_QUEUE_POS, predictions
        )

    if target_variable == "queue_full":
        if bool_threshold is not None:
            predictions = np.where(predictions > bool_threshold, True, False).astype(
                bool
            )

    if target_variable == "lanes":
        if bool_threshold is not None:
            predictions = np.where(predictions > bool_threshold, True, False).astype(
                bool
            )

    assert isinstance(predictions, np.ndarray)

    return predictions


def CNN_predict(model_filename, imdata, test_filenames, target_variable):

    """
    Get the prediction from an already trained CNN-model
    (Needs the cut and painted images, but not anything sent through VGG16.)
    """

    assert type(model_filename) == str
    assert type(imdata) == XData
    assert type(test_filenames) == list
    assert_target_variable(target_variable)

    X = imdata.get_collection_array(test_filenames)
    model = load_model(model_filename)  # tensorflow.keras.models.load_model

    y = model.predict(X)
    y = convert_predictions_to_evaluable(y, target_variable)

    assert len(y) == len(test_filenames)
    ser = pd.Series(data=y, index=test_filenames)

    return ser


def VGG16_predict(
    model_filename, xd, test_filenames, target_variable, bool_threshold=None
):

    """
    Doing predictions to evaluate the results of
    These needs to be based on the pre-processed data which has been through the VGG16 convolutional base.
    """

    if not (bool_threshold is None or isinstance(bool_threshold(float))):
        raise TypeError(
            "bool_threshold needs to be None or a float between 0.0 and 1.0."
        )

    model = load_model(model_filename)
    test_filenames = list(
        set(test_filenames).intersection(set(xd.get_stored_filenames()))
    )

    X = xd.get_collection_array(test_filenames)
    predictions = model.predict(X).flatten()
    ser = pd.Series(data=predictions, index=test_filenames)

    return ser


def draw_queue_end_pos(im, r, fn):

    assert type(im) == np.ndarray  # The image
    assert type(r) == Series_y  # The results
    assert type(fn) == str  # filename

    y1 = 0
    y2 = im.shape[0]
    end_pos = int(r.get_y(fn))
    col = r.color

    im = cv2.line(im, (end_pos, y1), (end_pos, y2), col, 3)

    return im


def y_pos_gen(first_y, delta_y=20):

    """Draw where the end of the line is."""

    assert type(first_y) == int
    assert type(delta_y) == int

    y = first_y
    while True:
        yield y
        y += delta_y


def draw_assertion(im, r, filename, y_pos):

    assert type(im) == np.ndarray  # The image
    assert type(r) == Series_y  # The results
    assert type(filename) == str  # filename for this image
    assert type(y_pos) == int  # Position to write end of queue (False/True)


def draw_queue_full(im, r, filename, y_pos, x=1100):

    """Draw the conclusion, is the queue full (or not)."""

    draw_assertion(im, r, filename, y_pos)
    queue_full = r.results[filename]
    assert type(queue_full) in [bool, np.bool_]

    im = cv2.putText(
        im, str(queue_full), (x, y_pos), cv2.FONT_HERSHEY_PLAIN, 2, r.color, 2
    )

    return im


def draw_lanes(im, r, filename, y_pos, x=800):

    """Draw the conclusion, is the queue full (or not)."""

    draw_assertion(im, r, filename, y_pos)
    lanes = r.results[filename]
    assert type(lanes) in [bool, np.bool_]

    im = cv2.putText(
        im,
        (lambda l: "2" if l else "1")(lanes),
        (x, y_pos),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        r.color,
        2,
    )

    return im


def evaluate_on_images(results, filenames):

    """Go into the mode where you can see the actual result and predictions."""

    load_new = True
    i = 0

    while True:

        if load_new:
            filename = filenames[i]
            full_path = os.path.join(PICTURES_DIR, filename)
            im_org = cv2.imread(full_path, cv2.IMREAD_COLOR)
            load_new = False

        im = im_org.copy()

        y_gen_queue_full = y_pos_gen(50, delta_y=25)
        y_gen_lanes = y_pos_gen(50, delta_y=25)

        for r in results:
            if r.target_variable == "queue_end_pos":
                im = draw_queue_end_pos(im, r, filename)
            elif r.target_variable == "queue_full":
                im = draw_queue_full(im, r, filename, next(y_gen_queue_full))
            elif r.target_variable == "lanes":
                im = draw_lanes(im, r, filename, next(y_gen_lanes))
            else:
                raise Exception("Check {0}".format(r.target_variable))

        # Display image
        cv2.imshow("Image", im)

        print("Show image!")

        k = cv2.waitKey(0) & 0xFF

        if k % 256 < 255:
            print(k % 256)

        if k % 256 == 27:
            print("Pressed ESC")
            cv2.destroyAllWindows()
            break

        if k % 256 == 32:
            print("Pressed SPACE")
            i += 1
            if i == len(filenames):
                i = 0
            load_new = True

        if k % 256 == 122:
            print("Pressed Z")
            i -= 1
            if i < 0:
                i = len(filenames) - 1
            load_new = True


def save_plot(plt, prefix: str) -> str:
    """Save a plot with a timestamp."""

    timestamp = str(datetime.datetime.now())[:19]
    for replacement in [(":", ""), (".", "_"), (" ", "_")]:
        timestamp = timestamp.replace(*replacement)
    filename = f"{prefix}_{timestamp}.png"
    plt.savefig(filename)
    print(f"The {prefix} plot is saved to {filename}.")

    return filename


def evaluate_with_residuals(results, save_prefix="residuals"):

    """Calculate and show the residual errors between an actual, continuous value and the predictions."""

    assert type(results) == list  # list of Series_y
    assert len(results) >= 1
    target_variables = list(set([r.target_variable for r in results]))

    for target_variable in target_variables:

        print(f"Target variable = {target_variable}")

        relevant_results = [r for r in results if r.target_variable == target_variable]

        # Find and organize the results which if relevant for this target_variable

        predictions = [r for r in relevant_results if not r.actual]
        assert len(predictions) == (len(relevant_results) - 1)
        actual = [r for r in relevant_results if r.actual][0]
        assert type(actual) == Series_y

        fig = plt.figure(figsize=(6, 4), dpi=200)
        axes = []

        row = 1
        for pred in predictions:

            residuals = actual.results - pred.results

            assert set(actual.results.index) == set(pred.results.index)
            pred.results = pred.results.loc[actual.results.index]

            ax = fig.add_subplot(len(predictions), 1, row)
            ax.set_xlim([0, 1200])
            ax.set_title("Predictions given actuals: " + pred.name)
            ax.scatter(actual.results, pred.results, s=1)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Prediction")
            ax.plot([0, 1200], [0, 1200])
            ax.text(
                1000, 200, "Optimist", ha="center", va="center", size=12, color="grey"
            )
            ax.text(
                200, 1000, "Pessimist", ha="center", va="center", size=12, color="grey"
            )
            ax.grid()

            print(pred.name)
            print("Mean miss: {0:.1f}".format(residuals.mean()))
            print("Stddev miss: {0:.1f}".format(residuals.std()))

            axes.append(ax)
            row += 1

        timestamp = str(datetime.datetime.now())[:19]
        for replacement in [(":", ""), (".", "_"), (" ", "_")]:
            timestamp = timestamp.replace(*replacement)

        plt.tight_layout()
        save_plot(plt, save_prefix)


def evaluate_with_stats(results):

    """Simple text output of results."""

    assert type(results) == list  # list of Series_y
    assert len(results) >= 1
    assert (
        len(set([r.target_variable for r in results])) == 1
    )  # Only one target_variable

    # Find and organize the results which if relevant for this target_variable
    predictions = [r for r in results if not r.actual]
    assert len(predictions) == (len(results) - 1)
    actual = [r for r in results if r.actual][0]
    assert type(actual) == Series_y

    for pred in predictions:
        print(pred.name)
        residuals = actual.results - pred.results
        print("Mean miss: {0}".format(residuals.mean()))
        print("Stddev miss: {0}".format(residuals.std()))


def evaluate_with_confusion_matrix(results, save_prefix="conf_matrix"):

    """Generate and show a confusion matrix for categorical predictions."""

    assert type(results) == list  # list of Series_y
    assert len(results) >= 1
    assert (
        len(set([r.target_variable for r in results])) == 1
    )  # Only one target_variable

    # Find and organize the results which if relevant for this target_variable
    predictions = [r for r in results if not r.actual]
    actual = [r for r in results if r.actual][0]

    fig = plt.figure(figsize=(6, 4), dpi=200)
    axes = []

    row = 1
    for pred in predictions:

        ######
        # EVERYTHING OUTSIDE HERE IS EQUAL IN THE PREVIOUS FUNCTION. CLEAN THIS UP
        # confusion_matrix does not handle Series which are not sorted similarly
        cm = confusion_matrix(
            actual.results.sort_index(), pred.results.round().astype(bool).sort_index()
        )
        print(actual.results[-20:])
        print(pred.results[-20:].round().astype(bool))
        print(set(pred.results.index) == set(actual.results.index))

        ax = fig.add_subplot(len(predictions), 1, row)
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
        ax.set_title(pred.name)
        # /END
        #######

        axes.append(ax)
        row += 1

    timestamp = str(datetime.datetime.now())[:19]
    for replacement in [(":", ""), (".", "_"), (" ", "_")]:
        timestamp = timestamp.replace(*replacement)

    plt.tight_layout()
    save_plot(plt, save_prefix)


def results_to_df(results):

    """
    Combine all the results into a DataFrame.
    """

    d = {}
    for y_ser in results:
        d[y_ser.name] = y_ser.results
    df = pd.DataFrame(d)
    return df


def get_evaluation_graph(results, method):

    assert method in ["residual_histogram", "confusion_matrix"]

    assert type(results) == list
    assert len(results) >= 2

    target_variables = list(set([r.target_variable for r in results]))
    assert len(target_variables) >= 1

    number_of_cols = len(target_variables)


def get_histogram_ax(
    fig, row, col, row_total, col_total, actual_result, prediction_result
):

    """Create a single histogram ax which will be pasted together at a higher level."""

    print(prediction_result.name, len(prediction_results.results))

    residuals = actual.results - pred.results

    plt.tight_layout()

    ax = fig.add_subplot(row_total, col_total, row)
    ax.set_xlim([-1200, 1200])
    ax.set_ylim([0, 15])
    ax.set_title(prediction_result.name)
    ax.hist(residuals, bins=24)

    return fig, ax


def get_results(
    labels_filename, xd_preconv_filename, target_variables, set_type="Test"
):

    """
    Gather actual and predicted values for a list of target_variables.
    This function includes all the models which are capable of predicting on those target_variables.
    """

    ACTUAL_COL = (255, 255, 255)  # WHITE
    MODEL1_COL = (255, 0, 0)  # BLUE
    MODEL2_COL = (0, 0, 255)  # RED

    with open(xd_preconv_filename, "rb") as f:
        xd_preconv = pickle.load(f)

    results = []

    for target_variable in target_variables:

        filenames = get_filenames(
            labels_filename, target_variable, set_type=set_type
        )  #

        print(f"len filenames: {len(filenames)}")

        if target_variable == "queue_end_pos":

            print(f"len filenames: {len(filenames)}")

            # Actual
            r = Series_y("actual.queue_end_pos", True, ACTUAL_COL, target_variable)
            r.set_results(get_y_data(LABELS_FILENAME, filenames, target_variable))
            results.append(r)

            r = Series_y(
                "CNN_EoQ_VGG16_noaug_weighted_20210408.h5",
                False,
                MODEL1_COL,
                target_variable,
            )
            r.set_results(
                VGG16_predict(
                    r"CNN_EoQ_VGG16_noaug_weighted_20210408.h5",
                    xd_preconv,
                    filenames,
                    target_variable,
                )
            )
            results.append(r)

        elif target_variable == "queue_full":

            r = Series_y("actual.queue_full", True, ACTUAL_COL, target_variable)
            r.set_results(get_y_data(LABELS_FILENAME, filenames, target_variable))
            results.append(r)

            r = Series_y(
                "CNN_Queue_full_20210408.h5", False, MODEL1_COL, target_variable
            )
            h = VGG16_predict(
                r"CNN_Queue_full_20210408.h5", xd_preconv, filenames, target_variable
            )
            r.set_results(h)
            results.append(r)

        elif target_variable == "lanes":

            r = Series_y("actual.lanes", True, ACTUAL_COL, target_variable)
            y = get_y_data(LABELS_FILENAME, filenames, target_variable)
            y = convert_float_lanes_to_boolean(y, input_is_12=True)
            r.set_results(y)
            results.append(r)

            r = Series_y(
                "CNN_Lanes_VGG16_weighted_20210408.h5",
                False,
                MODEL1_COL,
                target_variable,
            )
            h = VGG16_predict(
                r"CNN_Lanes_VGG16_weighted_20210408.h5",
                xd_preconv,
                filenames,
                target_variable,
            )
            r.set_results(h)
            results.append(r)

    clear_session()  # Clear tf memory usage - https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session

    return results


def main(labels_filename, xd_preconv_filename, eval_cfg):

    for ec in eval_cfg:

        # Is the configuration valid?
        assert_evaluable_on(ec["evaluate_how"])
        assert_target_variable(ec["target_variables"])

        # Gather results (predictions and actual values)
        results = get_results(
            labels_filename, xd_preconv_filename, ec["target_variables"]
        )

        # Only use the filenames (images) which are included in all the result sets.
        filenames = list(
            functools.reduce(
                lambda l1, l2: set(l1).intersection(set(l2)),
                [r.get_filenames() for r in results],
            )
        )

        if ec["evaluate_how"] == "show_on_images":
            evaluate_on_images(results, filenames)

        if ec["evaluate_how"] == "with_residuals":
            evaluate_with_residuals(results, save_prefix="residuals")

        if ec["evaluate_how"] == "with_confusion_matrix":
            evaluate_with_confusion_matrix(results, save_prefix="conf_matrix")

        if ec["evaluate_how"] == "dump_to_file":
            results_df = results_to_df(results)
            filename = "results.csv"
            results_df.to_csv(filename)
            print(f"The results have been written to {filename}")


if __name__ == "__main__":

    """
    Go through one or more trained model and show how they perform on the test data,
    possibly compared with each other.
    """

    # Virker (2021-04-30)
    # eval_cfg = [{"evaluate_how": "show_on_images",
    #             "target_variables": ["queue_end_pos"]}]

    # Virker (2021-04-20)
    # eval_cfg = [{"evaluate_how": "with_residuals",
    #             "target_variables": ["queue_end_pos"]}]

    # Virker (2021-04-20)
    # eval_cfg = [{"evaluate_how": "with_confusion_matrix",
    #             "target_variables": ["queue_full"]}]

    # Virker (2021-04-20)
    eval_cfg = [{"evaluate_how": "with_confusion_matrix",
                 "target_variables": ["lanes"],
                 "files": "min_predictions"}]

    # Virker (2021-04-20)
    # eval_cfg = [
    #     {
    #         "evaluate_how": "dump_to_file",
    #         "target_variables": ["lanes"],
    #         "files": "min_predictions",
    #     }
    # ]

    main(LABELS_FILENAME, PRECONV_FILENAME, eval_cfg)

    print("That was fun!")
