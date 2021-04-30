import os
import filecmp

from train_models import (
    crop_and_save_images,
    run_convbase_on_images,
    IMAGE_SHAPE,
)
from common import ROI

org_dir = os.path.join(os.getcwd(), "test_data", "org_images")
cropped_dir = os.path.join(os.getcwd(), "test_data", "cropped_images")
pre_conv_file = os.path.join(os.getcwd(), "test_data", "pre_conv", "pre_conv.bin")


def create_test_images(dir: str, size=20):
    """
    Create random noise test images.
    """

    for i in range(size):
        im = np.random.randint(
            255, size=IMAGE_SHAPE, dtype=np.uint8
        )  # Er dette formatet de lagres på?

    cv2.imwrite("im.png", im)


def test_crop_and_save_images():

    crop_and_save_images(org_dir, cropped_dir, ROI, force_all=False)

    new_files = [fn for fn in os.listdir(cropped_dir) if "_correct" not in fn]
    for fn in new_files:
        file_new = os.path.join(cropped_dir, fn)
        file_correct = os.path.join(
            cropped_dir, f"{fn.split('.')[0]}_correct.{fn.split('.')[1]}"
        )
        assert filecmp.cmp(file_new, file_correct, shallow=False)
        os.remove(file_new)


def test_run_convbase_on_images():

    run_convbase_on_images(cropped_dir, pre_conv_file, IMAGE_SHAPE, force_all=False)

    pre_conv_file_correct = (
        f"{pre_conv_file.split('.')[0]}_correct.{pre_conv_file.split('.')[1]}"
    )
    assert filecmp.cmp(pre_conv_file, pre_conv_file, pre_conv_file_correct)
    os.remove(pre_conv_file)
