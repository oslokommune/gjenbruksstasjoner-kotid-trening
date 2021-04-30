# gjenbruksstasjoner-kotid-trening
The actual model training, resulting i model files. It has been tested in Ubuntu 20.04.

## Installation
It is recommended to work in a virtual environment, e.g. by using the `venv` module, before installing the requirements.  
`$ python3 -m venv .venv`  
`$ source .venv/bin/activate` (when done, `deactivate` the venv).  

Install the required libraries:  
`$ pip install -r requirements.txt`  

## Train a model
Modify the HARDCODED PARAMETERS in `train_models.py` to reflect the location of the downloaded images and `labels_data.csv` file.
Which type of model should be trained and the configuration of it also needs to be set in the `if __name__ == "__main__"` block.  
Three types of models can be trained:
* End of queue - interprets where in the image where the queue ends. This is measured in x position, as a continuous model.
* Lanes - is there 1 or 2 lanes of cars in the images? This is implemented as a binary model (False = 1 lane, True = 2 lanes).
* Queue full - is the queue so long that the end of it can't be seen? This is implemented as a binary model.   

It may also be uesful to know that all the downloaded images gets cropped and put in a specified folder (like `cropped_images`). The cropped images then gets run through the pre-convolutional network and saved into one binary file (like `VGG16_preconv.bin`). This file can become very large and may take several hours to generate. For the future, it will probably be better to write the pre-convoluted images to separate files. The data in the binary files gets read and used as the X-data during training and evaluation.

Run `python3 train_models.py`.  

Remember that the model files have to be renamed before being uploaded to the production pipeline.

## Evaluate the model
The metrics for the various models can be measured on the `test` data set. This can be used to view predictions vs real value on actual images, looking at residuals, confusion matrices or dumping the predictions to a file. Also, the model filenames to be used are currently also hardcoded in the `get_results` function (sorry about that!).  

Run `python3 evaluate_models.py`. 

## Development testing
Some tests have been implemented using Pytest. Run `pytest` to execute these.  
For further development, it is highly recommended to add more tests.  
