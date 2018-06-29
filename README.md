# Blank PyTorch Deep Learning Piping

*Author: Nick Hershey*

## Setup

To install all required items
```
pip install -r requirements.txt
```

## Update Based on Your Data and Architecture Choices

1. __model/data_loader.py__: Update __fetch_dataloader__ to return train, test, and validation data sets. Then, update the three functions in the DataSet subclass:
    a. __init__ - any initial start up work ,usually getting a list of file names
    b. __len__ - return the number of data points in the data set
    c. __getitem__ - return a data point, usually after opening a file (x tensor, y)

1. __model/net.py__:  
    1. __init__ - to include the right layers for each network architecture
    1. __forward__ - forward propagate through each architecture
    1. __loss_fn__ - output the loss function you'd like to minimize for training. You can add more metrics (such as F1) by adding the function and the name of the function at the bottom.

3. __experiments__: For each architecture/hyperparameter experiment you'd like to run, create a folder with a __params.json__ in it.

## To Run

1. __train.py__: Trains the neural network
```
python train.py --data_dir {where data is stored} --model_dir experiments/{base,conv,lstm}_model`
```

2. __evaluate.py__: Used after each epoch of training to test each of the accuracies

3. __search_hyperparams.py__: Used to test a parameter defined in one of the experiments. I haven't really run any yet.
```
python search_hyperparams.py --data_dir {where data is stored} --parent_dir experiments/{learning_rate,filter_size,etc.}
```

4. __synthesize_results.py__: Display the results of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```
