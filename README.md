# CS23S038_DL_1
This project aims to implement a feedforward neural network for image classification, utilizing various gradient descent algorithms and their variants, including momentum, Nesterov accelerated gradient descent, Adam, Nadam, RMSprop, and stochastic gradient descent. The task involves classifying images into one of ten labels. The project incorporates the use of wandb for data visualization to generate graphs and reports, facilitating insightful observations on the performance of different methods.

** Add API KEY in train.py **

### libraries used:
In this project, numpy was employed for mathematical computations involved in forward propagation, back propagation, and loss function calculations. Keras and TensorFlow were leveraged to access the Fashion MNIST dataset.

### Installations:
`pip install -r requirements.txt` was used for the purpose of installments.

### Run:

- Start execution from train.py by `python train.py --wandb_entity myname --wandb_project myprojectname`

### Args Supported:

- 	activation function
-   batch size 
-   epochs 
-   hidden layers
-   learning rate 
-   number of neurons
-   optimizer function

### Training and Evaluation of Model:

#### Model Training Function:

The `model_train` function is utilized to train the model with specific hyperparameters. This function accepts arguments such as the number of epochs, learning rate, number of neurons, number of hidden layers, activation function, batch size, optimizer, as well as the training and validation datasets.

#### Training Process:

During training, a `Layers` class stores the information (pre-activation and activation) of every layer of the neural network. This `layers` list is passed into forward propagation, where the model makes predictions. Subsequently, backward propagation is called to update the parameters of each layer with the objective of minimizing the loss function.

#### Testing:

Once the model is trained, we perform one pass of forward propagation with the test dataset to report the accuracy on the test dataset.

#### Predict Function:

The `predict` function takes the input, true labels, and the layers list, and returns the predicted labels (`y_hat`) and the accuracy of the model.

#### Optimizers:

Different optimizers, such as "adam", "sgd", "nesterov", "mgd", "rmsprop", and "nadam", are implemented as separate functions. These optimizer functions are passed as arguments to the `model_train` function, providing flexibility in choosing the optimization algorithm for training the neural network.


## Directory Structure:

CS23S038_DL_1/
- train.py
- main.py
- raw_code.ipynb

`train.py:` Python script containing arguments and the main function to call the model_train function.

`main.py:` Python script containing optimization functions and logic.

`raw_code.ipynb:` Main Jupyter Notebook from which the Python script (cs23s036_dl_assignment1.py) was generated.
