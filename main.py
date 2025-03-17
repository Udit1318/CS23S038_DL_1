
"""--------------Activation Functions--------------"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def d_sigmoid(x):
  return (1 - sigmoid(x)) * sigmoid(x)

def tanh(x):
  return np.tanh(x)

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

def relu(x):
  return np.where(np.asarray(x) > 0, x, 0)

def d_relu(x):
    return np.where(x <= 0, 0, 1)

def softmax(x):
    e_x = np.exp(x)
    return e_x/e_x.sum()

"""--------------Loss Function:---------------"""

#cross-entropy
def cross_entropy_loss(y, y_hat, i): #y_hat is a (10 * 1) matrix containing probabilities corresponding to each class
  return -np.log(y_hat[y[i]]) #y[i] is the true label number(say y[i] --> 5) --> -(1 * log(0.8)) rest 0, and hence only that term will be non-zero in cross entropy

#squared-error
def squared_error(y, y_hat, i):
  e_l = np.zeros((y_hat.shape[0], 1))
  e_l[y[i]] = 1;

  #y_hat[y[i]] = (1-y_hat[y[i]])**2
  #return np.sum(np.square(y_hat))
  loss = np.sum((y_hat - e_l) ** 2)
  return loss

"""###Layer Class : parameters initialization for each layer"""

class Layer:  # W, b, act, d_act, dW, db ---> each layer stores then future adds da and dh

    activationFunc = { #types of activation functions and there grad
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid),
        'relu' : (relu, d_relu),
        'softmax' : (softmax, None)
    }

    def __init__(self, inputs, neurons, activation):

        #Xavier initialization sets the initial values of weights in a way that prevents gradients from vanishing or exploding during training.
        np.random.seed(33)  #ensures that the sequence of random numbers generated is reproducible.
        sd = np.sqrt(2 / float(inputs + neurons)) #calculates the standard deviation (sd) used for Xavier initialization of the weights.
        self.W = np.random.normal(0, sd, size=(neurons, inputs))  #initializes the weights (W) of the layer using a normal distribution with mean 0 and standard deviation sd --> shape of output array (neurons * inputs)
        self.b = np.zeros((neurons, 1)) #init bias to 0 ---> shape(neurons * 1)
        self.act, self.d_act = self.activationFunc.get(activation) #activation-func and diff_act-func is taken from argument.
        self.dW = 0 #gradients of the loss function with respect to the weights and biases of the layer, respectively
        self.db = 0

"""Forward propagation"""

def forward_propagation(h, layers):
  m = len(layers) #all layers present ---> input + hidden + output

  layers[0].a = np.dot(layers[0].W, h) #first layer pre-activation
  layers[0].h = layers[0].act(layers[0].a) #first layer activation

  for j in range(1, m-1):
    layers[j].a = np.dot(layers[j].W, layers[j-1].h) #hidden layers pre-activation
    layers[j].h = layers[j].act(layers[j].a) #hidden layers activation

  j+=1
  layers[j].a = np.dot(layers[j].W, layers[j-1].h) #last layers pre-activation
  layers[j].h = softmax(layers[j].a) #output layer activation using softmax fucntion ---> returns probability
  return layers[m-1].h #returns the probabilty given by softmax function of each class

"""Backward_propagation"""

def backward_propagation(l, y_hat, layers, inp): # l ---> label number of true class

  #one-hot vector
  e_l = np.zeros((y_hat.shape[0], 1)) #init a vector of size (y_hat * 1) all set to 0
  e_l[l] = 1 #set 1 corresponding to the true class label ---> true one-hot encoded vector

  #layers[len(layers)-1].da = -(e_l - y_hat) #gradient w.r.t activation of last layer (a_L) cross-entropy
  layers[len(layers)-1].da = np.multiply(2 * np.multiply(y_hat, (1 - y_hat)), (y_hat - e_l)) #gradient w.r.t activation of last layer (a_L) cross-entropy

  for j in range(len(layers)-1, 0, -1): #grads from L-1 to 1 layer

    layers[j].dW += np.dot(layers[j].da, (layers[j-1].h).T)
    layers[j].db += layers[j].da

    layers[j-1].dh = np.dot((layers[j].W).T, layers[j].da)
    layers[j-1].da = np.multiply(layers[j-1].dh, layers[j-1].d_act(layers[j-1].a))

  layers[0].dW += np.dot(layers[0].da, inp.T)
  layers[0].db += layers[0].da

  return layers


""" ----------------------------Optimizers--------------------------------  """

"""SGD / Batch Gradient Descent"""

def sgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    m = x_train.shape[0] #60,000
    costs = []
    for epoch in range(epochs):

      cost = 0
      for i in range(m):

        inp = x_train[i].reshape(784, 1) #each 28 * 28 image is converted into 784 * 1 column vector.

        #Feedforward
        h = inp
        h = forward_propagation(h, layers) #each column(image) is passed as input along with the layers.

        #Calulate training loss
        cost += cross_entropy_loss(y_train, h, i)

        #Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1)) # wrt i-th datapoint ---> y_train[i] --> true class label, h --> prob vector

        #mini-batch gradient decent
        if (i+1) % batch_size == 0:
          for layer in layers:
            layer.W = layer.W - learning_rate * layer.dW/batch_size # W for next iteration is current W - eta * dW
            layer.b = layer.b - learning_rate * layer.db/batch_size

            layer.dW = 0
            layer.db = 0

      costs.append(cost/m) #normalised cross-entropy loss after each epoch

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers) #Run the trained model after every epoch in validation data at once ---> entire [6000 * 784].T matrix is passed to forward prop at once, it output ---> [10 * 6000] y_hat matrix

      val_loss = 0

      for i in range(len(y_val)):
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i) #check the validation loss after every epoch

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0) # takes/assigns it to that class which have maximum probabity ----- 1 * 6000
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0] #calculate validation accuracy where every we made a correct pridiction upon total points after every epoch

      #wandb logs
      wandb.log({"epoch": epoch, "train_loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Training loss: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers


"""Momentum Gradient descent"""

def mgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    gamma = 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:  #inititalize w, b to 0
        layer.update_W = 0
        layer.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #momentum gradient decent
        if (i+1) % batch_size == 0:
          for layer in layers:

            layer.update_W = gamma*layer.update_W + learning_rate*layer.dW/batch_size #current delta is some gamma times previous history plus current delta
            layer.update_b = gamma*layer.update_b + learning_rate*layer.dW/batch_size

            layer.W = layer.W - layer.update_W
            layer.b = layer.b - layer.update_b

            layer.dW = 0
            layer.db = 0

            layer.update_W = 0
            layer.update_b = 0


      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0

      for i in range(len(y_val)):
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "train_loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Training loss: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers

"""Nesterov Gradient Descent"""

def nesterov(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    gamma = 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:
        layer.update_W = 0
        layer.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)

        #calculate W_lookaheads
        if (i+1) % batch_size == 0: #first move by history and then calculate grad at this point and then move accordingly
          for layer in layers:
            layer.W = layer.W - gamma * layer.update_W #moved by history(momentum)
            layer.b = layer.b - gamma * layer.update_b

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1)) #calculate grad at this moved point

        #nesterov gradient decent
        if (i+1) % batch_size == 0:
          for layer in layers:

            layer.update_W = gamma*layer.update_W + learning_rate*layer.dW/batch_size #final update
            layer.update_b = gamma*layer.update_b + learning_rate*layer.dW/batch_size

            layer.W = layer.W - layer.update_W
            layer.b = layer.b - layer.update_b

            layer.dW = 0
            layer.db = 0

            layer.update_W = 0
            layer.update_b = 0

      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0

      for i in range(len(y_val)):
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "train_loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Training loss: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers

"""RMSProp"""

def rmsprop(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    epsilon, beta = 1e-8, 0.9
    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:
        layer.update_W = 0
        layer.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #rmsprop gradient decent
        if (i+1) % batch_size == 0:
          for layer in layers:

            layer.update_W = beta*layer.update_W + (1-beta)*(layer.dW/batch_size)**2
            layer.update_b = beta*layer.update_b + (1-beta)*(layer.db/batch_size)**2

            layer.W = layer.W - (learning_rate / np.sqrt(layer.update_W + epsilon)) * (layer.dW/batch_size)
            layer.b = layer.b - (learning_rate / np.sqrt(layer.update_b + epsilon)) * (layer.db/batch_size)

            layer.dW = 0
            layer.db = 0

            layer.update_W = 0
            layer.update_b = 0


      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0

      for i in range(len(y_val)):
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "train_loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Training loss: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers

"""Adam"""

def adam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    epsilon, beta1, beta2 = 1e-8, 0.9, 0.99
    t = 0

    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:
        layer.m_W, layer.m_b, layer.v_W, layer.v_b, layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += squared_error(y_train, h, i)

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #adam gradient decent
        if (i+1) % batch_size == 0:
          t+=1

          for layer in layers:

            layer.m_W = beta1 * layer.m_W + (1-beta1)*layer.dW/batch_size
            layer.m_b = beta1 * layer.m_b + (1-beta1)*layer.db/batch_size

            layer.v_W = beta2 * layer.v_W + (1-beta2)*((layer.dW/batch_size))**2
            layer.v_b = beta2 * layer.v_b + (1-beta2)*((layer.db/batch_size))**2

            layer.m_W_hat = layer.m_W/(1-math.pow(beta1, t))
            layer.m_b_hat = layer.m_b/(1-math.pow(beta1, t))

            layer.v_W_hat = layer.v_W/(1-math.pow(beta2, t))
            layer.v_b_hat = layer.v_b/(1-math.pow(beta2, t))

            layer.W = layer.W - (learning_rate/np.sqrt(layer.v_W_hat + epsilon))*layer.m_W_hat
            layer.b = layer.b - (learning_rate/np.sqrt(layer.v_b_hat + epsilon))*layer.m_b_hat

            layer.dW = 0
            layer.db = 0

            layer.m_W, layer.m_b, layer.v_W, layer.v_b, layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0


      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0
      for i in range(len(y_val)):
        val_loss += squared_error(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "train_loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Training loss: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers

"""NAdam"""

def nadam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size):

    epsilon, beta1, beta2 = 1e-8, 0.9, 0.99
    gamma = 0.9
    t = 0

    m = x_train.shape[0]
    costs = []

    for epoch in range(epochs):

      for layer in layers:
        layer.m_W, layer.m_b, layer.v_W, layer.v_b, layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0
        layer.update_W = 0
        layer.update_b = 0

      cost = 0

      for i in range(m):

        inp = x_train[i].reshape(784, 1)

        # Feedforward
        h = inp
        h = forward_propagation(h, layers)

        # Calulate cost to plot graph
        cost += cross_entropy_loss(y_train, h, i)

        #calculate W_lookaheads
        if (i+1) % batch_size == 0:
          for layer in layers:
            layer.W = layer.W - gamma * layer.m_W
            layer.b = layer.b - gamma * layer.m_b

        # Backpropagation
        backward_propagation(y_train[i], h, layers, x_train[i].reshape(784, 1))

        #adam gradient decent
        if (i+1) % batch_size == 0:
          t+=1

          for layer in layers:

            layer.m_W = beta1 * layer.m_W + (1-beta1)*layer.dW/batch_size
            layer.m_b = beta1 * layer.m_b + (1-beta1)*layer.db/batch_size

            layer.v_W = beta2 * layer.v_W + (1-beta2)*((layer.dW/batch_size))**2
            layer.v_b = beta2 * layer.v_b + (1-beta2)*((layer.db/batch_size))**2

            layer.m_W_hat = layer.m_W/(1-math.pow(beta1, t))
            layer.m_b_hat = layer.m_b/(1-math.pow(beta1, t))

            layer.v_W_hat = layer.v_W/(1-math.pow(beta2, t))
            layer.v_b_hat = layer.v_b/(1-math.pow(beta2, t))

            layer.m_dash_W = beta1 * layer.m_W_hat + (1-beta1)*layer.dW/batch_size
            layer.m_dash_b = beta1 * layer.m_b_hat + (1-beta1)*layer.db/batch_size

            layer.W = layer.W - (learning_rate/np.sqrt(layer.v_W_hat + epsilon))*layer.m_dash_W
            layer.b = layer.b - (learning_rate/np.sqrt(layer.v_b_hat + epsilon))*layer.m_dash_b

            layer.dW = 0
            layer.db = 0

            layer.m_W, layer.m_b, layer.v_W, layer.v_b, layer.m_W_hat, layer.m_b_hat, layer.v_W_hat, layer.v_b_hat = 0, 0, 0, 0, 0, 0, 0, 0


      costs.append(cost/m)

      #predict on validation data
      prediction = forward_propagation(x_val.T, layers)

      val_loss = 0
      for i in range(len(y_val)):
        val_loss += cross_entropy_loss(y_val, prediction[:, i].reshape(10,1), i)

      val_loss = val_loss/len(y_val)
      prediction = prediction.argmax(axis=0)
      val_accuracy =  np.sum(prediction == y_val)/y_val.shape[0]

      #wandb logs
      wandb.log({"epoch": epoch, "train_loss": costs[len(costs)-1], "val_accuracy": val_accuracy, "val_loss": val_loss})

      print("-----------------epoch "+str(epoch)+"-----------------")
      print("Training loss: ", cost/m)
      print("Validation accuracy: ", val_accuracy)
      print("Validation loss: ", val_loss)

    return costs, layers

"""Putting all togather:

-----------------Optimizer----------------
"""

def optimizor(layers, optimizer, epochs, learning_rate, x_train, y_train, x_val, y_val, batch_size):

  if optimizer == "sgd":
    return sgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "mgd":
    return mgd(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "nesterov":
    return nesterov(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "rmsprop":
    return rmsprop(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "adam":
    return adam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  elif optimizer == "nadam":
    return nadam(epochs, layers, learning_rate, x_train, y_train, x_val, y_val, batch_size)
  else:
    print("No optimization algorithm named "+optimizer+" found")
    return "Error", "Error"

"""Function to Predict"""

def predict(input, y, layers): #After the model is trained do one pass of forward pass in test data and note loss

  prediction = forward_propagation(input, layers)

  loss = 0


  for i in range(len(y)):
    loss += squared_error(y, prediction[:, i].reshape(10,1), i)

  prediction = prediction.argmax(axis=0)
  accuracy = np.sum(prediction == y)/y.shape[0]

  return prediction, accuracy, loss/len(y)

"""----------Import dataset and putting in appropriate format-----------"""

from keras.datasets import fashion_mnist
(x_train_org, y_train_org), (x_test_org, y_test_org) = fashion_mnist.load_data()

"""**Normalizing the DataSet**"""

x_train_org = x_train_org / 255.0
x_test_org = x_test_org / 255.0

"""Flattening the data"""

x_train_temp = x_train_org.reshape(x_train_org.shape[0], -1)  #reshapes x-train to 60000 * 784 ---> .shape return [no. of rows, no. of cols]
y_train_temp = y_train_org
x_test = x_test_org.reshape(x_test_org.shape[0], -1) #reshape keeps the numbers of rows(60000) same and no of columns are infered based on shape of data.
y_test = y_test_org

"""Splliting dataset into training and validation"""

x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.1, random_state=33)

"""---------------Train model-------------"""

def model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val):

  layers= [Layer(x_train.shape[1], neurons, activation)]
  for _ in range(0, h_layers-1):
    layers.append(Layer(neurons, neurons, activation))
  layers.append(Layer(neurons, 10, 'softmax'))

  costs, layers = optimizor(layers, optimizer, epochs, learning_rate, x_train, y_train, x_val, y_val, batch_size)

  output_test, accuracy_test, test_loss = predict(x_test.T, y_test, layers)

  print("-------------****************----------------")
  print("Test accuracy: ", accuracy_test)
  print("Test loss: ", test_loss)

  return output_test


# Define the sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {"values": [5, 10]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "h_layers": {"values": [3, 4, 5]},
        "neurons": {"values": [32, 64, 128]},
        "optimizer": {"values": ['sgd', 'mgd', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        "batch_size": {"values": [16, 32, 64]},
        "activation": {"values": ['sigmoid', 'tanh', 'relu']}
    }
}

# Initialize Sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="UDIT_DL_1")


def model_train_wandb(config=None):
    """Function to train the model using WandB Sweep parameters."""
    
    with wandb.init(config=config):
        config = wandb.config  # Get hyperparameter values from WandB
        
        # Generate a unique run name based on hyperparameters
        run_name = f"-hl{config.h_layers}-bs{config.batch_size}-ac_{config.activation}"
        wandb.run.name = run_name

        # Define layers for the neural network
        layers = [Layer(x_train.shape[1], config.neurons, config.activation)]
        for _ in range(config.h_layers - 1):
            layers.append(Layer(config.neurons, config.neurons, config.activation))
        layers.append(Layer(config.neurons, 10, 'softmax'))

        # Train the model
        costs, layers = optimizor(
            layers, config.optimizer, config.epochs, config.learning_rate,
            x_train, y_train, x_val, y_val, config.batch_size
        )

        # Evaluate on test data
        output_test, accuracy_test, test_loss = predict(x_test.T, y_test, layers)

        # Log results
        print("----------------------------------")
        print("Test accuracy:", accuracy_test)
        print("Test loss:", test_loss)
        # wandb.log({"test_accuracy": accuracy_test, "test_loss": test_loss})

        # Optionally, log confusion matrix
        # labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        #           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_test,
        #                                                   preds=output_test, class_names=labels)})


# #if __name__ == "__main__":
# #    try:
#         # Run the sweep
#         wandb.agent(sweep_id, function=model_train_wandb, count=50)
#     except BrokenPipeError:
#         print("⚠️ Broken Pipe Error: Connection was interrupted.")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
