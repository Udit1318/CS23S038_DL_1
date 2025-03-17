from main.py import *
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='DL')
    parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='---')
    parser.add_argument('-a', '--activation', help='choices: ["sigmoid", "tanh", "ReLU"]', choices = ["sigmoid", "tanh", "ReLU"],type=str, default='tanh')
    parser.add_argument('-o', '--optimizer', help = 'choices: ["adam", "sgd","nesterov","mgd", "rmsprop", "nadam"]', choices = ["sgd", "adam","nesterov","mgd", "rmsprop", "nadam"],type=str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.001)
    parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=10)
    parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', type=int, default=128)
    parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=32)
    args = vars(parser.parse_args())
    return args


if __name__=='__main__':
	
	args = argument_parser()

	wandb.login(key="-----ADD API KEY-----")
	wandb.init(project=args['wandb_project'], entity=args['wandb_entity'])

	activation = args['activation']
	batch_size = args['batch_size']
	epochs = args['epochs']
	h_layers = args['num_layers']
	learning_rate = args['learning_rate']
	neurons = args['hidden_size']
	optimizer = args['optimizer']
	output_test = model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val)

	wandb.finish()
