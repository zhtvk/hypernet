import lasagne
from copy import deepcopy

def model1(input_height, input_width, output_dim, batch_size):
    l_in = lasagne.layers.InputLayer(
        shape = (batch_size, 1, input_height, input_width),
    )
    
    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters = 32,
        filter_size = (9, 16),
        nonlinearity = lasagne.nonlinearities.tanh,
        W = lasagne.init.GlorotUniform(),
    )
    
    l_conv2 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters = 32,
        filter_size = (1, 16),
        nonlinearity = lasagne.nonlinearities.tanh,
        W = lasagne.init.GlorotUniform(),
    )
    
    l_conv3 = lasagne.layers.Conv2DLayer(
        l_conv2,
        num_filters = 32,
        filter_size = (1, 16),
        nonlinearity = lasagne.nonlinearities.tanh,
        W = lasagne.init.GlorotUniform(),
    )
    
    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv3,
        num_units = 800,
        nonlinearity = lasagne.nonlinearities.tanh,
    )
    
    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units = 800,
        nonlinearity = lasagne.nonlinearities.tanh,
    )
    
    l_out = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units = output_dim,
        nonlinearity = lasagne.nonlinearities.softmax,
    )
    
    return l_out

net = {
    'dataset_path' : None,
    'neighbors' : 9,
    'channels' : 220,
    'num_epochs' : 400,
    'batch_size' : 50,
    'learning_rate' : 1e-2,
    'l2_reg' : 1e-5,
    'gamma' : 1e-4,
    'power' : 0.75,
    'momentum' : 0.9,
    'model' : None,
    'snapshot': None,
}

# 10 percent dataset
net1 = deepcopy(net)
net1['dataset_path'] = './data/indian10p.p.gz'
net1['model'] = model1
net1['snapshot'] = './trained_models/net1.p.gz'







    
