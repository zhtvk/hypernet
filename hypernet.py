from __future__ import print_function

import gzip
import itertools
import pickle
import numpy as np
import lasagne
from lasagne.objectives import aggregate, categorical_crossentropy
from lasagne.layers import get_output
from lasagne.layers import get_all_param_values, set_all_param_values
from lasagne.regularization import regularize_network_params
import theano
import theano.tensor as T
import time
import os
import sys

import sklearn.metrics as metrics


def load_data(dataset, neighbors, channels):
    
    f = gzip.open(dataset, 'rb')
    data = pickle.load(f)
    f.close()
    
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    X_train = X_train.reshape((X_train.shape[0], 1, neighbors, channels))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, neighbors, channels))
    X_test = X_test.reshape((X_test.shape[0], 1, neighbors, channels))

    k = np.unique(np.hstack((y_train, y_valid, y_test))).shape[0]
    
    return dict(
        X_train = theano.shared(lasagne.utils.floatX(X_train)),
        X_tensor_type = T.tensor4,
        y_train = T.cast(theano.shared(y_train), 'int32'),
        X_valid = theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid = T.cast(theano.shared(y_valid), 'int32'),
        X_test = theano.shared(lasagne.utils.floatX(X_test)),
        y_test = T.cast(theano.shared(y_test), 'int32'),
        num_examples_train = X_train.shape[0],
        num_examples_valid = X_valid.shape[0],
        num_examples_test = X_test.shape[0],
        num_classes = k,
        input_height = X_train.shape[2],
        input_width = X_train.shape[3],
        output_dim = k,
    )

def create_iter_functions(dataset, output_layer,
                          batch_size,                          
                          momentum,
                          l2_reg):
    
    batch_index = T.iscalar('batch_index')
    X_batch = dataset['X_tensor_type']('X')
    y_batch = T.ivector('y')
    batch_slice = slice(
        batch_index * batch_size, (batch_index + 1) * batch_size)

    loss_train = aggregate(
        categorical_crossentropy(get_output(output_layer, X_batch),
                                 y_batch)) + \
        l2_reg * regularize_network_params(output_layer,
                                           lasagne.regularization.l2)
    loss_eval = aggregate(
        categorical_crossentropy(get_output(output_layer, X_batch,
                                            deterministic = True),
                                 y_batch))
    
    pred = T.argmax(
        get_output(output_layer, X_batch, deterministic = True), axis = 1)

    accuracy = T.mean(T.eq(pred, y_batch), dtype = theano.config.floatX)
    
    all_params = lasagne.layers.get_all_params(output_layer)

    learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)
    
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)
    
    iter_train = theano.function(
        [batch_index, learning_rate], [loss_train],
        updates = updates,
        givens = {
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )
    
    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens = {
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )
    
    iter_test = theano.function(
        [batch_index], [accuracy],
        givens = {
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )   
    
    return dict(
        train = iter_train,
        valid = iter_valid,
        test = iter_test,
    )

def train(iter_funcs, dataset, batch_size, learning_rate, gamma, power):
    
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    LR = learning_rate
    
    for epoch in itertools.count(1):

        batch_train_losses = []
        for b in range(num_batches_train):
            iter = (epoch - 1) * batch_size + b
            batch_train_loss = iter_funcs['train'](b, learning_rate)
            batch_train_losses.append(batch_train_loss)
            learning_rate = LR * (1 + gamma * iter) ** (-power)
            
        avg_train_loss = np.mean(batch_train_losses)
        
        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)
            
        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)
        
        yield {
            'number': epoch,
            'learning_rate': learning_rate,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }

def test(iter_funcs, dataset, batch_size):

    num_batches_test = dataset['num_examples_test'] // batch_size

    batch_test_accuracies = []
    for b in range(num_batches_test):
        batch_test_accuracy = iter_funcs['test'](b)
        batch_test_accuracies.append(batch_test_accuracy)

    avg_test_accuracy = np.mean(batch_test_accuracies)

    return avg_test_accuracy

def predict(dataset, output_layer):
    
    X = dataset['X_tensor_type']('X')
    y = T.ivector('y')
    
    pred = T.argmax(
        get_output(output_layer, X, deterministic = True), axis = 1)

    fn_pred = theano.function(
        [], [y, pred],
        givens = {
            X: dataset['X_test'],
            y: dataset['y_test'],
        },
    )

    return fn_pred()

def test_net(snapshot):

    state = load_net(snapshot)

    if state == None:
        print('Could not load saved net ', snapshot)
        return

    print('Loading data and saved model...', end = ' ')
    tstart = time.time()
    dataset = load_data(state['dataset_path'],
                        state['neighbors'], state['channels'])    

    output_layer = state['model'](
        input_height = dataset['input_height'],
        input_width = dataset['input_width'],
        output_dim = dataset['output_dim'],
        batch_size = None,
    )
    set_all_param_values(output_layer, state['learned_params'])
    print('{:.3f} s'.format(time.time() - tstart))

    print('Evaluating net...', end = ' ')
    tstart = time.time()
    targets, predictions = predict(dataset, output_layer)
    print('{:.3f} s'.format(time.time() - tstart))

    print('')
    print('Test accuracy: {:.3f}%'.format(
        metrics.accuracy_score(targets, predictions) * 100.))
    print('Weighted F1-score: {:.3f}'.format(
        metrics.f1_score(targets,
                         predictions,
                         average = 'weighted')))
    
def train_net(dataset_path, neighbors, channels, model, num_epochs, batch_size,
              learning_rate, l2_reg, gamma, power, momentum, snapshot):
    
    print('Loading data...')
    dataset = load_data(dataset_path, neighbors, channels)
    
    print('Building model and compiling functions...')
    output_layer = model(
        input_height = dataset['input_height'],
        input_width = dataset['input_width'],
        output_dim = dataset['output_dim'],
        batch_size = batch_size,
    )
    
    iter_funcs = create_iter_functions(dataset, output_layer,
                                       batch_size, momentum, l2_reg)
    
    print('Starting training...')
    tstart = time.time()
    trun = tstart

    best_valid_loss = np.inf
    best_test_accuracy = 0.
    best_epoch = None
    
    try:
        for epoch in train(iter_funcs, dataset, batch_size,
                           learning_rate, gamma, power):
            
            print('Epoch {}/{} ({:.3f} s)'.format(
                epoch['number'], num_epochs, time.time() - tstart))
            tstart = time.time()
            print(' {:<30}{:>20.6f}'.format(
                'training loss:', epoch['train_loss']))
            print(' {:<30}{:>20.6f}'.format(
                'validation loss:', epoch['valid_loss']))
            print(' {:<30}{:>20.6f}%'.format('validation accuracy:',
                                             epoch['valid_accuracy'] * 100.))

            if epoch['valid_loss'] < best_valid_loss:
                test_accuracy = test(iter_funcs, dataset, batch_size)
                print(' {:<30}{:>20.6f}%'.format('testing accuracy:',
                                                 test_accuracy * 100.))
                best_valid_loss = epoch['valid_loss']

                if test_accuracy > best_test_accuracy:
                    best_epoch = epoch['number']
                    best_test_accuracy = test_accuracy

                    if snapshot != None:
                        state = dict (
                            dataset_path = dataset_path,
                            neighbors = neighbors,
                            channels = channels,
                            model = model,
                            learned_params = get_all_param_values(output_layer),
                            epoch = epoch['number'],
                            batch_size = batch_size,
                            learning_rate = learning_rate,
                            l2_reg = l2_reg,
                            gamma = gamma,
                            power = power,
                            momentum = momentum,
                        )
                        save_net(snapshot, state)
                    
            if epoch['number'] >= num_epochs:
                break
            
    except KeyboardInterrupt:
        raise
    
    print('Highest testing accuracy (@ epoch {}): {:.6f}%'.format(
        best_epoch,
        best_test_accuracy * 100.))
    print('Running {} took {:.2f} m'.format(os.path.split(__file__)[1],
                                            (time.time() - trun) / 60.))
    
    return output_layer

def save_net(snapshot, state):

    try:
        f = gzip.open(snapshot, 'wb')
        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
    except IOError:
        print('Unable to write to file: {}'.format(snapshot))
    else:
        print('Created snapshot: {}'.format(snapshot))
        f.close()

def load_net(snapshot):

    state = None
    try:
        f = gzip.open(snapshot, 'rb')
        state = pickle.load(f)
    except IOError:
        print('Unable to read file: {}'.format(snapshot))
    else:
        f.close()

    return state

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: ' + sys.argv[0] + ' <net1> [<net2> ...]')
        print('   or: ' + sys.argv[0] + ' <net.p.gz>')
        print('Train one or more networks sequentially,'
              ' or test a saved network.') 
        sys.exit()

    if os.path.splitext(sys.argv[1])[1] == '':
        
        import models
        
        args = set(sys.argv[1 : ])
        members = dir(models)
        train_models = [x for x in members if x in args]

        try: 
            for m in train_models:
            
                print('Network:', m)
                d = models.__dict__.get(m)
            
                for k, v in d.iteritems():
                    print(' {}: {}'.format(k, v))
                
                train_net(**d)
                print('-------------------------------------------------------')

        except KeyboardInterrupt:
            pass       

    else:
        test_net(sys.argv[1])
        
    
