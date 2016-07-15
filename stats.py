import numpy as np
import gzip, pickle
import sys

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'Usage: ', sys.argv[0], ' <dataset>'
        sys.exit()

    f = gzip.open(sys.argv[1], 'rb')
    train, val, test = pickle.load(f)
    f.close()

    print 'Dataset: ' + sys.argv[1]
    print 'Total samples: ', train[0].shape[0] + val[0].shape[0] + test[0].shape[0]
    print '--------------------------------------------------------------------------------'

    print 'Training set:'
    L = np.unique(train[1]).astype(np.int64)
    total = 0
    for l in L:
        class_data = train[0][train[1] == l, :]
        num_samples = class_data.shape[0]
        print 'Class ', l, ': ', num_samples
        total += num_samples
    print 'Training samples: ', total
    print '--------------------------------------------------------------------------------'

    print 'Validation set:'
    L = np.unique(val[1]).astype(np.int64)
    total = 0
    for l in L:
        class_data = val[0][val[1] == l, :]
        num_samples = class_data.shape[0]
        print 'Class ', l, ': ', num_samples
        total += num_samples
    print 'Validation samples: ', total
    print '--------------------------------------------------------------------------------'

    print 'Test set:'
    L = np.unique(test[1]).astype(np.int64)
    total = 0
    for l in L:
        class_data = test[0][test[1] == l, :]
        num_samples = class_data.shape[0]
        print 'Class ', l, ': ', num_samples
        total += num_samples
    print 'Test samples: ', total
    print '--------------------------------------------------------------------------------'

    
