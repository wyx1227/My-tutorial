import os
import sys
import timeit
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images, load_data

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
from toy_dataset import toy_dataset


class dA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)


    def kl_divergence(self, p, p_hat):
        term1 = p * T.log(p)
        term2 = p * T.log(p_hat)
        term3 = (1-p) * T.log(1 - p)
        term4 = (1-p) * T.log(1 - p_hat)
        return term1 - term2 + term3 - term4

    def sparsity_penalty(self, h, sparsity_level=0.05, sparsity_regularization=0.01):
        sparsity_lvl = theano.shared(
                        value=sparsity_level* numpy.ones(
                            self.n_hidden,
                            dtype=theano.config.floatX
                        ),
                        name='sparsity_level',
                        borrow=True
                    )        
        avg_act = h.mean(axis=0)
        kl_div = self.kl_divergence(sparsity_lvl, avg_act)
        sparsity_penalty = sparsity_regularization * kl_div.sum()
        return sparsity_penalty

    def get_cost_updates(self, corruption_level, l2, learning_rate, sparsity_level, sparsity_regularization):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        sparsity_penal = self.sparsity_penalty(y, sparsity_level, sparsity_regularization)
        cost += sparsity_penal   
        
        if l2: cost += l2 * T.sum(self.W**2)


        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
        

def test_toy(learning_rate=0.1, training_epochs=15,
             sparsity_regularization=0.001,
             sparsity_level=0.05,             
             n_hidden=30,
             dataset='../datasets/mnist.pkl.gz',
             batch_size=20,
             output_folder='toy_sA_plots'):
 
    print 'Creating dataset...'
    train_set_x = toy_dataset(p=0.001, size=10000, seed=238904)
    test_set_x = toy_dataset(p=0.001, size=10000, seed=238905)
    train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
    test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)
    numpy.random.shuffle(train_set_x)
    numpy.random.shuffle(test_set_x)
    train_set_x = theano.shared(train_set_x)
    test_set_x = theano.shared(test_set_x)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  
    x = T.matrix('x') 

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=4 * 4,
        n_hidden=n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate,
        l2=0.0001,
        sparsity_level=sparsity_level,
        sparsity_regularization=sparsity_regularization
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    print 'Starting training with %d epochs' %training_epochs
    plotting_time = 0.
    start_time = time.clock()
    print 'Starting training at %f ' %start_time
 
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

        plotting_start = timeit.default_timer()
        image = Image.fromarray(
                    tile_raster_images(
                        X=da.W.get_value(borrow=True).T,
                        img_shape=(4, 4),
                        tile_shape=(10, 10),
                        tile_spacing=(1, 1)
                    )
                )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start) 
                
    end_time = time.clock()
    training_time = (end_time - start_time)
    print 'Ending training at %f ' %end_time
    print 'Training took %.2f minutes' % ((end_time - start_time)/ 60.)

    os.chdir('../')



def test_mnist(learning_rate=0.1, training_epochs=15,
               n_hidden=500,
               sparsity_regularization=0.001,
               sparsity_level=0.05,                     
               dataset='../datasets/mnist.pkl.gz',
               batch_size=20,
               output_folder='MNIST_sA_plots'):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  
    x = T.matrix('x') 

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=500
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.3,
        learning_rate=learning_rate,
        l2=0.0001,
        sparsity_level=sparsity_level,
        sparsity_regularization=sparsity_regularization        
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    print 'Starting training with %d epochs' %training_epochs
    plotting_time = 0.
    start_time = time.clock()
    print 'Starting training at %f ' %start_time
 
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

        plotting_start = timeit.default_timer()
        image = Image.fromarray(
                    tile_raster_images(
                        X=da.W.get_value(borrow=True).T,
                        img_shape=(28, 28),
                        tile_shape=(10, 10),
                        tile_spacing=(1, 1)
                    )
                )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start) 
                
    end_time = time.clock()
    training_time = (end_time - start_time)
    print 'Ending training at %f ' %end_time
    print 'Training took %.2f minutes' % ((end_time - start_time)/ 60.)

    os.chdir('../')


if __name__ == '__main__':
    test_toy()
    test_mnist()