import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from utils import tile_raster_images, load_data

try:
    import PIL.Image as Image
except ImportError:
    import Image


class cA(object):
    def __init__(self, numpy_rng,
                 theano_rng=None,
                 input=None,
                 n_visible=784,
                 n_hidden=100,
                 n_batchsize=1,
                 W=None,
                 bhid=None,
                 bvis=None):
        

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize
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
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                                   dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_jacobian(self, hidden, W):
        return T.reshape(hidden * (1 - hidden),
                         (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                             W, (1, self.n_visible, self.n_hidden))

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, contraction_level, learning_rate):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        J = self.get_jacobian(y, self.W)
        self.L_rec = - T.sum(self.x * T.log(z) +
                             (1 - self.x) * T.log(1 - z),
                             axis=1)

        self.L_jacob = T.sum(J ** 2) / self.n_batchsize

        cost = T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)


def test_cA(learning_rate=0.01, training_epochs=20,
            dataset='../datasets/mnist.pkl.gz',
            batch_size=10, output_folder='cA_plots', contraction_level=.1):

    datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] 
    n_train_batches /= batch_size

    index = T.lscalar() 
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    rng = numpy.random.RandomState(123)

    ca = cA(numpy_rng=rng, input=x,
            n_visible=28 * 28, n_hidden=500, n_batchsize=batch_size)

    cost, updates = ca.get_cost_updates(contraction_level=contraction_level,
                                        learning_rate=learning_rate)

    train_ca = theano.function(
        [index],
        [T.mean(ca.L_rec), ca.L_jacob],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_ca(batch_index))

        c_array = numpy.vstack(c)
        print 'Training epoch %d, reconstruction cost ' % epoch, numpy.mean(
            c_array[0]), ' jacobian norm ', numpy.mean(numpy.sqrt(c_array[1]))

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(tile_raster_images(
        X=ca.W.get_value(borrow=True).T,
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))

    image.save('cae_filters.png')

    os.chdir('../')

if __name__ == '__main__':
    test_cA()