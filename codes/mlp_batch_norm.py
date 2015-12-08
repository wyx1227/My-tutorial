import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression

from utils import load_data, tile_raster_images

from toy_dataset import toy_dataset

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input
        if W is None:
            #Glorot 2010 uniform initialization
            scale = numpy.sqrt(6. / (n_in + n_out))
            W_values = numpy.asarray(
                rng.uniform(
                    low=-scale,
                    high=scale,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class MLP(object):

    def __init__(self,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
                 n_outs=10,
                 numpy_rng=None,
                 theano_rng=None,
                 epsilon_batchnorm=1e-6,
                 mode_batchnorm=0,
                 momentum_batchnorm=0.9):
        
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')  
        self.y = T.ivector('y')  


        self.epsilon_batchnorm = epsilon_batchnorm #-----
        self.mode_batchnorm = mode_batchnorm #------------
        self.momentum_batchnorm = momentum_batchnorm #----
        

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        self.errors = self.logLayer.errors(self.y)


    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index') 

        gparams = T.grad(self.finetune_cost, self.params)

        updates = []
        
        #--------------------------------------------------
        
        #input_shape = self.input_shape  # starts with samples axis
        #input_shape = input_shape[1:]

        #self.gamma = self.init((input_shape))
        #self.beta = K.zeros(input_shape)

        #self.params = [self.gamma, self.beta]
        #self.running_mean = K.zeros(input_shape)
        #self.running_std = K.ones((input_shape))
        
        #X = self.get_input(train=True)
        #m = K.mean(X, axis=0)
        #std = K.mean(K.square(X - m) + self.epsilon, axis=0)
        #std = K.sqrt(std)
        #mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        #std_update = self.momentum 
        #self.running_mean = K.zeros(input_shape)* self.running_std + (1-self.momentum) * std
        #self.updates = [(self.running_mean, mean_update),
                        #(self.running_std, std_update)]        
        
        #def get_output(self, train):
            #X = self.get_input(train)
            #if self.mode == 0:
                #X_normed = ((X - self.running_mean) /
                            #(self.running_std + self.epsilon))
            #elif self.mode == 1:
                #m = K.mean(X, axis=-1, keepdims=True)
                #std = K.std(X, axis=-1, keepdims=True)
                #X_normed = (X - m) / (std + self.epsilon)
            #out = self.gamma * X_normed + self.beta
            #return out        
        
        #--------------------------------------------------
        
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_mnist(finetune_lr=0.1,
             training_epochs=100,
             dataset='../datasets/mnist.pkl.gz',
             batch_size=10):
    
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    numpy_rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30)) 
    
    print '... building the model'
    mlp = MLP(numpy_rng=numpy_rng,
              theano_rng=theano_rng,
              n_ins=28 * 28,
              hidden_layers_sizes=[1000],
              n_outs=10)

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = mlp.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    patience = 4 * n_train_batches  
    patience_increase = 2.    
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:

                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

if __name__ == '__main__':
    test_mnist()

