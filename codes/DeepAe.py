try:
    import PIL.Image as Image
except ImportError:
    import Image

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from logistic_sgd import LogisticRegression

from utils import load_data, tile_raster_images

from toy_dataset import toy_dataset

from mlp import HiddenLayer
from BB_rbm_CD import RBM

class DBN(object):

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500]):

        self.sigmoid_layers = []
        self.sigmoid_layers_prime = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')  


        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            rbm_layer = RBM(numpy_rng=numpy_rng,
                                        theano_rng=theano_rng,
                                        input=layer_input,
                                        n_visible=input_size,
                                        n_hidden=hidden_layers_sizes[i])
            
            self.rbm_layers.append(rbm_layer)            

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid,
                                        W=rbm_layer.W,
                                        b=rbm_layer.hbias)

            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)
            

        for i in xrange(self.n_layers-1,-1,-1):
            if i == 0:
                output_size = n_ins
            else:
                output_size = hidden_layers_sizes[i - 1]

            if i == self.n_layers-1:
                layer_input = self.sigmoid_layers[-1].output
            else:
                layer_input = self.sigmoid_layers_prime[-1].output
                
            rbm_layer = self.rbm_layers[i]

            sigmoid_layer_prime = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=hidden_layers_sizes[i],
                                        n_out=output_size,
                                        activation=T.nnet.sigmoid,
                                        W=rbm_layer.W.T,
                                        b=rbm_layer.vbias)

            self.sigmoid_layers_prime.append(sigmoid_layer_prime)
            
            self.params.extend([sigmoid_layer_prime.b])

        self.finetune_cost = self.get_reconstruction_cost(self.x)

    def get_reconstruction_cost(self, x):
        reconstructed = self.sigmoid_layers_prime[-1].output        
        L = - T.sum(self.x * T.log(reconstructed) + (1 - self.x) * T.log(1 - reconstructed), axis=1)
        cost = T.mean(L)
        return cost


    def pretraining_functions(self, train_set_x, batch_size, k):

        index = T.lscalar('index')  
        learning_rate = T.scalar('lr')  

        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            cost, updates = rbm.get_cost_updates(learning_rate, k=k)

            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):

        (train_set_x, _) = datasets[0]
        (valid_set_x, _) = datasets[1]
        (test_set_x, _) = datasets[2]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index') 

        gparams = T.grad(self.finetune_cost, self.params)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i = theano.function(
            [index],
            outputs=self.finetune_cost,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            outputs=self.finetune_cost,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


        
        
def test_toy(finetune_lr=0.1,
             pretraining_epochs=3,
             pretrain_lr=0.01,
             k=1,
             training_epochs=10,
             dataset='../datasets/mnist.pkl.gz',
             output_folder='toy_DeepAe_plots',
             batch_size=10):
   
    print 'Creating dataset...'
    train_set_x = toy_dataset(p=0.001, size=20000, seed=238904)
    valid_set_x = toy_dataset(p=0.001, size=5000, seed=238905)
    test_set_x = toy_dataset(p=0.001, size=5000, seed=238906)
    train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
    valid_set_x = numpy.asarray(valid_set_x, dtype=theano.config.floatX)
    test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)
    numpy.random.shuffle(train_set_x)
    numpy.random.shuffle(valid_set_x)
    numpy.random.shuffle(test_set_x)
    train_set_x = theano.shared(train_set_x)
    valid_set_x = theano.shared(valid_set_x)   
    test_set_x = theano.shared(test_set_x)    
    
    datasets=[]
    datasets.append((train_set_x, None))
    datasets.append((valid_set_x, None))
    datasets.append((test_set_x, None))
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    numpy_rng = numpy.random.RandomState(123)
    
    print '... building the model'
    dbn = DBN(numpy_rng=numpy_rng, n_ins=4 * 4,
              hidden_layers_sizes=[25, 10])
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)       
    
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)
    
    print '... pre-training the model'
    start_time = timeit.default_timer()
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)
    
    end_time = timeit.default_timer()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
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
    #------------------------------------------------------------------
        
    recontruct_train_fn = theano.function(
        inputs=[],
        outputs=dbn.sigmoid_layers_prime[-1].output,
        givens={
                dbn.x: train_set_x[100 : 120]
        }            
    )           
    
    recontruct_test_fn = theano.function(
        inputs=[],
        outputs=dbn.sigmoid_layers_prime[-1].output,
        givens={
            dbn.x: test_set_x[100 : 120]
        }            
    )
    
    image = Image.fromarray(tile_raster_images(
        X=numpy.concatenate((test_set_x[100 : 120].eval(),recontruct_test_fn())),
        img_shape=(4, 4), tile_shape=(2,20),
        tile_spacing=(1, 1)))
    image.save('reconstructed_test.png')
    
    image = Image.fromarray(tile_raster_images(
        X=numpy.concatenate((train_set_x[100 : 120].eval(),recontruct_train_fn())),
        img_shape=(4, 4), tile_shape=(2, 20),
        tile_spacing=(1, 1)))
    image.save('reconstructed_train.png') 
    
    os.chdir('../') 


def test_mnist(finetune_lr=0.1,
             pretraining_epochs=3,
             pretrain_lr=0.01,
             k=1,
             training_epochs=10,
             dataset='../datasets/mnist.pkl.gz',
             output_folder='mnist_DeepAe_plots',
             batch_size=10):
    
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    numpy_rng = numpy.random.RandomState(123)
    
    print '... building the model'
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[800, 400])
    
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)       
        

    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = timeit.default_timer()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
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
    #------------------------------------------------------------------
        
    recontruct_train_fn = theano.function(
        inputs=[],
        outputs=dbn.sigmoid_layers_prime[-1].output,
        givens={
                dbn.x: test_set_x[100 : 130]
        }            
    )           
    
    recontruct_test_fn = theano.function(
        inputs=[],
        outputs=dbn.sigmoid_layers_prime[-1].output,
        givens={
            dbn.x: test_set_x[100 : 130]
        }            
    )
    image = Image.fromarray(tile_raster_images(
        X=numpy.concatenate((test_set_x[100 : 120].eval(),recontruct_test_fn())),
        img_shape=(28, 28), tile_shape=(2,20),
        tile_spacing=(1, 1)))
    image.save('reconstructed_test.png')
    
    image = Image.fromarray(tile_raster_images(
        X=numpy.concatenate((train_set_x[100 : 120].eval(),recontruct_train_fn())),
        img_shape=(28, 28), tile_shape=(2, 20),
        tile_spacing=(1, 1)))
    image.save('reconstructed_train.png') 
    
    os.chdir('../') 


if __name__ == '__main__':
    test_mnist()
    test_toy()
