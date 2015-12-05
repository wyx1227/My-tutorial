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

from mlp import HiddenLayer
from BB_rbm_CD import RBM

class DBN(object):

    def __init__(self,
                 n_ins=784,
                 hidden_layers_sizes=[500, 500],
                 n_outs=10,
                 numpy_rng=None,
                 theano_rng=None):
        
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

            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        self.errors = self.logLayer.errors(self.y)

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


def test_toy(finetune_lr=0.1,
             pretraining_epochs=10,
             pretrain_lr=0.01,
             k=1,
             training_epochs=100,
             dataset='../datasets/mnist.pkl.gz',
             batch_size=10):
   
    print 'Creating dataset...'
    train_set_x = toy_dataset(p=0.001, size=10000, seed=238904)
    valid_set_x = toy_dataset(p=0.001, size=10000, seed=238905)
    test_set_x = toy_dataset(p=0.001, size=10000, seed=238906)
    train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
    valid_set_x = numpy.asarray(valid_set_x, dtype=theano.config.floatX)
    test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)
    numpy.random.shuffle(train_set_x)
    numpy.random.shuffle(valid_set_x)
    numpy.random.shuffle(test_set_x)
    train_set_x = theano.shared(train_set_x)
    valid_set_x = theano.shared(valid_set_x)   
    test_set_x = theano.shared(test_set_x)    
                
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    print '... building the model'
    dbn = DBN(numpy_rng=rng,
              theano_rng=theano_rng,
              n_ins=4 * 4,
              hidden_layers_sizes=[100, 50, 50],
              n_outs=10)

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

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)



    for i in xrange(dbn.n_layers):
        
        X=dbn.rbm_layers[i].W.get_value(borrow=True).T,
        
        image = Image.fromarray(tile_raster_images(
            X=dbn.rbm_layers[i].W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('filters_corruption_30.png')


def test_mnist(finetune_lr=0.1,
             pretraining_epochs=100,
             pretrain_lr=0.01,
             k=1,
             training_epochs=1000,
             dataset='../datasets/mnist.pkl.gz',
             batch_size=10):
    
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[1000, 1000, 1000],
              n_outs=10)

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

# ploting

    image = Image.fromarray(
                tile_raster_images(
                    X=rbm.W.get_value(borrow=True).T,
                    img_shape=(4, 4),
                    tile_shape=(10, 10),
                    tile_spacing=(1, 1)
                )
            )
    image.save('filters_at_epoch_%i.png' % epoch)    


def sample_DBN():
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    
    plot_every = 1000
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    updates.update({persistent_vis_chain: vis_samples[-1]})
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )    

    image_data = numpy.zeros(
        (5 * n_samples + 1, 5 * n_chains - 1),
        dtype='uint8'
    )
    for idx in xrange(n_samples):

        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[5 * idx:5 * idx + 4, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(4, 4),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    image = Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')             
                 


if __name__ == '__main__':
    test_toy()
    test_mnist()

