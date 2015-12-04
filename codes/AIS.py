import numpy
import theano
from theano import tensor, config
from theano.tensor import nnet
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def rbm_ais(rbm_params, n_runs, visbias_a=None, data=None,
            betas=None, key_betas=None, rng=None, preproc=None,
            seed=23098):
    """
    Implements Annealed Importance Sampling for Binary-Binary RBMs, as
    described in:
    * Neal, R. M. (1998) ``Annealed importance sampling'', Technical Report No.
      9805 (revised), Dept. of Statistics, University of Toronto, 25 pages
    * Ruslan Salakhutdinov, Iain Murray. "On the quantitative analysis of deep
      belief networks".
      Proceedings of the 25th International Conference on Machine Learning,
      p.872-879, July 5--9, 2008, Helsinki, Finland
    Parameters
    ----------
    rbm_params: list
        list of numpy.ndarrays containing model parameters:
        [weights,visbias,hidbias]
    n_runs: int
        number of particles to use in AIS simulation (size of minibatch)
    visbias_a: numpy.ndarray
        optional override for visible biases. If both visbias_a and data
        are None, visible biases will be set to the same values of the
        temperature 1 model. For best results, use the `data` parameter.
    data: matrix, numpy.ndarray
        training data used to initialize the visible biases of the base-rate
        model (usually infinite temperature), to the log-mean of the
        distribution (maximum likelihood solution assuming a zero-weight
        matrix). This ensures that the base-rate model is the "closest" to the
        model at temperature 1.
    betas: numpy.ndarray
        vector specifying inverse temperature of intermediate distributions (in
        increasing order). If None, defaults to AIS.dflt_betas
    key_betas: numpy.ndarray
        if not None, AIS.run will save the log AIS weights for all temperatures
        in `key_betas`.  This allows the user to estimate logZ at several
        temperatures in a single pass of AIS.
    rng: None or RandomStream
        Random number generator object to use.
    seed: int
        if rng is None, initialize rng with this seed.
    """
    (weights, visbias, hidbias) = rbm_params

    if rng is None:
        rng = numpy.random.RandomState(seed)

    if data is None:
        if visbias_a is None:
            # configure base-rate biases to those supplied by user
            visbias_a = visbias
        else:
            visbias_a = visbias_a
    else:
        # set biases of base-rate model to ML solution
        data = preproc(data)
        data = numpy.asarray(data, dtype=config.floatX)
        data = numpy.mean(data, axis=0)
        data = numpy.minimum(data, 1 - 1e-5)
        data = numpy.maximum(data, 1e-5)
        visbias_a = -numpy.log(1. / data - 1)
    hidbias_a = numpy.zeros_like(hidbias)
    weights_a = numpy.zeros_like(weights)
    # generate exact sample for the base model
    v0 = numpy.tile(1. / (1 + numpy.exp(-visbias_a)), (n_runs, 1))
    v0 = numpy.array(v0 > rng.random_sample(v0.shape), dtype=config.floatX)

    # we now compute the log AIS weights for the ratio log(Zb/Za)
    ais = rbm_z_ratio((weights_a, visbias_a, hidbias_a),
                      rbm_params, n_runs, v0,
                      betas=betas, key_betas=key_betas, rng=rng)
    dlogz, var_dlogz = ais.estimate_from_weights()

    # log Z = log_za + dlogz
    ais.log_za = weights_a.shape[1] * numpy.log(2) + \
                 numpy.sum(numpy.log(1 + numpy.exp(visbias_a)))
    ais.log_zb = ais.log_za + dlogz
    return (ais.log_zb, var_dlogz), ais








# DO OTRO

def ais_data(fname, do_exact=True):

    rbm = load_rbm(fname)
  

    # load data to set visible biases to ML solution
    from pylearn.datasets import MNIST
    dataset = MNIST.train_valid_test()
    data = numpy.array(dataset.train.x, dtype=theano.config.floatX)

    # run ais using B=0 model with ML visible biases
    t1 = time.time()
    (logz, log_var_dz), aisobj = ais.rbm_ais(rbm.param_vals, n_runs=100, seed=123, data=data)
    print 'AIS logZ         : %f' % logz
    print '    log_variance : %f' % log_var_dz
    print 'Elapsed time: ', time.time() - t1

    if do_exact: 
        # exact log-likelihood
        exact_logz = rbm_tools.compute_log_z(rbm)
        print 'Exact logZ = %f' % exact_logz

        numpy.testing.assert_almost_equal(exact_logz, logz, decimal=0)

def test_ais():
    ais_data('mnistvh.mat')
    #ais_nodata('mnistvh.mat')
    #ais_nodata('mnistvh_500.mat', do_exact=False)