# https://github.com/gdesjardins/deep_tempering/blob/master/data/test_modes.py
import numpy
import theano
floatX = theano.config.floatX

def toy_dataset(p=0.01, size=10000, seed=238904, w=[.25,.25,.25,.25]):
    """
    Generates the dataset used in [Desjardins et al, AISTATS 2010]. The dataset
    is composed of 4x4 binary images with four basic modes: full black, full
    white, and [black,white] and [white,black] images. Modes are created by
    drawing each pixel from the 4 basic modes with a bit-flip probability p.
    
    :param p: probability of flipping each pixel p: scalar, list (one per mode) 
    :param size: total size of the dataset
    :param seed: seed used to draw random samples
    :param w: weight of each mode within the dataset
    """

    # can modify the p-value separately for each mode
    if not isinstance(p, (list,tuple)):
        p = [p for i in w]

    rng = numpy.random.RandomState(seed)
    data = numpy.zeros((size,16))

    # mode 1: black image
    B = numpy.zeros((1,16))
    # mode 2: white image
    W = numpy.ones((1,16))
    # mode 3: white image with black stripe in left-hand side of image
    BW = numpy.ones((4,4))
    BW[:, :2] = 0
    BW = BW.reshape(1,16)
    # mode 4: white image with black stripe in right-hand side of image
    WB = numpy.zeros((4,4))
    WB[:, :2] = 1
    WB = WB.reshape(1,16)

    modes = [B,W,BW,WB]
    data = numpy.zeros((0,16))
    
    # create permutations of basic modes with bitflip prob p
    for i, m in enumerate(modes):
        n = size * w[i]
        bitflip = rng.binomial(1,p[i],size=(n,16))
        d = numpy.abs(numpy.repeat(m, n, axis=0) - bitflip)
        data = numpy.vstack((data,d))

    return data