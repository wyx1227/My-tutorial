from rbm import RBM

import theano.tensor as T

class GBRBM(RBM):
    def __init__(self,
                 input,
                 n_visible=16,
                 n_hidden=20,                 
                 W=None, hbias=None, vbias=None,
                 numpy_rng=None, theano_rng=None):

            # initialize parent class (RBM)
            RBM.__init__(self,
                         input=input,
                         n_visible=n_visible,
                         n_hidden=n_hidden,
                         W=W, hbias=hbias, vbias=vbias,
                         numpy_rng=numpy_rng, theano_rng=theano_rng)

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = 0.5 * T.dot((v_sample - self.vbias), (v_sample - self.vbias).T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        '''
            Since the input data is normalized to unit variance and zero mean, we do not have to sample
            from a normal distribution and pass the pre_sigmoid instead. If this is not the case, we have to sample the
            distribution.
        '''     
        v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]