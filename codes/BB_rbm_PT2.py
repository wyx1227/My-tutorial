import numpy

import theano
import theano.tensor as T
import time

import toy_dataset

from theano.tensor.shared_randomstreams import RandomStreams

class RBM(object):
    def __init__(
        self,
        input=None,
        n_visible=16,
        n_hidden=20,
        n_chains = 5,
        W=None,
        hbias=None,
        vbias=None,
        beta = None,
        numpy_rng=None,
        theano_rng=None
    ):
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = numpy.asarray(
                0.01 * numpy_rng.randn(n_visible, n_hidden),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            hbias = theano.shared(
                value=numpy.zeros(n_hidden, dtype=theano.config.floatX),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            vbias = theano.shared(
                value=numpy.zeros(n_visible, dtype=theano.config.floatX),
                name='vbias',
                borrow=True
            )
            
        if beta is None:
            beta = theano.shared(
                value=numpy.linspace(1, 0, num=n_chains, dtype=theano.config.floatX),
                name='vbias',
                borrow=True
            )
            # no orginal tava assim numpy.linspace(1, beta_lbound, n_beta)

        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]

        self.n_beta  = theano.shared(n_beta, name='n_beta')
        
        
        #persistent_vis_chain = theano.shared(
            #numpy.asarray(
                #test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
                #dtype=theano.config.floatX
            #)
        #)        


    def energy(self, v_sample, h_sample):
        ''' Function to compute E(v,h) '''
        E_w = T.sum(T.dot(v_sample, self.W)*h_sample, axis=1)
        E_vbias = T.dot(v_sample, self.vbias)
        E_hbias = T.dot(h_sample, self.hbias)
        return - E_w - E_vbias - E_hbias


    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.nnet.softplus(wx_b),axis = 1)
        return -hidden_term - vbias_term

    def propup(self, vis, non_linear=True, temper=False):
        activation = T.dot(vis, self.W) + self.hbias
        if temper:
            activation *= self.beta
        return T.nnet.sigmoid(activation) if non_linear else activation

    def sample_h_given_v(self, v0_sample, temper=False):
        h1_mean = self.propup(v0_sample, temper=temper)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype = theano.config.floatX)
        return [h1_mean, h1_sample]

    def propdown(self, hid, non_linear=True, temper=False):
        activation = T.dot(hid, self.W.T) + self.vbias
        if temper:
            activation *= self.beta
        return T.nnet.sigmoid(activation) if non_linear else activation

    def sample_v_given_h(self, h0_sample, temper=False):
        v1_mean = self.propdown(h0_sample, temper=temper)
        v1_sample = self.theano_rng.binomial(size = v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype = theano.config.floatX)
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample, temper=False):
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample, temper=temper)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample, temper=temper)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample, temper=False):
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample, temper=temper)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample, temper=temper)
        return [h1_mean, h1_sample, v1_mean, v1_sample]
 
    
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        chain_start = persistent
        
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        chain_end = nv_samples[-1]
        
        
## SWAP VEM AKI!!!!        
        h1_mean, h1_sample, v1_mean, v1_sample = self.gibbs_vhv(v0_sample, temper=True) 
        E = self.energy(v1_sample, h1_sample)     
        
        
# do PT original
#            # retrieve pointer to particles at given temperature
#            tp1 = mixstat_out[bi, ti1]
#            tp2 = mixstat_out[bi, ti2]
#
#            # calculate exchange probability between two particles
#            r_beta = beta_out[tp2] - beta_out[tp1]
#            r_E = E[tp2] - E[tp1]
#            log_r = r_beta * r_E
#            r = numpy.exp(log_r)
#            swap_prob = numpy.minimum(1, r)
#            swap = self.rng.rand() < swap_prob
#
#            # extract index list of particles to swap
#            idx = numpy.where(swap == True)
#            sti1, sti2 = ti1[idx], ti2[idx]
#            stp1, stp2 = tp1[idx], tp2[idx]
#
#            # move pointers around to reflect swap
#            mixstat_out[bi,sti1] = stp2  # move high to low
#            mixstat_out[bi,sti2] = stp1  # move low to high
#            # update temperatures as well
#            beta_out_stp1 = copy.copy(beta_out[stp1])
#            beta_out[stp1] = beta_out[stp2]
#            beta_out[stp2] = beta_out_stp1
        

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
    
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(
                lr,
                dtype=theano.config.floatX
            )
    
        updates[persistent] = nh_samples[-1]            
        monitoring_cost = self.get_pseudo_likelihood_cost(updates)

        return monitoring_cost, updates
 
    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)

        fe_xi = self.free_energy(xi)

        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        fe_xi_flip = self.free_energy(xi_flip)

        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost



    def pt_step(self, v0_sample, beta, mixstat, labels, swapstat, 
                rtime, avg_rtime, nup, ndown):

        # perform Gibbs steps for all particles
        h1_mean, h1_sample, v1_mean, v1_sample = self.gibbs_vhv(v0_sample, temper=True) 
        E = self.energy(v1_sample, h1_sample)

        if self.n_beta.value > 1:

            # propose swap between chains (k,k+1) where k is odd
            beta, mixstat, labels, swapstat, rtime, avg_rtime = \
                    self.pt_swaps(beta, mixstat, E[self.batch_size:], 
                                  labels, swapstat,
                                  rtime, avg_rtime, self.tau, offset=0)

            # update labels and histograms
            nup, ndown = pt_update_histogram(mixstat, labels, nup, ndown, self.tau)
            
            # propose swap between chains (k,k+1) where k is even
            beta, mixstat, labels, swapstat, rtime, avg_rtime = \
                    self.pt_swaps(beta, mixstat, E[self.batch_size:],
                                  labels, swapstat,
                                  rtime, avg_rtime, self.tau, offset=1)
            
            # update labels and histograms
            nup, ndown = pt_update_histogram(mixstat, labels, nup, ndown, self.tau)

        return [h1_sample, v1_sample, v1_mean, 
                beta, mixstat, E, labels, swapstat, 
                rtime, avg_rtime, nup, ndown]

def test_rbm(learning_rate=0.1, training_epochs=15, n_chains=20, batch_size=20, n_hidden=20):
    
    train_set_x = toy_dataset(p=0.001, size=10000, seed=238904)
    test_set_x = toy_dataset(p=0.001, size=10000, seed=238905)
    train_set_x = numpy.asarray(train_set_x, dtype=theano.config.floatX)
    test_set_x = numpy.asarray(test_set_x, dtype=theano.config.floatX)
    train_set_x = theano.shared(train_set_x)
    test_set_x = theano.shared(test_set_x)
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    index = T.lscalar()
    x = T.matrix('x')
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)
    
    rbm = RBM(input=x, n_visible=4 * 4, n_hidden=n_hidden, 
              numpy_rng=rng, theano_rng=theano_rng)    
    
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)
    
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )
    
    start_time = time.clock()
    
    for epoch in xrange(training_epochs):
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)
    
    end_time = time.clock()
    print 'Training took %f minutes' % ((end_time - start_time)/ 60.)

    
    
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]
    
    test_idx = rng.randint(number_of_test_samples - n_chains)
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
    vis_mf, vis_sample = sample_fn()


if __name__ == '__main__':
    test_rbm()