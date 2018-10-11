import sys
import argparse
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions.dist_math import bound

class CommitmentProcess(pm.Categorical):
    """

    """

    def __init__(self, PI=None, Q=None, renewal_mask=None, num_states=3,
                 *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.PI = PI
        self.Q = Q
        self.renewal_mask = renewal_mask
        self.k = num_states
        self.mode = tt.cast(0,dtype='int64')

    def logp(self, x):

        log_p = 0.
        for i in range(0, self.shape[0]):
            p_it = self.PI[x[i,][:-1]]
            x_t = x[i,][1:]
            x_0 = tt.stack([x[i,][0]])
            mask = self.renewal_mask

            log_p_i = pm.Categorical.dist(self.Q).logp(x_0) + tt.sum(pm.Categorical.dist(p_it).logp(x_t))

            # Restrction: if not churned, cannot be in state 0 at a renewal period
            log_p = log_p + bound(log_p_i, tt.dot(mask, x_t)>0)

        return log_p

class UsageProcess(pm.Discrete):
    """

    """

    def __init__(self, alpha=None, th0=None, G=None, states=None, num_states=3,
                 *args, **kwargs):
        super(UsageProcess, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.th0 = th0
        self.G = G
        self.states = states
        self.num_states = num_states
        self.mean = 0.


    def logp(self, x):
        states = self.states

        # build theta vector from the components
        theta = tt.concatenate([tt.stack([self.th0]), tt.exp(self.G)])
        zero = tt.zeros_like(theta)

        for i in range(1, num_states):
            theta = theta + tt.concatenate([zero[-i:], theta[0:self.num_states-i]])

        # build lambda matrix: theta is row vector, alpha is column
        # labmda is the outer-product of the two
        Lambda = tt.outer(self.alpha,theta)

        log_p = 0.
        for i in range(0, self.shape[0]):
            lam_it = Lambda[i][states[i,]]
            y_it = x[i]
            log_p_i = tt.sum(pm.Poisson.dist(mu=lam_it).logp(y_it))

            log_p = log_p + log_p_i

        return log_p




description = "fits pymc3 model"
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', dest='file', help='he observation data file', required=True)
parser.add_argument('-o', dest='output_dir', help='directory path for saved trace', required=True)
parser.add_argument('--chains', help='number of chains to sample from', default='1')
parser.add_argument('--draws', help='number of draws for sampling', default='3000')
parser.add_argument('--renewal', help='renewal period', default='3')
parser.add_argument('--num-states', dest='num_states', help='number of states to model', default='3')
args = parser.parse_args()

print('Running on PyMC3 v{}'.format(pm.__version__))

file = args.file
print('reading from file ' + file)


alldata = pd.read_csv(file)
observed_usage=alldata.iloc[:,1:].as_matrix()

renewal_period = int(args.renewal)

# Sampling params
chains = int(args.chains)
draws = int(args.draws)

# Model params
num_states = int(args.num_states)
num_custs = observed_usage.shape[0]
obs_len = observed_usage.shape[1]
renewal_mask = np.where(((np.arange(1,obs_len)+1) % renewal_period)!=0,0,1)

# This lets us avoid trying out a lot of invalid state possibilities
states_test_val = np.ones((num_custs, obs_len))

from scipy import optimize
with pm.Model() as model:
    Q = pm.Dirichlet('Q', a=np.ones((num_states)) + 1., shape=(num_states))
    PI = pm.Dirichlet('PI', a=np.ones((num_states,num_states)) + 1., shape=(num_states,num_states))
    r = pm.Gamma('r', alpha=0.01, beta=0.01)
    A = pm.Gamma('A',alpha=r, beta=r, shape=((num_custs)))
    th0 = pm.Uniform('th0',lower=0.0,upper=10.0)
    G = pm.Normal('G',mu=np.zeros(num_states-1), sd=np.ones(num_states-1)*10000., shape=(num_states-1))

    states = CommitmentProcess('states', PI=PI, Q=Q, renewal_mask = renewal_mask, num_states=num_states, shape=(num_custs,obs_len), testval=states_test_val)
    usage = UsageProcess('usage', alpha=A, th0=th0, G=G, states=states, num_states=num_states, shape=(num_custs), observed=observed_usage)

    start = pm.find_MAP(method='Powell')
    step1 = pm.Metropolis(vars=[r,PI,Q,A,G,th0,usage])
    step2 = pm.CategoricalGibbsMetropolis(vars=[states])
    trace = pm.sample(draws, start=start, step=[step1,step2], chains=chains)

print('saving to ' + args.output_dir)
pm.backends.ndarray.save_trace(trace, directory=args.output_dir, overwrite=True)
