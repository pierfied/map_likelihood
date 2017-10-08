import numpy as np
import ctypes
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer

c_p_int = ctypes.POINTER(ctypes.c_int)
c_p_double = ctypes.POINTER(ctypes.c_double)

class SampleChain(ctypes.Structure):
    _fields_ = [('num_samples', ctypes.c_int),
                ('num_params', ctypes.c_int),
                ('accept_rate', ctypes.c_double),
                ('samples', ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ('log_likelihoods', ctypes.POINTER(ctypes.c_double))]

class LikelihoodArgs(ctypes.Structure):
    _fields_ = [('num_params', ctypes.c_int),
                ('f', ctypes.POINTER(ctypes.c_double)),
                ('y_inds', ctypes.POINTER(ctypes.c_int)),
                ('N', ctypes.POINTER(ctypes.c_double)),
                ('nx', ctypes.c_int),
                ('ny', ctypes.c_int),
                ('nz', ctypes.c_int),
                ('mu', ctypes.c_double),
                ('inv_cov', ctypes.POINTER(ctypes.c_double)),
                ('expected_N', ctypes.c_double)]

# Various parameters.
n_side = 10
expected_N = 10.
sigma_y = 1.
mu = -0.5 * (sigma_y ** 2)
inv_cov = np.array([1/(sigma_y ** 2),0,0,0])
num_params = n_side ** 3

# Create the fake survey.
y_true = np.random.normal(mu,sigma_y,(n_side,n_side,n_side))
N = np.random.poisson(expected_N*np.exp(y_true)).astype(np.float64)
f_map = np.ones((n_side,n_side,n_side))
y_inds = np.arange(n_side**3,dtype=np.int32)
d_photo = N/expected_N - 1
d_photo = np.maximum(d_photo,-0.999)
y_photo = np.log(1 + d_photo)

# Load the likelihood/sampling library.
sampler_lib = ctypes.cdll.LoadLibrary('../cmake-build-debug/libmaplikelihood.so')

# Setup the sampler function.
sampler = sampler_lib.sample_map
sampler.argtypes = [ctypes.POINTER(ctypes.c_double),
                    LikelihoodArgs, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_double]
sampler.restype = SampleChain

# Create the LikelihoodArgs
args = LikelihoodArgs()
args.num_params = num_params
args.f = f_map.ravel().ctypes.data_as(c_p_double)
args.y_inds = y_inds.ctypes.data_as(c_p_int)
args.N = N.ravel().ctypes.data_as(c_p_double)
args.nx = n_side
args.ny = n_side
args.nz = n_side
args.mu = mu
args.inv_cov = inv_cov.ctypes.data_as(c_p_double)
args.expected_N = expected_N

# Sampler Parameters
num_samps = 10000
num_steps = 1
num_burn = 1000
epsilon = 0.1

# Run the sampler.
# y_photo = np.random.randn(n_side ** 3)
y0 = np.random.randn(n_side ** 3).ctypes.data_as(c_p_double)
y0 = y_photo.ctypes.data_as(c_p_double)
results = sampler(y0, args, num_samps, num_steps, num_burn, epsilon)

print(results.accept_rate)

likelihoods = np.array([results.log_likelihoods[i]
                        for i in range(num_samps)])

plt.plot(range(num_samps),likelihoods)
plt.show()

chain = np.array([[results.samples[i][j] for j in range(num_params)]
                  for i in range(num_samps)])

c = ChainConsumer()
c.add_chain(chain)
c.configure(sigma2d=False)

bounds = c.analysis.get_summary()
y_mle = [b[1] for b in bounds.values()]

plt.clf()
plt.scatter(y_true,y_photo,label='$y_0$')
plt.scatter(y_true,y_mle,label='$y_{MLE}$')
plt.xlabel('$y_{true}$')
plt.ylabel('$y_{samp}$')
plt.legend()
plt.tight_layout()
plt.savefig('scatter.png')
plt.show()

c = ChainConsumer()
samp_params = np.random.choice(num_params,5)
c.add_chain(chain[:,samp_params])
c.configure(sigma2d=False)
fig = c.plotter.plot(figsize='column',truth=y_true.ravel()[samp_params])
# plt.tight_layout()
plt.savefig('corner.png')
plt.show()