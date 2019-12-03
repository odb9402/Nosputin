import tensorflow as tf
import gpflow
import numpy as np
from gpflow.kernels import Kernel, Stationary, Sum, Product  # used to derive new kernels
from gpflow import Param
from gpflow import transforms

from gpflow import settings
float_type = settings.dtypes.float_type
from gpflow.test_util import notebook_niter, is_continuous_integration

################################## Hyperparameters ##################################

learningRate = 0.001
trainingWeeks = 10
predWeeks = 2
epochs = 1000
evalEvery = 20
global is_test
is_test = False
eps = 1e-9
################################## Deep kernel model

def fully_nn_layer(x, input_dim, output_dim):
    global is_test
    x = tf.reshape(x, [-1, input_dim])
    
    fully1 = tf.layers.dense(inputs=x, units=1000)
    #fully1 = tf.contrib.layers.batch_norm(fully1, is_training= not is_test, decay=0.9, zero_debias_moving_mean=True)
    fully1 = tf.nn.relu(fully1)
    fully1 = tf.layers.dropout(fully1, rate=0.1, training=not is_test)
    
    fully2 = tf.layers.dense(inputs=fully1, units=500)
    #fully2 = tf.contrib.layers.batch_norm(fully2, is_training= not is_test, decay=0.9, zero_debias_moving_mean=True)
    fully2 = tf.nn.relu(fully2)
    fully2 = tf.layers.dropout(fully2, rate=0.1, training=not is_test)
    
    fully3 = tf.layers.dense(inputs=fully2, units=50, activation=tf.nn.relu)
    #fully3 = tf.contrib.layers.batch_norm(fully3, is_training= not is_test, decay=0.9, zero_debias_moving_mean=True)
    fully3 = tf.nn.relu(fully3)
    fully3 = tf.layers.dropout(fully3, rate=0.1, training=not is_test)
    
    fully4 = tf.layers.dense(inputs=fully3, units=output_dim, activation=None)
    
    return fully4


class DeepKernel(gpflow.kernels.Kernel):
    def __init__(self, kern, f):
        super().__init__(kern.input_dim)
        self.kern = kern
        self._f = f
    
    def f(self, X):
        if X is not None:
            with tf.variable_scope('forward', reuse=tf.AUTO_REUSE):
                return self._f(X)
    
    def _get_f_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='forward')
    
    @gpflow.autoflow([gpflow.settings.float_type, [None,None]])
    def compute_f(self, X):
        return self.f(X)
    
    def K(self, X, X2=None):
        return self.kern.K(X, X2)
    
    def Kdiag(self, X):
        return self.kern.Kdiag(X)

    
class KernelSpaceInducingPoints(gpflow.features.InducingPointsBase):
    pass

# same Kuu as regular inducing points
gpflow.features.Kuu.register(KernelSpaceInducingPoints, DeepKernel)(
    gpflow.features.Kuu.dispatch(gpflow.features.InducingPoints, gpflow.kernels.Kernel)
)

# Kuf is in NN output space
@gpflow.features.dispatch(KernelSpaceInducingPoints, DeepKernel, object)
def Kuf(feat, kern, Xnew):
    with gpflow.params_as_tensors_for(feat):
        return kern.K(feat.Z, kern.f(Xnew))

class NNComposedKernel(DeepKernel):
    """
    This kernel class applies f() to X before calculating K
    """
    
    def K(self, X, X2=None, train=True):
        return super().K(self.f(X), self.f(X2))
    
    def Kdiag(self, X):
        return super().Kdiag(self.f(X))
    
# we need to add these extra functions to the model so the tensorflow variables get picked up
class NN_SVGP(gpflow.models.SVGP):
    @property
    def trainable_tensors(self):
        return super().trainable_tensors + self.kern._get_f_vars()

    @property
    def initializables(self):
        return super().initializables + self.kern._get_f_vars()

    
class NN_VGP(gpflow.models.VGP):
    @property
    def trainable_tensors(self):
        return super().trainable_tensors + self.kern._get_f_vars()

    @property
    def initializables(self):
        return super().initializables + self.kern._get_f_vars()    


class NN_GPR(gpflow.models.GPR):
    @property
    def trainable_tensors(self):
        return super().trainable_tensors + self.kern._get_f_vars()

    @property
    def initializables(self):
        return super().initializables + self.kern._get_f_vars()

class NN_SGPR(gpflow.models.SGPR):
    @property
    def trainable_tensors(self):
        return super().trainable_tensors + self.kern._get_f_vars()
    @property
    def initializables(self):
        return super().initializables + self.kern._get_f_vars()


def square_dist(X, X2):
    Xs = tf.reduce_sum(tf.square(X), 1)
    X2s = tf.reduce_sum(tf.square(X2), 1)
    return -2 * tf.matmul(X, X2, transpose_b=True) + tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))    
    

################################ Spectral Mixture Kernel
class SMKernelComponent(Stationary):
    """
    Spectral Mixture kernel.
    """
    def __init__(self, input_dim, variance=1.0, lengthscales=None, 
                 frequency=1.0, active_dims=None, ARD=False):
        Stationary.__init__(self, input_dim=input_dim, variance=variance, lengthscales=lengthscales,
                            active_dims=active_dims, ARD=ARD)
        self.frequency = Param(frequency, transforms.positive, dtype=float_type)
        self.frequency.prior = gpflow.priors.Exponential(1.0)
        self.variance.prior = gpflow.priors.LogNormal(0, 1)

    @gpflow.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X
        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        freq = tf.expand_dims(self.frequency, 0)
        freq = tf.expand_dims(freq, 0)  # 1 x 1 x D
        r = tf.reduce_sum(2.0 * np.pi * freq * (f - f2), 2)
        return self.variance * tf.exp(-2.0*np.pi**2*self.scaled_square_dist(X, X2)) * tf.cos(r)
    
    @gpflow.params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    @gpflow.params_as_tensors
    def scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return super()._clipped_sqrt(dist)
    
    #@staticmethod
    #def _clipped_sqrt(r2):
    #    return tf.sqrt(tf.maximum(r2, 1e-40))


def SMKernel(Q, input_dim, active_dims=None, variances=None, frequencies=None,
             lengthscales=None, max_freq=1.0, max_len=1.0, ARD=False):
    """
    Initialises a SM kernel with Q components. Optionally uses a given initialisation,
    otherwise uses a random initialisation.
    max_freq: Nyquist frequency of the signal, used to initialize frequencies
    max_len: range of the inputs x, used to initialize length-scales
    """
    if variances is None:
        variances = [1./Q for _ in range(Q)]
    if frequencies is None:
        frequencies = [np.random.rand(input_dim)*max_freq for _ in range(Q)]
    if lengthscales is None:
        lengthscales = [np.abs(max_len*np.random.randn(input_dim if ARD else 1)) for _ in range(Q)]
    kerns = [SMKernelComponent(input_dim, active_dims=active_dims, variance=variances[i], 
                               frequency=frequencies[i], lengthscales=lengthscales[i], ARD=ARD)
             for i in range(Q)]
    return Sum(kerns)
    
    
class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    def log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0:1], Y[:, 1:2]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def conditional_mean(self, F):
        raise NotImplementedError

    def conditional_variance(self, F):
        raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0:1], Y[:, 1:2]
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(NoiseVar) \
               - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / (NoiseVar + eps)

    
    
################################## Model variables ##################################
is_training = tf.placeholder(tf.bool)
p_dropout = tf.placeholder(tf.float32)
