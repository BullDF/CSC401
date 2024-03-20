import fnmatch
import os
import random

import numpy as np

dataDir = "/u/cs401/A3/data/"


class theta:
    def __init__(self, name, M=8, d=13):
        """
        Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """
        Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma`
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        print("TODO")

    def reset_omega(self, omega):
        """
        Resets omega.

        Parameters
        ----------
        omega : probability of an observation generated by certain component,
                should be of shape [M, 1] or [M]

        Returns
        -------
        N/A, resets self.omega to given omega
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """
        Resets mu.

        Parameters
        ----------
        mu : mean for the M components,
             should be of shape [M, d]

        Returns
        -------
        N/A, resets self.mu to given mu
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """
        Resets sigma.

        Parameters
        ----------
        Sigma : covariance for the M components,
                should be of shape [M, d]

        Returns
        -------
        N/A, resets self.Sigma to given Sigma
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(x, myTheta: theta, m=None):
    """
    Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    **IMPORTANT**: Return shape:
        (single row for specific m): if x.shape == [d] and m is not None, then
            return value is float (or equivalent)
        (single row for all m): if x.shape == [d] and m is None, then
            return shape is [M]
        (vectorized for all M) if x.shape == [T, d] and m is None, then
            return shape is [M, T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.

    Parameters
    ----------
    x : data, could either be a single vector of dimension d or a pack of T vectors
        which makes it of dimension [T, d]
    myTheta : parameters of class theta that we defined
    m : index of Gaussian, if m = None, you should handle it for all m in M

    Returns
    -------
    log_bmx : log probability of d-dimensional vector x (See equation 1 of the handout)
    """
    d = x.shape[-1]
    # Single Row for specific m (log_bmx in [1])
    if len(x.shape) == 1 and m is not None:
        log_bmx = 0
    # Single Row for all m (log_bmx in [M])
    elif len(x.shape) == 1:
        M = myTheta.mu.shape[0]
        log_bmx = np.zeros(M)
        for i in range(M):
            log_bmx[i] = log_b_m_x(x, myTheta, i)
    # Vectorized (log_bmx in [M, T])
    else:
        M = myTheta.mu.shape[0]
        T = x.shape[0]
        log_bmx = np.zeros((T, M))
        for t in range(T):
            log_bmx[t] = log_b_m_x(x[t], myTheta)
        log_bmx = log_bmx.T

    return log_bmx

def log_p_m_x(log_Bs, myTheta):
    """
    Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: Log_Bs should be the vectorized version of log_bmx above

    Parameters
    ----------
    log_Bs : log probability of d-dimensional vector x (See equation 1 of the handout)
    myTheta : parameters of class theta that we defined

    Returns
    -------
    log_Ps : the matrix of log probabilities i.e. log of p(m|X;theta)
    """
    print("TODO")


def logLik(log_Bs, myTheta):
    """
    Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

    X can be training data, when used in train( ... ), and
    X can be testing data, when used in test( ... ).

    We don't actually pass X directly to the function because we instead pass:

    log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

    See equation 3 of the handout

    Parameters
    ----------
    log_Bs : log probability of d-dimensional vector x (See equation 1 of the handout)
    myTheta : parameters of class theta that we defined

    Returns
    -------
    log_Lik : the log likelihood (See equation 3 of the handout)
    """
    print("TODO")


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)
    print("TODO : Initialization")
    # for ex.,
    # myTheta.reset_omega(omegas_with_constraints)
    # myTheta.reset_mu(mu_computed_using_data)
    # myTheta.reset_Sigma(some_appropriate_sigma)

    print("TODO: Rest of training")

    return myTheta


def test(mfcc, correctID, models, k=5):
    """
    Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'

    If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

    e.g.,
               S-5A -9.21034037197
    the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    print("TODO")

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)

    print(f"Accuracy = {accuracy}")
