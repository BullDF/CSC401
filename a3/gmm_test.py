from scipy.stats import multivariate_normal
import scipy
import numpy as np
from a3_gmm import theta, log_b_m_x, log_p_m_x

def test_log_b_m_x():
    # Define parameters
    M = 2
    d = 3
    T = 4
    X = np.random.rand(T, d)
    myTheta = theta("Test", M, d)
    myTheta.reset_omega(np.array([1/M] * M))
    myTheta.reset_mu(np.random.rand(M, d))
    myTheta.reset_Sigma(abs(np.random.rand(M, d)))

    # Compute log-likelihoods using your function
    log_Bs_your_function = log_b_m_x(X, myTheta)

    # Compute log-likelihoods directly using scipy
    log_Bs_scipy = np.zeros((M, T))
    for m in range(M):
        rv = multivariate_normal(mean=myTheta.mu[m], cov=myTheta.Sigma[m])
        log_Bs_scipy[m] = rv.logpdf(X)

    # Check if the two results are close
    # assert False, (log_Bs_your_function, log_Bs_scipy)
    assert np.allclose(log_Bs_your_function, log_Bs_scipy), "The two results are not close."

def test_expectation_step():
    # Define parameters
    M = 2
    d = 3
    T = 4
    X = np.random.rand(T, d)
    myTheta = theta("Test", M, d)
    myTheta.reset_omega(np.array([1/M] * M))
    myTheta.reset_mu(np.random.rand(M, d))
    myTheta.reset_Sigma(abs(np.random.rand(M, d)))

    # Compute posterior probabilities using your function
    log_Bs_your_function = log_b_m_x(X, myTheta)
    log_posteriors_your_function = log_p_m_x(log_Bs_your_function, myTheta)
    posteriors_your_function = np.exp(log_posteriors_your_function).T

    # Compute posterior probabilities directly using scipy
    log_likelihoods = np.zeros((T, M))
    for m in range(M):
        rv = multivariate_normal(mean=myTheta.mu[m], cov=myTheta.Sigma[m])
        log_likelihoods[:, m] = np.log(myTheta.omega[m]) + rv.logpdf(X)
    log_posteriors_scipy = log_likelihoods - scipy.special.logsumexp(log_likelihoods, axis=1, keepdims=True)
    posteriors_scipy = np.exp(log_posteriors_scipy)

    # Check if the two results are close
    assert np.allclose(posteriors_your_function, posteriors_scipy), f"The two results are not close. ({posteriors_your_function}, {posteriors_scipy})"

# Run the tests
test_log_b_m_x()
test_expectation_step()