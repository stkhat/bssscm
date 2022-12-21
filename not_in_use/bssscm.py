"""
Note:
- Variables following an inverse-gamma distribution are sampled via 1/np.random.gamma (for Numba compatibility)
  Then the scale parameter of a inverse gamma is specified as 1/scale.
"""

import numpy as np
from tqdm import tqdm
from hmc_tmg import generate_general_tmg
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from numpy import linalg
import sys
from random import gauss
from numpy.linalg import cholesky
from numba import jit
from numba import types
from numba.typed import Dict
import matplotlib.ticker as ptick

TOL = 1e-11

# plt.style.use('default')
plt.rcParams['font.size'] = 12


# @jit(nopython=False, cache=True)
# def generate_general_tmg(Fc, gc, M, mean_r, initial, samples=1, cov=True):
#     """
#     Generates samples of truncated Gaussian distributed random vectors with general covariance matrix under
#     constraint
#     Fc * x + g >= 0.
#     Random vector length will be equal to the mean vector length, specified as a parameter.
#     Example usage - generation of non-negative truncated normal random vectors of size 5, with identity
#     covariance matrix:
#         >> import numpy as np
#         >> size = 5
#         >> mean = [0.1] * size
#         >> cov_mtx = np.identity(size)
#         >> Fc = np.identity(size)
#         >> g = np.zeros((size,1))
#         >> initial = np.ones((size,1))
#         >> print(HMCTruncGaussian().generate_general_tmg(Fc, g, cov_mtx, mean, initial))
#         [[1.5393077420852723, 0.83193549862758009, 0.17057082476061466, 0.35605405861148831, 0.54828265215645966]]
#     :param Fc: constraint matrix
#     :param g: constraint vector
#     :param mean: mean vector of distribution (note: this is the mean after truncation of a normal distribution)
#     :param cov_mtx: covariance matrix of distribution
#     :param initial: initial/starting point
#     :param samples: number of samples to output (default=1).
#     :return: list of samples
#     """

#     dim_cond = gc.shape[0]

#     # try:
#     R = cholesky(M)
#     # except LinAlgError:
#     #     raise ValueError("Error: covariance or precision matrix is not positive definite")

#     mu = np.matrix(mean_r)
#     if mu.shape[1] != 1:
#         mu = mu.transpose()

#     g = np.matrix(gc) + np.matrix(Fc)*mu
#     F = np.matrix(Fc)*R.transpose()
#     initial_sample = linalg.solve(R.transpose(), initial - mu)

#     dim = len(mu)     # dimension of mean vector; each sample must be of this dimension

#     # define all vectors in column order; may change to list for output
#     sample_matrix = []

#     # # more for debugging purposes
#     # if np.any(F*initial_sample + g < 0):
#     #     print("Error: inconsistent initial condition")
#     #     return

#     # count total number of times boundary has been touched
#     bounce_count = 0

#     # squared Euclidean norm of constraint matrix columns
#     Fsq = np.sum(np.square(F), axis=1)
#     Ft = F.transpose()
#     # generate samples
#     for i in range(samples):
#         # print("General HMC")
#         stop = False
#         j = -1
#         # use gauss because it's faster
#         initial_velocity = np.matrix([gauss(0, 1) for _ in range(dim)]).transpose()
#         previous = initial_sample.__copy__()

#         x = previous.__copy__()
#         T = np.pi/2
#         tt = 0

#         while True:
#             a = np.real(initial_velocity.__copy__())
#             b = x.__copy__()

#             fa = F*a
#             fb = F*b

#             u = np.sqrt(np.square(fa) + np.square(fb))
#             # has to be arctan2 not arctan
#             phi = np.arctan2(-fa, fb)

#             # find the locations where the constraints were hit
#             pn = np.abs(np.divide(g, u))
#             t1 = sys.maxsize*np.ones((dim_cond, 1))

#             collision = False
#             inds = [-1] * dim_cond
#             for k in range(dim_cond):
#                 if pn[k] <= 1:
#                     collision = True
#                     pn[k] = 1
#                     # compute the time the coordinates hit the constraint wall
#                     t1[k] = -1*phi[k] + np.arccos(np.divide(-1*g[k], u[k]))
#                     inds[k] = k
#                 else:
#                     pn[k] = 0

#             if collision:
#                 # if there was a previous reflection (j > -1)
#                 # and there is a potential reflection at the sample plane
#                 # make sure that a new reflection at j is not found because of numerical error
#                 if j > -1:
#                     if pn[j] == 1:
#                         cum_sum_pn = np.cumsum(pn).tolist()
#                         temp = cum_sum_pn[0]

#                         index_j = int(temp[j])-1
#                         tt1 = t1[index_j]

#                         if np.abs(tt1) < TOL or np.abs(tt1 - 2*np.pi) < TOL:
#                             t1[index_j] = sys.maxsize

#                 mt = np.min(t1)

#                 # update j
#                 j = inds[int(np.argmin(t1))]
#             else:
#                 mt = T

#             # update travel time
#             tt += mt

#             if tt >= T:
#                 mt -= tt - T
#                 stop = True

#             # print(a)
#             # update position and velocity
#             x = a*np.sin(mt) + b*np.cos(mt)
#             v = a*np.cos(mt) - b*np.sin(mt)

#             if stop:
#                 break

#             # update new velocity
#             reflected = F[j,:]*v/Fsq[j,0]
#             initial_velocity = v - 2*reflected[0,0]*Ft[:,j]

#             bounce_count += 1

#         # need to transform back to unwhitened frame
#         if cov:
#             sample = R.transpose()*x + mu
#         else:
#             sample = linalg.solve(R, x) + mu

#         sample = sample.transpose().tolist()
#         sample_matrix.append(sample[0])

#     return sample_matrix

@jit(nopython=True, cache=True)
def sample_v(V_s, Z_tilde_vec, theta_s, s_eta_s, s_xi_s, T):
    J = Z_tilde_vec.shape[0]

    s_pos_mcmc = (s_eta_s**-2 + 2*s_xi_s**-2)**(-1/2)  #size=2 for Xi
    for t in range(1,T-1):
        zd = Z_tilde_vec[:,t+1]-Z_tilde_vec[:,t]
        vd = V_s[:,t+1] + V_s[:,t-1]
        mu = s_pos_mcmc**2 * (zd/s_eta_s**2 + vd/s_xi_s**2)
        # v = np.random.normal(loc=mu, scale=s_pos_mcmc)
        # V_s[:,t] = v
        for j in range(J):
            v = np.random.normal(loc=mu[j], scale=s_pos_mcmc)
            V_s[j,t] = v

    s_pos_mcmc = (s_eta_s**-2 + s_xi_s**-2)**(-1/2) #size=1 for Xi
    t=0
    zd = Z_tilde_vec[:,t+1]-Z_tilde_vec[:,t]
    vd = V_s[:,t+1]-theta_s
    mu = s_pos_mcmc**2 * (zd/s_eta_s**2 + vd/s_xi_s**2)
    # v = np.random.normal(loc=mu, scale=s_pos_mcmc)
    # V_s[:,t] = v
    for j in range(J):
        v = np.random.normal(loc=mu[j], scale=s_pos_mcmc)
        V_s[j,t] = v

    t=T-1
    zd = Z_tilde_vec[:,t+1]-Z_tilde_vec[:,t]
    vd = V_s[:,t-1] + theta_s
    mu = s_pos_mcmc**2 * (zd/s_eta_s**2 + vd/s_xi_s**2)
    # v = np.random.normal(loc=mu, scale=s_pos_mcmc)
    # V_s[:,t] = v
    for j in range(J):
        v = np.random.normal(loc=mu[j], scale=s_pos_mcmc)
        V_s[j,t] = v


    return V_s

@jit(nopython=True, cache=True)
def sample_theta(V_s, s_xi_s):
    v_vec_diff = V_s[:,1:-1]-V_s[:,:-2]
    mean = v_vec_diff.mean()
    scale = s_xi_s/np.sqrt(v_vec_diff.size)
    theta_s = np.random.normal(loc=mean, scale=scale)
    return theta_s

@jit(nopython=True, cache=True)
def sample_s_xi(V_s, theta_s):
    a_xi, b_xi = 1, 10**-15 # ここ色々設定変えて試してみる

    v_vec_diff = V_s[:,1:-1]-V_s[:,:-2]
    shape_pos = a_xi+(v_vec_diff.size)/2
    rate_pos = b_xi + 1/2*((v_vec_diff-theta_s)**2).sum()
    s_xi_sq_mcmc = 1/np.random.gamma(shape=shape_pos,scale=1/rate_pos)
    s_xi_s = np.sqrt(s_xi_sq_mcmc)

    return s_xi_s

@jit(nopython=True, cache=True)
def sample_s_eta(Z_tilde_vec, V_s):
    a_eta, b_eta = 10**-5, 10**-15
    eta_diff = Z_tilde_vec[:,1:-1] - Z_tilde_vec[:,:-2] - V_s[:,:-2]
    shape_pos = a_eta+(eta_diff.size)/2
    rate_pos = b_eta + 1/2*(eta_diff**2).sum()
    s_eta_sq_mcmc = 1/np.random.gamma(shape=shape_pos,scale=1/rate_pos)
    s_eta_s = np.sqrt(s_eta_sq_mcmc)

    return s_eta_s

@jit(nopython=True, cache=True)
def sample_s_e(Z, Z_tilde_vec):
    a_e, b_e = 1, 10**-15
    diff = Z[:,1:] - Z_tilde_vec[:,1:]
    shape_pos = a_e+(diff.size)/2
    rate_pos = b_e + 1/2*(diff**2).sum()
    s_e_sq_mcmc = 1/np.random.gamma(shape=shape_pos,scale=1/rate_pos)
    s_e_s = np.sqrt(s_e_sq_mcmc)
    # tmp.append(s_e_s)
    return s_e_s

@jit(nopython=True, cache=True)
def sample_sigma_sq(Z_tilde, beta, lam_sq, tau_sq, Y, J, T, do_horsehoe, a_sig=10**-15, b_sig=10**-15):
    # D = np.diag(1/lam_sq) / tau_sq
    shape = a_sig + T/2 #+ do_horsehoe*J/2 
    scale = b_sig + 0.5*np.sum(((Y-Z_tilde.T.dot(beta))[1:])**2) #+ do_horsehoe*0.5*beta.dot(D).dot(beta)
    sigma_sq = 1/np.random.gamma(shape=shape,scale=1/scale,size=1)
    return sigma_sq

@jit(nopython=True, cache=True)
def sample_beta(Z_tilde, lam_sq, tau_sq, sigma_sq, Y, X0_V_X0t, x1_V_X0t, w, J, do_horseshoe):
    ZZt = Z_tilde[:,1:].dot(Z_tilde[:,1:].T)
    D = np.diag(1/lam_sq) / tau_sq * sigma_sq
    A = ZZt + w*X0_V_X0t + do_horseshoe*D
    A_inv = np.linalg.inv(A)
    mean = A_inv.dot(Z_tilde[:,1:].dot(Y[1:]) + w*x1_V_X0t)
    Cov = sigma_sq * A_inv

    # beta = np.random.multivariate_normal(mean, Cov)

    # For sampling in Numba No-Python Mode
    R = np.linalg.cholesky(Cov)
    b = np.empty((J), dtype=np.float_)
    for j in range(J):
        b[j] = np.random.normal()
    beta = R.dot(b)+mean

    return beta

# @jit(nopython=False, cache=True)
# def sample_beta_ehmc(Z_tilde, lam_sq, tau_sq, sigma_sq, Y, X0_V_X0t, x1_V_X0t, w, Jc, Fc, gc, b, initial, do_horseshoe):
#     ZZt = Z_tilde[:,1:].dot(Z_tilde[:,1:].T)
#     D = np.diag(1/lam_sq) / tau_sq
#     A = ZZt + w*X0_V_X0t + do_horseshoe*D
#     A_inv = np.linalg.inv(A)
#     mean = A_inv.dot(Z_tilde[:,1:].dot(Y[1:]) + w*x1_V_X0t)

#     Cov_inv = sigma_sq**-1 * A #np.linalg.inv(V)
#     S = np.linalg.inv(Jc.T.dot(Cov_inv).dot(Jc))
#     mu_c = S.dot(Jc.T).dot(Cov_inv).dot(mean-b)

#     samples = generate_general_tmg(Fc, gc, S, mu_c, initial, samples=1)
#     samples = np.array(samples)
#     beta_tmg = np.hstack([samples, 1-samples.sum(axis=1).reshape(-1,1)])

#     return beta_tmg[0]

@jit(nopython=True, cache=True)
def sample_lam_sq(rho, beta, tau_sq, sigma_sq,J):
    lam_sq = np.empty((J), dtype=np.float_)
    scale = 1/rho + beta**2 / (2*tau_sq)
    for j in range(J):
        lam_sq[j] = 1/np.random.gamma(shape=1,scale=1/scale[j])
    return lam_sq

@jit(nopython=True, cache=True)
def sample_rho(lam_sq,J):
    rho = np.empty((J), dtype=np.float_)
    scale = 1+1/lam_sq
    for j in range(J):
        rho[j] = 1/np.random.gamma(shape=1,scale=1/scale[j])
    return rho

@jit(nopython=True, cache=True)
def sample_tau_sq(psi,beta,lam_sq,sigma_sq,J):
    shape= (J+1)/2
    scale = 1/psi + np.sum(beta**2 / (2 * lam_sq))
    tau_sq = 1/np.random.gamma(shape=shape,scale=1/scale)
    return tau_sq

@jit(nopython=True, cache=True)
def sample_psi(tau_sq):
    scale = 1+1/tau_sq
    psi = 1/np.random.gamma(shape=1,scale=1/scale)
    return psi

@jit(nopython=True, cache=True)
def check():
    n_mcmc=100
    tmp = np.array([0]*n_mcmc, dtype=np.int_)
    print(tmp)


@jit(nopython=False, cache=True)
def run_mcmc(Y, x1, X0, Z, Z_tilde, T, J, K, w, init_theta, init_s_xi, init_s_eta, init_s_e, init_beta, init_sigma_sq, n_mcmc=100, do_horseshoe=False, do_ehmc=False):

    # init
    ## State-space models
    theta_samples = np.empty((n_mcmc), dtype=np.float_)
    s_xi_samples = np.empty((n_mcmc), dtype=np.float_)
    s_eta_samples = np.empty((n_mcmc), dtype=np.float_)
    s_e_samples = np.empty((n_mcmc), dtype=np.float_)
    s_xi_s = init_s_xi
    s_eta_s = init_s_eta
    theta_s = init_theta
    s_e_s = init_s_e
    V_s = np.zeros((J,T+1))
    # V_samples = np.empty([n_mcmc, J, T+1])
    # V_s[:,0] = V[:,0] # TODO: update here

    ## Regression with Horseshoe prior
    beta_samples = np.empty((n_mcmc, J), dtype=np.float_)
    sigma_sq_samples = np.empty((n_mcmc), dtype=np.float_)
    tau_sq_samples = np.empty((n_mcmc), dtype=np.float_)
    lam_sq_0_samples = np.empty((n_mcmc), dtype=np.float_)
    beta_s = init_beta # np.random.multivariate_normal(np.zeros(J), 0.01*np.identity(J))
    lam_sq_s = np.ones(J)
    rho_s = np.ones(J)
    tau_sq_s = 1
    psi_s = 1
    sigma_sq_s = init_sigma_sq

    # EHMC params
    l1=1
    size = J-1
    Fc = np.identity(size)
    Fc = np.concatenate((Fc, -np.ones((1,size))))
    gc = np.zeros((size, 1))
    gc = np.concatenate((gc, l1*np.ones((1,1))))
    Jc = np.identity(size)
    Jc = np.concatenate((Jc, -np.ones((1, size))))
    b = np.zeros(J)
    b[-1]=1
    initial = np.ones((J-1, 1))/(J*5)

    # For covariate balancing
    Cov_X_inv = np.linalg.inv(X0.T.dot(X0)/K)
    X0_V_X0t = X0.dot(Cov_X_inv).dot(X0.T)
    x1_V_X0t = x1.dot(Cov_X_inv).dot(X0.T)


    # check
    ZZt = Z_tilde[:,1:].dot(Z_tilde[:,1:].T)
    A = ZZt + w*X0_V_X0t
    A_inv = np.linalg.inv(A)
    # A_inv.dot(Z_tilde[:,1:].dot(Y[1:]) + w*x1_V_X0t)

    # print(f'reg: {A_inv.dot(Z_tilde[:,1:].dot(Y[1:]))}')
    # print(f'bal: {A_inv.dot(w*x1_V_X0t)}')

    # sanity check
    if Fc.shape[0] != gc.shape[0]:
        raise ValueError("Error: constraint dimensions do not match")

    # MCMC sampling
    for i_mcmc in range(n_mcmc):
        
        # v
        V_s = sample_v(V_s, Z_tilde, theta_s, s_eta_s, s_xi_s, T)

        # theta
        theta_s = sample_theta(V_s, s_xi_s)

        # s_xi
        s_xi_s = sample_s_xi(V_s, theta_s)

        # s_eta
        s_eta_s = sample_s_eta(Z_tilde, V_s)

        # s_e
        s_e_s = sample_s_e(Z, Z_tilde)

        # beta
        if do_horseshoe:
            lam_sq_s = sample_lam_sq(rho_s, beta_s, tau_sq_s, sigma_sq_s, J)
            rho_s = sample_rho(lam_sq_s,J)
            tau_sq_s = sample_tau_sq(psi_s,beta_s,lam_sq_s,sigma_sq_s,J)
            psi_s = sample_psi(tau_sq_s)
        
        if do_ehmc:
            # beta_s = sample_beta_ehmc(Z_tilde, lam_sq_s, tau_sq_s, sigma_sq_s, Y, X0_V_X0t, x1_V_X0t, w, Jc, Fc, gc, b, initial, do_horseshoe)
            raise ValueError('EHMC not supported')
        else:
            beta_s = sample_beta(Z_tilde, lam_sq_s, tau_sq_s, sigma_sq_s, Y, X0_V_X0t, x1_V_X0t, w, J, do_horseshoe)
            # beta_s = init_beta
    
        # sigma_sq
        sigma_sq_s = sample_sigma_sq(Z_tilde, beta_s, lam_sq_s, tau_sq_s, Y, J, T, do_horseshoe)

        # Save data 
        # V_samples[i_mcmc] = V_s
        theta_samples[i_mcmc] = theta_s
        s_xi_samples[i_mcmc] = s_xi_s
        s_eta_samples[i_mcmc] = s_eta_s
        s_e_samples[i_mcmc] = s_e_s
        beta_samples[i_mcmc] = beta_s
        sigma_sq_samples[i_mcmc] = sigma_sq_s
        tau_sq_samples[i_mcmc] = tau_sq_s
        lam_sq_0_samples[i_mcmc] = lam_sq_s[0]

        if i_mcmc%250==0:
            print(f'{i_mcmc} iterations done.')

    dict_samples = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float64[:,:]
        )

    dict_samples = {'theta': theta_samples,
                    's_xi': s_xi_samples,
                    's_eta': s_eta_samples,
                    's_e': s_e_samples,
                    'sigma_sq': sigma_sq_samples,
                    'tau_sq': tau_sq_samples,
                    'lam_sq_0': lam_sq_0_samples,
                    'beta': beta_samples}

    return dict_samples

def mcmc_dict_to_array(dict_samples, n_mcmc, J):
    n_params = len(dict_samples.keys())
    params_mcmc = np.empty([n_mcmc, n_params+J-1])
    keys=[]
    i=0
    for key in dict_samples.keys():
        if key=='beta':
            params_mcmc[:,i:(i+J)] = dict_samples[key]
            i+=1
            keys+=[f'beta_{j}' for j in range(J)]
        else:
            params_mcmc[:,i] = dict_samples[key]
            i+=1
            keys.append(key)

    return params_mcmc, keys

def plot_posterior(params_mcmc:np.array, keys:list, dict_params_truth:dict, out_fig:str='mcmc_plot.png',n_show:int=4,show_sum:bool=False):

    plt.gca().get_xaxis().set_major_formatter(ptick.ScalarFormatter(useMathText=True))

    n_param = len(dict_params_truth)
    n_show = np.min([n_param, n_show])
    n_rows = n_show+1 if show_sum else n_show
    # keys_params = list(dict_params_truth.keys())

    plt.figure(figsize=(16, 4*n_rows))
    for i in range(n_show):
        key_param = keys[i]
        plt.subplot(n_rows,2,2*i+1)
        plt.hist(params_mcmc[:,i])
        plt.axvline(dict_params_truth[key_param], label='truth', color='red')
        plt.axvline(params_mcmc[:,i].mean(), label='post mean', color='blue')
        plt.title(key_param, fontsize=15)
        plt.legend()
        plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        plt.subplot(n_rows,2,2*i+2)
        plt.plot(params_mcmc[:,i])

    # Only used when params_mcmc includes beta
    if show_sum: 
        i+=1
        plt.subplot(n_rows,2,2*i+1)
        plt.hist(params_mcmc.sum(axis=1))
        plt.axvline(dict_params_truth.sum(), label='truth', color='red')
        plt.axvline(params_mcmc.sum(axis=1).mean(), label='post mean', color='blue')
        plt.title(f'params_sum', fontsize=15)
        plt.legend()
        plt.subplot(n_rows,2,2*i+2)
        plt.plot(params_mcmc.sum(axis=1))
    
    plt.suptitle(f'MCMC plots for params = {dict_params_truth}', fontsize=20, wrap=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_fig)
    # plt.show()
    plt.close()