"""
Base code:
https://github.com/ptuls/HMC_TruncatedGaussian/blob/master/python/hmc_tmg.py
"""

import numpy as np
import numpy.linalg as lin
import sys

from random import gauss
from numpy.linalg import cholesky
EPS = 1e-11

def generate_general_tmg(Fc, gc, M, mean_r, initial, samples=1, cov=True):
    """
    Generates samples of truncated Gaussian distributed random vectors with general covariance matrix under
    constraint
    Fc * x + g >= 0.
    Random vector length will be equal to the mean vector length, specified as a parameter.
    Example usage - generation of non-negative truncated normal random vectors of size 5, with identity
    covariance matrix:
        >> import numpy as np
        >> size = 5
        >> mean = [0.1] * size
        >> cov_mtx = np.identity(size)
        >> Fc = np.identity(size)
        >> g = np.zeros((size,1))
        >> initial = np.ones((size,1))
        >> print(HMCTruncGaussian().generate_general_tmg(Fc, g, cov_mtx, mean, initial))
        [[1.5393077420852723, 0.83193549862758009, 0.17057082476061466, 0.35605405861148831, 0.54828265215645966]]
    :param Fc: constraint matrix
    :param g: constraint vector
    :param mean: mean vector of distribution (note: this is the mean after truncation of a normal distribution)
    :param cov_mtx: covariance matrix of distribution
    :param initial: initial/starting point
    :param samples: number of samples to output (default=1).
    :return: list of samples
    """
    # sanity check
    dim_cond = gc.shape[0]
    if Fc.shape[0] != dim_cond:
        print("Error: constraint dimensions do not match")
        return

    try:
        R = cholesky(M)
    except lin.LinAlgError:
        print("Error: covariance or precision matrix is not positive definite")
        return

    # using covariance matrix
    if cov:
        mu = np.matrix(mean_r)
        if mu.shape[1] != 1:
            mu = mu.transpose()

        g = np.matrix(gc) + np.matrix(Fc)*mu
        F = np.matrix(Fc)*R.transpose()
        initial_sample = lin.solve(R.transpose(), initial - mu)
    # using precision matrix
    else:
        r = np.matrix(mean_r)
        if r.shape[1] != 1:
            r = r.transpose()

        mu = lin.solve(R, lin.solve(R.transpose(), r))
        g = np.matrix(gc) + np.matrix(Fc)*mu
        F = lin.solve(R, np.matrix(Fc))
        initial_sample = initial - mu
        initial_sample = R*initial_sample

    dim = len(mu)     # dimension of mean vector; each sample must be of this dimension

    # define all vectors in column order; may change to list for output
    sample_matrix = []

    # more for debugging purposes
    if any(F*initial_sample + g < 0):
        print("Error: inconsistent initial condition")
        return

    # count total number of times boundary has been touched
    bounce_count = 0

    # squared Euclidean norm of constraint matrix columns
    Fsq = np.sum(np.square(F), axis=1)
    Ft = F.transpose()
    # generate samples
    for i in range(samples):
        # print("General HMC")
        stop = False
        j = -1
        # use gauss because it's faster
        initial_velocity = np.matrix([gauss(0, 1) for _ in range(dim)]).transpose()
        previous = initial_sample.__copy__()

        x = previous.__copy__()
        T = np.pi/2
        tt = 0

        while True:
            a = np.real(initial_velocity.__copy__())
            b = x.__copy__()

            fa = F*a
            fb = F*b

            u = np.sqrt(np.square(fa) + np.square(fb))
            # has to be arctan2 not arctan
            phi = np.arctan2(-fa, fb)

            # find the locations where the constraints were hit
            pn = np.abs(np.divide(g, u))
            t1 = sys.maxsize*np.ones((dim_cond, 1))

            collision = False
            inds = [-1] * dim_cond
            for k in range(dim_cond):
                if pn[k] <= 1:
                    collision = True
                    pn[k] = 1
                    # compute the time the coordinates hit the constraint wall
                    t1[k] = -1*phi[k] + np.arccos(np.divide(-1*g[k], u[k]))
                    inds[k] = k
                else:
                    pn[k] = 0

            if collision:
                # if there was a previous reflection (j > -1)
                # and there is a potential reflection at the sample plane
                # make sure that a new reflection at j is not found because of numerical error
                if j > -1:
                    if pn[j] == 1:
                        cum_sum_pn = np.cumsum(pn).tolist()
                        temp = cum_sum_pn[0]

                        index_j = int(temp[j])-1
                        tt1 = t1[index_j]

                        if np.abs(tt1) < EPS or np.abs(tt1 - 2*np.pi) < EPS:
                            t1[index_j] = sys.maxsize

                mt = np.min(t1)

                # update j
                j = inds[int(np.argmin(t1))]
            else:
                mt = T

            # update travel time
            tt += mt

            if tt >= T:
                mt -= tt - T
                stop = True

            # print(a)
            # update position and velocity
            x = a*np.sin(mt) + b*np.cos(mt)
            v = a*np.cos(mt) - b*np.sin(mt)

            if stop:
                break

            # update new velocity
            reflected = F[j,:]*v/Fsq[j,0]
            initial_velocity = v - 2*reflected[0,0]*Ft[:,j]

            bounce_count += 1

        # need to transform back to unwhitened frame
        if cov:
            sample = R.transpose()*x + mu
        else:
            sample = lin.solve(R, x) + mu

        sample = sample.transpose().tolist()
        sample_matrix.append(sample[0])

    return sample_matrix


# =========================================
# Not in use
# =========================================

# def find_travel_time(a, b, F, g, dim, T, j):
#     # Find
#     fa = F*a
#     fb = F*b

#     u = np.sqrt(np.square(fa) + np.square(fb))
#     # has to be arctan2 not arctan
#     phi = np.arctan2(-fa, fb)

#     # print(a)
#     # find the locations where the constraints were hit
#     pn = np.abs(np.divide(g, u))
#     print(f'pn: {pn}')
#     t1 = sys.maxsize*np.ones((dim, 1))

#     collision = False
#     inds = [-1] * dim
#     for k in range(dim):
#         if pn[k] <= 1:
#             collision = True
#             pn[k] = 1
#             # compute the time the coordinates hit the constraint wall
#             t1[k] = -1*phi[k] + np.arccos(np.divide(-1*g[k], u[k]))
#             inds[k] = k
#         else:
#             pn[k] = 0

#     if collision:
#         # if there was a previous reflection (j > -1)
#         # and there is a potential reflection at the sample plane
#         # make sure that a new reflection at j is not found because of numerical error
#         if j > -1:
#             if pn[j] == 1:
#                 cum_sum_pn = np.cumsum(pn).tolist()
#                 temp = cum_sum_pn[0]

#                 index_j = int(temp[j])-1
#                 tt1 = t1[index_j]

#                 if np.abs(tt1) < EPS or np.abs(tt1 - 2*np.pi) < EPS:
#                     t1[index_j] = sys.maxsize

#         mt = np.min(t1)

#         # update j
#         j = inds[int(np.argmin(t1))]
#     else:
#         mt = T

#     return mt, j


# def generate_general_tmg_multi_cond(Fc_dict, gc_dict, M, mean_r, initial, samples=1, cov=True):
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
#     cond_keys = Fc_dict.keys()
#     J = len(cond_keys)


#     # sanity check

#     if J!=len(gc_dict):
#         print("Error: constraint numbers do not match")
#         return
#     for key in cond_keys:
#         s = gc_dict[key].shape[0]
#         if Fc_dict[key].shape[0] != s:
#             print("Error: constraint dimensions do not match")
#             return

#     try:
#         R = cholesky(M)
#     except lin.LinAlgError:
#         print("Error: covariance or precision matrix is not positive definite")
#         return

#     # using covariance matrix
#     if cov:
#         mu = np.matrix(mean_r)
#         if mu.shape[1] != 1:
#             mu = mu.transpose()

#         initial_sample = lin.solve(R.transpose(), initial - mu)

#         g_list = []
#         F_list = []
#         for key in cond_keys:
#             g = np.matrix(gc_dict[key]) + np.matrix(Fc_dict[key])*mu
#             F = np.matrix(Fc_dict[key])*R.transpose()
#             g_list.append(g)
#             F_list.append(F)

#     # # using precision matrix
#     # else:
#     #     r = np.matrix(mean_r)
#     #     if r.shape[1] != 1:
#     #         r = r.transpose()

#     #     mu = lin.solve(R, lin.solve(R.transpose(), r))
#     #     g = np.matrix(gc) + np.matrix(Fc)*mu
#     #     F = lin.solve(R, np.matrix(Fc))
#     #     initial_sample = initial - mu
#     #     initial_sample = R*initial_sample

#     dim = len(mu)     # dimension of mean vector; each sample must be of this dimension

#     # define all vectors in column order; may change to list for output
#     sample_matrix = []

#     # more for debugging purposes
#     for c in range(J):
#         if any(F_list[c]*initial_sample + g_list[c]) < 0:
#             print("Error: inconsistent initial condition")
#             return

#     # count total number of times boundary has been touched
#     bounce_count = 0

#     # squared Euclidean norm of constraint matrix columns
#     Fsq_list = [np.sum(np.square(F), axis=0) for F in F_list]
#     Ft_list = [F.transpose() for F in F_list]
#     # generate samples
#     for i in range(samples):
#         # print("General HMC")
#         stop = False
#         # j = -1
#         j_list = [-1 for c in range(J)]

#         # use gauss because it's faster
#         initial_velocity = np.matrix([gauss(0, 1) for _ in range(dim)]).transpose()
#         previous = initial_sample.__copy__()

#         x = previous.__copy__()
#         T = np.pi/2
#         tt = 0
#         c_hit = 0

#         while True:
#             a = np.real(initial_velocity.__copy__())
#             b = x.__copy__()

#             mt_list = []
#             for c in range(J):
#                 g = g_list[c]
#                 F = F_list[c]
#                 j = j_list[c]
#                 print(f'len of g: {len(g)}')
#                 mt_tmp, j_tmp = find_travel_time(a, b, F, g, len(g), T, j)
#                 mt_list.append(mt_tmp)
#                 j_list[c] = j_tmp

#             # update travel time
#             c_hit = np.argmin(np.array(mt_list))
#             mt = mt_list[c_hit]
#             j = j_list[c_hit]
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
#             reflected = F_list[c_hit][j,:]*v/Fsq_list[c_hit][0,j]
#             initial_velocity = v - 2*reflected[0,0]*Ft_list[c_hit][:,j]

#             bounce_count += 1

#         # need to transform back to unwhitened frame
#         if cov:
#             sample = R.transpose()*x + mu
#         else:
#             sample = lin.solve(R, x) + mu

#         sample = sample.transpose().tolist()
#         sample_matrix.append(sample[0])

#     return sample_matrix