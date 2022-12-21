import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from hmc_tmg import generate_general_tmg

# plt.style.use('default')
plt.rcParams['font.size'] = 12

def get_post_mu_V(X,Y,sigma):
    M = np.linalg.inv(X.T.dot(X))
    mu = M.dot(X.T).dot(Y)
    V = sigma**2 * M

    return mu, V

def ln_p(beta, mu, V):
    return stats.multivariate_normal.logpdf(beta, mu, V)

def nabla_ln_p(beta, mu, V):
    V_inv = np.linalg.inv(V)
    return -V_inv.dot(beta-mu)

# =========================
# Gibbs sampling
# =========================
def update_beta_gibbs(X, Y, sigma):
    mu, V = get_post_mu_V(X,Y,sigma)
    beta_new = np.random.multivariate_normal(mu, V)
    return beta_new

def run_gibbs_sampler(X, Y, sigma, n_mcmc, n_warmup=2000, seed=1):
    np.random.seed(seed)

    # init array
    beta01_sample = np.empty([n_warmup+n_mcmc, 2])
    beta23_sample = np.empty([n_warmup+n_mcmc, 2])
    b01 = np.random.multivariate_normal(np.zeros(2), np.identity(2))
    b23 = np.random.multivariate_normal(np.zeros(2), np.identity(2))

    X01 = X[:,:2]
    X23 = X[:,2:]
    print('Running MCMC algorithm...')
    for i_mcmc in tqdm(range(n_warmup+n_mcmc)):
        b01 = update_beta_gibbs(X01, (Y-X23.dot(b23)), sigma)
        b23 = update_beta_gibbs(X23, (Y-X01.dot(b01)), sigma)
        beta01_sample[i_mcmc] = b01
        beta23_sample[i_mcmc] = b23
    print('Done.')

    # stack arrays and discard warmup sampels
    beta_mcmc = np.hstack([beta01_sample, beta23_sample])[n_warmup:]
    
    return beta_mcmc

def run_gibbs_sampler_horseshoe(X, Y, sigma, n_mcmc, n_warmup=2000, seed=1, do_horseshoe=True):
    np.random.seed(seed)

    # init
    dim_beta = X.shape[1]
    XtX = X.T.dot(X)

    beta_sample = np.empty([n_warmup+n_mcmc, dim_beta])
    beta = np.random.multivariate_normal(np.zeros(dim_beta), np.identity(dim_beta))
    lam_sq = np.ones(dim_beta)
    nu = np.ones(dim_beta)
    tau_sq = 1
    xi = 1

    def sample_beta(lam_sq, tau_sq, sigma, do_horseshoe):
        D = np.diag(1/lam_sq) / tau_sq
        A = XtX + do_horseshoe*D
        A_inv = np.linalg.inv(A)
        mean = A_inv.dot(X.T).dot(Y)
        cov = sigma**2 * A_inv
        beta = np.random.multivariate_normal(mean,cov)

        return beta
    
    def sample_lam_sq(nu, beta, tau_sq, sigma):
        rate = 1/nu + beta**2 / (2*tau_sq*sigma**2)
        lam_sq = 1/np.random.gamma(shape=1,scale=1/rate,size=dim_beta)
        return lam_sq
    
    def sample_nu(lam_sq):
        rate = 1+1/lam_sq
        nu = 1/np.random.gamma(shape=1,scale=1/rate,size=dim_beta)
        return nu

    def sample_tau_sq(xi,beta,lam_sq,sigma):
        shape= (dim_beta+1)/2
        rate = 1/xi + np.sum(beta**2 / (2 * sigma**2 * lam_sq))
        tau_sq = 1/np.random.gamma(shape=shape,scale=1/rate)
        return tau_sq

    def sample_xi(tau_sq):
        rate = 1+1/tau_sq
        xi = 1/np.random.gamma(shape=1,scale=1/rate)
        return xi

    print('Running MCMC algorithm...')
    for i_mcmc in tqdm(range(n_warmup+n_mcmc)):
        beta = sample_beta(lam_sq, tau_sq, sigma, do_horseshoe)
        if do_horseshoe:
            lam_sq = sample_lam_sq(nu, beta, tau_sq, sigma)
            nu = sample_nu(lam_sq)
            tau_sq = sample_tau_sq(xi,beta,lam_sq,sigma)
            xi = sample_xi(tau_sq)
        beta_sample[i_mcmc] = beta
    print('Done.')

    # Discard warmup sampels
    beta_mcmc = beta_sample[n_warmup:]
    
    return beta_mcmc


# =========================
# Hamiltonian Monte Carlo
# =========================
def run_hmc_sampler(X, Y, sigma, n_mcmc, n_warmup=2000, delta=0.0025, L=200, seed=1):
    np.random.seed(seed)

    # init array
    beta01_sample = np.empty([n_warmup+n_mcmc, 2])
    beta23_sample = np.empty([n_warmup+n_mcmc, 2])
    phi0 = np.zeros(2)
    S0 = np.identity(2)
    b01 = np.random.multivariate_normal(phi0, S0)
    b23 = np.random.multivariate_normal(phi0, S0)

    X01 = X[:,:2]
    X23 = X[:,2:]
    count_accepted = 0
    for i_mcmc in range(n_warmup+n_mcmc):
        # update b01
        b01 = update_beta_gibbs(X01, (Y-X23.dot(b23)), sigma)

        # update b23 
        mu, V = get_post_mu_V(X23, Y-X01.dot(b01), sigma)
        u0 = np.random.multivariate_normal(phi0, S0)
        u23 = u0
        b23_lf = b23
        for l in range(L):
            b23_lf = b23_lf + delta/2*u23
            u23 = u23+delta*nabla_ln_p(b23_lf, mu, V)
            b23_lf = b23_lf + delta/2*u23

        target = ln_p(b23, mu, V) - ln_p(b23_lf, mu, V)
        adjust = ln_p(u0, phi0, S0) - ln_p(u23, phi0, S0)
        log_ratio = target+adjust
        accept_percent = np.min([1, np.exp(log_ratio)])
        ind_accept = np.random.choice([1,0], p=[accept_percent, 1-accept_percent])
        if ind_accept:
            b23=b23_lf
            count_accepted+=1

        beta01_sample[i_mcmc] = b01
        beta23_sample[i_mcmc] = b23

    # stack arrays and discard warmup sampels
    beta_mcmc = np.hstack([beta01_sample, beta23_sample])[n_warmup:]
    acceptance_ratio = count_accepted/(n_mcmc+n_warmup)
    
    return beta_mcmc, acceptance_ratio


def run_ehmc_sampler_horseshoe(X, Y, sigma, n_mcmc, n_warmup=2000, seed=1, do_horseshoe=True):
    np.random.seed(seed)

    # init
    dim_beta = X.shape[1]
    XtX = X.T.dot(X)

    beta_sample = np.empty([n_warmup+n_mcmc, dim_beta])
    beta = np.random.multivariate_normal(np.zeros(dim_beta), 0.1**2 * np.identity(dim_beta))
    lam_sq = np.ones(dim_beta)
    nu = np.ones(dim_beta)
    tau_sq = 1
    xi = 1

    # EHMC setting
    l1=1
    size = dim_beta-1
    Fc = np.identity(size)
    Fc = np.concatenate([Fc, -np.ones((1,size))])
    gc = np.zeros((size, 1))
    gc = np.concatenate([gc, l1*np.ones((1,1))])
    initial = np.ones((size, 1))/(dim_beta*5)
    Jc = np.identity(dim_beta-1)
    Jc = np.concatenate([Jc, -np.ones((1, dim_beta-1))])
    b = np.zeros(dim_beta)
    b[-1]=1


    def sample_beta_hmc(lam_sq, tau_sq, sigma, do_horseshoe):
        D = np.diag(1/lam_sq)/tau_sq
        A = XtX + do_horseshoe*D
        A_inv = np.linalg.inv(A)
        mu = A_inv.dot(X.T).dot(Y)
        V = sigma**2 * A_inv

        V_inv = sigma**-2 * A #np.linalg.inv(V)
        S = np.linalg.inv(Jc.T.dot(V_inv).dot(Jc))
        mu_c = S.dot(Jc.T).dot(V_inv).dot(mu-b)

        samples = generate_general_tmg(Fc, gc, S, mu_c, initial, samples=1)
        samples = np.array(samples)
        beta_tmg = np.hstack([samples, 1-samples.sum(axis=1).reshape(-1,1)])

        return beta_tmg[0]
    
    def sample_lam_sq(nu, beta, tau_sq, sigma):
        rate = 1/nu + beta**2 / (2*tau_sq*sigma**2)
        lam_sq = 1/np.random.gamma(shape=1,scale=1/rate,size=dim_beta)
        return lam_sq
    
    def sample_nu(lam_sq):
        rate = 1+1/lam_sq
        nu = 1/np.random.gamma(shape=1,scale=1/rate,size=dim_beta)
        return nu

    def sample_tau_sq(xi,beta,lam_sq,sigma):
        shape= (dim_beta+1)/2
        rate = 1/xi + np.sum(beta**2 / (2 * sigma**2 * lam_sq))
        tau_sq = 1/np.random.gamma(shape=shape,scale=1/rate)
        return tau_sq

    def sample_xi(tau_sq):
        rate = 1+1/tau_sq
        xi = 1/np.random.gamma(shape=1,scale=1/rate)
        return xi

    print('Running MCMC algorithm...')
    for i_mcmc in tqdm(range(n_warmup+n_mcmc)):
        beta = sample_beta_hmc(lam_sq, tau_sq, sigma, do_horseshoe)
        if do_horseshoe:
            lam_sq = sample_lam_sq(nu, beta, tau_sq, sigma)
            nu = sample_nu(lam_sq)
            tau_sq = sample_tau_sq(xi,beta,lam_sq,sigma)
            xi = sample_xi(tau_sq)
        
        beta_sample[i_mcmc] = beta
    print('Done.')

    # Discard warmup sampels
    beta_mcmc = beta_sample[n_warmup:]
    
    return beta_mcmc


# plot
def plot_posterior(params_mcmc, params_truth, out_fig='mcmc_plot.png',n_show=4,show_sum=False):
    n_param = len(params_truth)
    n_show = np.min([n_param, n_show])
    n_rows = n_show+1 if show_sum else n_show

    plt.figure(figsize=(16, 4*n_rows))
    for i in range(n_show):
        plt.subplot(n_rows,2,2*i+1)
        plt.hist(params_mcmc[:,i])
        plt.axvline(params_truth[i], label='truth', color='red')
        plt.axvline(params_mcmc[:,i].mean(), label='post mean', color='blue')
        plt.title(f'params_{i+1}', fontsize=15)
        plt.legend()
        plt.subplot(n_rows,2,2*i+2)
        plt.plot(params_mcmc[:,i])

    if show_sum:
        i+=1
        plt.subplot(n_rows,2,2*i+1)
        plt.hist(params_mcmc.sum(axis=1))
        plt.axvline(params_truth.sum(), label='truth', color='red')
        plt.axvline(params_mcmc.sum(axis=1).mean(), label='post mean', color='blue')
        plt.title(f'params_sum', fontsize=15)
        plt.legend()
        plt.subplot(n_rows,2,2*i+2)
        plt.plot(params_mcmc.sum(axis=1))
    
    plt.suptitle(f'MCMC plots for params = {params_truth}', fontsize=20)
    plt.savefig(out_fig)
    plt.show()
    plt.close()


if __name__=='__main__':
    beta = np.array([0.6,0.3,0.1]+[0]*7) #np.array([0.6, 0.4, 0, 0])
    dim_beta = len(beta)
    n_sample = 1000
    n_mcmc = 2000
    n_warmup = 500
    sigma=0.2

    np.random.seed(2)
    X = np.random.multivariate_normal(mean=np.zeros(dim_beta), cov=np.identity(dim_beta), size=n_sample)
    Y =  X.dot(beta) + np.random.normal(size=n_sample, scale=sigma)

    # # Gibbs sampler
    # beta_mcmc_gibbs = run_gibbs_sampler(X, Y, sigma, n_mcmc, n_warmup)
    # plot_posterior(beta_mcmc_gibbs, beta, 'figure/mcmc_gibbs.png')

    # # HMC sampler
    # beta_mcmc_hmc, acceptance_ratio = run_hmc_sampler(X, Y, sigma, 1000, 500, delta=0.0015, L=200)
    # print(f'Acceptance ratio: {acceptance_ratio:.3f}')
    # plot_posterior(beta_mcmc_hmc, beta)

    # # Horseshoe prior
    # beta_mcmc_horseshoe = run_gibbs_sampler_horseshoe(X, Y, sigma, n_mcmc, n_warmup,do_horseshoe=True)
    # plot_posterior(beta_mcmc_horseshoe, beta, 'figure/mcmc_horseshoe.png', 6)
 
    # beta_mcmc = run_gibbs_sampler_horseshoe(X, Y, sigma, n_mcmc, n_warmup, do_horseshoe=False)
    # plot_posterior(beta_mcmc, beta, 'figure/mcmc_gibbs.png', 6)

    # Horseshoe prior with EHMC
    beta_mcmc_horseshoe = run_ehmc_sampler_horseshoe(X, Y, sigma, n_mcmc, n_warmup, do_horseshoe=True)
    plot_posterior(beta_mcmc_horseshoe, beta, 'figure/mcmc_ehmc_horseshoe.png', 5, show_sum=True)
 
    beta_mcmc = run_ehmc_sampler_horseshoe(X, Y, sigma, n_mcmc, n_warmup, do_horseshoe=False)
    plot_posterior(beta_mcmc, beta, 'figure/mcmc_ehmc.png', 5, show_sum=True)
