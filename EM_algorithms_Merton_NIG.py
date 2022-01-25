from scipy.optimize import fsolve, minimize, NonlinearConstraint
import numpy as np
import joblib
from scipy.stats import norm
import pandas as pd
from scipy.optimize import fsolve

########################################################
# import rpy2's package module
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects

# # import R's utility package
# utils = rpackages.importr('utils')
#
# # select a mirror for R packages
# utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('GeneralizedHyperbolic')

# # R vector of strings
# from rpy2.robjects.vectors import StrVector
#
# # Selectively install what needs to be install.
# # We are fancy, just because we can.
# names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))

GH = rpackages.importr('GeneralizedHyperbolic')

rlog = robjects.r['log']
########################################################


def inverse_Call_BS(A_t, E_t, L_t, T, sigma):
    r = 0
    ## T in df is given as T-t ##
    d1 = (np.log(A_t/L_t)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
    return A_t * norm.cdf(d1, scale = 1, loc = 0) - L_t * np.exp(-r * T) * norm.cdf(d1 - sigma*np.sqrt(T), scale = 1, loc = 0) - E_t

def solve_inverse_call_BS(data):
    return data.apply(lambda x: fsolve(lambda y: inverse_Call_BS(y, x['E_t'], x['L_t'], x['T'], x['sigma']), x['E_t']-x['L_t'])[0], axis=1)

def loglikelihood_BS(param, data):
    r = 0
    sigma = param[0]; mu = param[1]
    h = 1 / 365
    sum1 = sum(np.log([norm.pdf(x=data['log_inc'][i], scale=sigma*np.sqrt(h), loc=(mu-sigma**2/2.)*h) for i in range(2, data.shape[0])]))
    sum2 = - sum(data['log'][1:])
    sum3 = - sum(np.log([norm.cdf((data['log_a_d'][i]+(r+sigma**2/2.)*data['T'][i])/(sigma*np.sqrt(data['T'][i])), scale = 1, loc = 0) for i in range(2, data.shape[0])]))
    return sum1 + sum2 + sum3

def maximize_loglikelihood_BS(data, prev_param):
    x0 = prev_param
    cons = ({'type': 'ineq', 'fun': lambda param: param[0]-0.00001})
    #return minimize(lambda y: -loglikelihood_BS(y, data), x0, method='COBYLA', constraints=cons, options={'rhobeg': 10, 'maxiter': 1000})
    return minimize(lambda y: -loglikelihood_BS(y, data), x0, method='Nelder-Mead')#, options={'maxiter':200})

def asset_new_Merton(T_pd, E_new, L_new, sigma):
    return fsolve(lambda y: inverse_Call_BS(y, E_new, L_new, T_pd, sigma), E_new-L_new)[0]

def PD_Merton(T_pd, E_new, L_new, mu, sigma):
    # calculating new asset
    A_new = asset_new_Merton(T_pd, E_new, L_new, sigma)
    # calculating probability of default
    PD = norm.cdf(np.log(L_new / A_new), scale=sigma*np.sqrt(T_pd), loc=(mu-sigma**2/2.)*T_pd)*100
    return PD

def EM_algorithm_BS(data, E_new, L_new, T_pd, diff=0.001, iter = 15, initial_val = [1,0]):
    ## scale the time data to be given in years (1 year instead of 365 days) ##
    data['T'] = data['T'] / 366
    ## replace 0 with 0.001 because of computations ##
    data['T'] = data['T'].replace(0, 0.001)
    ## prepare parameter data ##
    sigma = initial_val[0]; mu = initial_val[1]
    data['sigma'] = sigma
    data['mu'] = mu
    df_parameters = pd.DataFrame([[sigma, mu, float('nan')]], columns=['sigma', 'mu', 'loglikelihood'])
    Cond = True
    df_assets_all = pd.DataFrame()
    i = 0
    while Cond:
        data['sigma'] = sigma;
        data['mu'] = mu;
        ## get assets for solving option price formula for given equity, debt, maturity, and sigma and mu (E step) ##
        df_assets = pd.DataFrame(solve_inverse_call_BS(data))
        ## make dataframe where we save asset timeseries data of each step ##
        df_assets_all = pd.concat([df_assets_all, df_assets], axis=1)
        ## prepare data for loglikelihood function ##
        df_assets = prepare_assets_df(df_assets, data)
        data_log_asset = pd.DataFrame(df_assets[['log_inc', 'log', 'log_a_d', 'T']].dropna())
        ## get new parameters that maximize loglikelihood function (M step) ##
        new_param = maximize_loglikelihood_BS(data_log_asset,
                                               [data['sigma'][0], data['mu'][0]])
        sigma = new_param.x[0]; mu = new_param.x[1]
        ## save parameters in a dataframe
        df2 = {'sigma': sigma, 'mu': mu, 'loglikelihood': -new_param.fun}
        df_parameters = df_parameters.append(df2, ignore_index=True)
        i = i + 1
        ## if convergence is reached then stop ##
        if i >= iter or (abs(data['sigma'][0] - sigma) <= diff and abs(data['mu'][0] - mu) <= diff):
            Cond = False
            ## calculating whether the company will default in a year from now ##
            ## calculate new asset of the beginning of that year ##
            A_new = asset_new_Merton(T_pd, E_new, L_new, sigma)
            # calculating probability of default
            pod = PD_Merton(T_pd, E_new, L_new, mu, sigma)
            print(sigma, mu)
            print(A_new)
            print(pod)
            return df_assets_all, df_parameters, A_new, pod

def inverse_Call_NIG(A_t, E_t, L_t, T, alpha, beta, delta, mu):
    r = 0
    if not (alpha >= 0.5 and abs(mu) <= delta * np.sqrt(2 * alpha - 1) and A_t>0):
        return None
    else:
        theta = - beta - 0.5 - (mu-r)/(2*delta)*np.sqrt((4*delta**2*alpha**2)/((mu-r)**2+delta**2)-1)
        if not (theta >= - alpha - beta and theta <= alpha - beta - 1):
            return None
        else:
            nig1 = 1.0 - GH.pnig(rlog(float(L_t/A_t[0]))[0], alpha=float(alpha), beta=float(beta+ theta + 1), delta=float(delta*T), mu=float(mu*T))[0]
            nig2 = 1.0 - GH.pnig(rlog(float(L_t/A_t[0]))[0], alpha=float(alpha), beta=float(beta+ theta), delta=float(delta*T), mu=float(mu*T))[0]
            return A_t * nig1 - L_t * np.exp(-r*T) * nig2 - E_t

def solve_inverse_call_NIG(data):
    return data.apply(lambda x: fsolve(lambda y: inverse_Call_NIG(y, x['E_t'], x['L_t'], x['T'], x['alpha'], x['beta'], x['delta'], x['mu']), x['E_t']+x['L_t'])[0], axis=1)

def loglikelihood_NIG(param, data):
    r=0
    alpha = param[0]; beta = param[1]; delta = param[2]; mu = param[3]
    theta = - beta - 0.5 - (mu - r) / (2 * delta) * np.sqrt((4 * delta ** 2 * alpha ** 2) / ((mu - r) ** 2 + delta ** 2) - 1)
    h = 1/365
    sum1 = sum(np.log([GH.dnig(float(data['log_inc'][i]), alpha=float(alpha), beta=float(beta), delta=float(delta*h), mu=float(mu*h))[0] for i in range(2, data.shape[0])]))
    sum2 = - sum(data['log'][1:])
    sum3 = - sum(np.log([1.0 - GH.pnig(float(data['log_d_a'][i]), alpha=float(alpha), beta=float(beta+theta+1), delta=float(delta*data['T'][i]), mu=float(mu*data['T'][i]))[0] for i in range(2, data.shape[0])]))
    return sum1+sum2+sum3

def maximize_loglikelihood_NIG(data, prev_param):
    x0 = prev_param
    ## constraints are from the paper Hubalek; Carlo (2006) ##

    cons = ({'type': 'ineq', 'fun': lambda param: param[0] - 0.50000001},     ## alpha > 1/2 ##
            {'type': 'ineq', 'fun': lambda param: abs(param[1]) - 0.000001},   ## |beta| > 0  ##
            {'type': 'ineq', 'fun': lambda param: abs(param[1]) - param[0]},  ## constraint on beta: |beta| < alpha ##
            {'type': 'ineq', 'fun': lambda param: abs(param[3]) - param[2] * np.sqrt(2 * param[0] - 1)},   ## 1st constr on mu |mu| < delta * sqrt(2 alpha -1) - Eq. (93) ##
            {'type': 'ineq', 'fun': lambda param: -(- 0.5 - param[3] / (2 * param[2]) * np.sqrt(     ## right constr on theta based on Eq. (94) and constraint Eq. (96) ##
        (4 * param[2] ** 2 * param[0] ** 2) / (param[3] ** 2 + param[2] ** 2) - 1)) - param[0]},
            {'type': 'ineq', 'fun': lambda param: - 0.5 - param[3] / (2 * param[2]) * np.sqrt(       ## left constr on theta based on Eq. (94) and constraint Eq. (96) ##
        (4 * param[2] ** 2 * param[0] ** 2) / (param[3] ** 2 + param[2] ** 2) - 1) - param[0] + 1})

    #return minimize(lambda param: -loglikelihood_NIG(param, data), x0,  method='COBYLA', constraints=cons, options={'rhobeg': 5, 'maxiter':1000})
    return minimize(lambda param: -loglikelihood_NIG(param, data), x0, method='Nelder-Mead', bounds=((1.0, 500),(-500, 500), (0.0001, 500), (-500, 500)))#, options={'maxiter': 200, 'disp': True, 'return_all':True})

def prepare_assets_df(df_assets, data):
    df_assets.columns = ['A_t']
    df_assets['log'] = np.log(df_assets['A_t'])
    df_assets['AssetPrev'] = df_assets['A_t'].shift()
    df_assets['increment'] = df_assets['A_t'] / df_assets['AssetPrev']
    ## log of the increment ##
    df_assets['log_inc'] = np.log(df_assets['increment'])
    ## log D_t/A_t ##
    df_assets['log_d_a'] = np.log(data['L_t'] / df_assets['A_t'])
    df_assets['log_a_d'] = np.log(df_assets['A_t']/data['L_t'])
    df_assets['T'] = data['T']
    return df_assets

def asset_new_NIG(T_pd, E_new, L_new, alpha, beta, delta, mu):
    return fsolve(lambda y: inverse_Call_NIG(y, E_new, L_new, T_pd, alpha, beta, delta, mu), E_new + L_new)[0]

def PD_NIG(T_pd, E_new, L_new, alpha, beta, delta, mu):
    # calculating new asset
    A_new = asset_new_NIG(T_pd, E_new, L_new, alpha, beta, delta, mu)
    # calculating probability of default
    PD = GH.pnig(rlog(float(L_new / A_new)), alpha=float(alpha), beta=float(beta), delta=float(delta * T_pd), mu=float(mu * T_pd))[0] * 100
    return PD

def EM_algorithm_NIG(data, E_new, L_new, T_pd = 1.0, diff = 0.001, iter = 15, initial_val=[1,0,1,0]):
    ## scale the time data to be given in years (1 year instead of 365 days) ##
    data['T'] = data['T'] / 366
    ## replace 0 with 0.001 because of computations ##
    data['T'] = data['T'].replace(0, 0.001)
    ## prepare parameter data ##
    alpha = initial_val[0]; beta = initial_val[1]; delta = initial_val[2]; mu = initial_val[3]
    print('initial params are: ', alpha, beta, delta, mu)
    data['alpha'] = alpha; data['beta'] = beta; data['delta'] = delta; data['mu'] = mu;
    df_parameters = pd.DataFrame([[alpha, beta, delta, mu, float('nan')]], columns=['alpha', 'beta', 'delta', 'mu', 'loglikelihood'])
    Cond = True
    df_assets_all = pd.DataFrame()
    i = 0
    while Cond:
        data['alpha'] = alpha; data['beta'] = beta; data['delta'] = delta; data['mu'] = mu;
        ## get assets for solving option price formula for given equity, debt, maturity, and current alpha, beta, delta and mu (E step)
        df_assets = pd.DataFrame(solve_inverse_call_NIG(data))
        ## make dataframe where we save asset timeseries data of each step
        df_assets_all = pd.concat([df_assets_all, df_assets], axis=1)
        ## prepare data for loglikelihood function
        df_assets = prepare_assets_df(df_assets, data)
        data_log_asset = pd.DataFrame(df_assets[['log_inc', 'log', 'log_d_a', 'T']].dropna())
        ## get new parameters that maximize loglikelihood function (M step)
        new_param = maximize_loglikelihood_NIG(data_log_asset, [data['alpha'][0], data['beta'][0], data['delta'][0], data['mu'][0]])
        alpha = new_param.x[0]; beta = new_param.x[1]; delta = new_param.x[2]; mu = new_param.x[3]
        ## save parameters in a dataframe
        df2 = {'alpha': alpha, 'beta': beta, 'delta': delta, 'mu': mu, 'loglikelihood': -new_param.fun}
        df_parameters = df_parameters.append(df2, ignore_index=True)
        i = i + 1
        ## if convergence is reached or then stop
        if i>= iter or (abs(data['alpha'][0]-alpha) <= diff and abs(data['beta'][0]-beta) <= diff and abs(data['delta'][0]-delta) <= diff and abs(data['mu'][0]-mu) <= diff):
            Cond = False
            # calculating new asset
            A_new = asset_new_NIG(T_pd, E_new, L_new, alpha, beta, delta, mu)
            # calculating probability of default
            pod = PD_NIG(T_pd, E_new, L_new, alpha, beta, delta, mu)
            print(alpha, beta, delta, mu)
            print(A_new)
            print(pod)
            return df_assets_all, df_parameters, A_new, pod