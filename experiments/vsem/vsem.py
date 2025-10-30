#
# vsem.py
# Implements the Very Simple Ecosystem (VSEM) model in Python, based on the
# implementation provided in the R `BayesianTools` package here:
# https://rdrr.io/cran/BayesianTools/man/VSEM.html
#
# Andrew Roberts
#

import numpy as np
import pandas as pd

def get_PAR_driver(n_days, rng=None):
    # Generates a synthetic time series at a daily time step that is supposed to
    # emulate photosynthetically active radiation (PAR) data. This time series is
    # intended for use as the model driver/forcing term of the VSEM ODE.
    # rng is a Numpy Random Generator object.

    if rng is None:
        rng = np.random.default_rng()

    time_steps = np.arange(n_days)
    PAR = 10 * np.abs(np.sin(time_steps/365 * np.pi) + 0.25 * rng.normal(size=n_days))
    return time_steps, PAR

def compute_lai(xv, lar):
    # Computes the leaf area index (LAI) calculation used in the VSEM model.
    # `lar` (leaf area ratio) is a constant and `xv`, the aboveground vegetation
    # pool, can be a vector.
    return lar * xv

def compute_nee(xv, PAR, gamma, lue, k, lar):
    # Computes the net ecosystem exchange (NEE) calculation used in the
    # VSEM model. This is vectorized so that the model driver `PAR` and
    # above ground vegetation `xv` can be vectors of equal length.
    return (1-gamma) * lue * PAR * (1 - np.exp(-k * compute_lai(xv, lar)))

def vsem_rhs(t, x, par, w):
    # Implements the "righthand side" of the VSEM ODE system; that is, the equations
    # describing the vector field. The interface of this function follows the
    # requirements of scipy.integrate.solve_ivp(), allowing for the opportunity
    # to use scipy's ODE integrators in place of the simple forward Euler scheme
    # with a daily timestep used by `solve_vsem()`. The state vector `x` is
    # 3-dimensional with elements ordered as: "vegetation pool", "soil pool",
    # and "root pool". The parameter `par` is 11-dimensional, with elements:
    # "light extinction coefficient", "leaf area ratio", "light use efficiency",
    # "autotropic respiration (fraction of GPP)", "aboveground vegetation longevity",
    # "belowground vegetation longevity", "Fraction of NPP allocated to aboveground
    # vegetation". The final 3 parameters are the initial conditions for the three
    # carbon pools, and are not utilized in this function.

    # Unpack current state and parameters
    xv, xs, xr = x
    k, lar, lue, gamma, tauv, taus, taur, av, cv, cs, cr = par

    # Compute NPP.
    npp = compute_nee(xv, w, gamma, lue, k, lar)

    # Compute state derivatives.
    dxv_dt = av*npp - xv/tauv
    dxs_dt = xr/taur + xv/tauv - xs/taus
    dxr_dt = (1-av)*npp - xr/taur

    return np.array([dxv_dt, dxs_dt, dxr_dt])

def solve_vsem(driver, par):
    # Implements a simple forward Euler scheme to provide a numerical approximation
    # of the VSEM ODE. The time step of the algorithm is daily, and the time step
    # of the driving term `driver` must also be daily. The number of time steps
    # (days) is determined by the length of `driver`. The initial condition is
    # taken from the last 3 entries of `par`. This function returns an array
    # of shape (number days, 5). The first 3 columns correspond to the carbon
    # pools (state variables): aboveground vegetation, soil, and belowground
    # vegetation (roots). The fourth column is net ecosystem exchange (NEE) and
    # the fifth is leaf area index (LAI).

    n_time_step = len(driver)
    init_cond = par[-3:]
    model_out = np.empty(shape=(n_time_step, 5))
    model_out[0,:3] = init_cond

    # Simulate trajectories of the three carbon pools.
    for day in range(1, n_time_step):
        x_curr = model_out[day-1,:3]
        model_out[day,:3] = np.add(x_curr, vsem_rhs(day, x_curr, par, driver[day]))

    # Compute additional functions of the pools: NEE and LAI.
    k, lar, lue, gamma, tauv, taus, taur, av, cv, cs, cr = par
    model_out[:,3] = compute_nee(model_out[:,0], driver, gamma, lue, k, lar)
    model_out[:,4] = compute_lai(model_out[:,0], lar)

    return model_out

def get_vsem_par_names():
    # Defines the official parameter names and parameter order for the VSEM
    # parameters.
    return ["KEXT","LAR","LUE","GAMMA","tauV","tauS","tauR","Av","Cv","Cs","Cr"]

def get_vsem_output_names():
    # Defines the official output variables and output variable order for the
    # VSEM model. This includes the three core state variables (carbon pools)
    # plus net ecosystem exchange and leaf area index, both of which are
    # functions of the aboveground vegetation state, as well as model
    # parameters.
    return ["Cv", "Cs", "Cr", "NEE", "LAI"]

def get_vsem_default_pars():
    # A convenience function to return default parameter values for the
    # VSEM model, including the initial condition values.

    # Default priors.
    par_defaults = [["KEXT", 0.5],
                    ["LAR", 1.5],
                    ["LUE", 0.002],
                    ["GAMMA", 0.4],
                    ["tauV", 1440.0],
                    ["tauS", 27370.0],
                    ["tauR", 1440.0],
                    ["Av", 0.5],
                    ["Cv", 3.0],
                    ["Cs", 15.0],
                    ["Cr", 3.0]]
    par_defaults = pd.DataFrame(par_defaults, columns=["par_name", "value"])

    return par_defaults


def get_vsem_default_priors():
    # A convenience function to return default priors for the VSEM model
    # parameters, including the initial conditions. Note that this does not
    # align with the bounds provided in the R BayesianTools function
    # `VSEMgetDefaults()`.

    # Default priors.
    par_priors = [["KEXT", "Uniform", 2e-01, 1.0],
                  ["LAR", "Uniform", 2e-01, 3.0],
                  ["LUE", "Uniform", 5e-04, 4e-03],
                  ["GAMMA", "Uniform", 2e-01, 6e-01],
                  ["tauV", "Uniform", 5e+02, 3e+03],
                  ["tauS", "Uniform", 4e+03, 5e+04],
                  ["tauR", "Uniform", 5e+02, 3e+03],
                  ["Av", "Uniform", 2e-01, 1.0],
                  ["Cv", "Uniform", 0.0, 10.0],
                  ["Cs", "Uniform", 0.0, 30.0],
                  ["Cr", "Uniform", 0.0, 10.0]]
    par_priors = pd.DataFrame(par_priors, columns=["par_name", "dist", "param1", "param2"])

    return par_priors


def fwd_vsem(par_cal, driver, par_cal_idx=None, par_default=None, simplify=True):
    # A convenience function defining a "forward map" for the VSEM model. This is
    # map from the calibration parameter (which includes initial conditions)
    # to the model outputs, which are the simulated trajectories of the 5
    # output variables. Note that the calibration parameters often represent a
    # subset of the set of 11 VSEM parameters. If this is the case, then
    # `par_cal_idx` must be specified, which is an array giving the indices
    # of the calibration parameters within the 11-dimensional full parameter
    # vector. In this case, the values of the calibration parameters will be set
    # to `par_cal`, while the values of the remaining ("fixed") parameters are
    # set to those specified in `par_default`. If `par_default` is None, then
    # it is set to `get_vsem_default_pars()`. Note that `par_default` must
    # be a vector of length 11, containing ALL of the parameters. `par_cal` will
    # replace the values of the calibration parameters within this vector.
    # Finally, note that `fwd_vsem()` is vectorized so that multiple runs can
    # be conducted at different values of the calibration parameters. In this
    # case `par_cal` should be an array of dimension (N_runs, N_cal), where
    # N_cal is the number of calibration parameters. For a single run, an
    # array of shape (1,N_cal) or (N_cal,) is allowed. For now, the same default
    # values for the fixed parameters are used across all runs, but this could
    # change in the future. Note that `get_vsem_par_names()` defines the official
    # ordering for the VSEM parameters. The return value of this function
    # is a Numpy array of shape (N_runs, N_time_step, N_outputs). If
    # there is only 1 run and `simplify` is True, then the output dimension is
    # simplified to `(N_time_step, N_outputs)`.

    # Ensure `par_cal` dimension is consistent with single run or multiple runs.
    n_dim = par_cal.ndim
    if n_dim==1:
        par_cal = par_cal.reshape((1,-1))
    elif n_dim > 2:
        raise ValueError("`par_cal` must be a Numpy array with dimension 1 or 2.")

    # If the set of calibration parameters is a subset of all VSEM parameters,
    # ensure that `par_cal_idx` is provided, and define defaults for the parameters
    # being fixed.
    par_names = get_vsem_par_names()
    n_par_vsem = len(par_names)
    n_par_cal = par_cal.shape[1]
    par_cal_is_subset = True
    if n_par_cal < n_par_vsem:
        assert par_cal_idx is not None
        assert len(par_cal_idx) == n_par_cal
        if par_default is None:
            par_default = get_vsem_default_pars()["value"].values
        else:
            assert len(par_default) == n_par_vsem
    elif n_par_cal > n_par_vsem:
        raise ValueError("Number of calibration parameters exceeds VSEM parameter dimension.")
    else:
        par_cal_is_subset = False

    # Execute forward model evaluations. For now this is done in a loop, but
    # parallelization may be added in the future.
    vsem_output_names = get_vsem_output_names()
    n_time_step = len(driver)
    n_run = par_cal.shape[0]
    n_output = len(vsem_output_names)
    model_output = np.empty((n_run,n_time_step,n_output))

    for i in range(n_run):
        if par_cal_is_subset:
            par_run = par_default.copy()
            par_run[par_cal_idx] = par_cal[i,:]
        else:
            par_run = par_cal[i,:]
        model_output[i,:,:] = solve_vsem(driver, par_run)

    # Optionally simplify dimension for the single-run case.
    if (n_run==1) and simplify:
        model_output = model_output[0,:,:]

    return model_output

def get_vsem_fwd_model(driver, n_par_cal, par_cal_idx=None, par_default=None, simplify=True):
    # A convenience function that returns a function representing the VSEM
    # forward model `fwd_vsem()` but only as an argument of `par_cal`, with the
    # remaining arguments fixed. This is convenient for parameter estimation
    # analyses where the remaining arguments (e.g., the model driver and the
    # fixed parameters) will be fixed throughout the analysis. Also includes
    # a check to ensure the parameter dimension is correct.

    def fwd(par_cal):
        if par_cal.ndim == 1:
            assert len(par_cal) == n_par_cal
        else:
            assert par_cal.shape[1] == n_par_cal
        return fwd_vsem(par_cal, driver, par_cal_idx, par_default, simplify)

    return fwd
