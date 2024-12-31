from arbitragelab.cointegration_approach import CointegrationSimulation

price_params = {
    "ar_coeff": 0.95,
    "white_noise_var": 0.5,
    "constant_trend":0.5
}

coint_params = {
    "ar_coeff": 0.2,
    "white_noise_var": 1,
    "constant_trend": 0,
    "beta": - 0.6
}

coint_simulator = CointegrationSimulation(20,250)

coint_simulator.load_params(price_params, target = 'price')
coint_simulator.load_params(coint_params, target='coint')

s1,s2,coint_errors = coint_simulator.simulate_coint(initial_price = 100, use_statsmodels = True)

plot = coint_simulator.plot_coint_series(s1[:,:], s2[:,:], coint_errors[:,:])