
        if 'S_16' in method:
            pbounds_rajomon['latency_threshold'] = (100, 30000)
        if 'S_14' in method:
            # pbounds_breakwater['breakwater_a'] = (0, 30)
            pbounds_rajomon['latency_threshold'] = (100, 60000)
            pbounds_rajomon['price_update_rate'] = (100, 40000)
            pbounds_rajomon['token_update_rate'] = (1000, 80000)
            pbounds_rajomon['price_step'] = (10, 300)

        if 'S_10' in method:
            pbounds_breakwater['breakwater_b'] = (0, 5)
            pbounds_breakwater['breakwater_slo'] = (100, 10000)