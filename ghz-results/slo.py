def get_slo(method, tight=False, all_methods=False):
    if 'motivation' in method:
        slo = 100
        return slo
    slo_buffer = 20
    min_lat = {
        "compose": 20,
        "user-timeline": 15,
        "home-timeline": 15,
        "S_149998854": 110,
        "S_102000854": 40,
        "S_161142529": 65,
        "hotels-http": 8,
        "reservation-http": 9,
        "user-http": 5,
        "recommendations-http": 6,
        # "motivation-aqm": 100,
        # "motivation-rl": 100,
        "all-methods-social": 20,
        "all-methods-hotel": 9,
    }
    tail95_lat = {
        "compose": 30, # under 4k req/s
        "user-timeline": 20, # under 6k req/s
        "home-timeline": 20, # under 6k req/s
        "S_149998854": 140,
        "S_102000854": 50,
        "S_161142529": 80,
        "hotels-http": 17, # under 8k req/s
        "reservation-http": 12, # under 8k req/s 
        "user-http": 7, # under 8k req/s
        "recommendations-http": 10, # under 8k req/s 
        # "motivation-aqm": 100,
        # "motivation-rl": None,
        "all-methods-social": 30,
        "all-methods-hotel": 17,
    }
    upper_bound = 250
    if all_methods:
        method = "all-methods-hotel" if "http" in method else "all-methods-social"
    slo = tail95_lat.get(method, None) * 4 if not tight else min_lat.get(method, None) + slo_buffer
    if slo > upper_bound:
        slo = upper_bound
    return slo