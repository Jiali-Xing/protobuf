# This file contains the SLOs for each method of the services.
# And the sustainable request load for each method.


def get_sustainable_load(method):
    # The sustainable load is the maximum request rate that the service can handle without violating the SLO.
    # The sustainable load is determined by the stabliity of the throughput. 
    # Results here are measured under 8 nodes.  
    sustainable_load = {
        # "compose": 4000,
        # "user-timeline": 6000,
        # "home-timeline": 6000,
        "S_149998854": 3000,
        "S_102000854": 4000,
        "S_161142529": 4000,
        # "hotels-http": 8000,
        # "reservation-http": 8000,
        # "user-http": 8000,
        # "recommendations-http": 8000,
        # "motivation-aqm": 8000,
        # "motivation-rl": 8000,
        # "all-methods-social": 8000,
        # "all-methods-hotel": 8000,
    }
    return sustainable_load.get(method, None)

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
    ave_lat = {
        # "compose": 25,
        # "user-timeline": 20,
        # "home-timeline": 20,
        "S_102000854": 45,
        "S_149998854": 120,
        "S_161142529": 70,
        # "hotels-http": 10,
        # "reservation-http": 10,
        # "user-http": 6,
        # "recommendations-http": 8,
        # "all-methods-social": 25,
        # "all-methods-hotel": 10,
    }
    tail95_lat = {
        "compose": 30, # under 4k req/s
        "user-timeline": 20, # under 6k req/s
        "home-timeline": 20, # under 6k req/s
        "S_102000854": 43,
        "S_149998854": 116,
        "S_161142529": 71,
        "hotels-http": 17, # under 8k req/s
        "reservation-http": 12, # under 8k req/s 
        "user-http": 7, # under 8k req/s
        "recommendations-http": 10, # under 8k req/s 
        # "motivation-aqm": 100,
        # "motivation-rl": None,
        "all-methods-social": 30,
        "all-methods-hotel": 17,
    }
    # upper_bound = 250
    # upper_bound_buffer = 100
    if all_methods:
        method = "all-methods-hotel" if "http" in method else "all-methods-social"
    # slo = tail95_lat.get(method, None) * 4 if not tight else min_lat.get(method, None) + slo_buffer
    slo = tail95_lat.get(method, None) * 5 if not tight else min_lat.get(method, None) + slo_buffer
    # if slo > tail95_lat.get(method, None) + upper_bound_buffer:
    #     slo = tail95_lat.get(method, None) + upper_bound_buffer
    return slo