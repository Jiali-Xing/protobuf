# This file is used to load the results of the experiments into a CSV file. Then group the results by setup and save them in a new CSV file.
# The CSV file is used to generate the plots in the paper

from utils import export_all_experiments_to_csv, calculate_group_statistics, extract_single_concurrent_request_to_csv, append_new_experiments_to_csv
from find_k import find_k_files


if __name__ == '__main__':
    '''
    export_all_experiments_to_csv([('0527_0000', '0610_1200')], 'alibaba-experiment_results.csv')
    # line above is for the 4 nodes experiment first time. below is retuned.
   calculate_group_statistics('alibaba-experiment_results.csv', 'grouped_ali.csv')
    '''
    # export_all_experiments_to_csv([('0607_1354', '0609_1200')], '4-nodes-experiment_results.csv')
    # calculate_group_statistics('4-nodes-experiment_results.csv', 'grouped_4n.csv')
    '''
    export_all_experiments_to_csv([('0609_1200', '0612_0000')], '4-nodes-monotonic-experiment_results.csv')
    calculate_group_statistics('4-nodes-monotonic-experiment_results.csv', 'grouped_4n_monotonic.csv')
    '''
 
    '''
    export_all_experiments_to_csv([('0615_1640', '0718_1943')], 'all-alibaba-experiment-s2-skipe.csv')
    aggregate_concurrent_runs('all-alibaba-experiment-s2-skipe.csv', 'all-alibaba-experiment-s2-skipe.csv')
    calculate_group_statistics('all-alibaba-experiment-s2-skipe.csv', 'grouped_all_ali.csv', k=5)

    # above was with spike only on S2. below is with spike on all services but at different times.
    # export_all_experiments_to_csv([('0618_1943', '0710_0100')], 'all-alibaba-experiment_results.csv')
    # aggregate_concurrent_runs('all-alibaba-experiment_results.csv', 'all-alibaba-experiment_results.csv')
    # calculate_group_statistics('all-alibaba-experiment_results.csv', 'grouped_all_ali.csv')

    # export_alibaba_experiments_to_csv([('0618_1943', '0624_0100')], 'alibaba-experiment_results.csv')

    # # export_all_experiments_to_csv([('0621_0401', '0623_1200')], 'hotel-experiment_results.csv')
    export_all_experiments_to_csv([('0623_0001', '0713_1200')], 'concurrent-requsts-experiment.csv')
    # # above was with spike only on S2. below is with spike on all services but at different times.
    # # export_all_experiments_to_csv([('0616_2308', '0620_0200')], 'hotel-experiment_results-both-spike.csv')

    aggregate_concurrent_runs('concurrent-requsts-experiment.csv', 'hotel-experiment_results.csv')
    calculate_group_statistics('hotel-experiment_results.csv', 'grouped_hotel.csv')

    # export_all_experiments_to_csv([('0707_0001', '0708_1200')], 'concurrent-breakwaterd-experiment.csv')
    # calculate_group_statistics('concurrent-breakwaterd-experiment.csv', 'grouped_breakwaterd.csv')

    export_single_request_to_csv([('0628_1800', '0713_0100')], 'single-requests-experiment.csv')
    calculate_group_statistics('single-requests-experiment.csv', 'grouped_single.csv')
    '''

    # export_all_experiments_to_csv('/z/large-nsdi/results', 'all-experiments.csv')
    # append_new_experiments_to_csv('/z/large-nsdi/results', 'all-experiments-stable.csv', 'all-experiments.csv')

    # aggregate_concurrent_runs('all-experiments.csv', 'hotel-experiment_results.csv')
    # calculate_group_statistics('hotel-experiment_results.csv', 'grouped_hotel.csv')
    extract_single_concurrent_request_to_csv(False, 'all-experiments.csv', 'single-requests-experiment.csv')
    extract_single_concurrent_request_to_csv(True, 'all-experiments.csv', 'concurrent-requests-experiment.csv')

    # find_k_files()
    
    calculate_group_statistics('concurrent-requests-experiment.csv', 'grouped_hotel.csv')

    # export_single_request_to_csv([('0728_0001', '0919_1200')], 'single-requests-experiment.csv')
    calculate_group_statistics('single-requests-experiment.csv', 'grouped_single.csv')
