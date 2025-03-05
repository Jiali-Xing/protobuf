# This file is used to load the results of the experiments into a CSV file. Then group the results by setup and save them in a new CSV file.
# The CSV file is used to generate the plots in the paper

from utils import export_all_experiments_to_csv, calculate_group_statistics, extract_single_concurrent_request_to_csv, append_new_experiments_to_csv

if __name__ == '__main__':
    # calculate_group_statistics('hotel-experiment_results.csv', 'grouped_hotel.csv')
    extract_single_concurrent_request_to_csv(False, 'all-experiments.csv', 'single-requests-experiment.csv')
    extract_single_concurrent_request_to_csv(True, 'all-experiments.csv', 'concurrent-requests-experiment.csv')

    calculate_group_statistics('concurrent-requests-experiment.csv', 'grouped_hotel.csv')

    # export_single_request_to_csv([('0728_0001', '0919_1200')], 'single-requests-experiment.csv')
    calculate_group_statistics('single-requests-experiment.csv', 'grouped_single.csv')
