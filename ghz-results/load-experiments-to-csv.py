# This file is used to load the results of the experiments into a CSV file. Then group the results by setup and save them in a new CSV file.
# The CSV file is used to generate the plots in the paper

from utils import export_all_experiments_to_csv, calculate_group_statistics


def generate_csv():
    # Define the time ranges for the experiments
    time_ranges = [
        ('0527_0000', '0531_2359'),
        # ('0531_0042', '0531_0044'),
    ]

    # Export all experiments to a CSV file
    export_all_experiments_to_csv(time_ranges, 'may-experiment_results.csv')


# def group_experiments():
#     calculate_group_statistics('experiment_results.csv', 'grouped_experiment_results.csv')

if __name__ == '__main__':
    export_all_experiments_to_csv([('0527_0000', '0610_1200')], 'alibaba-experiment_results.csv')
    export_all_experiments_to_csv([('0610_1200', '0612_0000')], 'all-alibaba-experiment_results.csv')
    # line above is for the 4 nodes experiment first time. below is retuned.
    export_all_experiments_to_csv([('0607_1354', '0609_1200')], '4-nodes-experiment_results.csv')
    export_all_experiments_to_csv([('0609_1200', '0611_0000')], '4-nodes-monotonic-experiment_results.csv')
    calculate_group_statistics('4-nodes-experiment_results.csv', 'grouped_4n.csv')
    calculate_group_statistics('4-nodes-monotonic-experiment_results.csv', 'grouped_4n_monotonic.csv')
    calculate_group_statistics('alibaba-experiment_results.csv', 'grouped_ali.csv')
    calculate_group_statistics('all-alibaba-experiment_results.csv', 'grouped_all_ali.csv')