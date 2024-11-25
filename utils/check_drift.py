import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def visualize_drift(reff_data, curr_data, data_version):
    n = len(curr_data.columns)

    if n<=5:
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(3, 6))
    else:
        n_row = math.ceil(n/2)
        fig, axes = plt.subplots(nrows=n_row, ncols=2, figsize=(5, 7))
        
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 8})

    ref_col = reff_data.columns
    # print("plotting...")
    for idx, col in enumerate(curr_data.columns):
        sns.kdeplot(reff_data[ref_col[idx]], ax=axes[idx] , color='orange', label='ref')
        sns.kdeplot(curr_data[col], ax=axes[idx], color='blue', label='curr')
        axes[idx].legend(fontsize=6, frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'metrics/drift/data_{data_version}.png')
    plt.close(fig)



def evaluate_drift(reff_data, curr_data, threshold:float=1.2, dataset_drift_share:float=0.5):
    drift = {}

    data_drift_report = Report(metrics=[
    DataDriftPreset(stattest='kl_div', stattest_threshold=.8, drift_share=dataset_drift_share),
    ])


    data_drift_report.run(current_data=curr_data, reference_data=reff_data)
    drift_report_dict = data_drift_report.as_dict()

    drift = {**drift_report_dict['metrics'][0]['result']}
    drift['column_drift_threshold'] = threshold
    
    for key, value in drift_report_dict['metrics'][1]['result']['drift_by_columns'].items():
        drift[key] = value['drift_score']

    return drift




if __name__ == "__main__":
    reff_data = pd.read_csv('data/TM1.csv')
    curr_data = pd.read_csv('data/TM2.csv')
    visualize_drift(reff_data, curr_data, 99)
    print("hello")