import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

import os
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# reff_data = 
# curr_data = 

def visualize_drift(reff_data, curr_data, data_version):
    n = len(curr_data.columns)

    if n<=5:
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(3, 6))
    else:
        n_row = math.ceil(n/2)
        fig, axes = plt.subplots(nrows=n_row, ncols=2, figsize=(5, 7))
        
    axes = axes.flatten()
    plt.rcParams.update({'font.size': 8})

    for idx, col in enumerate(reff_data.columns):
        sns.kdeplot(reff_data[col], ax=axes[idx] , color='blue', label='ref')
        sns.kdeplot(curr_data[col], ax=axes[idx], color='orange', label='curr')
        axes[idx].legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(f'metrics/drift/data_{data_version}.png')



def evaluate_drift(reff_data, curr_data, threshold:float=1.2):
    data_drift_report = Report(metrics=[
    DataDriftPreset(stattest='kl_div', stattest_threshold=threshold,)
    ])

    data_drift_report.run(current_data=curr_data, reference_data=reff_data)
    drift_report_dict = data_drift_report.as_dict()
