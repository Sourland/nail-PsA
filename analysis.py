import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import bootstrap
from sklearn.utils import resample
# Function to plot histograms together
# Function to plot histograms together with the same number of bins
def compare_histograms(df1, df2, column, label1, label2, output_dir, bins=None):
    plt.figure(figsize=(10, 6))
    
    # Determine common bins if not specified
    if bins is None:
        combined_data = pd.concat([df1[column], df2[column]])
        bins = np.histogram_bin_edges(combined_data, bins='auto')
    
    sns.histplot(df1[column], kde=True, label=label1, color='blue', alpha=0.5, bins=bins, stat='probability')
    sns.histplot(df2[column], kde=True, label=label2, color='red', alpha=0.5, bins=bins, stat='probability')
    
    plt.title(f'Comparison of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{column}_comparison.png'))
    plt.close()

# Function to calculate statistics
def calculate_statistics(df, column):
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0] if not df[column].mode().empty else None
    range_data = df[column].max() - df[column].min()
    std_dev = df[column].std()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    skewness = df[column].skew()
    kurt = df[column].kurt()
    return [mean, median, mode, range_data, std_dev, Q1, Q3, IQR, outliers.shape[0], skewness, kurt]

# Function to plot the statistics table
def plot_statistics_table(stats_data, column, output_dir):
    rows = ['Mean', 'Median', 'Mode', 'Range', 'Std Dev', 'Q1', 'Q3', 'IQR', 'Outliers Count', 'Skewness', 'Kurtosis']
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    table_data = [['Statistic', 'Value']] + list(zip(rows, stats_data))
    ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.25, 0.25])
    plt.title(f'Statistics for {column}')
    plt.savefig(os.path.join(output_dir, f'{column}_stats.png'))
    plt.close()

# Ensure the output directory exists
output_dir = 'results/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the datasets
df_dip_healthy = pd.read_csv('results/features_dip_healthy.csv')
df_dip_swollen = pd.read_csv('results/features_dip_swollen.csv')
df_pip_healthy = pd.read_csv('results/features_pip_healthy.csv')
df_pip_swollen = pd.read_csv('results/features_pip_swollen.csv')

# Columns to compare
columns_dip = ["DIP_Effective_Width_Index", "DIP_Effective_Width_Middle", "DIP_Effective_Width_Ring", "DIP_Effective_Width_Pinky"]
columns_pip = ["PIP_Effective_Width_Index", "PIP_Effective_Width_Middle", "PIP_Effective_Width_Ring", "PIP_Effective_Width_Pinky"]

for column in columns_dip:
    compare_histograms(df_dip_healthy, df_dip_swollen, column, 'Healthy', 'Swollen', output_dir)
    healthy_stats = calculate_statistics(df_dip_healthy, column)
    swollen_stats = calculate_statistics(df_dip_swollen, column)
    plot_statistics_table(healthy_stats, column, output_dir)
    plot_statistics_table(swollen_stats, column, output_dir)

# Compare PIP features
for column in columns_pip:
    compare_histograms(df_pip_healthy, df_pip_swollen, column, 'Healthy', 'Swollen', output_dir)
    healthy_stats = calculate_statistics(df_pip_healthy, column)
    swollen_stats = calculate_statistics(df_pip_swollen, column)
    plot_statistics_table(healthy_stats, column, output_dir)
    plot_statistics_table(swollen_stats, column, output_dir)


def calculate_sum_of_columns(df, columns, joint):
    df[joint] = df[columns].sum(axis=1)
    return df

# Calculate the sum of columns and then compare histograms
def compare_sum_of_columns(df1, df2, columns, label1, label2, output_dir, joint):
    sum1 = calculate_sum_of_columns(df1, columns, joint)
    sum2 = calculate_sum_of_columns(df2, columns, joint)
    compare_histograms(sum1, sum2, joint, label1, label2, output_dir)
    sum_stats1 = calculate_statistics(df1, joint)
    sum_stats2 = calculate_statistics(df2, joint)
    plot_statistics_table(sum_stats1, 'Sum of ' + ', '.join(columns), output_dir)
    plot_statistics_table(sum_stats2, 'Sum of ' + ', '.join(columns), output_dir)

# Compare DIP features sum
compare_sum_of_columns(df_dip_healthy, df_dip_swollen, columns_dip, 'Healthy', 'Swollen', output_dir, "Sum of DIP features")

# Compare PIP features sum
compare_sum_of_columns(df_pip_healthy, df_pip_swollen, columns_pip, 'Healthy', 'Swollen', output_dir, "Sum of PIP features")
