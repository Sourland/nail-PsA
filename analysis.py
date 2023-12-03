import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import bootstrap, shapiro, mannwhitneyu, ks_2samp, levene, ttest_ind
from sklearn.utils import resample


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
def plot_statistics_table(healthy_stats, swollen_stats, column, output_dir):
    rows = ['Mean', 'Median', 'Mode', 'Range', 'Std Dev', 'Q1', 'Q3', 'IQR', 'Outliers Count', 'Skewness', 'Kurtosis']
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Prepare the data for the healthy and swollen statistics
    table_data = [['Statistic', 'Healthy Value', 'Swollen Value']] + list(zip(rows, healthy_stats, swollen_stats))
    
    # Add the table to the plot
    table = ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.2, 0.2, 0.2])
    
    # Scale the columns to fit the data
    table.auto_set_column_width(col=list(range(len(table_data[0]))))
    
    plt.title(f'Statistics Comparison for {column}')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{column}_stats_comparison.png'))
    plt.close()


def save_statistics_to_csv(healthy_stats, swollen_stats, column, output_dir):
    # Define the rows for the statistics
    rows = ['Mean', 'Median', 'Mode', 'Range', 'Std Dev', 'Q1', 'Q3', 'IQR', 'Outliers Count', 'Skewness', 'Kurtosis']
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare the data for the healthy and swollen statistics
    table_data = [['Statistic', 'Healthy Value', 'Swollen Value']] + list(zip(rows, healthy_stats, swollen_stats))
    
    # Define the filename
    filename = os.path.join(output_dir, f'{column}_stats_comparison.csv')

    # Write the data to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table_data)


def perform_normality_test(df, column):
    """Perform Shapiro-Wilk Normality Test and return the result."""
    stat, p_value = shapiro(df[column])
    return p_value

def perform_mann_whitney_test(df1, df2, column):
    """Perform Mann-Whitney U Test and return the result."""
    u_stat, p_value = mannwhitneyu(df1[column], df2[column], alternative='two-sided')
    return u_stat, p_value

def cliffs_delta(x, y):
    """Calculate Cliff's Delta as a measure of effect size."""
    n_x, n_y = len(x), len(y)
    more = sum(xi > yi for xi in x for yi in y)
    less = sum(xi < yi for xi in x for yi in y)
    return (more - less) / (n_x * n_y)


def plot_cdf(df1, df2, column, output_dir):
    """Plot the Cumulative Distribution Function (CDF) for two datasets."""
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(df1[column], label='Healthy', color='blue')
    sns.ecdfplot(df2[column], label='Swollen', color='red')
    plt.title(f'Cumulative Distribution Function for {column}')
    plt.xlabel(column)
    plt.ylabel('CDF')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{column}_cdf.png'))
    plt.close()

def perform_ks_test(df1, df2, column):
    """Perform Kolmogorov-Smirnov Test and return the result."""
    statistic, p_value = ks_2samp(df1[column], df2[column])
    return statistic, p_value

def perform_levenes_test(df1, df2, column):
    """Perform Levene's Test for equality of variances and return the result."""
    statistic, p_value = levene(df1[column], df2[column])
    return statistic, p_value

def bootstrap_mean_confidence_interval(df, column, num_bootstrap=1000):
    """Calculate bootstrap mean confidence intervals."""
    bootstrapped_means = bootstrap((df[column],), np.mean, n_resamples=num_bootstrap)
    ci_lower = np.percentile(bootstrapped_means, 2.5)
    ci_upper = np.percentile(bootstrapped_means, 97.5)
    return ci_lower, ci_upper

def perform_two_sample_t_test(df1, df2, column):
    """Perform Independent Two-Sample T-Test and return the result."""
    t_stat, p_value = ttest_ind(df1[column], df2[column], equal_var=True)  # Use equal_var=False for Welch's T-test
    return t_stat, p_value

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

    t_statistic, t_p_value = perform_two_sample_t_test(df_dip_healthy, df_dip_swollen, column)
    print(f'Independent Two-Sample T-Test for {column} - Statistic: {t_statistic}, P-Value: {t_p_value}')

    # Perform and output results of Mann-Whitney U Test
    mwu_statistic, mwu_p_value = perform_mann_whitney_test(df_dip_healthy, df_dip_swollen, column)
    print(f'Mann-Whitney U Test for {column} - Statistic: {mwu_statistic}, P-Value: {mwu_p_value}')

    # Calculate and output effect size (Cliff's Delta)
    cliff_delta_value = cliffs_delta(df_dip_healthy[column], df_dip_swollen[column])
    print(f"Cliff's Delta for {column}: {cliff_delta_value}")

    # Perform and output results of Kolmogorov-Smirnov Test
    ks_statistic, ks_p_value = perform_ks_test(df_dip_healthy, df_dip_swollen, column)
    print(f'Kolmogorov-Smirnov Test for {column} - Statistic: {ks_statistic}, P-Value: {ks_p_value}')

    # Plot Cumulative Distribution Function (CDF)
    plot_cdf(df_dip_healthy, df_dip_swollen, column, output_dir)

    # Save statistics to CSV
    save_statistics_to_csv(healthy_stats, swollen_stats, column, output_dir)



# Compare PIP features
for column in columns_pip:
    compare_histograms(df_pip_healthy, df_pip_swollen, column, 'Healthy', 'Swollen', output_dir)
    healthy_stats = calculate_statistics(df_pip_healthy, column)
    swollen_stats = calculate_statistics(df_pip_swollen, column)

    t_statistic, t_p_value = perform_two_sample_t_test(df_pip_healthy, df_pip_swollen, column)
    print(f'Independent Two-Sample T-Test for {column} - Statistic: {t_statistic}, P-Value: {t_p_value}')
    
    # Perform and output results of Mann-Whitney U Test
    mwu_statistic, mwu_p_value = perform_mann_whitney_test(df_pip_healthy, df_pip_swollen, column)
    print(f'Mann-Whitney U Test for {column} - Statistic: {mwu_statistic}, P-Value: {mwu_p_value}')

    # Calculate and output effect size (Cliff's Delta)
    cliff_delta_value = cliffs_delta(df_pip_healthy[column], df_pip_swollen[column])
    print(f"Cliff's Delta for {column}: {cliff_delta_value}")

    # Perform and output results of Kolmogorov-Smirnov Test
    ks_statistic, ks_p_value = perform_ks_test(df_pip_healthy, df_pip_swollen, column)
    print(f'Kolmogorov-Smirnov Test for {column} - Statistic: {ks_statistic}, P-Value: {ks_p_value}')

    # Plot Cumulative Distribution Function (CDF)
    plot_cdf(df_pip_healthy, df_pip_swollen, column, output_dir)

    # Save statistics to CSV
    save_statistics_to_csv(healthy_stats, swollen_stats, column, output_dir)




