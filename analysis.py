import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists('results/plots'):
    os.makedirs('results/plots')

# Read the CSV into a DataFrame
df = pd.read_csv('results/features_dip.csv')

# Function to plot mean and sigmas
def plot_mean_and_sigmas(column, data_frame):
    mean_value = data_frame[column].mean()
    sigma_value = data_frame[column].std()

    plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.2f}')
    
    # Plot up to 3 sigmas
    for i in range(1, 4):
        plt.axvline(mean_value + i*sigma_value, color='g', linestyle='--', label=f'{i} Sigma = {mean_value + i*sigma_value:.2f}')
        plt.axvline(mean_value - i*sigma_value, color='g', linestyle='--', label=f'{i} Sigma = {mean_value - i*sigma_value:.2f}')

# Plot the distribution of each feature for DIP
for column in ["DIP_Effective_Width_Index", "DIP_Effective_Width_Middle", "DIP_Effective_Width_Ring", "DIP_Effective_Width_Pinky"]:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)  # Histogram with Kernel Density Estimation

    plot_mean_and_sigmas(column, df)

    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/plots/{column}.png')

    # Calculate the statistics
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]
    range_data = df[column].max() - df[column].min()
    std_dev = df[column].std()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    skewness = df[column].skew()
    kurt = df[column].kurt()

    # Create the data for the table
    rows = ['Mean', 'Median', 'Mode', 'Range', 'Std Dev', 'Q1', 'Q3', 'IQR', 'Outliers Count', 'Skewness', 'Kurtosis']
    data = [mean, median, mode, range_data, std_dev, Q1, Q3, IQR, outliers.shape[0], skewness, kurt]

    # Plot the table
    fig, ax = plt.subplots(figsize=(12, 4))  # set the size that you'd like (width, height)
    ax.axis('off')  # hide the axes
    table_data = []
    table_data.append(['Statistic', 'Value'])
    for row_label, row_data in zip(rows, data):
        table_data.append([row_label, row_data])
    ax.table(cellText=table_data, cellLoc = 'center', loc='center', colWidths=[0.25, 0.25])
    plt.title(f'Statistics for {column}')
    plt.savefig(f'results/plots/{column}_stats.png')

df = pd.read_csv('results/features_pip.csv')

# Plot the distribution of each feature for PIP
for column in ["PIP_Effective_Width_Index", "PIP_Effective_Width_Middle", "PIP_Effective_Width_Ring", "PIP_Effective_Width_Pinky"]:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)  # Histogram with Kernel Density Estimation
    
    plot_mean_and_sigmas(column, df)
    
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/plots/{column}.png')


    # Calculate the statistics
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]
    range_data = df[column].max() - df[column].min()
    std_dev = df[column].std()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    skewness = df[column].skew()
    kurt = df[column].kurt()

    # Create the data for the table
    rows = ['Mean', 'Median', 'Mode', 'Range', 'Std Dev', 'Q1', 'Q3', 'IQR', 'Outliers Count', 'Skewness', 'Kurtosis']
    data = [mean, median, mode, range_data, std_dev, Q1, Q3, IQR, outliers.shape[0], skewness, kurt]

    # Plot the table
    fig, ax = plt.subplots(figsize=(12, 4))  # set the size that you'd like (width, height)
    ax.axis('off')  # hide the axes
    table_data = []
    table_data.append(['Statistic', 'Value'])
    for row_label, row_data in zip(rows, data):
        table_data.append([row_label, row_data])
    ax.table(cellText=table_data, cellLoc = 'center', loc='center', colWidths=[0.25, 0.25])
    plt.title(f'Statistics for {column}')
    plt.savefig(f'results/plots/{column}_stats.png')

