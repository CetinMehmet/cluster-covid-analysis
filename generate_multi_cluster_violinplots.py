import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

"""
    This script is used to produce the violin plots for the surfing article and the Mehmet's research project.
    The differences made between figures are:
        1. The x parameter in the ax.text() function to adjust the position of the max PDF value.
        2. The parameters passed to the set_ylim has changed depending on the metric.
"""


DAS_PATH = "/var/scratch/lvs215/processed-surf-dataset/"


#%% Helper functions
def covid_non_covid(df):
    if df.index.dtype == "int64":
        df.index = pd.to_datetime(df.index, unit='s')

    covid_df = df.loc['2020-02-27 00:00:00' :, :]
    non_covid_df = df.loc[: '2020-02-26 23:59:45', :]
    covid_df.reset_index()
    non_covid_df.reset_index()
    return covid_df, non_covid_df

def get_custom_values(df):
    values = np.array([])
    for column in df.columns:
        arr = df[column].values
        mask = (np.isnan(arr) | (arr < 0))
        
        arr = np.round(arr[~mask], 1)  # Filter out NaN values and less than 0 and round the values for ram utilization
        values = np.append(values, arr)
    return values

def get_max_pdf(df):
    def normalize(df):
        df = df.value_counts(sort=False, normalize=True).rename_axis('target').reset_index(name='pdf')
        df["cdf"] = df["pdf"].cumsum()
        return df
        
    df_new = normalize(pd.DataFrame(df))
    index_max_pdf = df_new["pdf"].idxmax()
    max_value = df_new.iloc[index_max_pdf]
    return (max_value["pdf"], max_value["target"])


def plot_violin(covid_val, non_covid_val, ax, ylabel):
    sns.violinplot(
        data=[covid_val, non_covid_val], 
        palette=["lightcoral", "steelblue"],
        ax=ax, width=0.8, cut=0)

    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xticks(np.arange(2))
    ax.set_ylim(0, )
    ax.set_xticklabels(
        ('covid', 'non-covid'),
       ha='center', fontsize=16
    )

    if ylabel == "Load1":
        ax.set_ylim(0, 400)
        max_covid_val = np.max(covid_val)
        max_non_covid_val = np.max(non_covid_val)

        if max_covid_val > 400:
            ax.text(x=0-0.15, y=411, s=str(max_covid_val), fontsize=14, color="black", va="center")
        if max_non_covid_val > 400:
            ax.text(x=1-0.15, y=411, s=str(max_non_covid_val), fontsize=14, color="black", va="center")
    
    if ylabel == "Load1":
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    
    if ylabel == "Power consumption [W]":
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

df_free = pd.read_parquet(DAS_PATH + "node_memory_MemFree")
df_total = pd.read_parquet(DAS_PATH + "node_memory_MemTotal")
df_ram_covid, df_ram_non_covid = covid_non_covid(100 * (1 - (df_free / df_total)))
df_load1_covid, df_load1_non_covid = covid_non_covid(pd.read_parquet(DAS_PATH + "node_load1"))
df_power_covid, df_power_non_covid = covid_non_covid(pd.read_parquet(DAS_PATH + "surfsara_power_usage"))
df_temp_covid, df_temp_non_covid = covid_non_covid(pd.read_parquet(DAS_PATH + "surfsara_ambient_temp"))

df_ram_covid_vals, df_ram_non_covid_vals = get_custom_values(df_ram_covid), get_custom_values(df_ram_non_covid)
df_load1_covid_vals, df_load1_non_covid_vals = get_custom_values(df_load1_covid), get_custom_values(df_load1_non_covid)
df_power_covid_vals, df_power_non_covid_vals = get_custom_values(df_power_covid), get_custom_values(df_power_non_covid)
df_temp_covid_vals, df_temp_non_covid_vals = get_custom_values(df_temp_covid), get_custom_values(df_temp_non_covid)


fig, ((ax_ram, ax_power), (ax_temp, ax_load)) = plt.subplots(2, 2, figsize=(11, 5), sharex=True)
plt.tight_layout()

plot_violin(
    covid_val=df_power_covid_vals, 
    non_covid_val=df_power_non_covid_vals, 
    ax=ax_power, ylabel="Power consumption [W]")

plot_violin(
    covid_val=df_temp_covid_vals, 
    non_covid_val=df_temp_non_covid_vals, 
    ax=ax_temp, ylabel="Temperature [C]")

plot_violin(
    covid_val=df_ram_covid_vals,
    non_covid_val=df_ram_non_covid_vals,
    ax=ax_ram, ylabel="RAM utilization [%]")

plot_violin(
    covid_val=df_load1_covid_vals,
    non_covid_val=df_load1_non_covid_vals,
    ax=ax_load, ylabel="Load1")

plt.savefig("/home/cmt2002/cluster_analysis/plots/" + "multi_cluster_plots.pdf", dpi=100)

print("Done!")



