import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


DAS_PATH = "/var/scratch/lvs215/processed-surf-dataset/"


if len(sys.argv) != 2:
    print("1 argument must be given")
    exit(1)

elif sys.argv[1] != "node" and sys.argv[1] != "time" and sys.argv[1] != "all":
    print("Arguments that can be passed: 'node', 'time', 'all' ")
    exit(1)

plot_type = sys.argv[1]


df_total = pd.read_parquet(DAS_PATH + "node_memory_MemTotal")
df_free = pd.read_parquet(DAS_PATH + "node_memory_MemFree")

# Load parquets
df_memory = 1 - (df_free / df_total) 
df_load = pd.read_parquet(DAS_PATH + "node_load1")
df_temp = pd.read_parquet(DAS_PATH + "surfsara_ambient_temp")
df_power = pd.read_parquet(DAS_PATH + "surfsara_power_usage")

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
        
    arr = arr[~mask]  # Filter out NaN values and less than 0
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

#%%
df_load_covid, df_load_non_covid = covid_non_covid(df_load)
df_power_covid, df_power_non_covid = covid_non_covid(df_power)
df_temp_covid, df_temp_non_covid = covid_non_covid(df_temp)
df_memory_covid, df_memory_non_covid = covid_non_covid(df_memory)
#%%

mean_per_node = "node"
mean_per_timestamp = "time"

if plot_type == mean_per_node:
    savefig_title = "covid_cluster_mean_per_node_violinplot.pdf"
    df_load_covid_m = df_load_covid.mean()
    df_load_non_covid_m = df_load_non_covid.mean()
    df_power_covid_m = df_power_covid.mean()
    df_power_non_covid_m =  df_power_non_covid.mean()
    df_temp_covid_m = df_temp_covid.mean()
    df_temp_non_covid_m = df_temp_non_covid.mean()
    df_memory_covid_m = df_memory_covid.mean()
    df_memory_non_covid_m = df_memory_non_covid.mean()

elif plot_type == mean_per_timestamp:
    savefig_title = "covid_cluster_mean_per_timestamp_violinplot.pdf"
    df_load_covid_m = df_load_covid.mean(axis=1)
    df_load_non_covid_m = df_load_non_covid.mean(axis=1)
    df_power_covid_m = df_power_covid.mean(axis=1)
    df_power_non_covid_m =  df_power_non_covid.mean(axis=1)
    df_temp_covid_m = df_temp_covid.mean(axis=1)
    df_temp_non_covid_m = df_temp_non_covid.mean(axis=1)
    df_memory_covid_m = df_memory_covid.mean(axis=1)
    df_memory_non_covid_m = df_memory_non_covid.mean(axis=1)
    
else:
    savefig_title = "covid_cluster_all_values_violinplot.pdf"
    df_load_covid_m = get_custom_values(df_load_covid)
    df_load_non_covid_m = get_custom_values(df_load_non_covid)
    #df_power_covid_m = get_custom_values(df_power_covid)
    #df_power_non_covid_m = get_custom_values(df_power_non_covid)
    #df_temp_covid_m = get_custom_values(df_temp_covid)
    #df_temp_non_covid_m = get_custom_values(df_temp_non_covid)
    #df_memory_covid_m = get_custom_values(df_memory_covid)
    #df_memory_non_covid_m = get_custom_values(df_memory_non_covid)

#%%

def plot_violin(covid_val, non_covid_val, ax, title, ylabel):
    sns.violinplot(
        data=[covid_val, non_covid_val], 
        palette=["lightcoral", "steelblue"],
        ax=ax, width=0.95, cut=0)

    ax.set_ylabel(ylabel, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.text(x=-0.45, y=int(get_max_pdf(covid_val)[1])+1, s="{:.2f}".format(get_max_pdf(covid_val)[0]), fontsize=16, color="black", va="center")
    ax.text(x=0.4, y=int(get_max_pdf(non_covid_val)[1])+1, s="{:.2f}".format(get_max_pdf(non_covid_val)[0]), fontsize=16, color="black", va="center")
    ax.set_xticks(np.arange(2))
    ax.set_ylim(0, 50)
    ax.set_xticklabels(
        ('covid', 'non-covid'),
       ha='center', fontsize=22
    )
    max_covid_val = np.max(covid_val)
    max_non_covid_val = np.max(non_covid_val)
    if max_covid_val > 50:
        ax.text(x=0-0.05, y=51, s=str(round(max_covid_val, 1)), fontsize=16)
    
    if max_non_covid_val > 50:
        ax.text(x=1-0.05, y=51, s=str(round(max_non_covid_val, 1)), fontsize=16)

fig, (ax_load, ax_power, ax_temp, ax_memory) = plt.subplots(4, 1, figsize=(12, 32))
fig.tight_layout(h_pad=10.0, w_pad=6.0)

plot_violin(df_load_covid_m, df_load_non_covid_m, ax_load, "Load1", "Load1")
#plot_violin(df_power_covid_m, df_power_non_covid_m, ax_power, "Power consumption", "Power consumption [watt]")
#plot_violin(df_temp_covid_m, df_temp_non_covid_m, ax_temp, "Ambient temperature", "Temperature [celsius]")
#plot_violin(df_memory_covid_m, df_memory_non_covid_m, ax_memory, "RAM utilization", "Utilized fraction")
plt.subplots_adjust(wspace=0.15, hspace=0.2, left=0.11, bottom=0.15, right=0.98, top=0.96)
plt.savefig(savefig_title, dpi=100)
