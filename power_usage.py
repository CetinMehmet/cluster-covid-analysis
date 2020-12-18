import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import glob
import pyarrow.parquet as pq


DAS_PATH = "var/scratch/lvs215/process-surf-dataset/"
LOCAL_PATH = "/Users/cetinmehmet/Desktop/processed-surf-dataset/"
FULL_SEASON = 221
DAY = 4 * 60 * 24 # 4 * 60 is an hour

df_load1 = pd.read_parquet(DAS_PATH + "node_load1")
df_power = pd.read_parquet(DAS_PATH + "surfsara_power_usage")

RACKS = [
    "r10", "r11", "r12", "r13", "r14",
    "r15", "r23", "r25", "r26", "r27",
    "r31", "r32", "r33",
    'r30', 'r31', 'r32', 'r33', 'r34'
]
GPU_RACKS = [
    'r30', 'r31', 'r32', 'r33', 'r34'
]


def get_nodes(rack, df):
    nodes = set()
    for node in df.columns:
        if node.split("n")[0] == rack:
            nodes.add(node)

    return nodes

for rack_title in RACKS:
    df_rack_load = df_load1[get_nodes(rack_title, df_load1)].replace(-1, 0)
    df_rack_power = df_power[get_nodes(rack_title, df_power)].replace(-1, 0)
    rows = len(df_rack_load.columns)
    fig, ax_arr = plt.subplots(rows, 1, figsize = (32, rows * 10), constrained_layout=True)
    
    for i in range(0, rows):
        if df_rack_load[i].iloc[:, i:i+1].values != []:
            ax_arr[i].plot(df_rack_load.iloc[:, i:i+1].values, label="Load1")
            ax_arr[i].set_xticks(range(0, len(df_power.index), DAY*5))
            ax_arr[i].set_xticklabels(range(1, FULL_SEASON, 5), fontsize=20)
            ax_arr[i].set_title("Node " + df_rack_load.iloc[:, i:i+1].columns[0], fontsize=24)
            ax_arr[i].set_ylabel("Load1", fontsize=24)
            ax_arr[i].set_xlabel("Days", fontsize=22)
            ax_arr[i].tick_params(axis='y', which='major', labelsize=26)
            ax_arr[i].tick_params(axis='y', which='minor', labelsize=22)
            ax_arr[i].set_ylim(0, 100) # Above 100 is irrelevant
            ax_arr[i].legend(loc="upper left", fontsize=24)
        
        if df_rack_power.iloc[:, i:i+1].values != []:
            ax_t.plot(df_rack_power.iloc[:, i:i+1].values, color="red", label="Power usage")
            ax_t.tick_params(axis='y', which='major', labelsize=26)
            ax_t.tick_params(axis='y', which='minor', labelsize=22)
            ax_t = ax_arr[i].twinx()
            ax_t.set_ylim(0, )
            ax_t.set_ylabel("Power consumption [W]", fontsize=24)
            ax_t.legend(loc="upper right", fontsize=24)


    if rack_title in GPU_RACKS:
        fig.suptitle("ML rack " + rack_title, fontsize=28)
    else:
        fig.suptitle("Generic rack " + rack_title, fontsize=28)

    plt.savefig("load1_vs_power_" + rack_title + ".pdf", dpi=100)

