import os
import argparse
import pymeshlab
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from datetime import datetime
import random
import json
from tqdm import tqdm
from fractions import Fraction

from variance import perform_anova, perform_levene, get_pstdev, get_pvar

pymeshlab.pmeshlab.load_plugin('filter_orientedbbox.dll')
# Note needs custom plugin from https://github.com/cnr-isti-vclab/meshlab-extra-plugins/releases
# THIS IS GPLv3

def process_mesh(ms, mesh_id):
    ms.set_current_mesh(mesh_id)
    ms.apply_filter('oriented_bounding_box', create_ch=True, set_transform=True)
    ms.set_current_mesh(ms.mesh_number() - 1)
    ms.compute_matrix_by_principal_axis()

    bbox = ms.current_mesh().bounding_box()
    min_corner = bbox.min()
    max_corner = bbox.max()

    x_dim = max_corner[0] - min_corner[0]
    y_dim = max_corner[1] - min_corner[1]
    z_dim = max_corner[2] - min_corner[2]

    min_bounding_square = max(x_dim, y_dim, z_dim)
    min_bounding_square = round(min_bounding_square, 4)

    a = max(x_dim, y_dim)
    b = min(x_dim, y_dim)
    h = z_dim
    theta = math.atan(b / a) * (180 / math.pi)
    xmin: np.float64

    if a / b > math.tan(math.radians(22.5)):
        xmin = np.float64(b)
    else:
        xmin = np.float64(2 * (h * math.cos(math.radians(45 - theta))))

    volume = np.float64(x_dim * y_dim * z_dim)
    return xmin, volume

def save_results(file_name, current_time, run, mesh_dictionary):
    with open(f"Variance\\{file_name}_bounding_box_{current_time}_run{run+1}.json", "w") as file:
        values = [np.float64(value[:-1]) for value in mesh_dictionary.values()]
        json.dump({"values": values}, file, indent=4)
        print(f"Results written to file for run {run+1}")

def perform_variance_analysis(mesh_dictionary):
    mesh_sizes = [np.float64(size[:-1]) for size in mesh_dictionary.values()]

    if len(mesh_sizes) > 1:
        mid_index = len(mesh_sizes) // 2
        group1 = mesh_sizes[:mid_index]
        group2 = mesh_sizes[mid_index:]

        pstdev = get_pstdev(mesh_sizes)
        print(f"Population Standard Deviation: {pstdev}")

        pvar = get_pvar(mesh_sizes)
        print(f"Population Standard Deviation: {pvar}")

        anova_f_statistic, anova_p_value = perform_anova(group1, group2)
        print(f"ANOVA F-statistic: {anova_f_statistic}, p-value: {anova_p_value}")

        levene_stat, levene_p_value = perform_levene(group1, group2)
        print(f"Levene's Test statistic: {levene_stat}, p-value: {levene_p_value}")
    else:
        variance_files = os.listdir("Variance")
        if len(variance_files) == 1:
            variance_file_path = os.path.join("Variance", variance_files[0])
            with open(variance_file_path, "r") as file:
                data = json.load(file)
                mesh_sizes = data["values"]

                if len(mesh_sizes) > 1:
                    mid_index = len(mesh_sizes) // 2
                    group1 = mesh_sizes[:mid_index]
                    group2 = mesh_sizes[mid_index:]

                    pstdev = get_pstdev(mesh_sizes)
                    print(f"Population Standard Deviation: {pstdev}")

                    pvar = get_pvar(mesh_sizes)
                    print(f"Population Standard Deviation: {pvar}")

                    anova_f_statistic, anova_p_value = perform_anova(group1, group2)
                    print(f"ANOVA F-statistic: {anova_f_statistic}, p-value: {anova_p_value}")

                    levene_stat, levene_p_value = perform_levene(group1, group2)
                    print(f"Levene's Test statistic: {levene_stat}, p-value: {levene_p_value}")
                else:
                    print("Not enough data to perform ANOVA and Levene's test.")

def plot_results(file_name, current_time, run, mesh_sizes, p80_value):
    cumulative_area = np.cumsum(mesh_sizes)
    total_area = cumulative_area[-1]
    cumulative_area_percent = (cumulative_area / total_area) * 100

    bins = np.linspace(0.025, 2, num=15)
    bin_indices = np.digitize(mesh_sizes, bins)

    bin_cumulative_area_percent = []
    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_cumulative_area_percent.append(cumulative_area_percent[bin_mask].max())
        else:
            bin_cumulative_area_percent.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bins[:-1], bin_cumulative_area_percent, marker='o', linestyle='-', color='skyblue',
            label='Cumulative Area % Passing')
    ax.axvline(x=p80_value, color='red', linestyle='--', linewidth=2, label=f'P80: {p80_value}m')
    ax.set_xscale('log')
    ax.set_xlabel('Particle Size (mm)', fontsize=12)
    ax.set_ylabel('Volume % Passing', fontsize=12)
    ax.set_title('Particle Size Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    tick_positions = np.arange(0.025, 1.0, 0.05)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{tick:.3f}' for tick in tick_positions], rotation=45)
    ax.tick_params(axis='x', pad=10)

    plt.tight_layout()
    plt.draw()
    plt.savefig(f"Graphs\\{file_name}_particle_size_distribution_{current_time}_run{run+1}.png")
    print(f"Results saved as image for run {run+1}")

