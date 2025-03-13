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

pymeshlab.pmeshlab.load_plugin('filter_orientedbbox.dll')

# Note needs custom plugin from https://github.com/cnr-isti-vclab/meshlab-extra-plugins/releases
# WARNING THIS IS GPLv3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Particle Sizing',
        description='Determine size of each particle in mesh'
    )

    parser.add_argument('-f', '--file', default=None, type=str, help='glb mesh file')
    parser.add_argument('-n', '--num_runs', default=1, type=int, help='Number of times to run the code')
    parser.add_argument('-s', '--seed', default=42, type=int')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    for run in range(args.num_runs):
        ms = pymeshlab.MeshSet()
        # ms.set_verbosity(True)
        ms.load_new_mesh(args.file)
        # ms.print_status()

        mesh_count = ms.mesh_number()

        print("Meshes to compute size: {}".format(mesh_count))

        meshes = []

        mesh_dictionary = {}
        mesh_volume_dictionary = {}

        for mesh_id in range(mesh_count):
            #print("Computing size for mesh {}".format(mesh_id))
            if ms[mesh_id].face_number() > 2:
                try:
                    ms.set_current_mesh(mesh_id)
                    ms.apply_filter('oriented_bounding_box', create_ch=True, set_transform=True)
                    ms.set_current_mesh(ms.mesh_number() - 1)
                    ms.compute_matrix_by_principal_axis()

                    # Get the bounding box dimensions
                    bbox = ms.current_mesh().bounding_box()
                    min_corner = bbox.min()
                    max_corner = bbox.max()

                    # Calculate the dimensions
                    x_dim = max_corner[0] - min_corner[0]
                    y_dim = max_corner[1] - min_corner[1]
                    z_dim = max_corner[2] - min_corner[2]

                    # Calculate the minimum bounding square 
                    min_bounding_square = max(x_dim, y_dim, z_dim)
                    min_bounding_square = round(min_bounding_square, 4)

                    # Trigonometric Check 
                    a = max(x_dim, y_dim)
                    b = min(x_dim, y_dim)
                    h = z_dim
                    theta = math.atan(b / a) * (180/math.pi)
                    xmin : float

                    if (a/b > math.tan(math.radians(22.5))):
                        xmin = b
                    else:
                        xmin = 2 * (h * math.cos(math.radians(45 - theta)))

                    square_aperture = xmin ** 2
                    #xmin = round(xmin, 4)

                    print(f"Mesh {mesh_id} - Bounding Box Dimensions: X: {x_dim}, Y: {y_dim}, Z: {z_dim}")
                    #print(f"Mesh {mesh_id} - OLD Minimum Bounding Square: {min_bounding_square}")
                    print(f"Mesh {mesh_id} - Minimum Bounding Square: {xmin}")
                    #print(f"Mesh {mesh_id} - Volume: {volume}")
                    print("")

                    # Add the mesh to the list of meshes
                    volume = x_dim * y_dim * z_dim
                    mesh_dictionary[mesh_id] = str(float(min_bounding_square)) + "m"
                    mesh_volume_dictionary[mesh_id] = volume

                except Exception as e:
                    print(f"Error processing mesh {mesh_id}: {e}")

        # Get the current date and time
        now = datetime.now()

        print(mesh_dictionary)

        # Format the date and time
        current_time = now.strftime("%y%m%d-%H%M%S")

        print("Current Date and Time:", current_time)

        mesh_sizes = [float(size[:-1]) for size in mesh_dictionary.values()]
        mesh_sizes.sort()

        # Calculate P80
        p80_index = int(0.8 * len(mesh_sizes))
        p80_value = mesh_sizes[p80_index]

        # Find the corresponding mesh ID for the P80 value
        p80_mesh_id = list(mesh_dictionary.keys())[p80_index]

        # # Calculate P50
        # p50_index = int(0.5 * len(mesh_sizes))
        # p50_value = mesh_sizes[p50_index]

        # # Find the corresponding mesh ID for the P50 value
        # p50_mesh_id = list(mesh_dictionary.keys())[p50_index]

        # mesh_volumes = list(mesh_volume_dictionary.values())
        # # Calculate cumulative area % passing based on volume
        # cumulative_volume = np.cumsum(mesh_volumes)
        # total_volume = cumulative_volume[-1]
        # cumulative_area_percent = (cumulative_volume / total_volume) * 100

        # Calculate cumulative area % passing
        cumulative_area = np.cumsum(mesh_sizes)
        total_area = cumulative_area[-1]
        cumulative_area_percent = (cumulative_area / total_area) * 100

        # Define logarithmic bins
        #bins = np.logspace(np.log10(min(mesh_sizes)), np.log10(max(mesh_sizes)), num=15)
        bins = np.linspace(0.025, 2, num = 15) # CHANGED FROM 1m TO 2m

        # Digitize the mesh sizes into bins
        bin_indices = np.digitize(mesh_sizes, bins)

        # Calculate cumulative area % passing for each bin
        bin_cumulative_area_percent = []
        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                bin_cumulative_area_percent.append(cumulative_area_percent[bin_mask].max())
            else:
                bin_cumulative_area_percent.append(0)

        # Filter out bins with zero cumulative area percentage
        filtered_bins = [bins[i] for i in range(len(bin_cumulative_area_percent)) if bin_cumulative_area_percent[i] > 0]
        filtered_bin_cumulative_area_percent = [percent for percent in bin_cumulative_area_percent if percent > 0]

        # Visualize the data in matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(bins[:-1], bin_cumulative_area_percent, marker='o', linestyle='-', color='skyblue', label='Cumulative Area % Passing')
        ax.axvline(x=p80_value, color='red', linestyle='--', linewidth=2, label=f'P80: {p80_value}m')
        #ax.axvline(x=p50_value, color='orange', linestyle='--', linewidth=2, label=f'P50: {p50_value}m')
        ax.set_xscale('log')
        ax.set_xlabel('Particle Size (mm)', fontsize=12)
        ax.set_ylabel('Volume % Passing', fontsize=12)
        ax.set_title('Particle Size Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)  # Reduced alpha for less opacity

        # ax.set_xticks(bins)
        # ax.set_xticklabels([f'{bin:.3f}' for bin in bins], rotation=45)

        # Set x-ticks to every increment between 0.025m and 1m
        tick_positions = np.arange(0.025, 1.0, 0.05)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'{tick:.3f}' for tick in tick_positions], rotation=45)
        ax.tick_params(axis='x', pad=10)

        plt.tight_layout()
        plt.draw()
        plt.savefig(f"particle_size_distribution_{current_time}_run{run+1}.png")
        print(f"Results saved as image for run {run+1}")
        #plt.show()

        # Write the results to a file
        with open(f"bounding_box_{current_time}_run{run+1}.json", "w") as file:
            # Sort the dictionary by value in descending order
            sorted_meshes = sorted(mesh_dictionary.items(), key=lambda item: float(item[1][:-1]), reverse=True)
            values = [float(value[:-1]) for key, value in sorted_meshes]
            json.dump({"values": values}, file, indent=4)
            print(f"Results written to file for run {run+1}")