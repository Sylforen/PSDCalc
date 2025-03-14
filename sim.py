import argparse
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime
import os
import json

from mbs import process_mesh, assess_variance, perform_variance_analysis, save_results
from obb import get_obb_lengths, set_random_seed  # Import the function from obb.py

def main():
    """
    | --(f)iles ===> default = valTest.glb (do not forget file extension!)
    | --(n)um_runs ===> default = 1
    | --(s)eed ===> default = 42
    """
    parser = argparse.ArgumentParser(
        prog='Particle Sizing',
        description='Determine size of each particle in mesh'
    )

    parser.add_argument('-f', '--files', nargs='+', default=["valTest.glb"], type=str, help="List of files to process")
    parser.add_argument('-n', '--num_runs', default=1, type=int, help="Number of runs")
    parser.add_argument('-s', '--seed', default=42, type=int, help="Random seed")
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_random_seed(args.seed)

    all_results = []
    dimensions_list = []  # Moved outside the loop to collect dimensions across all runs

    for file in args.files:
        for run in tqdm(range(args.num_runs), desc=f"Processing runs for {file}"):
            dimensions_asd = {}

            # Get OBB lengths using the function from obb.py
            obb_lengths = get_obb_lengths(file)
            if not obb_lengths:
                print(f"No valid meshes found in {file}")
                continue

            mesh_count = len(obb_lengths)
            print(f"Meshes to compute size in {file}: {mesh_count}")

            mesh_dictionary = {}
            mesh_volume_dictionary = {}

            single_run_dimensions = []
            for mesh_id in range(mesh_count):
                try:
                    dimensions, xmin, volume = process_mesh(file, mesh_id)
                    single_run_dimensions.append(dimensions)
                    dimensions_list.append(dimensions)
                    mesh_dictionary[mesh_id] = str(np.float64(xmin)) + "m"
                    mesh_volume_dictionary[mesh_id] = volume
                except Exception as e:
                    print(f"Error processing mesh {mesh_id} in {file}: {e}")

            for mesh in single_run_dimensions:
                dimensions_asd[mesh['mesh_id']] = mesh

            now = datetime.now()
            current_time = now.strftime("%y%m%d-%H%M%S")

            mesh_sizes = [np.float64(size[:-1]) for size in mesh_dictionary.values()]
            mesh_sizes.sort()  # Ensure the list is sorted before calculating p80_value

            if len(mesh_sizes) > 0:
                p80_index = int(0.8 * len(mesh_sizes))
                p80_value = mesh_sizes[min(p80_index, len(mesh_sizes) - 1)]
            else:
                p80_value = None

            # Save results and plot
            save_results(os.path.splitext(file)[0], current_time, run, mesh_dictionary)

            if run == args.num_runs - 1:
                results = perform_variance_analysis(mesh_dictionary)
                all_results.append((file, run, results))

            with open('stuff_{}.json'.format(run), 'w') as f:
                json.dump(dimensions_asd, f)

    # Display all results at the end
    for file, run, results in all_results:
        print(f"\nResults for {file}, run {run+1}:")
        if 'error' in results:
            print(results['error'])
        else:
            if 'pstdev' in results:
                print(f"Population Standard Deviation: {results['pstdev']}")
            if 'pvar' in results:
                print(f"Population Variance: {results['pvar']}")
            if 'anova_f_statistic' in results:
                print(f"ANOVA F-statistic: {results['anova_f_statistic']}, p-value: {results['anova_p_value']}")
            if 'levene_stat' in results:
                print(f"Levene's Test statistic: {results['levene_stat']}, p-value: {results['levene_p_value']}")

    # Assess variance in dimensions
    variance_results = assess_variance(dimensions_list)
    print("\nVariance in dimensions:")
    print(f"Variance in smallest_side_1: {variance_results['variance_smallest_side_1']}")
    print(f"Variance in smallest_side_2: {variance_results['variance_smallest_side_2']}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")