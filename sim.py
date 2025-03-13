import argparse
import numpy as np
import random
from tqdm import tqdm
import pymeshlab
from datetime import datetime
import os

from mbs import plot_results, save_results, perform_variance_analysis, process_mesh


def main():
    """
    | --(f)ile ===> default = valTest.glb (do not forget file extension!)
    | --(n)um_runs ===> default = 1
    | --(s)eed ===> default = 42
    """
    parser = argparse.ArgumentParser(
        prog='Particle Sizing',
        description='Determine size of each particle in mesh'
    )

    parser.add_argument('-f', '--file', default="valTest.glb", type=str)
    parser.add_argument('-n', '--num_runs', default=1, type=int)
    parser.add_argument('-s', '--seed', default=42, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    for run in tqdm(range(args.num_runs), desc="Processing runs"):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(args.file)

        mesh_count = ms.mesh_number()
        print("Meshes to compute size: {}".format(mesh_count))

        mesh_dictionary = {}
        mesh_volume_dictionary = {}

        for mesh_id in range(mesh_count):
            if ms[mesh_id].face_number() > 2:
                try:
                    xmin, volume = process_mesh(ms, mesh_id)
                    mesh_dictionary[mesh_id] = str(np.float64(xmin)) + "m"
                    mesh_volume_dictionary[mesh_id] = volume
                except Exception as e:
                    print(f"Error processing mesh {mesh_id}: {e}")

        now = datetime.now()
        current_time = now.strftime("%y%m%d-%H%M%S")

        mesh_sizes = [np.float64(size[:-1]) for size in mesh_dictionary.values()]
        p80_index = int(0.8 * len(mesh_sizes))
        p80_value = mesh_sizes[p80_index]

        plot_results(os.path.splitext(args.file)[0], current_time, run, mesh_sizes, p80_value)
        save_results(os.path.splitext(args.file)[0], current_time, run, mesh_dictionary)

        if run == args.num_runs - 1:
            perform_variance_analysis(mesh_dictionary)

if __name__ == '__main__':
    main()
