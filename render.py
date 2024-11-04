import os
import csv
import argparse
import numpy as np

from lib.config import ConfigLoader

def create_histograms(x, z, q, flags, image_dim, x_bounds, z_bounds):
    if len(x) != len(z) or len(x) != len(q):
        print(f"\033[0;31m-- Error: Mismatched lengths: x({len(x)}), z({len(z)}), q({len(q)})\033[0m")
        return None, None
    
    z_pixels, x_pixels = image_dim
    x_min, x_max = x_bounds
    z_min, z_max = z_bounds

    z_bins = np.linspace(z_min - 0.5, z_max + 0.5, z_pixels + 1)
    x_bins = np.linspace(x_min - 0.5, x_max + 0.5, x_pixels + 1)

    input_hist, _, _ = np.histogram2d(z, x, bins=[z_bins, x_bins], weights=q)
    input_hist = input_hist.astype(float)

    target_hist = np.zeros_like(input_hist, dtype='float')

    for i in range(len(x)):
        x_idx = np.clip(np.digitize(x[i], x_bins) - 1, 0, x_pixels - 1)
        z_idx = np.clip(np.digitize(z[i], z_bins) - 1, 0, z_pixels - 1)

        label = 0
        for index, flag in enumerate(flags[i]):
            if flag:
                label = index + 1

        target_hist[z_idx, x_idx] = label

    return input_hist, target_hist


def parse_data(data):
    try:
        if data[-1] == '1':
            data = data[:-1]

        n_hits, n_flags, n_meta = map(int, data[:3])
        run, subrun, event = map(int, data[3:6])
        height, width = map(int, data[6:8])
        x_vtx, z_vtx = map(float, data[8:10])
        drift_min, drift_max, wire_min, wire_max = map(float, data[10:14])

        hit_data = data[14:]

        exp_len = n_hits * (3 + n_flags)
        act_len = len(hit_data)

        print(f"\033[0;36m-- Expected length: {exp_len}, Actual length: {act_len}\033[0m")
        if act_len < exp_len:
            raise ValueError("\033[0;31m-- Inconsistent hit data length\033[0m") 

        hit_x = np.array(hit_data[0::(3 + n_flags)], dtype=float)
        hit_z = np.array(hit_data[1::(3 + n_flags)], dtype=float)
        hit_q = np.array(hit_data[2::(3 + n_flags)], dtype=float)

        hit_flags = []
        for i in range(n_flags):
            hit_flags.append(np.array(hit_data[(3 + i)::(3 + n_flags)], dtype=float))

        return (run, subrun, event), (height, width), (drift_min, drift_max, wire_min, wire_max), (hit_x, hit_z, hit_q, np.vstack(hit_flags).T)

    except (ValueError, IndexError) as e:
        print(f"\033[0;31m-- Error parsing event data: {e}\033[0m")
        return None, None


def process_event(data, output_dir):
    meta_data, hit_data = parse_data(data)
    if meta_data is None or hit_data is None:
        print(f"\033[0;31m-- Skipping event due to data parsing issues.\033[0m")
        return

    (run, subrun, event), (height, width), (drift_min, drift_max, wire_min, wire_max), (hit_x, hit_z, hit_q, hit_flags) = meta_data, hit_data
    input_hist, target_hist = create_histograms(hit_x, hit_z, hit_q, hit_flags, (height, width), (drift_min, drift_max), (wire_min, wire_max))

    input_dir = os.path.join(output_dir, "input_file")
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    input_name = os.path.join(input_dir, f"{run}_{subrun}_{event}.npz")
    target_name = os.path.join(target_dir, f"{run}_{subrun}_{event}.npz")

    np.savez_compressed(input_name, input_hist)
    np.savez_compressed(target_name, target_hist)


def process_file(config):
    input_file = config.input_file
    output_dir = config.output_dir
    n_events = config.n_events

    try:
        print(f"\033[0;34m-- Processing file: {input_file}...\033[0m")
        with open(input_file, 'r') as f:
            reader = csv.reader(f)
            for entry, data in enumerate(reader):
                if entry >= n_events:
                    break 
                process_event(data, output_dir)

                if (entry + 1) % 1000 == 0:
                    print(f"\033[0;36m-- Processed {entry + 1} events.\033[0m")

        print(f"\033[0;32m-- Completed processing file successfully!\033[0m")

    except Exception as e:
        print(f"\033[0;31m-- Error processing file {input_file}: {e}!\033[0m")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = ConfigLoader(args.config) 
    process_file(config)