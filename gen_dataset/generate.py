import argparse
import csv
import logging

import numpy as np
from src.ode_generator import ODEGenerator
from src.utils import try_log
from tqdm import tqdm

if __name__ == "__main__":
    generator = ODEGenerator()

    parser = argparse.ArgumentParser(description="Generate ODE dataset")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output_file", type=str, default="data/dataset.csv", help="Output CSV file name")
    args = parser.parse_args()

    num_samples = args.num_samples
    output_file = args.output_file

    try_log(f"Generating {num_samples} samples...", level=logging.INFO)

    global_state_seed = np.random.randint(0, 9999999)
    seed_gen_rng = np.random.RandomState(global_state_seed)
    try_log(f"Global state seed: {global_state_seed}", level=logging.INFO)

    eqn_encountered = set()
    samples = []

    for i in tqdm(range(num_samples)):
        while True:
            try:
                seed = seed_gen_rng.randint(0, 9999999)
                rng = np.random.RandomState(seed)

                out = generator.generate_ode_pair(rng)

                if out is None:
                    try_log("Invalid output. Regenerating...")
                    continue

                eq_str, eq_prefix, sol_str, sol_prefix = out

                if eq_str in eqn_encountered:
                    try_log("Duplicate equation encountered. Regenerating...")
                    continue

                eqn_encountered.add(eq_str)

                break
            except KeyboardInterrupt:
                try_log("Keyboard interrupt detected. Exiting...", level=logging.INFO)
                exit(0)
            except TimeoutError:
                continue
            except Exception as e:
                try_log(f"Unexpected error: {e}. Regenerating...", level=logging.ERROR)
                continue

        samples.append([i, seed, eq_str, eq_prefix, sol_str, sol_prefix])

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "seed", "equ_str", "equ_prefix", "sol_str", "sol_prefix"])
        writer.writerows(samples)

    try_log(f"Dataset generated and saved to {output_file}", level=logging.INFO)
