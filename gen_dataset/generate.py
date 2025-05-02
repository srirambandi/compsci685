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

    global_state_seed = np.random.randint(10e8)
    seed_gen_rng = np.random.RandomState(global_state_seed)
    try_log(f"Global state seed: {global_state_seed}", level=logging.INFO)

    seed_encounter = set()
    sol_encountered = set()
    samples = []

    for i in tqdm(range(num_samples)):
        while True:
            try:
                seed = seed_gen_rng.randint(10e8)
                rng = np.random.RandomState(seed)

                if seed in seed_encounter:
                    try_log(f"Seed {seed} already encountered. Regenerating...", level=logging.INFO)
                    continue
                seed_encounter.add(seed)

                sol_out = generator.generate_clean_solution(rng)

                if sol_out is None:
                    continue

                sol_str, sol_clean, sol_prefix = sol_out

                if sol_str in sol_encountered:
                    try_log(f"Solution {sol_str} already encountered. Regenerating...")
                    continue
                sol_encountered.add(sol_str)

                ode_out = generator.generate_ode(rng, sol_clean)

                if ode_out is None:
                    continue

                eq_str, eq_prefix = ode_out

                break
            except KeyboardInterrupt:
                try_log("Keyboard interrupt detected. Exiting...", level=logging.INFO)

                with open(output_file.replace(".csv", "_interrupted.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "seed", "equ_str", "equ_prefix", "sol_str", "sol_prefix"])
                    writer.writerows(samples)

                try_log(f"Dataset generated and saved to {output_file}", level=logging.INFO)
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
