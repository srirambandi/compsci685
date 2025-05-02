import numpy as np
import csv

# FORMAT: id,seed,equ_str,equ_prefix,sol_str,sol_prefix
DATASETS_FOLDER = "data"
DATASETS = [
    "data1", "data2"  # etc.
]
OUTPUT_FILE = "data/final.csv"

encountered = set()
seed_encountered = set()
final_dataset = []

for dataset in DATASETS:
    with open(f"{DATASETS_FOLDER}/{dataset}.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if (row[2], row[4]) not in encountered:
                encountered.add((row[2], row[4]))
                final_dataset.append(row[1:])
                seed_encountered.add(row[1])

np.random.shuffle(final_dataset)

with open(OUTPUT_FILE, "w", newline='') as f:
    writer = csv.writer(f, delimiter=",")
    for i, row in enumerate(final_dataset):
        writer.writerow([i] + row)

print(f"Final dataset size: {len(final_dataset)}")
