
# --------------------------------------------------
# Save Breed Confusion Analysis
# --------------------------------------------------

import csv
import os

# Top 5 confused breed pairs
confused_pairs = [
    ["British Shorthair", "Russian Blue", 21, 8, 29],
    ["American Pit Bull Terrier", "Staffordshire Bull Terrier", 27, 2, 29],
    ["Birman", "Ragdoll", 14, 12, 26],
    ["Bengal", "Maine Coon", 5, 15, 20],
    ["English Cocker Spaniel", "English Setter", 13, 5, 18]
]

# Create output folder
os.makedirs("results/metrics", exist_ok=True)

# Save results to CSV
with open("results/metrics/confused_breeds.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Breed A",
        "Breed B",
        "A predicted as B",
        "B predicted as A",
        "Total Confusions"
    ])

    writer.writerows(confused_pairs)

print("Confusion analysis saved successfully!")
