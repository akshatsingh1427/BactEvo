import csv
import random

time_steps = 1000
population = 100
resource = 1000

genotype_A = 0.6
genotype_B = 0.3
genotype_C = 0.1

mutation_rate = 0.01

data = []

for t in range(time_steps):

    # resource consumption
    resource -= random.uniform(1, 5)
    resource = max(resource, 0)

    # population growth
    growth = random.uniform(5, 15)
    population += growth

    # mutation frequency
    mutation_frequency = random.uniform(0.01, 0.1)

    # genotype changes
    genotype_A += random.uniform(-0.02, 0.02)
    genotype_B += random.uniform(-0.02, 0.02)
    genotype_C = 1 - (genotype_A + genotype_B)

    # normalize genotype densities
    total = genotype_A + genotype_B + genotype_C
    genotype_A /= total
    genotype_B /= total
    genotype_C /= total

    cooperation_index = random.uniform(0.1, 0.6)
    competition_index = random.uniform(0.1, 0.7)

    data.append([
        t,
        int(population),
        round(resource, 2),
        round(genotype_A, 3),
        round(genotype_B, 3),
        round(genotype_C, 3),
        round(mutation_frequency, 3),
        round(cooperation_index, 3),
        round(competition_index, 3)
    ])

# write CSV
with open("simulation_metrics.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "time_step",
        "total_population",
        "resource_concentration",
        "genotype_A_density",
        "genotype_B_density",
        "genotype_C_density",
        "mutation_frequency",
        "cooperation_index",
        "competition_index"
    ])

    writer.writerows(data)

print("simulation_metrics.csv generated with 1000 rows!")
