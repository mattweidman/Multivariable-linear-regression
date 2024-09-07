import csv
import numpy as np

VARIABLES = 5
ARRAY_LEN = 1000

# The first columns of x are random numbers between 0 and 100.
# The last column is all 1's so that weights can represent the intercept.
x = np.concatenate(
    [np.random.rand(ARRAY_LEN, VARIABLES) * 100, np.ones((ARRAY_LEN, 1))],
    axis=1)

# Weights increase by 2, starting at 2. The last weight is the y-intercept.
weights = np.arange(stop=(VARIABLES + 1)) * 2 + 2

std_dev = 13
residuals = np.random.normal(size=ARRAY_LEN) * std_dev

# y is an array of random numbers linearly related to x
y = x.dot(weights) + residuals

# Write all to file
with open('multivar.csv', 'w', newline='') as file:
    fieldnames = ['x' + str(i) for i in range(VARIABLES)] + ['y']
    csvwriter = csv.DictWriter(file, fieldnames=fieldnames)
    csvwriter.writeheader()
    for i in range(ARRAY_LEN):
        row = {}
        for j in range(VARIABLES):
            row['x' + str(j)] = x[i, j]
        row['y'] = y[i]
        csvwriter.writerow(row)