import csv
import math
import matplotlib.pyplot as plt
import numpy as np

# Read data from file
x_arr = []
y_arr = []
with open('multivar.csv', newline='') as file:
    csvreader = csv.reader(file)
    # skip the header
    next(csvreader)
    for row in csvreader:
        x_row = [float(xi) for xi in row[:-1]]
        x_arr.append(x_row)
        y_arr.append(float(row[-1]))

# Convert to numpy
# Concatenate a column of 1's to x so that weights include the y-intercept.
x = np.concatenate([np.array(x_arr), np.ones((len(x_arr), 1))], axis=1)
y = np.array(y_arr)
n = len(y_arr)

# Compute weights.
# Derivation: https://spia.uga.edu/faculty_pages/mlynch/teaching/ols/OLSDerivation.pdf
weights = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
print("Weights:", weights)

# Compute variance and standard deviation.
y_predicted = x.dot(weights)
variance = np.average((y - y_predicted)**2)
print("variance:", variance)
print("standard deviation:", math.sqrt(variance))