import sys
import csv
import numpy as np
import pandas as pd


PWM_file = sys.argv[1]
U1_file_1 = sys.argv[2]
U1_file_2 = sys.argv[2]

output_file_name = PWM_file[:-4] + '_U1.csv'


g_data = pd.DataFrame()
for chunk in pd.read_csv(U1_file_1, header=0, sep=',', chunksize=1000):
    g_data = pd.concat([g_data, chunk], ignore_index=False)
try:
    g_data = g_data.drop('labels', axis=1)
except KeyError:
    print('No labels in gene data')

try:
    g_data = g_data.drop('Unnamed: 0', axis=1)
except KeyError:
    print('No Unnamed: 0 found in gene data')

header = list(g_data.columns.values)

gene_data = np.array(g_data)


g_data_1 = pd.DataFrame()
for chunk in pd.read_csv(U1_file_1, header=0, sep=',', chunksize=1000):
    g_data_1 = pd.concat([g_data_1, chunk], ignore_index=False)
try:
    g_data_1 = g_data_1.drop('labels', axis=1)
except KeyError:
    print('No labels in gene data')

try:
    g_data_1 = g_data_1.drop('Unnamed: 0', axis=1)
except KeyError:
    print('No Unnamed: 0 found in gene data')

header_1 = list(g_data_1.columns.values)

gene_data_1 = np.array(g_data)


g_data_2 = pd.DataFrame()
for chunk in pd.read_csv(U1_file_2, header=0, sep=',', chunksize=1000):
    g_data_2 = pd.concat([g_data_2, chunk], ignore_index=False)
try:
    g_data_2 = g_data_2.drop('labels', axis=1)
except KeyError:
    print('No labels in gene data')

try:
    g_data_2 = g_data_2.drop('Unnamed: 0', axis=1)
except KeyError:
    print('No Unnamed: 0 found in gene data')

header_2 = list(g_data_2.columns.values)

gene_data_2 = np.array(g_data_2)


out_header = header + header_1 + header_2
out_file = np.hstack((gene_data, gene_data_1, gene_data_2))

with open(output_file_name, "w") as f:
    w = csv.writer(f)
    w.writerow(out_header)
    w.writerows(out_file)
