from __future__ import division

import csv
import sys
import math
import pickle
import numpy as np
import pandas as pd


polya_db_positive_dataset = '../Data/FASTA_files/CrypticModel/' + sys.argv[1]
output_file_name = sys.argv[1][:-3] + '_RNAhyb.csv'

kmer_dictionary = 'RNAhyb_short_output_protocole2.pickle'

bin_size = 20
step_len = 10
seq_len = 140

with open(kmer_dictionary, 'rb') as handle:
    kmer_dict = pickle.load(handle)

kmer_len = 7

# Calculate total amount of bins
n_bins = int((seq_len - bin_size) / step_len)


##################################################
# Reading the data
##################################################

data = pd.read_csv(polya_db_positive_dataset, sep='\t', header=None)

data = data.apply(lambda x: x.astype(str).str.upper())

CPA_data = list(data[1])

CPA_data = [x.upper() for x in CPA_data]

# Exclude sequences that have "NNN" in them
NNN_seq = []
for line in CPA_data:
    if "N" in line:
        NNN_seq.append(line)

CPA_data = [x for x in CPA_data if x not in NNN_seq]

CPA_data = [x[180:-180] for x in CPA_data]

# Make proper data len
seq_len = len(CPA_data[0])


### RNAhyb scripts
def get_sample_score(x):
    this_seq = CPA_data[x]
    this_seq_row = []

    for i in range(seq_len - kmer_len + 1):
        subset = this_seq[i:i + kmer_len]
        score = kmer_dict[subset]
        this_seq_row.append(score)

    return this_seq_row


def calc_RNAhyb_scores():
    output_file = []
    for x in range(len(CPA_data)):
        cc = get_sample_score(x)
        output_file.append(cc)

    return output_file


data_scores = calc_RNAhyb_scores()


# Get average in bins
data_scores_bin = []
for z in range(len(data_scores)):
    the_bin = data_scores[z]
    bin_scores = []
    for x in range(n_bins):
        bin_start = int(x * step_len)
        bin_end = int(bin_start + bin_size)
        current_bin = the_bin[bin_start:bin_end]

        max_bin = np.max(current_bin)

        bin_scores.append(max_bin)

    data_scores_bin.append(bin_scores)


def get_aff(sequence):
    for x in range(len(sequence)):
        sequence[x] = math.exp(-1*sequence[x])
    return sequence


# Convert MFE scores into affinity scores
data_scores_bin = [get_aff(score) for score in data_scores_bin]


def feature_names():

    # Splicing sites
    bin_names_splice = []
    for i in range(0, n_bins):
        name = "RNAhyb bin " + str(i + 1)
        bin_names_splice.append(name)

    return np.asarray(bin_names_splice)


names = feature_names()


with open(output_file_name, "w") as f:
    w = csv.writer(f)
    w.writerow(names)
    w.writerows(data_scores_bin)
