from __future__ import division

import csv
import sys
import numpy as np
import pandas as pd


polya_db_positive_dataset = '../Data/FASTA_files/CrypticModel/' + sys.argv[1]
output_file_name = sys.argv[1][:-3] + '_MaxEnt.csv'

kmer_scores_file = '9_mers_scores.txt'


score_threshold = 0
perl_scan_length = 9
bin_size = 20
step_len = 10
seq_len = 140

# Calculate total amount of bins
n_bins = int((seq_len - perl_scan_length - bin_size) / step_len)


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


#########################################################
# Load 9mer scores and convert them into dict
#########################################################
def load_9mer_scores():
    with open(kmer_scores_file) as f:
        reader = csv.reader(f, delimiter="\t")
        kmer_scores = list(reader)

    for x in range(len(kmer_scores)):
        kmer_scores[x][1] = float(kmer_scores[x][1])

    # Turn the list into dict
    kmer_scores = {k: v for k, v in kmer_scores}

    return kmer_scores


#########################################################
# MaxEnt dictionary implementation
#########################################################
def calculate_maxent_scores_in_overlapping_bins(x, kmer_scores):
    # First, get current window
    part = CPA_data[x]

    this_sequence_row = []

    for i in range(0, len(part) - perl_scan_length + 1):
        current_sub_bin = part[i:i + perl_scan_length]

        current_score = kmer_scores[current_sub_bin]
        this_sequence_row.append(current_score)

    return this_sequence_row


def calc_maxent_scores():

    kmer_scores = load_9mer_scores()

    pq1vec = []
    for x in range(0, len(CPA_data)):
        cc = calculate_maxent_scores_in_overlapping_bins(x, kmer_scores)
        pq1vec.append(cc)

    scores_1 = np.asarray(pq1vec)

    return scores_1


#####################################################
# Feature names function
#####################################################
def feature_names():

    # Splicing sites
    bin_names_splice = []
    for i in range(0, n_bins):
        name = "MaxEnt bin " + str(i + 1)
        bin_names_splice.append(name)

    return np.asarray(bin_names_splice)


#####################################################
# Run and save functions
#####################################################
scores = calc_maxent_scores()

def get_high_hits(features):
    print(type(features))

    for x in range(len(features)):
        for j in range(len(features[x])):
            if features[x][j] < score_threshold:
                features[x][j] = 0
            else:
                features[x][j] = 1

    return features

scores = get_high_hits(scores)

data_scores_bin = []
for z in range(len(scores)):
    the_bin = scores[z]
    bin_scores = []
    for x in range(n_bins):
        bin_start = int(x * step_len)
        bin_end = int(bin_start + bin_size)
        current_bin = the_bin[bin_start:bin_end]

        max_bin = np.max(current_bin)

        bin_scores.append(max_bin)

    data_scores_bin.append(bin_scores)

names = feature_names()

with open(output_file_name, "w") as f:
    w = csv.writer(f)
    w.writerow(names)
    w.writerows(data_scores_bin)
