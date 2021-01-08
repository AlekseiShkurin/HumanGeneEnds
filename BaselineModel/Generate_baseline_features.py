from __future__ import division
import csv
import itertools

from Bio.Seq import Seq
import pandas as pd
import numpy as np
import sys

# Get name of the FASTA file to process
dataset = sys.argv[1]

# Get the name of the resulting feature matrix
output_file_name = sys.argv[2]

bin_size = 30
step_len = 10


######################################
# Load data
######################################

data = pd.read_csv(dataset, sep='\t', header=None)

data = data.apply(lambda x: x.astype(str).str.upper())

CPA_data = list(data[1])

CPA_data = [x.upper() for x in CPA_data]

# Exclude sequences that have "NNN" in them
NNN_seq = []
for line in CPA_data:
    if "N" in line:
        NNN_seq.append(line)

CPA_data = [x for x in CPA_data if x not in NNN_seq]

# Calculate total number of bins
seq_len = len(CPA_data[0])

n_bins = int((seq_len - bin_size) / step_len)


#################################################
# BEEML implementation
#################################################


def revcomplementSeq(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ("".join(complement.get(base, base) for base in reversed(seq)))


def eucld_dist(x, y):
    x = np.array(x)
    y = np.array(y)
    dist = np.linalg.norm(x - y)
    return (dist)


IUPAC_withAmb = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
                 'R': [0.5, 0, 0.5, 0], 'Y': [0, 0.5, 0, 0.5], 'S': [0, 0.5, 0.5, 0], 'W': [0.5, 0, 0, 0.5],
                 'K': [0, 0, 0.5, 0.5],
                 'M': [0.5, 0.5, 0, 0], 'B': [0, 0.333, 0.333, 0.333], 'D': [0.333, 0, 0.333, 0.333],
                 'H': [0.333, 0.333, 0, 0.333],
                 'V': [0.333, 0.333, 0.333, 0], 'N': [0.25, 0.25, 0.25, 0.25]
                 }
JustBases = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
JustBases_N = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
               'N': [0.25, 0.25, 0.25, 0.25]}


def PFM2IUPAC(PFMmat, IUPAC_Dict, trim=False):
    MotifStr = ''
    for row in PFMmat:
        SmallestDistance_Base = ''
        SmallestDistance_Value = 1000000
        for Base, Vector in IUPAC_Dict.items():
            d = eucld_dist(row, Vector)
            if d < SmallestDistance_Value:
                SmallestDistance_Base = Base
                SmallestDistance_Value = d
        MotifStr += SmallestDistance_Base
    if trim == True:
        return (TRIM_Ns(MotifStr))
    else:
        return (MotifStr)


def TRIM_Ns(IUPACstr):
    TryingForward = True
    TryingReverse = True
    while TryingForward != False:
        if IUPACstr.startswith('N'):
            IUPACstr = IUPACstr[1:]
        else:
            TryingForward = False

    while TryingReverse != False:
        if IUPACstr.endswith('N'):
            IUPACstr = IUPACstr[:-1]
        else:
            TryingReverse = False
    return (IUPACstr)


def ReadCisBPMotif(filename):
    count = 0
    PFM = ''
    with open(filename) as inpfm:
        for line in inpfm:
            if line[0] == 'P':
                continue
            else:
                count += 1
                line = [float(l) for l in line.strip().split('\t')[1:]]
                if count == 1:
                    PFM = np.array(line)
                else:
                    PFM = np.vstack((PFM, np.array(line)))
    return (PFM)


class TFMotif:
    def __init__(self, ID=None, motif_FPM=None):
        self.ID = ID
        self.mat = motif_FPM

    def __str__(self):
        return ('Motif Name: ' + self.ID + '\n' +
                'Consensus Sequence (Strong): ' + TRIM_Ns(PFM2IUPAC(self.mat, JustBases)) + '\n' +
                'Consensus Sequence (Ambiguous): ' + PFM2IUPAC(self.mat, IUPAC_withAmb))

    def convert2EnergyMatrix(self):
        import numpy as np
        PSEUDO = 0.00001
        self.emat = self.mat + PSEUDO
        self.emat = -1 * np.log(self.emat)
        # loop through lines to add the max constant
        for pos in range(0, self.mat.shape[0]):
            maxl = max(self.mat[pos])
            lmaxl = np.log(maxl)
            self.emat[pos] += lmaxl


def BEEMLscan(energymatrix, seq, mu=0):
    from math import exp
    BaseIndex = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    motifLen = energymatrix.shape[0]
    FwdScores = []
    RevScores = []
    pos = 0

    e = 2.718281828

    while pos < (len(seq) - len(energymatrix) + 1):
        Ei = 0
        subseq = seq[pos:(pos + motifLen)]
        for subp in range(0, len(subseq)):
            base = subseq[subp]
            Ei += energymatrix[subp, BaseIndex[base]]
        subseqscore = 1 / (1 + exp(Ei - mu))
        # print subseqscore
        FwdScores.append(subseqscore)
        pos += 1

    # Scan reverse direction
    seq = revcomplementSeq(seq)
    pos = 0
    while pos < (len(seq) - len(energymatrix) + 1):
        Ei = 0
        subseq = seq[pos:(pos + motifLen)]
        for subp in range(0, len(subseq)):
            base = subseq[subp]
            Ei += energymatrix[subp, BaseIndex[base]]
        subseqscore = 1 / (1 + exp(Ei - mu))
        # print subseqscore
        RevScores.append(subseqscore)
        pos += 1
    return (FwdScores, RevScores)


##############################################################
# Calculate occurences of a kmer
##############################################################
def find_occ(motif, part):
    cnt = 0
    idx = 0
    while True:
        idx = part.find(motif, idx)
        if idx >= 0:
            cnt += 1
            idx += 1
        else:
            break
    return cnt


def inwin_calc_scores(motif, current_list, x):
    # First, get current window
    part = current_list[x]

    part_one_row = []
    for i in range(0, n_bins):
        bin_start = int(i * step_len)
        bin_end = int(bin_start + bin_size)
        current_bin = part[bin_start:bin_end]

        cc = find_occ(motif, current_bin)

        part_one_row.append(cc)

    return part_one_row


def calc_scores(motif):
    pq1vec = []
    for x in range(0, len(CPA_data)):
        cc = inwin_calc_scores(motif, CPA_data, x)
        pq1vec.append(cc)

    # Now, convert them to np arrays
    scores_1 = np.asarray(pq1vec)

    return scores_1


################################################################
# Calculate binding likelihood using BEEML
################################################################
def find_motkmer_counts(kmer, pos_data):
    counts = 0

    for x in range(0, len(pos_data)):
        # Count appearances of each k-mer in pos samples
        cnts = find_occ(str(kmer), pos_data[x])

        # Sum it
        counts = counts + cnts

    return counts


def generate_FPM(instances):
    from Bio import motifs

    # Get amount of times each k-mer appears in the data
    appears = []
    for x in range(len(instances)):
        count = find_motkmer_counts(instances[x], CPA_data)
        appears.append(count)

    # Create new instances that take into accounts amount
    # of time each k-mer appears
    new_instances = []
    for x in range(len(appears)):
        with_app = list(itertools.repeat(instances[x], appears[x]))
        new_instances.extend(with_app)

    m = motifs.create(new_instances)

    # Subset counts from it
    m_counts = m.counts

    # Reformat it from Biopython counts to FPM
    FPM = []
    x = 0
    all_sum = 0
    for y in range(0, 4):
        # Get a line (each line is a nucleotide)
        all_sum += m_counts[y][x]

    for x in range(0, 4):

        # Get a line (each line is a nucleotide)
        FPM_line = m_counts[x]

        for i in range(0, len(FPM_line)):
            if FPM_line[i] != 0:
                FPM_line[i] = FPM_line[i] / all_sum

        # Append
        FPM.append(FPM_line)

    final_FPM = np.asarray(FPM).T

    return final_FPM


def inwin_calc_aff(motif, current_list, x):
    # First, get current window
    part = current_list[x]

    part_out_one_row = []
    for i in range(0, n_bins):
        bin_start = int(i * step_len)
        bin_end = int(bin_start + bin_size)
        current_bin = part[bin_start:bin_end]

        fwd, rev = BEEMLscan(motif.emat, current_bin)
        sumScore = max(fwd)

        part_out_one_row.append(sumScore)

    return part_out_one_row


# Now, calculated it in the needed bins
def calc_affinities(instances):
    pq1vec = []

    bb = generate_FPM(instances)

    motif = TFMotif(ID='Cebpb', motif_FPM=bb)

    # To Score sequences with the BEEML method
    motif.convert2EnergyMatrix()

    for x in range(0, len(CPA_data)):
        cc = inwin_calc_aff(motif, CPA_data, x)
        pq1vec.append(cc)

    # Generate scores

    # Now, convert them to np arrays
    scores_1 = np.asarray(pq1vec)

    return scores_1


def calc_affinities_PWM(instances):
    pq1vec = []

    motif = TFMotif(ID='Cebpb', motif_FPM=instances)

    # To Score sequences with the BEEML method
    motif.convert2EnergyMatrix()

    for x in range(0, len(CPA_data)):
        cc = inwin_calc_aff(motif, CPA_data, x)
        pq1vec.append(cc)

    # Generate scores

    # Now, convert them to np arrays
    scores_1 = np.asarray(pq1vec)

    return scores_1


def read_RBP_FPMs(filename):
    FPM_names = pd.read_csv(filename, delimiter='\t', header=0, index_col=0)
    FPM_names = np.asarray(FPM_names)

    return FPM_names


def inbin_calc_aff(motif, n_bins, bin_size, step_len, current_list, x):
    # First, get current window
    part = current_list[x]

    this_motif_row = []
    for i in range(0, n_bins):
        bin_start = int(i * step_len)
        bin_end = int(bin_start + bin_size)
        current_bin = part[bin_start:bin_end]

        # Here goes current feature function
        fwd, rev = BEEMLscan(motif.emat, current_bin)
        sumScore = sum(fwd)
        this_motif_row.append(sumScore)

    return this_motif_row


#################################################
# Loading the PWMs
#################################################
# Specify kmers to check
CA_UA = ["CA", "TA"]

DSEs = ["GT", "TG", 'TT', 'GG', "T", "G", "TTT", "TGG", "TGT", "GTG"]

motif_kmers = [[Seq("AATAAA"), Seq("ATTAAA"), Seq("TATAAA"), Seq("TTTAAA"), Seq("AAGAAA"), Seq("AGTAAA"), Seq("CATAAA"),
                Seq("AATACA"), Seq("AATATA"), Seq("GATAAA"), Seq("AATGAA"), Seq("ACTAAA"), Seq("AATAGA")]]
poly_U_kmers = [[Seq('TATTTT'), Seq('TGTTTT'), Seq('TTTTTT')]]
poly_U_kmer = [[Seq('TTTT')]]
UGUA_kmers = [[Seq('TGTA')]]

# Get names of files for RBP analysis
PolyU = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/PolyU_PWM.txt')
NUDT = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/NUDT_PWM.txt')
UGUA_Hu = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/UGUA_Hu.txt')
DSE_PWM_1 = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/DSE_1_PWM.txt')
DSE_PWM_2 = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/DSE_2_PWM.txt')
DSE_PWM_3 = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/DSE_3_PWM.txt')
DSE_PWM_4 = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/DSE_4_PWM.txt')
PAS_Siepel = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/PAS_Siepel.txt')
PAS_Hu = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/PAS_Hu.txt')
PAS_PWM = read_RBP_FPMs('../Data/PWMs/BaselineModelPWMs/PAS_PWM.txt')


###########################################
# Execution Functions
###########################################
# Simple count scores
def kmer_scores_all(kmers_to_check):
    # Create empty variable for scores
    scores = None

    for x in range(0, len(kmers_to_check)):

        # Calculate scores for motif
        sco = calc_scores(kmers_to_check[x])

        # Column ctack all the scores
        if scores is None:
            scores = sco
        else:
            scores = np.hstack((scores, sco))

    return scores


# DSEs count scores
def calc_combined_scores(DSEs):
    pq1vec = []
    for x in range(len(CPA_data)):
        sample_score = []
        for y in range(len(DSEs)):
            score = inwin_calc_scores(DSEs[y], CPA_data, x)
            sample_score.append(score)
        sample_score = np.sum(sample_score, axis=0)
        pq1vec.append(sample_score)

    # Generate scores

    # Now, convert them to np arrays
    scores_1 = np.asarray(pq1vec)

    return scores_1


# K-mer motifs
def motif_scores_all(motif_kmers):
    # Create empty variable for scores
    scores = None

    for x in range(0, len(motif_kmers)):

        # Calculate scores for motif
        sco = calc_affinities(motif_kmers[x])

        # Column stack all the scores
        if scores is None:
            scores = sco
        else:
            scores = np.hstack((scores, sco))

    return scores


def feature_names():
    # First - generate names for each base for each bin

    # and for motifs
    PAS_estimate = []
    for i in range(0, n_bins):
        name = "PAS hex PWM bin " + str(i + 1)
        PAS_estimate.append(name)

    PAS_Hu_ = []
    for i in range(0, n_bins):
        name = 'PAS PWM Hu bin ' + str(i + 1)
        PAS_Hu_.append(name)

    PAS_Siep = []
    for i in range(0, n_bins):
        name = 'PAS PWM Siepel bin ' + str(i + 1)
        PAS_Siep.append(name)

    DSE_kmers = []
    for i in range(0, n_bins):
        name = 'DSE k-mers bin ' + str(i + 1)
        DSE_kmers.append(name)

    DSE_PWM_1_ = []
    for i in range(0, n_bins):
        name = 'DSE PWM 1 Hu bin ' + str(i + 1)
        DSE_PWM_1_.append(name)

    DSE_PWM_2_ = []
    for i in range(0, n_bins):
        name = 'DSE PWM 2 Hu bin ' + str(i + 1)
        DSE_PWM_2_.append(name)

    DSE_PWM_3_ = []
    for i in range(0, n_bins):
        name = 'DSE PWM 3 Hu bin ' + str(i + 1)
        DSE_PWM_3_.append(name)

    DSE_PWM_4_ = []
    for i in range(0, n_bins):
        name = 'DSE PWM 4 Hu bin ' + str(i + 1)
        DSE_PWM_4_.append(name)

    NUDT_name = []
    for i in range(0, n_bins):
        name = "NUDT PWM bin " + str(i + 1)
        NUDT_name.append(name)

    UGUA_Hu_ = []
    for i in range(0, n_bins):
        name = "UGUA PWM Hu bin " + str(i + 1)
        UGUA_Hu_.append(name)

    UGUA = []
    for i in range(0, n_bins):
        name = "UGUA kmer bin " + str(i + 1)
        UGUA.append(name)

    polyU_name = []
    for i in range(0, n_bins):
        name = "Poly-U PWM Hu bin " + str(i + 1)
        polyU_name.append(name)

    polyU_kmers_ = []
    for i in range(0, n_bins):
        name = "Poly-U hex bin " + str(i + 1)
        polyU_kmers_.append(name)

    polyU_kmer_ = []
    for i in range(0, n_bins):
        name = "UUUU bin " + str(i + 1)
        polyU_kmer_.append(name)

    CA_UA = []
    for i in range(0, n_bins):
        name = "CA_UA bin " + str(i + 1)
        CA_UA.append(name)

    bin_names = PAS_estimate + PAS_Hu_ + PAS_Siep + DSE_kmers + DSE_PWM_1_ + DSE_PWM_2_ + DSE_PWM_3_ \
                + DSE_PWM_4_ + UGUA_Hu_ + NUDT_name + UGUA + polyU_name + polyU_kmers_ + polyU_kmer_ + CA_UA

    return np.asarray(bin_names)


##########################################
# Initiate and save
##########################################


def save_features():
    # PAS
    scores1 = calc_affinities_PWM(PAS_PWM)
    scores2 = calc_affinities_PWM(PAS_Hu)
    scores3 = calc_affinities_PWM(PAS_Siepel)

    # DSEs
    scores4 = calc_combined_scores(DSEs)
    scores5 = calc_affinities_PWM(DSE_PWM_1)
    scores6 = calc_affinities_PWM(DSE_PWM_2)
    scores7 = calc_affinities_PWM(DSE_PWM_3)
    scores8 = calc_affinities_PWM(DSE_PWM_4)

    # UGUA
    scores9 = calc_affinities_PWM(UGUA_Hu)
    scores10 = calc_affinities_PWM(NUDT)
    scores11 = motif_scores_all(UGUA_kmers)

    # U-rich elements
    scores12 = calc_affinities_PWM(PolyU)
    scores13 = motif_scores_all(poly_U_kmers)
    scores14 = motif_scores_all(poly_U_kmer)

    # Cleavage Site
    scores15 = calc_combined_scores(CA_UA)

    scores = np.hstack((scores1, scores2))
    scores = np.hstack((scores, scores3))
    scores = np.hstack((scores, scores4))
    scores = np.hstack((scores, scores5))
    scores = np.hstack((scores, scores6))
    scores = np.hstack((scores, scores7))
    scores = np.hstack((scores, scores8))
    scores = np.hstack((scores, scores9))
    scores = np.hstack((scores, scores10))
    scores = np.hstack((scores, scores11))
    scores = np.hstack((scores, scores12))
    scores = np.hstack((scores, scores13))
    scores = np.hstack((scores, scores14))
    scores = np.hstack((scores, scores15))

    names = feature_names()

    with open(output_file_name, "w") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(scores)


save_features()
