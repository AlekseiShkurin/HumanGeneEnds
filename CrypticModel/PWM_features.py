from __future__ import division
import csv
import pandas as pd
import numpy as np
import sys
import os


PWM_list_1 = [fn for fn in os.listdir('../Data/PWMs/CrypticModelPWMs/Dominguez') if fn.endswith('.PWM')]
PWM_list_2 = [fn for fn in os.listdir('../Data/PWMs/CrypticModelPWMs/CISBP_RNA') if fn.endswith('.txt')]

PWM_list = PWM_list_1 + PWM_list_2

file_to_load = '../FASTA_files/CrypticModel/' + sys.argv[1]
output_file_name = sys.argv[1][:-3] + '.csv'

window_length = 140

bin_size = 20
step_len = 10

bins_per_window = int((window_length - bin_size) / step_len)


def load_data(filename):
    data = pd.read_csv(filename, sep='\t', header=None)

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

    return CPA_data


def generate_features(pos_data_sample):
    from Bio.Seq import Seq

    #############################################
    ### BEEML implementation
    #############################################
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

            # TODO def plotMotifLogo(self):
            # try to find out how to do this with weblogo ideally

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

    class DNAseq:
        def __init__(self, ID=None, seq=None, score=None):
            self.ID = ID
            self.seq = seq.upper()
            self.score = score
            self.rcseq = revcomplementSeq(self.seq)

        def __str__(self):
            return ("ID: " + self.ID + '\nSeq: ' + self.seq)

        def calcDNAshape(self):
            if len(self.seq) >= 5:
                self.MGW = calcMGW(self.seq)
                self.Roll = calcRoll(self.seq, self.rcseq)
                self.ProT = calcProT(self.seq)
                self.HelT = calcHelT(self.seq, self.rcseq)
            else:
                print("Sequence too short for DNA shape calculator")


    ################################################################
    # Calculate PWM binding affinity using BEEML
    ################################################################
    def inwin_calc_aff(motif, part):
        # First, get current window

        part_out_one_row = []

        for i in range(0, bins_per_window):
            bin_start = int(i * step_len)
            bin_end = int(bin_start + bin_size)
            current_bin = part[bin_start:bin_end]

            fwd, rev = BEEMLscan(motif.emat, current_bin)
            sumScore = max(fwd)

            part_out_one_row.append(sumScore)

        return part_out_one_row


    def calc_affinities_PWM(instances):
        pq1vec = []

        motif = TFMotif(ID='Cebpb', motif_FPM=instances)

        # To Score sequences with the BEEML method
        motif.convert2EnergyMatrix()

        # Generate scores
        cc = inwin_calc_aff(motif, pos_data_sample)
        pq1vec.append(cc)

        return cc

    ################################################
    # And RBP FPM analysis
    ################################################

    # Function to read PWM from file
    def read_RBP_FPMs(filename):
        FPM_names = pd.read_csv(filename, delimiter='\t', header=0, index_col=0)
        FPM_names = np.asarray(FPM_names)

        return FPM_names


    ####################################################################
    #####    Initiate and save   #####
    ####################################################################


    def save_features():
        output_line = []
        skipped_PWMs = []
        for PWM_ in PWM_list:
            pwm_file = 'PWMs/' + PWM_
            PWM = read_RBP_FPMs(pwm_file)
            try:
                output_line.extend(calc_affinities_PWM(PWM))
            except ValueError:
                skipped_PWMs.append(PWM_)

        return output_line, skipped_PWMs

    scores_all, skipped_PWMs = save_features()

    return scores_all, skipped_PWMs


################################################################
#                         EXECUTION
################################################################
fasta_data = load_data(file_to_load)

resulting_scores = []
for sample in fasta_data:
    scores, skipped_PWMs = generate_features(sample)

    resulting_scores.append(scores)


def feature_names():
    PWM_names = []
    for PWM_ in PWM_list:

        if PWM_ not in skipped_PWMs:
            for i in range(0, bins_per_window):
                name = PWM_ + '_' + str(i + 1)
                PWM_names.append(name)
        else:
            print(PWM_)
            continue

    return np.asarray(PWM_names)


headers = feature_names()


with open(output_file_name, "w") as f:
    w = csv.writer(f)
    w.writerow(headers)
    w.writerows(resulting_scores)
