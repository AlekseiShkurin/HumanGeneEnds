import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import sys
import pickle

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


dominant_testing = sys.argv[1]
dominant_training = sys.argv[2]
negative_training = sys.argv[3]
negative_testing = sys.argv[4]

outputfile = 'BaselineModel.pkl'
output_roc = 'BaselineModel_rocauc.csv'
output_rec = 'BaselineModel_prerec.csv'

pict_name_rocauc = 'BaselineModel_rocauc.png'
pict_name_prerec = 'BaselineModel_prerec.png'

n_trees = 30000


#########################
# LOAD DATA
#########################
negative_training_ = pd.DataFrame()
for chunk in pd.read_csv(negative_training, header=0, sep=',', chunksize=1000):
    negative_training_ = pd.concat([negative_training_, chunk], ignore_index=False)

negative_testing_ = pd.DataFrame()
for chunk in pd.read_csv(negative_testing_, header=0, sep=',', chunksize=1000):
    negative_testing_ = pd.concat([negative_testing_, chunk], ignore_index=False)

dominant_training_ = pd.DataFrame()
for chunk in pd.read_csv(dominant_training, header=0, sep=',', chunksize=1000):
    dominant_training_ = pd.concat([dominant_training_, chunk], ignore_index=False)

dominant_testing_ = pd.DataFrame()
for chunk in pd.read_csv(dominant_testing, header=0, sep=',', chunksize=1000):
    dominant_testing_ = pd.concat([dominant_testing_, chunk], ignore_index=False)

headers = list(dominant_testing_)


# Generate labels
dominant_training_labels = np.array([1] * len(dominant_training_))
dominant_testing_labels = np.array([1] * len(dominant_testing_))
negative_training_labels = np.array([0] * len(negative_training_))
negative_testing_labels = np.array([0] * len(negative_testing_))

dominant_training_labels = dominant_training_labels.reshape(len(dominant_training_labels), 1)
dominant_testing_labels = dominant_testing_labels.reshape(len(dominant_testing_labels), 1)
cryptic_training_labels = negative_training_labels.reshape(len(negative_training_labels), 1)
cryptic_testing_labels = negative_testing_labels.reshape(len(negative_testing_labels), 1)

training_labels = np.concatenate((dominant_training_labels, cryptic_training_labels), axis=None)
testing_labels = np.concatenate((dominant_testing_labels, cryptic_testing_labels), axis=None)

# Preprocess data
dominant_training_array = dominant_training_.values
dominant_testing_array = dominant_testing_.values
cryptic_training_array = negative_training_.values
cryptic_testing_array = negative_testing_.values

joint_training = np.vstack((dominant_training_array, cryptic_training_array))
joint_test = np.vstack((dominant_testing_array, cryptic_testing_array))

scaler = StandardScaler()

scaler.fit(joint_training)
pickle.dump(scaler, open('BaselineScaler.pkl', 'wb'))

scaled_training = scaler.transform(joint_training)
scaled_test = scaler.transform(joint_test)


def plot_rf_rocauc(fpr_rf, tpr_rf, roc_auc):
    plt.figure(1)
    plt.plot(fpr_rf, tpr_rf, label='(AUROC = %0.3f)' % roc_auc)
    plt.title('AUROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(pict_name_rocauc)


# Plot precision-recall
def plot_rf_prerec(precision, recall, average_precision):
    plt.figure(2)
    lw = 2
    plt.plot(recall, precision, lw=lw,
             label='Av.PREC = %0.3f' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUCPR')
    plt.legend(loc="upper right")
    plt.savefig(pict_name_prerec)


####################
# Train the model
####################
def fit_rf():
    # Define classifier
    rf = RandomForestClassifier(n_estimators=n_trees, oob_score=True, min_samples_split=5, class_weight="balanced",
                                n_jobs=-1)

    # Train forest
    rf.fit(scaled_training, training_labels)

    # Make predictions
    rf_prediction = rf.predict_proba(scaled_test)[:, 1]

    # Evaluate performance
    fpr_rf, tpr_rf, _ = roc_curve(testing_labels, rf_prediction)
    roc_auc = auc(fpr_rf, tpr_rf)

    # Plot and save
    plot_rf_rocauc(fpr_rf, tpr_rf, roc_auc)

    # Now, calculate and return pre/rec parameters
    precision, recall, _ = precision_recall_curve(testing_labels, rf_prediction)
    average_precision = average_precision_score(testing_labels, rf_prediction)

    # Plot and save
    plot_rf_prerec(precision, recall, average_precision)

    return rf, fpr_rf, tpr_rf, roc_auc, precision, recall, average_precision


# Save the model
def save_rf(rf, filename):
    pickle.dump(rf, open(filename, 'wb'))


rf, fpr_rf, tpr_rf, roc_auc, precision, recall, average_precision = fit_rf()

roc_auc_1 = [roc_auc, 0]
average_precision_1 = [average_precision, 0]

save_rf(rf, outputfile)

# Save the rocauc
with open(output_roc, "w") as f:
    w = csv.writer(f)
    w.writerow(fpr_rf)
    w.writerow(tpr_rf)
    w.writerow(roc_auc_1)

# Save the prerec
with open(output_rec, "w") as f:
    w = csv.writer(f)
    w.writerow(precision)
    w.writerow(recall)
    w.writerow(average_precision_1)

