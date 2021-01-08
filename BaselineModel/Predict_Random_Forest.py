import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle


file_to_test = sys.argv[1]

# Load model and scaler
with open('BaselineModel.pkl', 'rb') as fp:
    RF_classifier = pickle.load(fp)

with open('BaselineScaler.pkl', 'rb') as fp:
    scaler = pickle.load(fp)

# Load data to be tested
data_testing = pd.DataFrame()
for chunk in pd.read_csv(file_to_test, header=0, sep=',', chunksize=1000):
    data_testing_ = pd.concat([data_testing, chunk], ignore_index=False)

data_scaled = scaler.transform(data_testing)

# Make prediction
rf_prediction = RF_classifier.predict_proba(data_scaled)[:, 1]

# Save results
with open('BaselinePredictions.csv', "w") as f:
    w = csv.writer(f)
    w.writerow(rf_prediction)
