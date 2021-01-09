# HumanGeneEnds

The scripts and files in this repository allow you to locally train the baseline and cryptic models described in : Shkurin, A,Go t and Hughes, TR, "Known sequence features can explain half of all human gene ends"     

NOTE: in case of the baseline CPA classifier, the resulting model will be around 20 GB in size, so make sure you have enough memory  

Below are the necessary steps to train each of the models

## Baseline CPA classifier

### Data preparation

Go to the BaselineModel directory and run the `Generate_baseline_features.py` file first. There are 4 FASTA files to be processed: Dominant_CPA_testing.fa, Dominant_CPA_training.fa, Negative_testing.fa, Negative_training.fa  

For each of the files submit a separate job, an example is shown below  

```
python Generate_baseine_features.py Dominant_CPA_training.fa
python Generate_baseine_features.py Negative_training.fa
python Generate_baseine_features.py Dominant_CPA_testing.fa
python Generate_baseine_features.py Negative_testing.fa

```

It is strongly recommended to parallelize this part of the analysis as calculating all the features for each of the training samples might take substantial amount of time. Ideally, you would use a computing cluster  

NOTE: the negative datasets are reduced to 80,000 samples to comply with GitHub file size limit (50MB). You can also train the model on complete negative dataset (training set 53MB and testing set 200MB) which can be downloaded here - https://hugheslab.ccbr.utoronto.ca/supplementary-data/HumanGeneEnds/

### Model training

Now, you can run the `Train_Random_Forest.py`, from the same directory. Pass each of the datasets created during the previous step as arguments. The order is: positive training dataset, negative training, positive testing, negative testing. An example is shown below

```
python Train_Random_Forest.py Dominant_CPA_training.csv Negative_training.csv Dominant_CPA_testing.csv Negative_testing.csv
```

Again, as with the data preparation step, it is recommended to use a computing cluster  

### Making predictions
Finally, to make predictions on new data using the resulting model, you can use `Predict_Random_Forest.py`  

The model is restricted to sequences of size 500 nt. First, put your FASTA file into FASTA_files/BaselineModel folder (or modify the script accordingly to load data from different directory) and then pass it into `Generate_baseine_features.py`, like shown below

```
python Generate_baseine_features.py TEST.fa
```

Then, pass the resulting file into the `Predict_Random_Forest.py`  

```
python Predict_Random_Forest.py TEST.csv
```

That will create a file with model predictions for each of the samples in your original FASTA file

## Cryptic CPA classifier

### Data preparation
Go into the CrypticModel directory. `PWM_features.py` file calculates binding likelihoods of all known human RBPs. Files `U1_MaxEntScan.py` and `U1_RNAhybrid.py` calculate U1 binding likelihood using and implementation of MaxEntScan and RNAhybrid scan dsigned specifically for this project (using pre-trained dictionaries of k-mers)  

First, run the `PWM_features.py` with each of the 4 datasets: Dominant_CPA_training.fa, Cryptic_CPA_training.fa, Dominant_CPA_testing.fa, Cryptic_CPA_testing.fa, like shown below

```
python PWM_features.py Dominant_CPA_training.fa
python PWM_features.py Cryptic_CPA_training.fa
python PWM_features.py Dominant_CPA_testing.fa
python PWM_features.py Cryptic_CPA_testing.fa
```

Like with the baseline model, it is recommended to parallelise this process and use a computing cluster  

Next, follow the same procedure with U1 feature scripts. Pass each of the 4 datasets in the same way as described above into `U1_MaxEntScan.py` and `U1_RNAhyb.py`  

If you calculated the U1 features and would like to add them to the PWM feature space, use `Add_U1_features.py` passing all 3 datasets together like shown below  

```
python Add_U1_features.py Dominant_CPA_training.csv Dominant_CPA_training_MaxEnt.csv Dominant_CPA_training_RNAhyb.csv
python Add_U1_features.py Cryptic_CPA_training.csv Cryptic_CPA_training_MaxEnt.csv Cryptic_CPA_training_RNAhyb.csv
python Add_U1_features.py Dominant_CPA_testing.csv Dominant_CPA_testing_MaxEnt.csv Dominant_CPA_testing_RNAhyb.csv
python Add_U1_features.py Cryptic_CPA_testing.csv Cryptic_CPA_testing_MaxEnt.csv Cryptic_CPA_testing_RNAhyb.csv
```

### Training the model
To train the model, run the `Train_Logistic_Regression.py` file passing the data in the following order: positive training dataset, negative training, positive testing, negative testing. An example is shown below

```
python Train_Logistic_Regression.py Dominant_CPA_training_U1.csv Cryptic_CPA_training_U1.csv Dominant_CPA_testing_U1.csv Cryptic_CPA_testing_U1.csv
```

The resulting model will be saved in the same directory
