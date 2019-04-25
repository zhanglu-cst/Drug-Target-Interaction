# Drug Target Interaction Predict -- DTI
----
A Dango based Web Platform for drug target interaction prediction.

The Platform can predict whether the protien sequence and the drug SMILES will be interact with each other.



## Train 
Before you use, you need to train some models.

Due to the file size limit of github, the trained model files can not upload to github, so, you need to train it yourself.

It is recommended that you open the root directory with PyCharm IDE, then run the train progress in PyCharm. Otherwith, you need to deal with some path problem.
Specifically, you may need to add the root directory to package search path by sys.path.append()

You need to run the following files to finish the training progress.
* `./train_model/train_encoder_D/Encoder_D_train.py`
* `./train_model/train_encoder_P/Encoder_P_train.py`
* `./train_model/train_NN/NN_classify_train.py`

NOTE: The NN_classify_train.py should run after the Encoder_D_train.py and Encoder_P_train.py



## Usage

`
python3 manage.py runserver 127.0.0.1:8000
`

Then open the chrome browser, and get to 127.0.0.1:8000

## Dataset
The dataset is downloaded from [DrugBank](https://www.drugbank.ca).   

Data files used in training process is in the data_process_to_fingerprint directory.


## Description of the directory
* API   
The directory contains some API of the model, it is directly called by Django APP.
* data_process_to_fingerprint   
The directory contains all of the dataset, and some function that transfer the origin drug SMILES and pretion sequence to fingerprints.
* DrugTargetInteractionWebPlatform, static, templates, APP    
All of them are some necessary component of Django.
* model_files   
The directory that store the model files
* train_model   
The code that train different component of our system.




### The work is based on the following papers:
- Wang, L , et al. "A Computational-Based Method for Predicting Drug-Target Interactions by Using Stacked Autoencoder Deep Neural Network." Journal of Computational Biology A Journal of Computational Molecular Cell Biology 25.3(2017).
- Wen, Ming , et al. "Deep-Learning-Based Drug–Target Interaction Prediction." Journal of Proteome Research 16.4(2017):1401-1409.
- Peng-Wei, K. C. C. Chan , and Z. H. You . "Large-scale prediction of drug-target interactions from deep representations." 2016 International Joint Conference on Neural Networks (IJCNN) IEEE, 2016.
