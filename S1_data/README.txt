Last edited: 23.08.2018
Author: Serife Seda Kucur (serife.kucur@artorg.unibe.ch)

Information regarding data the following files included in this folder (data):

BD_dataset.csv: The data set referred as BD in the paper
BD_training_patient_ids.csv: Patient identities used in 10 training sets for BD data set
BD_validation_patient_ids.csv: Patient identities used in 10 validation sets for BD data set
BD_test_patient_ids.csv: Patient identities used in 10 test sets for BD data set 
RT_dataset: The data set referred as RT in the paper
RT_training_patient_ids.csv: Patient identities used in 10 training sets for RT data set
RT_validation_patient_ids.csv: Patient identities used in 10 validation sets for RT data set
RT_test_patient_ids.csv: Patient identities used in 10 test sets for RT data set 

More detailed information for each file is below:

#####################################################################################################
BD_dataset.csv/RT_dataset.csv

Each row corresponds to information related to one examination (one visual field).
Columns are described below:
PATIENT ID	: The identity number of the corresponding patient
EYE ID		: The identity number of the eye. 0 for right eye, 1 for left eye.
MD		: Mean Defect of the visual field
sLV            : Square root of loss variance
GLAUCOMATOUS   : The diagnosis of the corresponding visual field. 1 if glaucomatous, 0 otherwise
X_d		: the x coordinate of the dth visual field location, d being the location id number
Y_d		: the y coordinate of the dth visual field location, d being the location id number
VF_d		: the visual field value of the dth location, i.e. age-normalized threshold value
		(absolute threshold - age-matched normal value), d being the location id number

#####################################################################################################

BD_training_patient_ids.csv/RT_training_patient_ids.csv

Each row corresponds to the patient identities in training set (referred as PATIENT ID in BD_dataset.csv/RT_dataset.csv) 
of each fold in 10-fold cross validation.

#####################################################################################################

BD_validation_patient_ids.csv/RT_validation_patient_ids.csv

Each row corresponds to the patient identities in validation set (referred as PATIENT ID in BD_dataset.csv/RT_dataset.csv) 
of each fold in 10-fold cross validation.

#####################################################################################################

BD_test_patient_ids.csv/RT_test_patient_ids.csv

Each row corresponds to the patient identities in test set (referred as PATIENT ID in BD_dataset.csv/RT_dataset.csv) 
of each fold in 10-fold cross validation.

For more information, please contact serife.kucur@artorg.unibe.ch
