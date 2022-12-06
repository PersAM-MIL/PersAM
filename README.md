# Description of code

## PersAM_train.py

Code to learn PersAM. We trained PersAM using 8 GPUs.\
When used, it is necessary to implement a process to normalize clinical_record.

## PersAM_test.py

Code to evaluate PersAM.

## model.py

Code describing the PersAM model.

## transformer_simple.py

Code describing the transformer used in PersAM.

## Dataset.py

Code to create dataset (bags).

## Required Data

./Data : Whole slide images (SVS) \
./csv/[DLBCL,FL,Reactive].csv : files written coordinate of each patch\
./[DLBCL,FL,Reactive].txt : files written slideID of each subtype\
./clinical_record.csv : clinical record data
