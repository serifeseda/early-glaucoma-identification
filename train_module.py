"""
Definitions of training functions called in main.py
"""
# Copyright (C) 2018  Serife Seda Kucur

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os, sys, csv
import numpy as np
import pickle
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend
from keras.models import load_model
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from utils import *
def call_cnn_model():
    """
    Compiles the CNN model to be trained
    Outputs:
        model: Compiled Keras model
    """

    # Model configuration
    l_in = Input(shape = (61,61))
    l_model = Conv2D(4, 3, padding='valid', kernel_initializer = 'glorot_uniform', activation = 'relu')(l_in)
    l_model = BatchNormalization()(l_model)
    l_model = Conv2D(4, 3, padding='valid', kernel_initializer = 'glorot_uniform', activation = 'relu')(l_model)
    l_model = BatchNormalization()(l_model)
    l_model = Conv2D(4, 3, padding='valid', kernel_initializer = 'glorot_uniform', activation = 'relu')(l_model)
    l_model = MaxPooling2D()(l_model)
    l_model = BatchNormalization()(l_model)
    l_model = Conv2D(4, 3, padding='valid', kernel_initializer = 'glorot_uniform', activation = 'relu')(l_model)
    l_model = BatchNormalization()(l_model)
    l_model = Conv2D(4, 3, padding='valid', kernel_initializer = 'glorot_uniform', activation = 'relu')(l_model)
    l_model = BatchNormalization()(l_model)
    l_model = MaxPooling2D()(l_model)
    l_model  = GlobalAveragePooling2D()(l_model)
    l_model = BatchNormalization()(l_model)
    l_model = Dense(32, activation = 'relu')(l_model)
    l_model = BatchNormalization()(l_model)
    l_model = Dense(32, activation = 'relu')(l_model)
    l_model = Dropout(0.5)(l_model)
    l_out  = Dense(1, activation = 'sigmoid')(l_model)

    # Build model
    model = Model(inputs = l_in, outputs = l_out)

    # Optimizer
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    # Compile the model
    model.compile(optimizer = opt, loss = 'binary_crossentropy')
   
    return model

def call_nn_model(input_dim):
    """
    Compiles the NN model to be trained. The model is fixed (described in the paper.)
    Inputs:
        input_dim: The dimensions of the input layer
    Outputs:
        model: Compiled model
    """
    
    # Model
    l_in = Input(shape = (input_dim,))
    l_model = Dense(32, activation = 'relu')(l_in)
    l_model = BatchNormalization()(l_model)
    l_model = Dropout(0.5)(l_model)
    l_model = Dense(32, activation = 'relu')(l_model)
    l_model = BatchNormalization()(l_model)
    l_model = Dropout(0.5)(l_model)
    l_out  = Dense(1, activation = 'sigmoid')(l_model)
     
    # Build the model
    model = Model(inputs = l_in, outputs = l_out)

    # Optimizer
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    # Compile mthe model
    model.compile(optimizer = opt, loss = 'binary_crossentropy')
   
    return model

def train_cnn(dataset, dirname):
    """
    Trains the CNN model (10-fold cross-validation)
    Inputs:
        dataset: Data set name,'BD' or 'RT'
        experiment_file: A text file describing the CNN model architecture
        dirname : The direcotry into which ther results will be put
        datafile: The training/test splits to be used. If not, it uses default one. This data file is an object with attributes 
                 related to trainining, test and validation splits (see  create_folds function and Data_Fold class in utils.py)
        lighter_cnn: If True, this has less number of hidden units in the top layers of the CNN (see construct_cnn_model)
    Outputs: Saves the best model and the average precision scores for baselines and CNN for each fold
    """
    # Set nuber of oflds and runs for each fold
    num_folds = 10
    num_runs  = 5
    
    # Create folder
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    # Load data splits
    data = load_object('./data/{}_data_folds.pkl'.format(dataset))

    # This matrix keeps the precisions for baselines and  CNN
    precs      = np.zeros((num_folds, num_runs, 4))
    precs_val  = np.zeros((num_folds, num_runs, 4))

    # Determine how much data will be augmented to balance the imbalanced data
    if dataset == 'BD':
        cl = 1
        n_repeat = 2
    elif dataset == 'RT':
        cl = 0
        n_repeat = 9
        
    # Training and validation by cross-validation
    for i in range(num_folds):

        # Get data splits for the current fold
        tr_input    = data[i].tr_input_imgs
        val_input   = data[i].val_input_imgs
        test_input  = data[i].test_input_imgs
        tr_output   = data[i].tr_output
        val_output  = data[i].val_output
        test_output = data[i].test_output
        test_md     = data[i].test_md
        test_slv    = data[i].test_slv
        val_md      = data[i].val_md
        val_slv     = data[i].val_slv

        # Augment the data
        tr_input, tr_output = resample_data(tr_input, tr_output, cl, n_repeat)

        # Standardize the data (0-mean, unit-variance)
        mean_tr = np.mean(tr_input)
        std_tr  = np.std(tr_input)
        tr_input = (tr_input - mean_tr)/std_tr
        val_input = (val_input - mean_tr)/std_tr
        test_input = (test_input - mean_tr)/std_tr
        
        # Compute baselines
        precs[i, :, 0] = average_precision_score(test_output, test_md)
        precs[i, :, 1] = average_precision_score(test_output, test_slv)
        precs[i, :, 2] = combine_global_indices(test_md, test_slv, test_output)

        for run in range(num_runs): 
                            
            # Model config
            model = call_cnn_model()

            # Print the number of sampels in each class after data augmentation
            print('Labels-1 : {:d}, labels-0 : {:d}'.format(int(tr_output.sum()), int((tr_output==0).sum())))

            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
            model_name  = '{}/best_model_config_no_{:d}_fold_{:d}_run_{:d}.h5'.format(dirname, model_no, i, run)
            check_it_out = ModelCheckpoint(model_name, save_best_only=True)

            # TRAIN
            model.fit(tr_input,tr_output, \
                            validation_data = (val_input, val_output),\
                            epochs = 50, \
                            batch_size = 32, \
                            shuffle = True, \
                            callbacks =  [early_stop, check_it_out])
            
            # Evaluate best model
            model_best = load_model(model_name)
            pred_cnn   = model_best.predict(test_input, batch_size = 32)
            precs[i, run, 3] = average_precision_score(test_output,  pred_cnn) 

            # Write the results to the summary file
            F = open('{}/result_summary.txt'.format(dirname), mode = 'a')
            result_sum = 'Fold {:d}: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} -- CNN: {:.3f}\n'.format(i,
                                        precs[i,run, 0], precs[i,run, 1],precs[i, run, 2], precs[i, run, 3])
            F.write(result_sum)
            F.close()
            
            print('Fold {:d}: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} --  CNN: {:.3f}'.format(i,
                                precs[i,run,0], precs[i,run,1], precs[i,run,2], precs[i, run, 3]))                 

            # Clear graph ??
            backend.clear_session()
            ####### Run iteration ended
        
        ########## Fold iteration ended   

        # Write last results
        precs_med = np.median(precs.reshape(num_runs * num_folds, 4), 0)
        precs_std = np.std(precs.reshape(num_runs * num_folds, 4), 0)

        F = open('{}/result_summary.txt'.format(dirname), mode = 'a')
        result_sum_med = 'Medians: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} -- CNN: {:.3f}\n'.format(
                                            precs_med[0], precs_med[1], precs_med[2], precs_med[3])
        result_sum_std = 'STDevs: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} -- CNN: {:.3f}\n'.format(
                                            precs_std[0], precs_std[1], precs_std[2], precs_std[3])
        F.write(result_sum_med)
        F.write(result_sum_std)
        F.close()
        
        # Save the results into npz file
        np.savez('{}/ap_scores_config_no_{:d}.npz'.format(dirname, model_no), precs = precs)

        # Plot results
        fig_x_labels = ['MD', 'SLV', 'MD+SLV', 'CNN'] 
        plt.figure()
        plt.boxplot(precs.reshape(num_runs * num_folds, 4), labels = fig_x_labels)
        plt.grid()
        plt.savefig('{}/ap_performance_config_no_{:d}'.format(dirname, model_no))
        # plt.show()
        plt.close()

        ########### model iteration ended

        return 0 

def train_nn(dataset, dirname, datafile=None, lighter_nn=False):
    """
    Trains the NN model (10-fold cross-validation)
    Inputs:
    dataset: Data set name,'BD' or 'RT'
    dirname : The direcotry into which ther results will be put
    datafile: The training/test splits to be used. If not, it uses default one. This data file is an object with attributes 
                related to trainining, test and validation splits (see  create_folds function and Data_Fold class in utils.py)
    lighter_nn: If True, this has less number of hidden units in the top layers of the CNN (see construct_cnn_model)
    Outputs: Saves the best model and the average precision scores for baselines and CNN for each fold
    """
    # Set 
    num_folds = 10
    num_runs  = 5

    # Create folder
    if not os.path.exists(dirname):
    os.makedirs(dirname)

    # Load data splits
    data = load_object('./data/{}_data_folds.pkl'.format(dataset))

    # This matrix keeps the precisions for baselines and  CNN
    precs     = np.zeros((num_folds, num_runs, 4))

    # Determine how much data will be augmented to balance the imbalanced data
    if dataset == 'BD':
        cl = 1
        n_repeat = 2
    elif dataset == 'RT':
        cl = 0
        n_repeat = 9

    for i in range(num_folds):

        # Get the data splits
        tr_input    = data[i].tr_input_vf
        val_input   = data[i].val_input_vf
        test_input  = data[i].test_input_vf
        tr_output   = data[i].tr_output
        val_output  = data[i].val_output
        test_output = data[i].test_output
        test_md     = data[i].test_md
        test_slv    = data[i].test_slv

        # Augment the data
        tr_input, tr_output = resample_data(tr_input, tr_output, cl, n_repeat)

        # Standardize
        mean_tr = np.mean(tr_input)
        std_tr  = np.std(tr_input)
        tr_input = (tr_input - mean_tr)/std_tr
        val_input = (val_input - mean_tr)/std_tr
        test_input = (test_input - mean_tr)/std_tr

        # Compute baselines
        precs[i, :, 0] = average_precision_score(test_output, test_md)
        precs[i, :, 1] = average_precision_score(test_output, test_slv)
        precs[i, :, 2] = combine_global_indices(test_md, test_slv, test_output)

        for run in range(num_runs): 
                            
            # Model config
            model = call_nn_model(test_input.shape[1])
            
            # Print model parameters
            print(model.count_params())

            # Print the number of sampels in each class after data augmentation
            print('Labels-1 : {:d}, labels-0 : {:d}'.format(int(tr_output.sum()), int((tr_output==0).sum())))
            
            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
            model_name  = '{}/best_model_fold_{:d}_run_{:d}.h5'.format(dirname,i,run)
            check_it_out = ModelCheckpoint(model_name, save_best_only=True)

            # TRAIN
            model.fit(tr_input,tr_output, \
                            validation_data = (val_input, val_output),\
                            epochs = 50, \
                            batch_size = 32, \
                            shuffle = True, \
                            callbacks =  [early_stop, check_it_out])

            # Evaluate the best model
            model_best = load_model(model_name)
            pred_cnn   = model_best.predict(test_input, batch_size = 32)
            precs[i, run, 3] = average_precision_score(test_output,  pred_cnn) 
            
            # Write the results to the summary file
            F = open('{}/result_summary.txt'.format(dirname), mode = 'a')
            result_sum = 'Fold {:d}: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} -- NN: {:.3f}\n'.format(i,
                                        precs[i,run, 0], precs[i,run, 1],precs[i, run, 2], precs[i, run, 3])
            F.write(result_sum)
            F.close()
            print('Fold {:d}: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} --  NN: {:.3f}'.format(i,
                                precs[i,run,0], precs[i,run,1], precs[i,run,2], precs[i, run, 3])) 
            
            # Clear graph ??
            backend.clear_session()
            ####### Run iteration ended

        ########## Fold iteration ended    

    # Write last results
    precs_med = np.median(precs.reshape(num_runs * num_folds, 4), 0)
    precs_std = np.std(precs.reshape(num_runs * num_folds, 4), 0)
    F = open('{}/result_summary.txt'.format(dirname), mode = 'a')
    result_sum_med = 'Medians: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} -- NN: {:.3f}\n'.format(
                                        precs_med[0], precs_med[1], precs_med[2], precs_med[3])
    result_sum_std = 'Stds: MD: {:.3f} -- SLV: {:.3f} -- MD+SLV: {:.3f} -- NN: {:.3f}\n'.format(
                                        precs_std[0], precs_std[1], precs_std[2], precs_std[3])
    F.write(result_sum_med)
    F.write(result_sum_std)
    F.close()

    # Save the results into npz file
    np.savez('{}/ap_scores.npz'.format(dirname), precs = precs)

    # Plot results
    fig_x_labels = ['MD', 'SLV', 'MD+SLV', 'NN'] 
    plt.figure()
    plt.boxplot(precs.reshape(num_runs * num_folds, 4), labels = fig_x_labels)
    plt.grid()
    plt.savefig('{}/ap_performance'.format(dirname))
    # plt.show()
    plt.close()

    return 0
    
 