"""
Some auxiliary functions
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


import sys 
from ast import literal_eval
import pickle
import numpy as np
import pandas as pd
import csv
from scipy.misc import imresize
from keras.callbacks import EarlyStopping, Callback, LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.core import Dropout
from keras.layers import Input, Dense, Flatten, Conv2D, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Concatenate, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from keras import backend
from keras.models import load_model 
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

class Data_Fold:
    """
    Data class with attributes below. 
    This puts in a structured way the raw data and images for one fold.
    """
    tr_input_imgs   = []
    val_input_imgs  = []
    test_input_imgs = []
    tr_input_vf     = []
    val_input_vf    = []
    test_input_vf   = []
    test_md  = []
    test_slv = []
    tr_output   = []
    val_output  = []
    test_output = []
    train_pat_idents = []
    val_pat_idents   = []
    test_pat_idents  = []
    train_idx        = []
    val_idx          = []
    test_idx         = []
def save_object(obj, filename):
    """
    Saves an object ot a file
    Inputs:
        obj: Object to write into the file
        filename: The filename to write into
    Outputs:
        File with filename
    """        
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def load_object(filename):
    """
    Load the object in a pickle file into the environment
    Inputs: 
        filename: the filename were object to load was saved
    Outputs:
        obj: the object of interest, can be a list of objects
    """
    with open(filename, 'rb') as input:
        obj = pickle.load(input)

    return obj

def create_data(dataset):
    """
    Creates a data object including all data folds used in training, validation 
    and test.
    Inputs:
        dataset: Dataset name. 'BD' for Budapest dataset, 'RT' for Rotterdam dataset
    Outputs:
        data_folds: Data fold object (see Data_Fold class)
    """

    if dataset == 'BD':
        num_locs = 38
    elif dataset == 'RT':
        num_locs == 54
    else:
        print('No dataset is available with that name!')
    
    df = pd.read_csv('./S1_data/{}_dataset.csv'.format(dataset))
    patient_id = np.asarray(df['PATIENT ID'])
    eye_id     = np.asarray(df['EYE ID'])[:,np.newaxis]
    md         = np.asarray(df['MD'])[:,np.newaxis]
    slv        = np.asarray(df['sLV'])[:,np.newaxis]
    labels     = np.asarray(df['GLAUCOMATOUS'])[:,np.newaxis]
    vf_data = np.zeros((len(labels), num_locs))
    x = np.zeros((len(labels), num_locs))
    y = np.zeros((len(labels), num_locs))
    for k in range(num_locs):
        vf_data[:, k] = np.asarray(df['VF_{:d}'.format(k+1)])
        x[:,k] = np.asarray(df['X_{:d}'.format(k+1)])
        y[:,k] = np.asarray(df['Y_{:d}'.format(k+1)])

    # Append global indices to VF data matrix
    vf_data = np.concatenate([vf_data, md, slv], axis=1)

    # Map all coordinates to right eye format
    x[eye_id[:,0]==1, :] = -1 * x[eye_id[:,0]==1, :]

    # Create voronoi images
    xy = np.concatenate([x[0, np.newaxis].T, y[0, np.newaxis].T], axis=1)
    vor_images = generate_voronoi_images_given_image_size(vf_data, xy)

    # Read training, validation and test splits
    tr_pat_ids = []
    val_pat_ids = []
    test_pat_ids = []
    with open('./S1_data/{}_training_patient_ids.csv'.format(dataset), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            tr_pat_ids.append(np.asarray(row).astype('int'))

    with open('./S1_data/{}_validation_patient_ids.csv'.format(dataset), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            val_pat_ids.append(np.asarray(row).astype('int'))

    with open('./S1_data/{}_test_patient_ids.csv'.format(dataset), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            test_pat_ids.append(np.asarray(row).astype('int'))

    # Create data object with data splits
    data_folds = []
    num_folds = 10
    for f in range(num_folds):

        train_idx = np.isin(patient_id, tr_pat_ids[f])
        val_idx = np.isin(patient_id, val_pat_ids[f])
        test_idx = np.isin(patient_id, test_pat_ids[f])

        data_fold = Data_Fold()

        # Voronoi images
        data_fold.tr_input_imgs   = vor_images[train_idx]
        data_fold.val_input_imgs  = vor_images[val_idx]
        data_fold.test_input_imgs = vor_images[test_idx]
        
        # Linear VF vectors (total deviation)
        data_fold.tr_input_vf      = vf_data[train_idx]
        data_fold.val_input_vf     = vf_data[val_idx]
        data_fold.test_input_vf    = vf_data[test_idx]
        
        # MD, SLV
        data_fold.train_md = md[train_idx]
        data_fold.val_md   = md[val_idx]
        data_fold.test_md  = md[test_idx]
        data_fold.train_slv = slv[train_idx]
        data_fold.val_slv   = slv[val_idx]
        data_fold.test_slv  = slv[test_idx]

        # Coordinates
        data_fold.xy       = xy

        # Labels
        data_fold.tr_output   = labels[train_idx]
        data_fold.val_output  = labels[val_idx]
        data_fold.test_output = labels[test_idx]
        
        # Patient and samples indices
        data_fold.train_pat_idents = patient_id[train_idx]
        data_fold.val_pat_idents   = patient_id[val_idx]
        data_fold.test_pat_idents  = patient_id[test_idx]
        data_fold.train_idx        = train_idx.copy()
        data_fold.val_idx          = val_idx.copy()
        data_fold.test_idx         = test_idx.copy()
        
        data_folds.append(data_fold)

    return data_folds
def compute_metrics(labels, predictions, fnr_given=0.05):
    """ Finds area under roc curve, average prec.score and false positive rate
    at a given false negative rate (fnr_given).
    Inputs:
        labels: True labels
        predictions: The classifier score (probability)
        fnr_given: (optional) The false negative rate at which false positive rate will be computed.
    Outputs:
        roc_auc: Area under curve under ROC curve
        av_prec_score: Average precision score
        fpr: False positive rate at given <fnr>
    """ 
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1) 
    fnr = 1 - tpr
    fpr_at_given_fnr = np.max(fpr[np.where(fnr>fnr_given)[0]])

    roc_auc = roc_auc_score(labels, predictions)
    pr_auc  = average_precision_score(labels, predictions) 
    
    return roc_auc, pr_auc, fpr_at_given_fnr
def combine_global_indices(md, slv, labels):
    """ combine_global_indices.
    computes evare precision score, area under roc curve and false positive rate 
    at a given false negative rate for combination of md and slv.
    Inputs: 
        md : mean deviation (or mean defect)
        slv: square-root of loss variance 
        labels: true labels
    Outputs:
        av_prec_score: average precision score
    """
    N = len(md)
    
    md = md.flatten()
    slv = slv.flatten()
    labels = labels.flatten()
    md_th_vec = np.linspace(md.min(), md.max(), 50)
    slv_th_vec = np.linspace(slv.min(), slv.max(), 50)
    md_c, slv_c = np.meshgrid(md_th_vec, slv_th_vec)
    md_c = md_c.flatten()
    slv_c = slv_c.flatten()
    
    tpr = np.zeros(len(slv_c))
    fpr = np.zeros(len(slv_c))  
    tnr = np.zeros(len(slv_c))  
    fnr = np.zeros(len(slv_c))  
    prec = np.zeros(len(slv_c))  
    recall = np.zeros(len(slv_c))  
    fnr = np.zeros(len(slv_c))  
    for k, (md_th, slv_th) in enumerate(zip(md_c, slv_c)):

        pred = (md > md_th) & (slv > slv_th)
        if pred.sum() == 0:
            prec[k] = 0 
        else:
            prec[k] = pred[labels==1].sum()/(pred==1).sum()
        recall[k] = pred[labels==1].sum()/(labels==1).sum()
  
    idx = np.argsort(recall)
    recall = np.sort(recall)
    prec = prec[idx]
 
    av_prec_score = np.sum([ prec[k] * (recall[k] - recall[k-1]) for k in range(1, len(recall))])

    return av_prec_score
def resample_data(imgs,labels,cl,n):
    """
    This resamples (copies) the samples of class <cl> and augments data.
    Inputs: 
        imgs: Voronoi images
        labels: Corresponding labels
        cl: Class (0 or 1) to be augmented
        n: The number of sampling for 'one' sample 
    Outputs: 
        imgs: The augmented images
        labels: The labels augmented corresponding to the images.

    """
    idx = (labels == cl).squeeze()
    imgs_p = imgs[idx]
    if len(imgs.shape) > 2:
        imgs_p = np.tile(imgs_p, (n,1,1,1)) 
    else:
        imgs_p = np.tile(imgs_p, (n,1))
    imgs = np.concatenate((imgs, imgs_p), axis=0)  
    labels_p = labels[idx]
    labels_p = np.tile(labels_p, (n,1))   
    labels = np.concatenate((labels, labels_p), axis=0)
    
    return imgs, labels
def generate_voronoi_images_given_image_size(data, xy_coordinates, image_size=(61,61)):

    """ 
    Generates voronoi images using given values for patching colors and given coordiantes for seed points.
    Here image size is fixed or given by the user.
    Output conventions according to Theano CNN implemntation (4D tensor -- (number_of_samples, number_of_channels, image_row, image_col))
    Inputs: 
        data: NxL data matrix, N number of samples, L number of dimensions
        xy_coordinates: x- and y- coordinates of the sees points, Lx2  matrix
        image_size: (optional) the output image size 
    Outputs:
        vor_images: NxMxRxC voronoi images, 4D tensor -- (number of samples N, number of channels M, number of rows R, number_of_columns C)
    """

    # Number of seed locations/points
    num_locs = xy_coordinates.shape[0]
    num_obs = data.shape[0]

    # start from 0
    x = np.zeros((num_locs,1))
    y = np.zeros((num_locs,1))
    x[:,0] = xy_coordinates[:,0]
    y[:,0] = xy_coordinates[:,1]
    x = x + int(image_size[0]/2)
    y = y + int(image_size[1]/2)
    voronoi_points = np.column_stack((x,y))

    # A grid of full space points including seed ones
    space_coordinates = np.mgrid[0:image_size[0], 0:image_size[1]]
    x_coord = space_coordinates[0,:].flatten() # columns
    y_coord = space_coordinates[1,:].flatten() # rows
    space_coordinates = np.vstack((x_coord,y_coord)).transpose()

    # Define an image
    img_col_size = image_size[0]
    img_row_size = image_size[1]
    img = np.zeros((img_row_size, img_col_size))

    # Fill in image
    n_channels = 1
    vor_images = np.zeros((num_obs, 1, img_row_size, img_col_size))
    for k in range(num_obs):
        value_vector = data[k,:]
        for img_col_ind, img_row_ind in space_coordinates:
            dist = (voronoi_points[:,0] - img_col_ind)**2 + (voronoi_points[:,1] - img_row_ind)**2
            idx = np.argmin(dist)
            img[img_row_ind, img_col_ind] = value_vector[idx]

        # Have to flip because of matrix conventions (y axis coordinates increase
        # from down to up but a matrix row indices increase from up to down)
        img = np.flipud(img)
        vor_images[k, 0, : , :] = img

    # Move axis for keras models
    vor_images = np.moveaxis(vor_images, 1, -1)

    return vor_images

def get_voronoi_indices(xy_coordinates, image_size=(61,61)):

    """ 
    Returns the cluster indices of a voronoi image.
    Returned matrix include in each of its pixel the group to which the pixel belongs to.
    (mostly the same with the function generate_voronoi_images_given_image_size())
    Inputs:
        xy_coordinates: The coordinates of the seed points 
        image_size: (optional) The output image size 
    Outputs:
        index_img: The index image (description above)
    """

    # Number of seed locations/points
    num_locs = xy_coordinates.shape[0]

    # start from 0
    x = np.zeros((num_locs,1))
    y = np.zeros((num_locs,1))
    x[:,0] = xy_coordinates[:,0]
    y[:,0] = xy_coordinates[:,1]
    x = x + int(image_size[0]/2)
    y = y + int(image_size[1]/2)
    voronoi_points = np.column_stack((x,y))

    # A grid of full space points including seed ones
    space_coordinates = np.mgrid[0:image_size[0], 0:image_size[1]]
    x_coord = space_coordinates[0,:].flatten() # columns
    y_coord = space_coordinates[1,:].flatten() # rows
    space_coordinates = np.vstack((x_coord,y_coord)).transpose()

    # Define an image
    img_col_size = image_size[0]
    img_row_size = image_size[1]
    index_img = np.zeros((img_row_size, img_col_size))

    # Fill in image
    for img_col_ind, img_row_ind in space_coordinates:
        dist = (voronoi_points[:,0] - img_col_ind)**2 + (voronoi_points[:,1] - img_row_ind)**2
        idx = np.argmin(dist)
        index_img[img_row_ind, img_col_ind] = idx

    # Have to flip because of matrix conventions (y axis coordinates increase
    # from down to up but a matrix row indices increase from up to down)
    index_img = np.flipud(index_img)
        
    return index_img
def get_best_threshold(labels, predictions):
    """ 
    Determines the best threshold according to the best F1 score
    Inputs:
        labels: True labels
        predictions: Classification output scores or probabilities
    Outputs:
        prec[idx] : best precision score
        thresholds[idx]: best threshold
    """

    prec, recall, thresholds = precision_recall_curve(labels, predictions, pos_label=1) 

    F1_score = []
    for threshold in thresholds:
        F1_score.append(f1_score(labels == 1, predictions >= threshold))
    
    idx = np.argmax(np.asarray(F1_score))
    return prec[idx], thresholds[idx] 