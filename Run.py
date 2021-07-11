# custom libraries (CreateDictionary() etc..)
from bow_libraries import *

## Standard python libraries
import random
from collections import Counter
import time
import shelve
import os, sys

## skimage, numpy etc
import numpy as np
from skimage import io, filters, transform, exposure
from skimage.feature import hog
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt


np.random.seed(7)


if __name__ == "__main__":
    print("\n\n######################################################\n\n\
Bag of Words image matching starting. Loading datasets..\n")
    
    try:
        ## Data preprocessing
        test_np = np.genfromtxt ('./data/fashion-mnist_test.csv', delimiter=",")
        test_data, test_labels = test_np[1:,1:], test_np[1:,0]
        test_data = test_data.reshape(10000, 28, 28).astype('float32')/255

        train_np = np.genfromtxt ('./data/fashion-mnist_train.csv', delimiter=",")
        train_data, train_labels = train_np[1:,1:], train_np[1:,0]
        train_data = train_data.reshape(60000, 28, 28).astype('float32')/255
    except:
        print("Error: train and test image data files not found.\nLocation needed:\n./data/fashion-mnist_train.csv\n./data/fashion-mnist_test.csv")
        print("\nLoading test_labels only and using pre-computed model..")
        preds = np.genfromtxt('predictions_final.csv', delimiter=",")
        test_labels = np.genfromtxt('./data/test_labels.csv', delimiter=",")
        metrics = GetMetrics(preds, test_labels)
        sys.exit(1)
    #####################################
    ## 1) Extract features 
    # Get full patch and featute set of training data. structure: [ (n * px * px, n * l) ] so each tuple is one image
    print("Beginning process.. Extracting features from train and test sets..")
    # note: set 1 / 2 were used initially to find optimal HOG parameters. Set 3 used for final predictions        
    train_set = []
    test_set = []
    # Set 1
    train1 = extract_all(train_data, patch_size=8, stride=4, hog_cell_size=4, hog_block_size=2, orients=8)
    test1 = extract_all(test_data, patch_size=8, stride=4, hog_cell_size=4, hog_block_size=2, orients=8)
    # # Set 2
    train2 = extract_all(train_data, patch_size=14, stride=7, hog_cell_size=7, hog_block_size=2, orients=12)
    test2 = extract_all(test_data, patch_size=14, stride=7, hog_cell_size=7, hog_block_size=2, orients=12)
    # Set 3
    train3 = extract_all(train_data, patch_size=28, stride=1, hog_cell_size=9, hog_block_size=1, orients=12)
    test3 = extract_all(test_data, patch_size=28, stride=1, hog_cell_size=9, hog_block_size=1, orients=12)

    


    print("Feature extraction complete, loading/creating dictionary..")

    #####################################    
    ## 2) Create (or load pre-saved) vocabulary

    # Note, vocab creation takes a long time. I used the shelve module to store a set of vocabularies
    # By default we load 
    # (To force generate a new vocabulary set raise an exception in the try block (i.e. force it go to go except block))
    try:
        v = shelve.open('vocabs')
        # Load vocab of 100 words.
        (v1f, v1p, v1_params) = v['v1']
        (v2f, v2p, v2_params) = v['v2']
        (v3f, v3p, v3_params) = v['v3']
        v.close()
        
        
    except:
        print("No vocab file found, creating...")

        all_train_p1, all_train_f1 = zip(*train1)
        all_train_p1, all_train_f1 = (np.concatenate(all_train_p1), np.concatenate(all_train_f1))
        (v1f, v1p) = CreateDictionary(all_train_p1, all_train_f1, 100)
        print("Dictionary 1 created..")
        all_train_p2, all_train_f2 = zip(*train2)
        all_train_p2, all_train_f2 = (np.concatenate(all_train_p2), np.concatenate(all_train_f2))
        (v2f, v2p) = CreateDictionary(all_train_p2, all_train_f2, 100)
        print("Dictionary 2 created..")
        all_train_p3, all_train_f3 = zip(*train3)
        all_train_p3, all_train_f3 = (np.concatenate(all_train_p3), np.concatenate(all_train_f3))
        (v3f, v3p) = CreateDictionary(all_train_p3, all_train_f3, 100)
        print("Dictionary 3 created..")
        # # Store for future access
        v = shelve.open('vocabs')
        v['v1'] = (v1f, v1p, v1_params) 
        v['v2'] = (v2f, v2p, v2_params) 
        v['v3'] = (v3f, v3p, v3_params)
        v.close()
    print(f"BoW dictionaries from the following HOG feature sets are being used:\n{v1_params}\n{v2_params}\n{v3_params}\n")
    ###



    # 2.B Store the representative patches as images in the folder
    # Only do this if images haven't already been generated
    if len(os.listdir("./centroid_images/dict1"))==0:
        for i in range(len(v1p)):
            plt.imsave(f"./centroid_images/dict1/{i}.jpeg", v1p[i])
    if len(os.listdir("./centroid_images/dict2"))==0:
        for i in range(len(v2p)):
            plt.imsave(f"./centroid_images/dict2/{i}.jpeg", v2p[i])
    if len(os.listdir("./centroid_images/dict3"))==0:    
        for i in range(len(v3p)):
            plt.imsave(f"./centroid_images/dict3/{i}.jpeg", v3p[i])


    print("Visual dictionary loaded/built. Computing histograms...")


    #####################################    
    ## 3) Compute histograms for training and testing images
    
    # Structure: [ (n * px * px, n * l) ] so each tuple is one image
    
    train_hists1, train_hists2, train_hists3 = ([], [], [])
    test_hists1, test_hists2, test_hists3 = ([],[],[])
    for _,f_set in train1:
        train_hists1.append(SumHistogram(f_set, v1f))
    for _,f_set in test1:
        test_hists1.append(SumHistogram(f_set, v1f))

    for _,f_set in train2:
        train_hists2.append(SumHistogram(f_set, v2f))
    for _,f_set in test2:
        test_hists2.append(SumHistogram(f_set, v2f))

    for _,f_set in train3:
        train_hists3.append(SumHistogram(f_set, v3f))
    for _,f_set in test3:
        test_hists3.append(SumHistogram(f_set, v3f))
    print("Histogram calculations complete.. Getting predictions..")
    #####################################    
    ## 4) Make predictions
    # NOTE: by default we load the predictions as this is the slowest process!
    # The prediction csv has been generated from running MajorityVote on the predictions generated by 3 different BoW dictionaries 
    # (from different HOG descriptors). This increased accuracy by ~2% from 79 to 81 (compared to HOGv2)
    # To force make predictions, raise an exception in the try block
    
    try:
        preds = np.genfromtxt ('predictions_32039131_final.csv', delimiter=",")
        print("Predictions successfully loaded")
    except:
        print("Prediction file not found/unsuccessful. Making predictions (24+ hours).")
        preds1 = MakePredictions(test_hists1, train_hists1, 6, train_labels)
        print("1/3 complete")
        preds2 = MakePredictions(test_hists2, train_hists2, 6, train_labels)
        print("2/3 complete")
        preds3 = MakePredictions(test_hists3, train_hists3, 6, train_labels)
        preds = MajorityVote(preds1, preds2, preds3)
        # Save the predictions for future:
        np.array(preds).tofile('predictions.csv', sep = ',')
        print("Predictions complete, writing to 'predictions.csv'\n Computing metrics..")
    
    
    metrics = GetMetrics(preds, test_labels)

    print("Complete.")





