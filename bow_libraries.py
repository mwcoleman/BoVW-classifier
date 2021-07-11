import pandas as pd
import numpy as np
from skimage import io, filters, transform, exposure
from skimage.feature import hog, corner_harris, corner_subpix, corner_peaks, canny, daisy
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
import random
from collections import Counter
import time

np.random.seed(7)


def CreateDictionary(patches, feat_vec, k):
    # A K-means clustering method to create a visual dictionary for BOW
    ## Inputs: Patches - Visual representations corresponding to the features
    #          feat_vec- a list of feature vectors  
    #     # Performs K means clustering

    # # # First flatten list of lists:
    # patches = [item for sublist in patches for item in sublist]
    # feat_vec = [item for sublist in feat_vec for item in sublist]


    ## Initialise centroids using the K++ proposal: 
    # sample from feature space with probability proportional 
    # to distance to existing centroids
    # ref: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
    # and  https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html

    print(f'Beginning clustering for k: {k}')
    start_time = time.time()
    def plus_plus(fv, k):
        np.random.seed(7)

        # First centroid is just the first feature vector.
        centroids = [fv[0]]

        for _ in range(1, k):
            # Calcuate probability based on the distance between points
            distance = np.array([min([np.inner(k-v,k-v) for k in centroids]) for v in fv])
            probs = distance/distance.sum()
            probs = probs.cumsum()
            
            # Rejection sampling to probabilistically pick centroid far away from existing
            r = np.random.rand()
            
            for j, p in enumerate(probs):
                if r < p:
                    i = j
                    break

            # Set the next centroid as the chosen vector from our feature vector space
            centroids.append(fv[i])

        return np.array(centroids)

    vec_num = feat_vec.shape[0]
    vec_len = feat_vec.shape[1]

    converged = False

    ## STEP 1: Initialise centroids 
    # creates k centroid vectors of same length as our feature vectors
    centroids = plus_plus(feat_vec, k)

    #STEP 2: Iterate through update process until converged:
    j = 0   
    while(not converged):
        iter_time = time.time()
        # Make a dictionary with centroid vector pairings. Gets reset every iteration
        clusters = {k: [] for k in range(centroids.shape[0])}

        j += 1
        # Step one - find the closest centroid for each point and group it. Do euclidian
        for i in range(vec_num):
            
            # Subtract the feature vector from each centroid, and calculate the l2 norm.
            # specify the axis of the length of the centroid vectors to sum over each
            l2 = np.linalg.norm(centroids - feat_vec[i], axis=1)        
            # Take the minimum of this to find the closest centroid
            k = np.where(l2==min(l2))[0][0]

            clusters[k].append(i)

        # Store old value for convergence check
        old_centroids = centroids.copy()

        # Step two - update the centroid as the mean of the cluster
        for k in clusters:

            # Only update centroid vector if its corresponding cluster is non-zero in size
            if (len(clusters[k]) != 0):
                # Updat  kth centroid with the mean of it's cluster 
                centroids[k] = np.mean([feat_vec[vec_index] for vec_index in clusters[k] ], axis=0)
            else:
                # In case where no points associated with centroid, choose to re-assign it
                centroids[k] = feat_vec[random.randint(0,vec_len)]

        # Consider converged when no updates have been performed or the change in vectors is small
        old_centroids[old_centroids==0]=0.0000001 # fix div by 0
        converged =  (not np.any( (np.absolute(old_centroids - centroids)/old_centroids) > .01)) or (j>100)    
        print(f"Iteration {j} Complete, Iter time: {time.time() - iter_time}")
        
        
    if j>100:
        print('converged due to iterations >100')
    else:
        print('converged due to threshold')
    print(f"Completed. Time taken (min): {(time.time() - start_time)/60}")
    ## Save the image patch whose feature descriptor is closest to each cluster centre
    representative_patches = []
    for k in centroids:
        l2 = np.linalg.norm(k - feat_vec, axis=1)   
        hog_index = np.where(l2==min(l2))[0][0]
        representative_patches.append(patches[hog_index])
    return (centroids, representative_patches)

def ComputeHistogram(feat_vec, vis_dict):
    ## Inputs: a feature vector representing an image, a visual dictionary created by k-means (create_dict funtion)
    # Outputs: a histogram of assignment to each of the 'words' in the visual dictionary.
    # Soft assignment:= assignment weight is based on l2 norm.
   
    l2 = np.linalg.norm(feat_vec - vis_dict, axis=1)   # end up with 100 distances
    l2[l2==0] = 0.0001  # fix div by zero problem
    # Penalise by distance
    h = ((1/l2))**2
  
    return(h)

def MatchHistogram(h1, h2, method):
    ## Computes the 'distance' between two histograms using either chi squared, or histogram intersection
    # Chisquared works better (empirically for this task)
    if method == 'intersection':
        return np.sum(np.minimum(h1,h2))
    
    elif method == 'chisquared':
        return np.sum( [ ((a-b)**2 / (a+b)) for (a,b) in zip(h1, h2)] ) 

def SumHistogram(img_feat_set, vis_dict):
    # Input: array of features for a single image, 
    # and a vocab to perform soft assignment on
    # Output: a summed histogram for that image
    h = [0 for k in vis_dict]
    for fv in img_feat_set:
        h += ComputeHistogram(fv, vis_dict)
    return h 

def get_features(image, patch_size, stride, hog_cell_size, hog_block_size, orients):
    # Inputs: A single image of dimension px * px
    # outputs: a tuple of (patches, features)  where number of patches and dimensions are dependent on the parameters specified. Number of features is equal to number of           #           patches
    
    
    def cut_patches(img, patch_size, stride):
        # Inputs: greyscale Image (hxwx1 np array), patch_size parameter specifies size of patches (nxn) default 7x7 px
        
        # Outputs: nxn x #patches  array. #patches equal to len(img_set) * (area of image / area of patch) e.g. 28^2 / 7^2 = 16 

        # https://scikit-image.org/docs/0.9.x/api/skimage.util.html#view-as-windows
        window = (patch_size, patch_size)
        stride = stride # we don't want any overlapping pixels
        image_cuts = view_as_windows(img, window, stride)
        # Flatten it into (n^2, patch_heigh, patch_width)
        cut_flattened = image_cuts.reshape((image_cuts.shape[0]**2, image_cuts.shape[2],image_cuts.shape[3] ))
        return cut_flattened
    
    def extract_HOG(patch, cell_size, block_size, orients):
        return hog(patch, orientations=orients, pixels_per_cell=(cell_size, cell_size),
                    cells_per_block=(block_size, block_size), visualize=False, multichannel=False, feature_vector=True)
    
    ## Main method
    ## Cut image into sliding window patches, obtain (n x 2dpatch)
    patches = cut_patches(image, patch_size, stride)

    ## Obtain HOG for each patch
    hogs = []
    for patch in patches:
        hogs.append(extract_HOG(patch, hog_cell_size, hog_block_size, orients))
    hogs = np.vstack(hogs)
    
    return (patches, hogs)

def extract_all(image_set, patch_size, stride, hog_cell_size, hog_block_size, orients):
    # Takes a image set, and some parameters. returns a list of tuples, each
    # ( (n * px * px), (n * l) ) 
    # where n is number of patches making up image, px is patch size, l is feature vector length
    
    pf_set = []
    for image in image_set:
        pf_set.append(get_features(image, patch_size, stride, hog_cell_size, hog_block_size, orients))

    return pf_set    

def MakePredictions(test_hists, train_hists, knn, train_labels):
    predict_label = []
    for i in range(len(test_hists)):
        hist_dists = []
        # Find the closest training image:
        for j in range(len(train_hists)):
            hist_dists.append( (j, MatchHistogram(test_hists[i], train_hists[j], method='chisquared')) )
    
        # Sort top 30 neighbours:
        sorted_n = sorted(hist_dists, key=lambda x: x[1])[:knn]
        nn_labels = [train_labels[i[0]] for i in sorted_n]
        count = Counter(nn_labels)
        maxval = max(count.values())
        majority = [k for k,v in count.items() if v==maxval][0]
        predict_label.append(majority)

    return predict_label

def GetMetrics(pred, label):
    # overall classification accuracy, class wise accuracy, precision and recall 
    pred = np.array(pred)
    label = np.array(label)
    N = label.shape[0]
    classes = [i for i in range(10)]
    
    accuracy = 100*np.sum(pred == label)/N

    cw_accuracy, cw_precision, cw_recall = ([], [], [])
    cw_tp, cw_fp, cw_tn, cw_fn = ([], [], [], [])

    for c in classes:
        
        tp = np.sum(pred[label==c] == c)
        fp = np.sum(label[pred==c] != c)
        tn = np.sum(pred[label!=c] != c)
        fn = np.sum(pred[label==c] != c)
        cw_tp.append(tp)
        cw_fn.append(fn)
        cw_fp.append(fp)
        cw_tn.append(tn)

        cw_accuracy.append(100*(tp/(tp+fn)))
        cw_precision.append(tp/(tp+fp))
        cw_recall.append(tp/(tp+fn))
      
    precision = sum(cw_precision)/10
    recall = sum(cw_recall)/10

    metrics = {'accuracy': accuracy, "cw_accuracy":cw_accuracy, "precision":precision, "recall":recall}
    # Overall Precision and recall from average of classwise
    print("\n:::Overall Performance:::")
    print(f"Accuracy: {round(metrics['accuracy'], 2)}% -- Precision: {round(metrics['precision'], 4)} -- Recall: {round(metrics['recall'], 4)}\n\n")
    print(":::Class wise accuracy:::")
    
    classnames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    for i in range(10):
        print(f"Class {i}: {round(metrics['cw_accuracy'][i], 2)}% -- ({classnames[i]})")
    
    
    return metrics    