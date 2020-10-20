import astroalign
from scipy.spatial import cKDTree
import pickle
import json
import skimage.transform
import os
import numpy as np
import cv2
from util import crop_image, get_average_precision_recall, get_normalise_direction_matrix, visualize


astroalign.MAX_CONTROL_POINTS = 50
astroalign.MIN_MATCHES_FRACTION = 0.8
astroalign.NUM_NEAREST_NEIGHBORS = 15
astroalign.PIXEL_TOL = 0.01

__cache_dir__ = "aa_cache"

def set_cache_dir(cache):
    """Sets the cache

    Args:
        cache (str): The directory where to cache values sued for matching.
    """
    __cache_dir__ = cache
    
def get_cache_dir():
    """Sets the cache directory

    Returns:
        str: The directory where the cache is stored
    """
    return __cache_dir__

def aamatch(spots_1, spots_2, source_invariants = None, source_asterisms = None, source_invariant_tree = None, target_invariants = None, target_asterisms = None, target_invariant_tree = None):
    """Attempts to match 2 lists of spots.

    Args:
        spots_1 (List[List[float]]): The list of spots in list 1
        spots_2 (List[List[float]]): The list of spots in list 2
        source_invariants (Object, optional): The cached triangle invariants for list 1 if available. Defaults to None.
        source_asterisms (Object, optional): The triangle asterisms for list 1 if available. Defaults to None.
        source_invariant_tree (Object, optional): The invariant kd-tree for list 1 if available. Defaults to None.
        target_invariants (Object, optional): The cached triangle invariants for list 2 if available. Defaults to None.
        target_asterisms (Object, optional): The triangle asterisms for list 2 if available. Defaults to None.
        target_invariant_tree (Object, optional): The invariant kd-tree for list 2 if available. Defaults to None.

    Returns:
        Tuple[np.ndarray, int, float, List[int], List[int]]: Tuple of the Affine transform matrix mapping one point onto the other, the number of inliers, the score and the matching indexes form list 1 and list 2
    """
    try:
        T, (s_pos, t_pos), (s_idx, t_idx) = astroalign.find_transform(spots_1, spots_2,
                                                                        source_invariants = source_invariants, source_asterisms = source_asterisms, source_invariant_tree = source_invariant_tree,
                                                                        target_invariants = target_invariants, target_asterisms = target_asterisms, target_invariant_tree = target_invariant_tree)
        return (T, len(s_pos), (len(s_pos) / len(spots_1)) * (len(t_pos) / len(spots_2)), s_idx, t_idx)
    except Exception as e:
        return None, 0 ,0, [], []
    
def retrieve_from_cache(imgKey):
    """Retrieves the precomputed values from cache if present

    Args:
        imgKey (str): The key for which to retrieve the precomputed values

    Returns:
        Dict|None: The object stored in the cache for the key
    """
    if os.path.isfile(os.path.join(get_cache_dir(), imgKey + ".aa.pickle")):
        with open(os.path.join(get_cache_dir(), imgKey + ".aa.pickle"), 'rb') as f:
            precomputedModelValues = pickle.load(f)
        return precomputedModelValues
    else:
        return None
    
def open_cached(modelDict, imageKey, spotsJsonKey = 'spotsJson', maskImgKey = 'mask', precomputedPickle = 'precompAA'):
    """Attempts to open a cached version of precomputed values for the comparison. Computes these if necessarty and saves to cache.

    Args:
        modelDict (Dict): The dictionary describing the fish
        imageKey (str): The key of the fish
        spotsJsonKey (str, optional): The name of the key under which the json file name containing the spots of the fish is saved. Defaults to 'spotsJson'.
        maskImgKey (str, optional): The name of the key under which the image name containing the mask of the fish is saved. Defaults to 'mask'.
        precomputedPickle (str, optional): name of the key under which the cached precomputed values are stored, if not in the standard location. Defaults to 'precompAA'.

    Returns:
        dict: The dictionary of precomputed values
    """
    precomputedModelValues = None
    if precomputedPickle in modelDict and modelDict[precomputedPickle] is not None :
        with open(modelDict[precomputedPickle], 'rb') as f:
            precomputedModelValues = pickle.load(f)
    else:
        precomputedModelValues = retrieve_from_cache(imageKey)
            
    if precomputedModelValues is None or "version" not in precomputedModelValues or precomputedModelValues["version"] != 1 or precomputedModelValues["nn"] != astroalign.NUM_NEAREST_NEIGHBORS or precomputedModelValues["max_control_points"] != astroalign.MAX_CONTROL_POINTS:
        precomputedModelValues = precomputeValues(modelDict, imageKey, spotsJSONKey = spotsJsonKey, maskImgKey = maskImgKey)
    
    return precomputedModelValues
        
def precomputeValues(fishDict, imgKey, spotsJSONKey = 'spotsJson', maskImgKey = 'mask', precomputedPickle = 'precompAA'):
    """Precomputes values necessary for comparison of the fish to another fish

    Args:
        fishDict (dict[str,str]): The dictionary describing all components of the fish image
        imgKey (str): The key of the fish imge
        spotsJSONKey (str, optional): The key in the dictionary indicating the filepath to the json file containing the spot information for the fish. Defaults to 'spotsJson'.
        maskImgKey (str, optional): The key in the dictionary indicating the filepath to the mask file containing the boolean mask for the fish. Defaults to 'mask'.
        precomputedPickle (str, optional): The key in the dictionary indicating the filepath to the precomputed values of the fish. Defaults to 'precompAA'.

    Returns:
        [type]: [description]
    """
    spots = None
    if fishDict[spotsJSONKey] is not None:
        with open(fishDict[spotsJSONKey], 'r') as f:
            spots = np.asarray(json.load(f))
        if len(spots) > 5:
            spots = spots[:,:2] # Remove size from spots
            mask = crop_image(cv2.imread(fishDict[maskImgKey], 0))
            R = get_normalise_direction_matrix(mask)
            tmpPoints = np.copy(np.asarray(spots))
            tmpPoints = np.dot(R[:,(0,1)], np.array([tmpPoints[:,0], tmpPoints[:,1]]))
            tmpPoints = np.array(list(zip(tmpPoints[0], tmpPoints[1])))
            warpedTargetMask = crop_image(cv2.warpAffine(mask, R, (mask.shape[1] + 500, mask.shape[0] + 500)))
            spotsProcessed = tmpPoints / np.max(warpedTargetMask.nonzero())

            invariants, asterisms = astroalign._generate_invariants(spotsProcessed[:astroalign.MAX_CONTROL_POINTS])
            kdTree = cKDTree(invariants)
            fishDict[precomputedPickle] = os.path.join(get_cache_dir(), imgKey + ".aa.pickle")
            precomputedObject = {
                    "max_control_points": astroalign.MAX_CONTROL_POINTS,
                    "invariants": invariants,
                    "asterisms": asterisms,
                    "kdtree": kdTree,
                    "spots": spots,
                    "spots_standardised": spotsProcessed,
                    "nn": astroalign.NUM_NEAREST_NEIGHBORS,
                    "version": 1
                }
            with open(os.path.join(get_cache_dir(), imgKey + ".aa.pickle"), 'wb') as f:
                pickle.dump(precomputedObject, f)
            return precomputedObject
    return None
    
    
def findClosestMatch(modelImageName, modelDict, imagesToComareDicts, spotsJsonKey = 'spotsJson', maskImgKey = 'mask', verbose = False, progress = False):
    """Finds the best matches for the model image in the images to compare dictionaries

    Args:
        modelImageName (str): The name of the image being compared
        modelDict (dict): The dictionary containing data about the image
        imagesToComareDicts (Dict[Dict]): The list of other image dictionaries to compare to.
        spotsJsonKey (str, optional): The key in the dictionary indicating the filepath to the json file containing the spot information for the fish. Defaults to 'spotsJson'.
        maskImgKey (str, optional): The key in the dictionary indicating the filepath to the mask file containing the boolean mask for the fish. Defaults to 'mask'.
        verbose (bool, optional): If to print verbose information to stdout. Defaults to False.
        progress (bool, optional): If to pring progress information to stdout. Defaults to False.

    Returns:
        List[Tuple[float, float, str]]: List of tuples of (score, maskCoverage, image name) for each image the model image was compared to.
    """
    modelPrecompValues = open_cached(modelDict, modelImageName, spotsJsonKey=spotsJsonKey, maskImgKey=maskImgKey)
    if modelPrecompValues is None:
        if verbose:
            print("Not enough points")
        return []
    
    centerTranslationMatrix = np.float32([[1, 0, 250], [0,1,250], [0, 0, 1]])
    
    targetSpots = cv2.imread(modelDict["spots"], 0)
    targetMask = crop_image(cv2.imread(modelDict[maskImgKey], 0))   
    targetMaskShifted = cv2.warpPerspective(targetMask, centerTranslationMatrix, (targetMask.shape[1] + 500, targetMask.shape[0] + 500))
    
    if verbose:
        print(modelImageName)
        visualize(modelPrecompValues["spots_standardised"], modelPrecompValues["spots_standardised"], annotateOrder = True, invertYAxis = False, figsize=(10,10))
    
    ranking = []
    for imageKey in imagesToComareDicts:
        if progress or verbose:
            print(imageKey)

        comparatorImageDict = imagesToComareDicts[imageKey]
        
        dataPrecompValues = open_cached(comparatorImageDict, imageKey, spotsJsonKey=spotsJsonKey, maskImgKey=maskImgKey)
        
        if dataPrecompValues is None:
            ranking.append((-1, 0, imageKey))
            continue
        
            

        if verbose:
            visualize(dataPrecompValues["spots_standardised"], dataPrecompValues["spots_standardised"], annotateOrder = True, invertYAxis = False, figsize=(10,10))

        T, num_points, score, s_idx, t_idx = aamatch(dataPrecompValues["spots_standardised"], modelPrecompValues["spots_standardised"],
                                                     dataPrecompValues["invariants"], dataPrecompValues["asterisms"], dataPrecompValues["kdtree"],
                                                     modelPrecompValues["invariants"], modelPrecompValues["asterisms"], modelPrecompValues["kdtree"])
         
        maskCoverage = 0
        spotsCoverage = 0
        if T is not None:
            if verbose:
                transformedCoords = []
                for coord in dataPrecompValues["spots_standardised"]:
                    coordNew = np.array([coord[0], coord[1], 1])
                    coordNew = T.params @ coordNew
                    transformedCoords.append(coordNew[:2])

                maxValue = max(np.max(dataPrecompValues["spots_standardised"]), np.max(dataPrecompValues["spots_standardised"]), np.max(np.asarray(transformedCoords)))
                visualize(dataPrecompValues["spots_standardised"] ,dataPrecompValues["spots_standardised"], annotateOrder = True, invertYAxis = False, figsize=(10,10))
                visualize(modelPrecompValues["spots_standardised"], modelPrecompValues["spots_standardised"], annotateOrder = True, invertYAxis = False, figsize=(10,10))

                visualize(modelPrecompValues["spots_standardised"], np.asarray(transformedCoords), annotateOrder = False, invertYAxis = False, figsize=(10,10))
    
            model = skimage.transform.AffineTransform()
            model.estimate(dataPrecompValues["spots"][s_idx], modelPrecompValues["spots"][t_idx])
            dataMask = crop_image(cv2.imread(comparatorImageDict[maskImgKey], 0))     
            
            warped = cv2.warpPerspective(dataMask, centerTranslationMatrix @ model.params, (targetMaskShifted.shape[1], targetMaskShifted.shape[0]))

            pMask, rMask = get_average_precision_recall(np.array([warped]), np.array([targetMaskShifted]), verbose = verbose)
            maskCoverage = 2 * (pMask * rMask) / (pMask + rMask)
        
        
        ranking.append((score * maskCoverage, maskCoverage, imageKey))
    return ranking
