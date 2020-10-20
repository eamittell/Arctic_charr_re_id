import UnetExtractor
from collections.abc import Iterable
from algorithms import Algorithm
import os
from pathlib import Path
import astroalignMatch
import grothMatcherCustom

def _translatePath(inPath):
    """Default function for translating an original image path to a result path for spots and masks. Creates intermediate folders if no such end folder exists.
    inPath (str|Path) The path to translate
    
    Returns: (str) The translated path
    """
    outPath = "results%s%s" % (os.sep, os.sep.join(inPath.split(os.sep)[-3:]))
    if not os.path.exists(os.path.dirname(outPath)):
        os.makedirs(os.path.dirname(outPath))
    return outPath

def _imgDirectoryTranslatorWrapper(root):
    def _imgDirectoryTranslator(key):
        """Default function for translating a key to an image directory
        key (str|Path) The key to convert to a file path

        Returns: (str|None) the filepath assosciatred with the key
        """
        # Split into cave, year, month, img name
        keyElements = key.split("-")
        dirPath = os.path.join(root, "%s_%s" % (keyElements[1], keyElements[2]), "Cave%s" % (keyElements[0][1:]))
        # Can have extension of .jpg, .JPG, .JPEG, so just check which one it is. Ignore hidden files (starting with .)
        filteredImageNames = [path for path in os.listdir(dirPath) if keyElements[3] in path and "." != path[0]]
        if len(filteredImageNames) > 0:
            return os.path.join(dirPath, filteredImageNames[0])
        else:
            return None
        
    return _imgDirectoryTranslator

class Matcher():
    
    def __init__(self, 
                 imgRoot = "all_images",
                 grothCache = "groth_cache",
                 astroalignCache = "aa_cache",
                 maskModelWeights = "unet_mask.hdf5",
                 maskBatchSize = 20,
                 maskResultOutputDir = None,
                 maskFileSuffix = ".mask.png",
                 customMaskExtractor = None,
                 spotUnetWeights = "unet_spots.hdf5",
                 spotBatchSize = 10,
                 spotThreshold = 0.01,
                 spotResultOutputDir = None,
                 spotFileSuffix = ".spots.png",
                 spotJsonFileSuffix = ".spots.json",
                 customSpotExtractor = None,
                 gpuAcceleration = True,
                 keyToPathTranslator = None):
        
        """Constructor for a new matcher
         imgRoot             (str|Path) The directory where the base images are stored. If custom keyToPathTranslator is provided, this has no effect
         grothCache          (str|Path) Directory where the custom groth algorithm should store it's cache files
         astroalignCache     (str|Path) Directory where the astroalign algorithm should store it's cache files
         maskModelWeights    (str|Path) Path to the stored weights for the mask Unet model. Can be omitted if an initialised extractor is provided in the
                                        customMaskExtractor argument.
         maskBatchSize       (int)      The batch size to use when processing masks. Speeds up extraction of multiple masks at once, but uses more memory
                                        (GPU or RAM depending if GPU acceleration is used)
         maskResultOutputDir (func)     Function of the from (str | Path) -> (str | Path) which converts an input image path into a path where the result of the mask
                                        extraction operation should be stored
         maskFileSuffix      (str)      The suffix that is used to identify mask files
         customMaskExtractor (Object)   An object that contains a function of form ([str], batch_size=int, outputDir=func (str | Path) -> (str | Path)) -> None
                                        which extracts a mask from the original image to the location indicated by the function passed as the outputDir with the suffix
                                        given in maskFileSuffix argument
         spotUnetWeights     (str|Path) Path to the stored weights for the spot Unet model. Can be omitted if an initialised extractor is provided in the
                                        customSpotExtractor argument.
         spotBatchSize       (int)      The batch size to use when processing spots. Speeds up extraction of spots from multiple fish at once, but uses more memory
                                        (GPU or RAM depending if GPU acceleration is used)
         spotThreshold       (float)    Value in range [0,1] that is applied to the spots prediction to as a threshold for determining which part of the prediction should
                                        be taken as spots. Higher values result in higher precision, lower values result in higher recall.
         spotResultOutputDir (func)     Function of the from (str | Path) -> (str | Path) which converts an input image path into a path where the result of the spot
                                        extraction operation should be stored
         spotFileSuffix      (str)      The suffix that is used to identify raw spot files
         spotJsonFileSuffix  (str)      The suffix that is used to identify json spot files
         customSpotExtractor (Object)   An object that contains a function of form
                                        ([str], [str], batch_size=int, threshold=float, outputDir=func (str | Path) -> (str | Path)) -> None
                                        which extracts spots from the original image and mask to the location indicated by the function passed as the outputDir with the
                                        suffix given in spotFileSuffix and spotJsonFileSuffix arguments
         gpuAcceleration     (bool)     If to use CUDA acceleration when extracting masks an spots. If true, uses the first available GPU device, else it uses the
                                        first available CPU device. Can be ommitted if passing a custom spot and mask extractor.
         keyToPathTranslator (func)     Function of form (str) -> (str | Path) which takes in an image key, and converts it to the path of the location of the base image.
         """
        
        self.grothCache = grothCache
        self.astroalignCache = astroalignCache
        self.keyToPathTranslator = _imgDirectoryTranslatorWrapper(imgRoot) if keyToPathTranslator is None else keyToPathTranslator

        self.maskBatchSize = maskBatchSize
        self.maskResultOutputDir = _translatePath if maskResultOutputDir is None else maskResultOutputDir
        self.maskFileSuffix = maskFileSuffix

        self.spotBatchSize = spotBatchSize
        self.spotThreshold = spotThreshold
        self.spotResultOutputDir =  _translatePath if spotResultOutputDir is None else spotResultOutputDir
        self.spotFileSuffix = spotFileSuffix
        self.spotJsonFileSuffix = spotJsonFileSuffix
        
        # If a custom spot or mask extractor was passed, use that over the default unet extractor. Can be used to supply a Unet extractor with options other than default.
        if customMaskExtractor is not None:
            self.maskExtractor = customMaskExtractor
        else:
            self.maskExtractor = UnetExtractor.UnetMaskExtractor(modelWeights = maskModelWeights, useGPU = gpuAcceleration)
        
        if customSpotExtractor is not None:
            self.spotExtractor = customSpotExtractor
        else:
            self.spotExtractor = UnetExtractor.UnetSpotExtractor(modelWeights = spotUnetWeights, useGPU = gpuAcceleration)
        
    def matching(self, query_imgs, matching_imgs, algorithm = Algorithm.CUSTOM_GROTH, rankingLimit = None):
        """Matches a set of images against a second set of images.
        query_imgs    ([str]|str)            The image keys (or key) that are to be matched
        matching_imgs ([str]|str)            The image keys (or key) that are to be matched to
        algorithm     (algorithms.Algorithm) The matching algorithm to use
        rankingLimit  (int|None)             The max number of reuslts to return
        
        Returns dictionary of form:
        {str: [
            {"file_name": str,
             "ranking": int,
             "score": float}
        ]}
        """
        # Support both singular input and multiple input
        if isinstance(query_imgs, str): 
            query_imgs = [query_imgs]
        if isinstance(matching_imgs, str):
            matching_imgs = [matching_imgs]
            
        all_imgs = query_imgs + matching_imgs
            
        # Check if any images need to have masks extracted
        maskPaths = self.__ensureMasksExtracted(all_imgs)
        # Check if any images need to have spots extracted
        spotPaths, spotJsonPaths = self.__ensureSpotsExtracted(all_imgs)
        #Construct dictionaries for each key
        inputDictionary = {}
        
        for i in range(len(query_imgs)):
            # The precomp and precompAA are fields for custom cahce files outside of the normal ache directory. Just leave blank. Algorithms will check their cache anyway
            # The label fields are left blank as they are only used for training. Also no labels exist
            inputDictionary[query_imgs[i]] = {
                    'img': self.keyToPathTranslator(query_imgs[i]),
                    'mask': maskPaths[i],
                    'maskLabel': None,
                    'spotsLabel': None,
                    'spots': spotPaths[i],
                    'spotsJson': spotJsonPaths[i],
                    'precomp': None,
                    'precompAA': None
            }
            
        comparatorDictionary = {}
        for i in range(len(matching_imgs)):
            comparatorDictionary[matching_imgs[i]] = {
                    'img': self.keyToPathTranslator(matching_imgs[i]),
                    'mask': maskPaths[len(query_imgs) + i],
                    'maskLabel': None,
                    'spotsLabel': None,
                    'spots': spotPaths[len(query_imgs) + i],
                    'spotsJson': spotJsonPaths[len(query_imgs) + i],
                    'precomp': None,
                    'precompAA': None
            }
        
        results = {}
        # Run the correct algorithm
        if algorithm == Algorithm.CUSTOM_GROTH:
            grothMatcherCustom.set_cache_dir(self.grothCache)
            for key in query_imgs:
                results[key] = []
                result = grothMatcherCustom.findClosestMatch(key, inputDictionary[key], comparatorDictionary, local_triangle_k = 25)
                
                #Order results
                orderedResult = sorted(result, key=lambda x: x[0], reverse = True)
                
                #Restrict to number results requested
                limit = rankingLimit
                if limit is None:
                    limit = len(orderedResult)
                for rank in range(min(limit, len(orderedResult))):
                    r = orderedResult[rank]
                    results[key].append({
                        "file_name": r[2],
                        "ranking": rank + 1,
                        "score": r[0]
                    })
        else:
            astroalignMatch.set_cache_dir(self.grothCache)
            for key in query_imgs:
                results[key] = []
                result = astroalignMatch.findClosestMatch(key, inputDictionary[key], comparatorDictionary)
                #Order results
                orderedResult = sorted(result, key=lambda x: x[0], reverse = True)
                #Restrict to number results requested
                limit = rankingLimit
                if limit is None:
                    limit = len(orderedResult)
                for rank in range(min(limit, len(orderedResult))):
                    r = orderedResult[rank]
                    results[key].append({
                        "file_name": r[2],
                        "ranking": rank + 1,
                        "score": r[0]
                    })
                    
        return results
        
        
        
    def __ensureMasksExtracted(self, imgs):
        """Ensures all images assosciated with the given keys have extracted masks, extracting the masks where necessary
        imgs ([str]) List of images for which to make sure masks exist, extracting masks when no mask is present.
        
        Returns: [str] of mask paths for each img in imgs
        """

        pathsToProcess = []
        maskPaths = []
        for key in imgs:
            imgPath = self.keyToPathTranslator(key)

            if imgPath is None:
                raise IOError("No image found for key %s" % key)
            maskResultsPath = self.maskResultOutputDir(imgPath)
            
            if not os.path.isfile(maskResultsPath + self.maskFileSuffix): #Check if file generated exists
                pathsToProcess.append(imgPath)
            maskPaths.append(maskResultsPath + self.maskFileSuffix)
        
        if len(pathsToProcess) > 0:
            self.maskExtractor.generate_masks(pathsToProcess, batch_size = self.maskBatchSize, outputDir = self.maskResultOutputDir)
        
        return maskPaths
        
    def __ensureSpotsExtracted(self, imgs):
        """Ensures all images assosciated with the given keys have extracted spots, extracting the spots where necessary
        imgs ([str]) List of images for which to make sure spots exist, extracting spots when no spot file is present.
        
        Returns: ([str], [str]) of spot paths and spot json paths for each img in imgs
        """
        imgPathsToProcess = []
        maskPathsToProcess = []
        spotPaths = []
        spotJsonPaths = []
        
        for key in imgs:
            imgPath = self.keyToPathTranslator(key)
            
            if imgPath is None:
                raise IOError("No image found for key %s" % key)
            maskResultsPath = self.maskResultOutputDir(imgPath)
            
            if not os.path.isfile(maskResultsPath + self.maskFileSuffix): #Check if mask file exists
                raise IOError("No mask found when generating spots for key %s" % key)
                
            spotResultsPath = self.spotResultOutputDir(imgPath)
            
            if not os.path.isfile(spotResultsPath + self.spotJsonFileSuffix) or not os.path.isfile(spotResultsPath + self.spotFileSuffix): #Check if spots file exists
                imgPathsToProcess.append(imgPath)
                maskPathsToProcess.append(maskResultsPath + self.maskFileSuffix)
            spotPaths.append(spotResultsPath + self.spotFileSuffix)
            spotJsonPaths.append(spotResultsPath + self.spotJsonFileSuffix)
                
        if len(imgPathsToProcess) > 0:
            self.spotExtractor.generate_spots(imgPathsToProcess,
                                              maskPathsToProcess,
                                              batch_size = self.spotBatchSize,
                                              threshold = self.spotThreshold,
                                              outputDir = self.spotResultOutputDir)
        return spotPaths, spotJsonPaths
        