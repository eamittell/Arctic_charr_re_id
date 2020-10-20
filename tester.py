import util
import DBUtil
import grothMatcherCustom
import astroalignMatch
import os
import time
import json
import sys
from enum import Enum

class Algorithm(Enum):
    RANSAC_AFFINE = 1
    MODIFIED_GROTH = 2

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        


cave = 21
algorithm = Algorithm.RANSAC_AFFINE

resultFilename = "results.%s.C%d.json" % ("aa" if algorithm == Algorithm.RANSAC_AFFINE else "customGroth", cave)
print(resultFilename)


totalResults = {}
prevDir = [2012, "June"]
currentDir = [2012, "Aug"]



for i in range(15):
    print(currentDir, prevDir)
    testingImages = DBUtil.get_images(cave, years=[currentDir[0]], months=[currentDir[1]], rootDirs=["../all_images/", "patched/results/"])
    databaseImages = DBUtil.get_images(cave, years=range(2012, prevDir[0] + 1), lastMonth = 0 if prevDir[1] == "June" else 1, rootDirs=["../all_images/", "patched/results/"])
    tagToFish, fishToTag = DBUtil.connectFish(cave, years=range(2012, currentDir[0] + 1), lastMonth = 0 if currentDir[1] == "June" else 1, rootDirs=["../all_images/", "patched/results/"])
    totalDirectoryProcessingTime = 0
    for keyPath in testingImages.keys():
        
        potentialMatches = tagToFish[fishToTag[keyPath]]
        potentialMatches.remove(keyPath)
        print("Testing: " + str(keyPath) + " (" + str(fishToTag[keyPath]) + "), Potential matches in DB: " + str(potentialMatches))
        sys.stdout.flush()
        currentTime = time.time()
        results = None
        if algorithm == Algorithm.RANSAC_AFFINE:
            results = astroalignMatch.findClosestMatch(keyPath, testingImages[keyPath], databaseImages, verbose = False, progress = False)
        elif algorithm == Algorithm.MODIFIED_GROTH:
            results = grothMatcherCustom.findClosestMatch(keyPath, testingImages[keyPath], databaseImages, progress = False, local_triangle_k = 25) #Added local_triangle_k
        else:
            print("Unrecognised algorithm")
        
        elapsedTime = time.time() - currentTime
        totalDirectoryProcessingTime += elapsedTime
        results = [list(entry) for entry in results]
        [entry.append(fishToTag[entry[-1]]) for entry in results]
        
#         order = sorted(results, key=lambda x: x[0], reverse = True)

#         order = [(entry[0], entry[1], fishToTag[entry[1]]) for entry in order if entry[0] != 0]
#         #order = [(entry[0], entry[1], fishToTag[entry[1]], entry[2]) for entry in order if entry[0] != 0]
        print(elapsedTime)
#         print(order)
        sys.stdout.flush()
        totalResults[keyPath] = {
            "tag": fishToTag[keyPath],
            "potentialMatches": potentialMatches,
            "results": results
        }
        
        
        with open(resultFilename, "w") as f:
            json.dump(totalResults, f, cls=NpEncoder)
        
        
    averageDirectoryTime = totalDirectoryProcessingTime / len(testingImages)
    print("Directory finished: ", currentDir[0], "/", currentDir[1], " ", totalDirectoryProcessingTime, averageDirectoryTime, len(databaseImages))
    sys.stdout.flush()
    
    prevDir[0] = currentDir[0]
    prevDir[1] = currentDir[1]
    
    currentDir[0] = currentDir[0] + 1 if currentDir[1] == "Aug" else currentDir[0]
    currentDir[1] = "June" if currentDir[1] == "Aug" else "Aug"
    
print("Finished")
    

        
