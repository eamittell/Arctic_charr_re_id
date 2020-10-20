import os
import pandas as pd
from enum import Enum

class DateOrder(Enum):
    YEAR_MONTH = 1,
    MONTH_YEAR = 2
    
def generate_dates(years, months, firstMonth = 0, lastMonth=1, order = DateOrder.YEAR_MONTH, sep = "_"):
    """Generate dates for the given years and months

    Args:
        years (Iterable[int]): The years for which to generate dates
        months (Iterable[str]): The months for which to generate dates
        firstMonth (int, optional): The index of the month (in the months list) from which to start generating dates in the first year. Defaults to 0.
        lastMonth (int, optional): The index of the month (in the months list) to which the dates shuold be generated in the final year. Defaults to 1.
        order (DateOrder, optional): The order to yield dates. Defaults to DateOrder.YEAR_MONTH.
        sep (str, optional): The seperator to use between components. Defaults to "_".

    Yields:
        str: The specified dates
    """
    for year in years:
        monthIdx = 0
        for month in months:
            if year == years[0] and monthIdx < firstMonth:
                pass
            yield str(year) + sep + str(month) if order == DateOrder.YEAR_MONTH else str(month) + sep + str(year), year, monthIdx
            
            if year == years[-1] and monthIdx == lastMonth:
                break
            monthIdx += 1

def _dir_generator(caveNum, rootDirs=["../all_images/", "results"], years=range(2012, 2020), months=["June", "Aug"], firstMonth = 1, lastMonth=1, verbose = False):
    """Generates directory pths for the given parameters

    Args:
        caveNum (int): The number of the cave to generate directories for.
        rootDirs (list, optional): The root directories to generate directories from. Defaults to ["../all_images/", "results"].
        years ([type], optional): The years for which to generate directory paths. Defaults to range(2012, 2020).
        months (list, optional): The months for which to generate directoy paths. Defaults to ["June", "Aug"].
        firstMonth (int, optional): The index of the month (in the months list) from which to start generating dates in the first year. Defaults to 1.
        lastMonth (int, optional): The index of the month (in the months list) to which the dates shuold be generated in the final year. Defaults to 1.
        verbose (bool, optional): If to print verbose information. Defaults to False.

    Yields:
        str: Directory paths for the specified dates
    """
    for rootDir in rootDirs:
        for date, year, monthIdx in generate_dates(years = years, months = months, firstMonth=firstMonth, lastMonth=lastMonth):
            directory = os.path.join(rootDir,date, "Cave" + str(caveNum))
            if os.path.isdir(directory):
                yield directory, str(caveNum), year, monthIdx
            elif verbose:
                print("%s is not a directory" % directory)
                
                
def _assignToFish(images, fileKey, key, value):
    """Helper function to assign value to fish which may not exist"""
    if fileKey not in images:
        images[fileKey] = {"img": None, "mask": None, "maskLabel": None, "spotsLabel": None, "spots": None, "spotsJson": None, "precomp": None, "precompAA": None}
    images[fileKey][key] = value

def get_images(caveNum, rootDirs=["../all_images/", "results"], years=range(2012, 2020), months=["June", "Aug"], firstMonth = 0, lastMonth = 1, verbose = False):
    """Creates database of images from directories
    """
    images = {}
    for directory, cave, year, monthIdx in _dir_generator(caveNum, rootDirs = rootDirs, years = years, months = months, firstMonth = firstMonth, lastMonth = lastMonth, verbose = verbose):
        for file in sorted(os.listdir(directory)):
            filePath = os.path.join(directory, file)
            if os.path.isfile(filePath) and ('IMG' in file or 'DSC' in file) and (".xcf" not in file):
                fileComponents = file.split(".")
                fileKey = "C%s-%s-%s-%s" % (cave, str(year), months[monthIdx], fileComponents[0])
                

                if len(fileComponents) == 2 and fileComponents[-1].lower() in ["jpg", "jpeg", "png", "bmp"]:
                    _assignToFish(images, fileKey, "img", filePath)
                elif len(fileComponents) >= 2 and fileComponents[-1].lower() == "pickle":
                    if len(fileComponents) >= 3 and fileComponents[1] == 'aa':
                        _assignToFish(images, fileKey, "precompAA", filePath)
                    else:
                        _assignToFish(images, fileKey, "precomp", filePath)
                elif len(fileComponents) >= 3 and fileComponents[2] == "mask":
                    if len(fileComponents) >= 4 and fileComponents[3] == "acc":
                        _assignToFish(images, fileKey, "maskLabel", filePath)
                    else:
                        _assignToFish(images, fileKey, "mask", filePath)
                elif len(fileComponents) >= 3 and fileComponents[2] == "spots" and fileComponents[-1] == "json":
                    _assignToFish(images, fileKey, "spotsJson", filePath)
                elif len(fileComponents) >= 3 and fileComponents[2] == "spots" or fileComponents[1] == "spots":
                    if (len(fileComponents) >= 4 and fileComponents[3] == "acc") or fileComponents[2] == "acc":
                        _assignToFish(images, fileKey, "spotsLabel", filePath)
                    else:
                        _assignToFish(images, fileKey, "spots", filePath)

                        
    return images

def connectFish(caveNum, rootDirs=["../all_images/", "results"], years=range(2012, 2020), months=["June", "Aug"], tagTranslation = {}, firstMonth = 0, lastMonth=1, verbose = False):
    """Connects fish based on csv files in the given directories
    """
    fish = {}
    inverseMapping = {}
    for directory, cave, year, monthIdx in _dir_generator(caveNum, rootDirs = rootDirs, years = years, months = months, firstMonth = firstMonth, lastMonth = lastMonth, verbose = verbose):
        metadataFound = False
        for file in os.listdir(directory):
            filePath = os.path.join(directory, file)
            if os.path.isfile(filePath) and (file.startswith("filename") or file.startswith("newfilename")):
                df = None
                if file.split(".")[-1].lower() == 'csv':
                    df = pd.read_csv(filePath, delimiter = ';', quotechar="'")
                elif file.split(".")[-1].lower() == 'xlsx':
                    df = pd.read_excel(filePath, sheet_name="Sheet1")
                
                if df is not None:
                    if verbose:
                        print(filePath)
                    try:
                        for index, row in df.iterrows():
                            tag = str(row["ID"])
                            if tag.lower() == "nan":
                                if "comments" in row and str(row["comments"]).lower() != "nan":
                                    tag = "c:%s" % str(row["comments"])
                                else:
                                    tag = None
                            if tagTranslation is not None and tag in tagTranslation:
                                tag = tagTranslation[tag]

                            if str(row["filename"]).lower() == "nan":
                                if verbose:
                                    print("Row %d in metadatafile %s does not list an image filename!" % (index, filePath))
                            else:

                                keyPath = "C%s-%s-%s-%s" % (cave, str(year), months[monthIdx], row["filename"].split(".")[0]) 
                                ## !!------------------ File entry in CSV file is wrong. Fix here as cannot modfiy source files --------------------!! ##
                                if keyPath == 'C21-2017-Aug-IMG_3262':
                                    keyPath = 'C21-2017-Aug-IMG_3263'
                                ## !!------------------                            End of fix                                   --------------------!! ##
                                inverseMapping[keyPath] = tag
                                try:
                                    fish[tag].append(keyPath)
                                except:
                                    fish[tag] = [keyPath]
                        metadataFound = True
                    except Exception as e:
                        if verbose:
                            print("Error occured. Looking for different metadata file...")
                        metadataFound = False
                if metadataFound:
                    break
                    
        if not metadataFound and verbose:
            print("Cannot find metadata file for directory: %s" % directory)
                
                        
    return fish, inverseMapping

