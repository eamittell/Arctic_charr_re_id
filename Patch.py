from PIL import Image
import math
import numpy as np


class PatchedImage():
    
    
    def __init__(self, patch_size, scale_limit, img, background_color = 0):
        resizedImg, self.scale_multiplier = scaleImg(img, patch_size, scale_limit)
        self.patch_size = patch_size
        self.scale_limit = scale_limit
        self.resizedImgSize = resizedImg.size
        self.size = (int(math.ceil(resizedImg.size[0] / patch_size[0])), int(math.ceil(resizedImg.size[1] / patch_size[1])))
        self.patches = list(splitToPatches(resizedImg, patch_size, background_color))
        
    def mapPatches(self, func):
        self.patches = [func(patch) for patch in self.patches]
        
    def streamPatches(self):
        for patch in self.patches:
            yield patch
        
    def joinPatches(self):
        recreated_scaled_img = Image.new(self.patches[0].mode, self.resizedImgSize, 0)
        loc = [0, 0]
        for patch in self.patches:
            croppedPatch = patch.crop((0, 0, min(self.patch_size[0], self.resizedImgSize[0] - loc[0]), min(self.patch_size[1], self.resizedImgSize[1] - loc[1])))
            recreated_scaled_img.paste(croppedPatch, (loc[0], loc[1]))
            loc[0] += self.patch_size[0]
            if loc[0] >= recreated_scaled_img.size[0]:
                loc[1] += self.patch_size[1]
                loc[0] = 0
        return recreated_scaled_img.resize((round(recreated_scaled_img.size[0] / self.scale_multiplier), round(recreated_scaled_img.size[1] / self.scale_multiplier)))
        
    
    def asarray(self, dtype = None, fullArray = True):
        if fullArray:
            if dtype is not None:
                return np.asarray([np.array(patch, dtype=dtype) for patch in self.patches], dtype=dtype)
            else:
                return np.asarray([np.array(patch) for patch in self.patches])
        else:
            if dtype is not None:
                return [np.array(patch, dtype = dtype) for patch in self.patches]
            else:
                return [np.array(patch) for patch in self.patches]
    
    def fromarray(self, npArrayPatches):
        if npArrayPatches.shape[-1] == 1:
            npArrayPatches = npArrayPatches.reshape(npArrayPatches.shape[0], npArrayPatches.shape[1], -1)
        self.patches = [Image.fromarray(arrayPatch) for arrayPatch in npArrayPatches]
        
    
def scaleImg(img, patch_size, scale_limit):
    scale_multiplier = 1
    resized_img = None
    #If roughly size of patch in x or y dimension, then scale
    if img.size[1] <= img.size[0] and img.size[1] <= patch_size[1] + scale_limit and img.size[1] > patch_size[1]:
        #Scale to fit y axis
        scale_multiplier = patch_size[1] / img.size[1]
        resized_img = img.resize((round(scale_multiplier * img.size[0]), round(scale_multiplier * img.size[1])))
    elif img.size[0] <= patch_size[0] + scale_limit and img.size[0] > patch_size[0]:
        #Scale to fit x axis
        scale_multiplier = patch_size[0] / img.size[0]
        resized_img = img.resize((round(scale_multiplier * img.size[0]), round(scale_multiplier * img.size[1])))
    else:
        resized_img = img
        
    return (resized_img, scale_multiplier)


def splitToPatches(img, patch_size, background_color):
    currentPos = [0, 0]
    while currentPos[1] < img.size[1]:
        patch = Image.new(img.mode, patch_size, background_color)
        patch.paste(img.crop((currentPos[0], currentPos[1], min(currentPos[0] + patch_size[0], img.size[0]), min(currentPos[1] + patch_size[1], img.size[1]))))
        yield patch
        currentPos[0] += patch_size[0]
        if currentPos[0] >= img.size[0]:
            currentPos[1] += patch_size[1]
            currentPos[0] = 0