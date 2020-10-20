import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import csv
import math
import json
from PIL import Image, ExifTags

def visualize(base, selected = None, triangsBase = None, triangsSelected = None, annotateOrder = False, invertYAxis = False, figsize=(5,5)):
    fig = plt.figure(figsize=figsize)
    fig.add_axes([0, 0, 1, 1])
    ax = fig.axes[0]
    ax.set_xlim(xmin = -0.1, xmax = 1.1)
    ax.set_ylim(ymin = -0.1, ymax = 1.1)
    base = np.copy(base)
    selected = np.copy(selected)
    if invertYAxis:
        base[:,1] = 0.7 - base[:,1]
        selected[:,1] = 0.7 - selected[:,1]
    ax.scatter(base[:,0],  base[:,1], color='red', label='Base', marker="+")
    if selected is not None:
        ax.scatter(selected[:,0],  selected[:,1], color='blue', label='Selected', marker="x")#s = reversed(np.arange(3, len(selected) + 3)), marker="x")
    
    if triangsBase is not None:
        for triang in triangsBase:
            poly = plt.Polygon(triang, ec='red', fc = 'none', alpha = 0.1)
            ax.add_patch(poly)
       
    if triangsSelected is not None:
        for triang in triangsSelected:
            poly = plt.Polygon(triang, ec='blue', fc='none', alpha = 0.1)
            ax.add_patch(poly)
            
    if annotateOrder:
        for i in range(0, len(selected)):
            ax.annotate(i + 1, (selected[i][0], selected[i][1]))
    
    ax.legend(loc='upper left', fontsize='large')
    plt.draw()
    plt.show()

def cv2_imshow(image, title=None, cmap=None, dpi=100):
    image = image.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 2 and cmap == None:
        #Is grayscale
        cmap="gray"
        
    shape=np.shape(image)[0:2][::-1]
    size = [float(i)/dpi for i in shape]

    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig,[0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap=cmap)

    plt.show()
    
def find_center_mask(image, min_mask_area = 10000):
    h, w = image.shape
    
    centerX = int(w / 2)
    centerY = int(h / 2)
    contours, hier = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contourSorted = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_mask_area:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            distanceFromCenter = math.sqrt(pow(centerX - cx, 2) + pow(centerY - cy, 2))
            contourSorted.append((distanceFromCenter, cnt))
    contourSorted = sorted(contourSorted, key=lambda x : x[0])
    if len(contourSorted) > 0:
        cnt = contourSorted[0][1]
        newImage = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(newImage,[cnt],0,255,-1)
        return newImage
    return image

def get_image_mask(img):
    blurred = cv2.GaussianBlur(img, (125, 125), 125)
    ret, thresh = cv2.threshold(blurred,80,140,cv2.THRESH_BINARY_INV)
    # Fill in holes in mask
    mainMask = find_center_mask(thresh)
    contour,hier = cv2.findContours(mainMask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mainMask,[cnt],0,255,-1)
    return mainMask

def get_difference_of_gaussians(img, s = 6, s2 = None):
    if s2 is None:
        s2 = s * 2
    smoothImg = cv2.GaussianBlur(img , (21, 21), 5)
    return cv2.GaussianBlur(smoothImg, (21, 21), s) - cv2.GaussianBlur(smoothImg, (21, 21), s2)

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    numChannels = img.shape[2] if len(img.shape) > 2 else 1
    mask = img>tol
    if numChannels > 1:
        return img[np.ix_(mask.any(1),mask.any(0),[i for i in range(numChannels)])]
    else:
        return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_to_mask(img,mask,tol=0):
    #mask and img should be same size
    # img is 2D image data
    # tol  is tolerance
    numChannels = img.shape[2] if len(img.shape) > 2 else 1
    mask = mask>tol
    if numChannels > 1:
        return img[np.ix_(mask.any(1),mask.any(0),[i for i in range(numChannels)])]
    else:
        return img[np.ix_(mask.any(1),mask.any(0))]

def getBinaryMaskPrecisionRecall(mask, target):
    TP_mat = np.logical_and(generated, label)
    FP_mat = np.logical_and(generated, np.invert(label))
    TN_mat = np.logical_and(np.invert(generated), np.invert(label))
    FN_mat = np.logical_and(np.invert(generated), label)
    TP = np.sum(TP_mat)
    FP = np.sum(FP_mat)
    TN = np.sum(TN_mat)
    FN = np.sum(FN_mat)
        
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    
    return (precision, recall)


def resize_ratio(img, targetRatio):
    w =  img.shape[1]
    h =  img.shape[0]
    currentRatio = w/h
    if (targetRatio > 1 and currentRatio < 1) or (targetRatio < 1 and currentRatio > 1):
        # rotate image first
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle90, scale)
        
    currentRatio = img.shape[1]/img.shape[0]
    
    
def getPointsFromImage(img, limit = None):
    contour,hier = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    spotsDrawn = 0
    points = []
    for cnt in sorted(contour, key=cv2.contourArea, reverse=True):
        
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        points.append((cx, cy))
        spotsDrawn += 1
    return points

def mergePointsToImage(data, target, border = 20):
    maxPointData = np.amax(data, axis=0) 
    maxPointTarget = np.amax(target, axis=0)
    
    minPointData = np.amin(data, axis=0) 
    minPointTarget = np.amin(target, axis=0)
    print(minPointData[0], type(minPointData[0]))
    print(maxPointData[0])
    minX = min(minPointData[0][0], minPointTarget[0][0])
    minY = min(minPointData[0][1], minPointTarget[0][1])
    
    maxX = max(maxPointData[0][0], maxPointTarget[0][0])
    maxY = max(maxPointData[0][1], maxPointTarget[0][1])
    
    dataShifted = np.apply_along_axis(lambda p : np.array[p[0] - minX + border, p[1] - minY + border], 0, data)
    targetShifted = np.apply_along_axis(lambda p : np.array[p[0] - minX + border, p[1] - minY + border], 0, target)
    
    newImageTarget = np.zeros((maxY - minY + border * 2, maxX - minX + border * 2), dtype=np.uint8)
    for p in targetShifted:
        try:
            newImage[int(p[1]), int(p[0])] = 255
        except Exception as e:
            print(e)
    
    newImageData = np.zeros((maxY - minY + border * 2, maxX - minX + border * 2), dtype=np.uint8)
    for p in dataShifted:
        try:
            newImage[int(p[1]), int(p[0])] = 255
        except Exception as e:
            print(e)
            
    return cv2.merge([newImageTarget,newImageData,np.zeros((maxY - minY + border * 2, maxX - minX + border * 2), dtype=np.uint8)])
    



def pointcloud_to_image(pc, xSize, ySize):
    translatedImage = np.zeros((ySize, xSize), dtype=np.uint8)
    for p in np.asarray(pc.points):
        try:
            translatedImage[int(p[0]), int(p[1])] = 255
        except Exception as e:
            pass
    return translatedImage


def expand2square(pil_img, background_color = 0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def get_actual_img_bounds(img):
    image_data_bw = np.array(img)
    if (image_data_bw.ndim > 2):
        image_data_bw = image_data_bw.max(axis=2)
        
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    return cropBox
    
def crop_to_bounds(img, bounds):
    image_data = np.array(img)
    if (image_data.ndim == 2):
        return Image.fromarray(image_data[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1])
    else:
        return Image.fromarray(image_data[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1 , :])
    
def get_average_precision_recall(generatedMasks, labels, names = None, verbose = False, dpi = 250):
    precisionTotal = 0
    recallTotal = 0
    for i in range(len(generatedMasks)):
        if verbose:
            print(i, names[i] if names is not None else "")
        generated = generatedMasks[i]
        label = labels[i]
        TP_mat = np.logical_and(generated, label)
        FP_mat = np.logical_and(generated, np.invert(label))
        TN_mat = np.logical_and(np.invert(generated), np.invert(label))
        FN_mat = np.logical_and(np.invert(generated), label)
        TP = np.sum(TP_mat)
        FP = np.sum(FP_mat)
        TN = np.sum(TN_mat)
        FN = np.sum(FN_mat)
        
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        if verbose:
            output = np.zeros(shape=(label.shape[0], label.shape[1], 3), dtype=np.uint8)
            output[:,:,2] = FP_mat.astype(dtype=np.uint8) * 255
            output[:,:,1] = TP_mat.astype(dtype=np.uint8) * 255
            output[:,:,0] = FN_mat.astype(dtype=np.uint8) * 255
            cv2_imshow(output, dpi = dpi)

            print('TP: ', TP, '\nFP: ', FP, '\nTN: ', TN, '\nFN: ', FN)
            print("Precision: ", precision, "\nRecall: ", recall, )
        precisionTotal += precision
        recallTotal += recall
    return precisionTotal / len(generatedMasks), recallTotal / len(generatedMasks)


def read_PIL_image(path, with_rotation = True):
    pic = Image.open(path) 
    if with_rotation and pic.size[0] < pic.size[1]:
        pic=pic.rotate(90, expand=True)
    return pic

def get_normalise_direction_matrix(img):
    y, x = np.nonzero(img)
    centerX = np.mean(x)
    centerY = np.mean(y)
    x = x - centerX
    y = y - centerY
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    theta = np.arctan((y_v1)/(x_v1)) * (180 / math.pi)
    return cv2.getRotationMatrix2D((centerX, centerY), theta, 1.0)