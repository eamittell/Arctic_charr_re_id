# https://github.com/karolzak/keras-unet
from keras_unet.models import custom_unet
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras.callbacks import ModelCheckpoint
from keras_unet.utils import get_augmented
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image
from keras_unet.utils import plot_imgs
import tensorflow as tf
import math
from matplotlib.pyplot import imshow
from util import *
from Patch import PatchedImage
import json
import os

class UnetSpotExtractor():
    
    def __init__(self, inputShape = (512, 512, 3), modelWeights = None, dropout = 0.05, useGPU = True):
        self.inputShape = inputShape
        self.model = custom_unet(
                        input_shape=inputShape,
                        use_batch_norm=False,
                        filters=64,
                        dropout=dropout,
                        num_layers=4,
                        output_activation='sigmoid')
        self.useGPU = useGPU
        if modelWeights is not None:
            try:
                self.model.load_weights(modelWeights)
                self.trained = True
            except IOError as e:
                print(e)
                self.trained = False
        else:
            self.trained = False
            
    def train(self, training_generator, validation_generator = None, epochs = 100, steps_per_epoch = 50, weights_save_file = 'unet_weights/unet_spots_{epoch:02d}.hdf5', loss='binary_crossentropy', metrics = ['accuracy', iou, iou_thresholded], additionalCallbacks = []):

        callbacks = additionalCallbacks
        if weights_save_file is not None:
            model_checkpoint = ModelCheckpoint(weights_save_file, monitor='loss', verbose=1, save_best_only=True)
            callbacks.append(model_checkpoint)

        deviceToUse = '/device:CPU:0'
        if self.useGPU:
            deviceToUse = '/device:GPU:0'
        with tf.device(deviceToUse):
            self.model.compile(optimizer = Adam(lr = 1e-4), loss = loss, metrics = metrics)
            history = self.model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_generator, validation_steps=2, callbacks=[model_checkpoint])
            self.trained = True
            return history
            
    def loadWeights(self, weightFile):
        self.model.load_weights(weightFile)
                     
                
    def predict(self, images, batchSize = 5):
        
        deviceToUse = '/device:CPU:0'
        if self.useGPU:
            deviceToUse = '/device:GPU:0'
        if self.trained:
            
            batches = np.array_split(images, math.ceil(len(images) / batchSize))
            print(len(images), batchSize, len(batches))
            results = []
            with tf.device(deviceToUse):
                for batch in batches:
                    results.append(self.model.predict(batch))
            return np.concatenate(results, axis = 0)
        else:
            print("Model not trained")
            return images
            
        
            
        
    def createAugmentedGenerator(self, imgs, labels, data_gen_args = dict(rotation_range=5,
                                                                            width_shift_range=0.05,
                                                                            height_shift_range=0.05,
                                                                            shear_range=0.1,
                                                                            zoom_range=0.2,
                                                                            horizontal_flip=True,
                                                                            vertical_flip=True,
                                                                            fill_mode='constant'),
                                                     validation_size=0.1, seed_1 = 42, seed_2 = 42,
                                                     seed_3 = 21, batch_size_train = 5, batch_size_validate = 5):
        trainX = imgs
        trainY = labels
        testX = None
        testY = None
        if validation_size > 0:
            trainX, testX, trainY, testY = train_test_split(imgs, labels, test_size=validation_size, random_state=seed_1)
        
        imageGenerator = get_augmented(
            trainX, trainY, batch_size=batch_size_train,
            data_gen_args = data_gen_args, seed=seed_2)

        if validation_size > 0:
            if 'channel_shift_range' in data_gen_args:
                del data_gen_args['channel_shift_range']
                
            if 'brightness_range' in data_gen_args:
                del data_gen_args['brightness_range']
            validationImageGenerator = get_augmented(
                testX, testY, batch_size=batch_size_validate,
                data_gen_args = data_gen_args, seed = seed_3)
            return imageGenerator, validationImageGenerator, testX, testY
        
        return imageGenerator
        
        
    # Training input looks different from normal input
    def prepareTrainingInput(self, imgs, annotations, masks, patchedImageMode = False, verbose = False):
        i = 0
        imgs_list = []
        annotations_list = []
        for image, annotation, fishMask in zip(imgs, annotations, masks):
            img = read_PIL_image(image)
            # If input to model is grayscale, convert to grayscale
            if self.inputShape[2] == 1:
                img = img.convert("L")
            
            fm = read_PIL_image(fishMask).convert('L')
            maskBounds = get_actual_img_bounds(fm)
            
            colorMode = "L" if self.inputShape[2] == 1 else "RGB"
            img_masked = Image.new(colorMode, img.size)
            img_masked.paste(img, mask = fm)
            img_masked = crop_to_bounds(img_masked, maskBounds)
            
            if patchedImageMode:
                patchedImage = PatchedImage((self.inputShape[0], self.inputShape[1]), min(self.inputShape[0], self.inputShape[1]) / 4, img_masked)
                imgs_list.extend(patchedImage.asarray(fullArray = False))
            else:
                ratio = img_masked.size[1]/img_masked.size[0]
                img_masked = expand2square(img_masked.resize((self.inputShape[0], round(self.inputShape[0] * ratio))))
                imgs_list.append(np.array(img_masked))


            annotationImg = read_PIL_image(annotation).convert('L')
            annot_masked = Image.new("L", annotationImg.size)
            annot_masked.paste(annotationImg, mask = fm)
            annot_masked = crop_to_bounds(annot_masked, maskBounds)
            if patchedImageMode:
                patchedImage = PatchedImage((self.inputShape[0], self.inputShape[1]), min(self.inputShape[0], self.inputShape[1]) / 4, annot_masked)
                annotations_list.extend(patchedImage.asarray(fullArray = False))
            else:
                annot_masked = expand2square(annot_masked.resize((self.inputShape[0], round(self.inputShape[0] * ratio))))
                annotations_list.append(np.array(annot_masked))
            if verbose:
                print(i, image, img_masked.size, annot_masked.size, img_masked.size[1]/img_masked.size[0])
            i += 1
        imgs_arr = np.asarray(imgs_list, dtype=np.float32) / 255
        annotations_arr = np.asarray(annotations_list, dtype = np.float32) / 255
        return (imgs_arr.reshape(imgs_arr.shape[0], imgs_arr.shape[1], imgs_arr.shape[2], self.inputShape[2]),
                annotations_arr.reshape(annotations_arr.shape[0], annotations_arr.shape[1], annotations_arr.shape[2], 1)
               )
    
    def generate_spots(self, imagePaths, maskPaths, suffix = ".spots.png", suffix_json_spots = '.spots.json', batch_size = 5, verbose = False, threshold = -1, outputDir=None, patchedImageMode = False):
        if self.trained:
            batchNum = 1
            batches_imgs = np.array_split(imagePaths, math.ceil(len(imagePaths) / batch_size))
            batches_masks = np.array_split(maskPaths, math.ceil(len(imagePaths) / batch_size))
            for batchNum in range(len(batches_imgs)):
                img_batch = batches_imgs[batchNum]
                mask_batch = batches_masks[batchNum]
                if verbose:
                    print("Batch: %i/%i" % (batchNum + 1, len(batches_masks)))
                imgs_list = []
                i = 0
                for image, mask in zip(img_batch, mask_batch):
                    img = read_PIL_image(image)
                    # If input to model is grayscale, convert to grayscale
                    if self.inputShape[2] == 1:
                        img = img.convert("L")
                    
                    if mask is None:
                        if verbose:
                            print("No mask avaliable for: %s" % image)
                        continue
                    fm = read_PIL_image(mask).convert('L')
                    
                    if len(np.where(np.asarray(fm).max(axis=0)>0)[0]) <= 0:
                        if verbose:
                            print("No mask present in: %s" % mask)
                        continue
                    
                    maskBounds = get_actual_img_bounds(fm)

                    colorMode = "L" if self.inputShape[2] == 1 else "RGB"
                    img_masked = Image.new(colorMode, img.size)
                    img_masked.paste(img, mask = fm)
                    img_masked = crop_to_bounds(img_masked, maskBounds)
                    
                    originalShape = img_masked.size

                    ratio = img_masked.size[1]/img_masked.size[0]
                    img_masked = expand2square(img_masked.resize((self.inputShape[0], round(self.inputShape[0] * ratio))))
                    
                    if verbose:
                        print(i, image, img_masked.size, originalShape)
                    if img_masked.size[0] != self.inputShape[0] or img_masked.size[1] != self.inputShape[1]:
                        if verbose:
                            print("Something went wrong: Invalid masked image size.")
                    else:
                        imgs_list.append((np.array(img_masked), image, ratio, originalShape))
                    i += 1
                if verbose:
                    print("Predicting...")
                if len(imgs_list) > 0:
                    only_img_list = np.asarray([x[0] for x in imgs_list])
                    spots = self.predict(only_img_list, batchSize = batch_size)
                    if threshold >= 0:
                        spots[spots > threshold] = 1
                        spots[spots <= threshold] = 0

                    imgs_list = [(imgs_list[i][0], imgs_list[i][1], imgs_list[i][2], imgs_list[i][3], spots[i]) for i in range(len(imgs_list))]
                    if verbose:
                        print("Saving...")

                    for img, imageName, ratio, originalShape, spots in imgs_list:
                        #Figure out right height
                        #spots = spots.reshape(spots.shape[0], spots.shape[1])
                        currentSpotsHeight = ratio * spots.shape[1]
                        spots = spots.reshape(spots.shape[0], spots.shape[1])
                        print(spots.shape, [round(spots.shape[1] / 2 - currentSpotsHeight / 2), round(spots.shape[1] / 2 + currentSpotsHeight / 2), 0, spots.shape[0]])
                        newSpots = crop_to_bounds(spots, [round(spots.shape[1] / 2 - currentSpotsHeight / 2), round(spots.shape[1] / 2 + currentSpotsHeight / 2), 0, spots.shape[0]])
                        newSpots = np.asarray(newSpots.resize(originalShape))
                        newSpots = np.asarray(newSpots * 255, dtype = np.uint8)
                        if outputDir is None:
                            print(imageName + suffix)
                            Image.fromarray(newSpots).save(imageName + suffix, suffix.split(".")[-1])
                            with open(imageName + suffix_json_spots, 'w') as f:
                                json.dump(getPointsFromImage(newSpots), f)
                        elif callable(outputDir):
                            print(outputDir(imageName)  + suffix)
                            Image.fromarray(newSpots).save(outputDir(imageName) + suffix, suffix.split(".")[-1])
                            with open(outputDir(imageName) + suffix_json_spots, 'w') as f:
                                json.dump(getPointsFromImage(newSpots), f)
                        else:
                            print(outputDir + os.sep + imageName.split(os.sep)[-1] + suffix)
                            Image.fromarray(newSpots).save(outputDir + os.sep + imageName.split(os.sep)[-1] + suffix, suffix.split(".")[-1])
                            with open(outputDir + os.sep + imageName.split(os.sep)[-1] + suffix_json_spots, 'w') as f:
                                json.dump(getPointsFromImage(newSpots), f)
                else:
                    print("No images in batch to process...")
                    
                    
                batchNum += 1
        else:
            print("Network not trained")
            
    def generate_spots_patched(self, imagePaths, maskPaths, suffix = ".spots.png", suffix_json_spots = '.spots.json', batch_size = 5, verbose = False, threshold = -1, outputDir=None):
        if self.trained:
            img_num = 0
            for image, mask in zip(imagePaths, maskPaths):
                if verbose:
                    print("(%i/%i) Image: %s" % (img_num + 1, len(imagePaths), image))
                
                img = read_PIL_image(image)
                # If input to model is grayscale, convert to grayscale
                if self.inputShape[2] == 1:
                    img = img.convert("L")

                if mask is None:
                    if verbose:
                        print("No mask avaliable for: %s" % image)
                    continue
                fm = read_PIL_image(mask).convert('L')

                if len(np.where(np.asarray(fm).max(axis=0)>0)[0]) <= 0:
                    if verbose:
                        print("No mask present in: %s" % mask)
                    continue

                maskBounds = get_actual_img_bounds(fm)

                colorMode = "L" if self.inputShape[2] == 1 else "RGB"
                img_masked = Image.new(colorMode, img.size)
                img_masked.paste(img, mask = fm)
                img_masked = crop_to_bounds(img_masked, maskBounds)

                patchedImage = PatchedImage((self.inputShape[0], self.inputShape[1]), min(self.inputShape[0], self.inputShape[1]) / 4, img_masked)

                if verbose:
                    print("Predicting...", img_masked.size, patchedImage.size)
                print(patchedImage.asarray(dtype = np.float32).shape)
                    
                spots = self.predict(patchedImage.asarray(dtype = np.float32) / 255, batchSize = batch_size)
                
                
                if verbose:
                    print("Saving...")
                    
                if threshold >= 0:
                    spots[spots > threshold] = 1
                    spots[spots <= threshold] = 0

                print(spots.shape)
                patchedImage.fromarray((spots * 255).astype(np.uint8))
                
                spotsFullImg = patchedImage.joinPatches()
                print(spotsFullImg.size)

                if outputDir is None:
                    if verbose:
                        print(image + suffix)
                    spotsFullImg.save(image + suffix, suffix.split(".")[-1])
                    with open(image + suffix_json_spots, 'w') as f:
                        json.dump(getPointsFromImage(np.asarray(spotsFullImg)), f)
                elif callable(outputDir):
                    if verbose:
                        print(outputDir(image)  + suffix)
                    spotsFullImg.save(outputDir(image) + suffix, suffix.split(".")[-1])
                    with open(outputDir(image) + suffix_json_spots, 'w') as f:
                        json.dump(getPointsFromImage(np.asarray(spotsFullImg)), f)
                else:
                    if verbose:
                        print(outputDir + os.sep + image.split(os.sep)[-1] + suffix)
                    spotsFullImg.save(outputDir + os.sep + image.split(os.sep)[-1] + suffix, suffix.split(".")[-1])
                    with open(outputDir + os.sep + image.split(os.sep)[-1] + suffix_json_spots, 'w') as f:
                        json.dump(getPointsFromImage(np.asarray(spotsFullImg)), f)
                    
                    
                img_num += 1
        else:
            print("Network not trained")
    
    
class UnetMaskExtractor():
    
    def __init__(self, inputShape = (512, 512, 3), modelWeights = None, useGPU = True, dropout = 0.05):
        self.inputShape = inputShape
        self.model = custom_unet(
                        input_shape=inputShape,
                        use_batch_norm=False,
                        filters=64,
                        dropout=dropout,
                        num_layers=4,
                        output_activation='sigmoid')
        self.useGPU = useGPU
        if modelWeights is not None:
            try:
                self.model.load_weights(modelWeights)
                self.trained = True
            except IOError as e:
                print(e)
                self.trained = False
        else:
            self.trained = False
            
    def train(self, training_generator, validation_generator = None, epochs = 100, steps_per_epoch = 50, weights_save_file = 'unet_weights/unet_mask_{epoch:02d}.hdf5', loss='binary_crossentropy', metrics = ['accuracy', iou, iou_thresholded], additionalCallbacks = []):
        
        if not self.trained:
            callbacks = additionalCallbacks
            if weights_save_file is not None:
                model_checkpoint = ModelCheckpoint(weights_save_file, monitor='loss', verbose=1, save_best_only=True)
                callbacks.append(model_checkpoint)
                

            deviceToUse = '/device:CPU:0'
            if self.useGPU:
                deviceToUse = '/device:GPU:0'
            with tf.device(deviceToUse):
                self.model.compile(optimizer = Adam(lr = 1e-4), loss = loss, metrics = metrics)
                history = self.model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_generator, validation_steps=2, callbacks=callbacks)
                self.trained = True
                return history
        else:
            print("Model already trained")
            
    def loadWeights(self, weightFile):
        self.model.load_weights(weightFile)
                     
                
    def predict(self, images, batchSize = 5, postProcess = True, threshold = 0.31):
        
        deviceToUse = '/device:CPU:0'
        if self.useGPU:
            deviceToUse = '/device:GPU:0'
        if self.trained:
            batches = np.array_split(images, math.ceil(len(images)/batchSize))
            results = []
            with tf.device(deviceToUse):
                for batch in batches:
                    if postProcess:
                        results.append(self.postProcessPredictions(self.model.predict(batch), threshold = threshold))
                    else:
                        results.append(self.model.predict(batch))
            
            return np.concatenate(results, axis = 0)
        else:
            print("Model not trained")
            return images
            
            
    def postProcessPredictions(self, predictions, threshold = 0.31, kernel_size=(3,3), iterations = 3, min_mask_area = 100):
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], predictions.shape[2])
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
        kernel = np.ones(kernel_size,np.uint8)
        dilated_images = []
        for img in predictions:
            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = img.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            im_floodfill = img.copy()
            # Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0,0), 1);

            # Invert floodfilled image
            im_floodfill_inv = 1 - im_floodfill
            # Combine the two images to get the foreground.
            im_out = np.bitwise_or(img.astype(np.uint8), im_floodfill_inv.astype(np.uint8))
            
            #Apply erosion followed by expansion to merge up separated predictions
            dilated = cv2.morphologyEx(im_out, cv2.MORPH_CLOSE, kernel, iterations = iterations)
            dilated_images.append(find_center_mask(np.asarray(dilated * 255, dtype = np.uint8), min_mask_area = min_mask_area))

        return np.stack(dilated_images, axis = 0)
        
        
            
        
    def createAugmentedGenerator(self, imgs, labels, data_gen_args = dict(rotation_range=5,
                                                                            width_shift_range=0.05,
                                                                            height_shift_range=0.05,
                                                                            shear_range=0.1,
                                                                            zoom_range=0.2,
                                                                            horizontal_flip=True,
                                                                            vertical_flip=True,
                                                                            fill_mode='constant'),
                                                     validation_size=0.1, seed_1 = 42, seed_2 = 42,
                                                     seed_3 = 21, batch_size_train = 5, batch_size_validate = 5):
        trainX = imgs
        trainY = labels
        testX = None
        testY = None
        if validation_size > 0:
            trainX, testX, trainY, testY = train_test_split(imgs, labels, test_size=validation_size, random_state=seed_1)
        
        
        imageGenerator = get_augmented(
            trainX, trainY, batch_size=batch_size_train,
            data_gen_args = data_gen_args, seed=seed_2)

        if validation_size > 0:
            if 'channel_shift_range' in data_gen_args:
                del data_gen_args['channel_shift_range']
                
            if 'brightness_range' in data_gen_args:
                del data_gen_args['brightness_range']
            print(data_gen_args)
            validationImageGenerator = get_augmented(
                testX, testY, batch_size=batch_size_validate,
                data_gen_args = data_gen_args, seed = seed_3)
            return imageGenerator, validationImageGenerator, testX, testY
        
        return imageGenerator
    
    def prepareInput(self, imgs, verbose = False):
        imgs_list = []
        i = 0
        for image in imgs:
            img = read_PIL_image(image)
            # If input to model is grayscale, convert to grayscale
            if self.inputShape[2] == 1:
                img = img.convert("L")

            ratio = img.size[1]/img.size[0]
            img = expand2square(img.resize((self.inputShape[0], round(self.inputShape[0] * ratio))))
            if verbose:
                print(i, image, img.size)
            imgs_list.append(np.array(img))
            i += 1
        return imgs_list
        
    # Training input looks different from normal input
    def prepareTrainingInput(self, imgs, annotations, verbose = False):
        i = 0
        imgs_list = []
        annotations_list = []
        for image, annotation in zip(imgs, annotations):
            img = read_PIL_image(image)
            # If input to model is grayscale, convert to grayscale
            if self.inputShape[2] == 1:
                img = img.convert("L")

            ratio = img.size[1]/img.size[0]
            if verbose:
                print(img.size, ratio)
            img = expand2square(img.resize((self.inputShape[0], round(self.inputShape[0] * ratio))))
            imgs_list.append(np.array(img))


            annotationImg = read_PIL_image(annotation).convert('L')
            annotationImg = expand2square(annotationImg.resize((self.inputShape[0], round(self.inputShape[0] * ratio))))
            annotations_list.append(np.array(annotationImg))
            if verbose:
                print(i, image, img.size, annotationImg.size, img.size[1]/img.size[0])
            i += 1
        imgs_arr = np.asarray(imgs_list, dtype=np.float32) / 255
        annotations_arr = np.asarray(annotations_list, dtype = np.float32) / 255
        return (imgs_arr.reshape(imgs_arr.shape[0], imgs_arr.shape[1], imgs_arr.shape[2], self.inputShape[2]),
                annotations_arr.reshape(annotations_arr.shape[0], annotations_arr.shape[1], annotations_arr.shape[2], 1)
               )
    
    def generate_masks(self, imagePaths, suffix = ".mask.png", batch_size = 5, verbose = False, outputDir=None, threshold = 0.31):
        if self.trained:
            batchNum = 1
            batches = np.array_split(imagePaths, math.ceil(len(imagePaths) / batch_size))
            for batch in batches:
                if verbose:
                    print("Batch: %i/%i" % (batchNum, len(batches)))
                imgs_list = []
                i = 0
                for image in batch:
                    origImg = read_PIL_image(image)
                    originalShape = origImg.size
                    # If input to model is grayscale, convert to grayscale
                    if self.inputShape[2] == 1:
                        origImg = origImg.convert("L")
                    ratio = origImg.size[1]/origImg.size[0]
                    tmp = origImg.resize((self.inputShape[0], round(self.inputShape[0] * ratio)))
                    img = expand2square(tmp)
                    if verbose:
                        print(i, image, originalShape, img.size, tmp.size)
                        i += 1
                    imgs_list.append(np.array((np.array(img), ratio, originalShape, image, np.asarray(origImg))))
                if verbose:
                    print("Predicting...")
                only_img_list = np.asarray([x[0] for x in imgs_list])
                masks = self.predict(only_img_list, batchSize = batch_size, threshold = threshold)
                
                imgs_list = [(imgs_list[i][0], imgs_list[i][1], imgs_list[i][2], imgs_list[i][3], imgs_list[i][4], masks[i]) for i in range(len(imgs_list))]
                if verbose:
                    print("Saving...")
                
                for img, ratio, originalShape, imageName, origImage, mask in imgs_list:
                    #Figure out right hieght
                    currentMaskHeight = ratio * mask.shape[1]
                    newMask = crop_to_bounds(mask, [round(mask.shape[1] / 2 - currentMaskHeight / 2), round(mask.shape[1] / 2 + currentMaskHeight / 2), 0, mask.shape[0]])
                    newMask = np.asarray(newMask.resize(originalShape))
                    if len(np.where(newMask.max(axis=0)>0)[0]) <= 0:
                        if verbose:
                            print("Could not extract fish mask succesfully from: %s" % imageName)
                    if outputDir is None:
                        Image.fromarray(newMask).save(imageName + suffix, suffix.split(".")[-1])
                    elif callable(outputDir):
                        Image.fromarray(newMask).save(outputDir(imageName) + suffix, suffix.split(".")[-1])
                    else:
                        Image.fromarray(newMask).save(outputDir + os.sep + imageName.split(os.sep)[-1] + suffix, suffix.split(".")[-1])
                batchNum += 1
        else:
            print("Network not trained")
    
def getPointsFromImage(img, limit = None):
    contour,hier = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    spotsDrawn = 0
    points = []
    for cnt in sorted(contour, key=cv2.contourArea, reverse=True):
        M = cv2.moments(cnt)
        cx = 0
        cy = 0
        if (M['m00'] == 0):
            cx = int(cnt[0][0][0])
            cy = int(cnt[0][0][1])
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        points.append((cx, cy, radius))
        spotsDrawn += 1
    return points


    
        