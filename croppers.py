import cv2
from sklearn.cluster import DBSCAN
from itertools import chain
from skimage.io import imread
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tf_unet import unet

class PatternMatchCropper:
    def __init__(self, threshold):
        self.tube_pattern = self.__preprocess_image(imread(os.path.join(os.path.dirname(__file__), "pipe-green.png"))[:25])
        self.inv_tube_pattern = np.flip(np.flip(self.tube_pattern, axis=0), axis=1)
        self.birb_pattern = self.__preprocess_image(imread(os.path.join(os.path.dirname(__file__), "yellowbird-downflap.png")))
        
        self.threshold = threshold
        
    def __preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(image, 50, 200)
    
    def __find_candidates(self, image, pattern, use_max=False):
        image = self.__preprocess_image(image)
        res = cv2.matchTemplate(image, pattern, cv2.TM_CCOEFF_NORMED)
        
        threshold = res.max() if use_max else self.threshold
        
        return np.where(res >= threshold)
    
    def __aggregate_candidates(self, candidates):
        if candidates[0].size == 0:
            return []
        
        dbscan = DBSCAN(min_samples=1, eps=1e1)
        points = np.vstack(candidates).T
        try:
            cluster_indices = dbscan.fit_predict(points)
        except:
            print(candidates)
            raise

        matches = pd.DataFrame(points[np.where(cluster_indices != -1)])
        matches["cluster"] = cluster_indices[np.where(cluster_indices != -1)]

        return np.flip(matches.groupby("cluster").mean().values.astype(int), axis=1)
    
    def __specify_tube_matches(self, matches, pattern, is_top=True):
        h, w = pattern.shape

        def specify_coords(coords):
            top_pt, bottom_pt = np.copy(coords), np.copy(coords)

            bottom_pt[0] += w

            if is_top:
                bottom_pt[1] = 400
            else:
                top_pt[1] = 0
                bottom_pt[1] += h

            return tuple(top_pt), tuple(bottom_pt)

        return list(map(specify_coords, matches))
    
    def __specify_birb_matches(self, matches, pattern):
        h, w = pattern.shape

        def specify_coords(coords):
            top_pt, bottom_pt = np.copy(coords), np.copy(coords)

            bottom_pt[0] += w
            bottom_pt[1] += h

            return tuple(top_pt), tuple(bottom_pt)

        return list(map(specify_coords, matches))
    
    def __find_tubes(self, image):
        cands = self.__find_candidates(image, self.tube_pattern)
        matches = self.__aggregate_candidates(cands)
        specified_matches = self.__specify_tube_matches(matches, self.tube_pattern, True)

        inv_cands = self.__find_candidates(image, self.inv_tube_pattern)
        inv_matches = self.__aggregate_candidates(inv_cands)
        inv_specified_matches = self.__specify_tube_matches(inv_matches, self.inv_tube_pattern, False)

        return chain(specified_matches, inv_specified_matches)

    def __find_birb(self, image):
        matches = self.__find_candidates(image, self.birb_pattern, use_max=True)
        return self.__specify_birb_matches(np.flip(np.vstack(matches).T, axis=1), self.birb_pattern)

    def __copy_regions(self, image, matches):
        new_image = np.zeros_like(image)

        for match in matches:
            (y1, x1), (y2, x2) = match
            new_image[x1:x2, y1:y2] = image[x1:x2, y1:y2]
        return new_image
    
    def crop_image(self, image):
        tube_matches = self.__find_tubes(image)
        birb_matches = self.__find_birb(image)
        matches = chain(tube_matches, birb_matches)
        
        return self.__copy_regions(image, matches)
    
    
    
class GrabCutCropper:
    def __get_region_proposals(self, image):
        means = image.mean(axis=0)[:, 2]
        points = np.where(means <= means.max() - 10)[0]
        
        if points.size == 0:
            return []

        regions = []
        begin, end = points[0], points[0]

        for point in points[1:]:
            if point - end > 5:
                regions.append((begin, end))
                begin = point
            end = point
        regions.append((begin, points[-1]))

        return regions

    def __make_fg_mask(self, image, regions):
        mask = np.zeros_like(image).astype(bool)

        for begin, end in regions:
            mask[:, begin-5:end+5] = True

        return mask

    def __find_mask_position(self, img, x, width):
        rect = (x, 0, width, 400)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask = np.zeros(img.shape[:2],np.uint8)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,3,cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        return mask

    def __find_mask(self, image, to_analyze):
        mask = np.zeros(image.shape[:2],np.uint8)

        for x, block_width in to_analyze:
            mask |= self.__find_mask_position(image, x, block_width)

        return mask

    def __find_tubes(self, image):
        regions = self.__get_region_proposals(image)
        refined_regions = []
        refine_width = 6

        for begin, end in regions:
            x = begin - refine_width
            width = end - begin + 2 * refine_width
            refined_regions.append((x, width))

        return self.__find_mask(image, refined_regions)
    
    def __find_birb(self, image):
        rect = (50, 10, 50, 380)

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        mask = np.zeros(image.shape[:2],np.uint8)
        cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')

        return mask
    
    def crop_image(self, image):
        mask = self.__find_tubes(image) | self.__find_birb(image)
        image_cut = image * mask[:,:,np.newaxis]
        
        return image_cut
    
    
from skimage.transform import resize

class SemanticSegmentationCropper:
    TUBE_WIDTH = 46
    
    def __init__(self, model_path="./unet_trained/model.ckpt"):
        self.net = unet.Unet(channels=3, n_class=3, layers=3, features_root=16)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.net.restore(self.sess, model_path)
        
        self.y_dummy = np.empty((1, 400, 288, 3))
        
    def __predict(self, image):
        return self.sess.run(self.net.predicter, feed_dict={self.net.x: [image], 
                                                            self.net.y: self.y_dummy, 
                                                            self.net.keep_prob: 1.})[0]
        
    def crop_image(self, image):
        div = 255 if image.max() > 2 else 1
        pred = self.__predict(image / div)
        mask = self.__process_prediction(pred)
        mask = self.__process_mask(mask)
        
        return mask * image
        
    def __process_prediction(self, prediction):
        mask = np.zeros(prediction.shape[:-1], dtype=int)

        for i in [1, 2]:
            threshold = prediction[:, :, i].max() - 0.05
            mask = mask | (prediction[:, :, i] > threshold)

        return mask
    
    def __process_mask(self, mask):
        mask = np.pad(mask, 20, mode="reflect")
        self.__repair_mask(mask)
        return mask[:, :, np.newaxis]
    
   

    def __change_row_left(self, mask, x):
        row = np.zeros(20)

        if mask[x, 20] == 1:
            streak = mask[x, 20:20+self.TUBE_WIDTH].sum()
            to_add = self.TUBE_WIDTH - streak
            row[-min(20, to_add):] = 1

        mask[x, :20] = row

    def __change_row_right(self, mask, x):
        row = np.zeros(20)

        if mask[x, -21] == 1:
            streak = mask[x, -21-self.TUBE_WIDTH:-21].sum()
            to_add = self.TUBE_WIDTH - streak
            row[:min(20, to_add)] = 1

        mask[x, -20:] = row

    def __repair_mask(self, mask):
        for x in range(mask.shape[0]):
            self.__change_row_left(mask, x)
            self.__change_row_right(mask, x)