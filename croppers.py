import cv2
from sklearn.cluster import DBSCAN
from itertools import chain
from skimage.io import imread
import numpy as np
import pandas as pd
import os

class PatternMatchCropper:
    def __init__(self):
        self.tube_pattern = self.__preprocess_image(imread(os.path.join(os.path.dirname(__file__), "pipe-green.png"))[:25])
        self.inv_tube_pattern = np.flip(np.flip(self.tube_pattern, axis=0), axis=1)
        self.birb_pattern = self.__preprocess_image(imread(os.path.join(os.path.dirname(__file__), "yellowbird-downflap.png")))
        
    def __preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(image, 50, 200)
    
    def __find_candidates(self, image, pattern):
        image = self.__preprocess_image(image)
        res = cv2.matchTemplate(image, pattern, cv2.TM_CCOEFF_NORMED)
        
        return np.where(res == res.max())
    
    def __aggregate_candidates(self, candidates):
        dbscan = DBSCAN(min_samples=1, eps=1e1)
        points = np.vstack(candidates).T
        cluster_indices = dbscan.fit_predict(points)

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
        matches = self.__find_candidates(image, self.birb_pattern)
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