from typing import Type

import cv2
import numpy as np

class SegGen:

    def __find_contours(self, mask):
        contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
        longest_contour = []
        for cnt in contours:
            if len(cnt) > len(longest_contour):
                longest_contour = cnt
        return longest_contour

    def __init__(self, image_file_name, mask_file_name):
        self.MAX_CONTOUR_PTS = 60
        self.img1 = cv2.imread(image_file_name)
        self.mask1 = cv2.imread(mask_file_name)
        self.contour1 = self.__find_contours(self.mask1)
        max_pts = min(self.MAX_CONTOUR_PTS, len(self.contour1))

        # roi is (x, y, w, h)
        self.obj1_roi = cv2.boundingRect(self.contour1)
        # offset is (x_off, y_off)
        self.off1 = (int(self.mask1.shape[1] / 2 - (self.obj1_roi[0] + self.obj1_roi[2] / 2)),
                     int(self.mask1.shape[0] / 2 - (self.obj1_roi[1] + self.obj1_roi[3] / 2)))

        # normalize contours
        self.norm_ctrs1 = self.contour1[0::int(len(self.contour1) / max_pts)]
        self.ctr_pts = len(self.norm_ctrs1)
        # center the contours in the image
        self.norm_ctrs1 += self.off1

        # reshaped contours, necessary for the transformer
        self.norm_ctrs1_r = self.norm_ctrs1[:, 0, :].reshape(1, -1, 2).astype(np.float32)

    # given an object (image), region of interest (rect) and offset
    # return new image containing the rect shifted by offset
    def shift_object(self, obj, obj_roi, off):
        obj_off = np.zeros_like(obj)
        (x, y, w, h) = (obj_roi[0], obj_roi[1], obj_roi[2], obj_roi[3])
        (x_o, y_o) = (obj_roi[0] + off[0], obj_roi[1] + off[1])
        obj_off[y_o:y_o + h, x_o:x_o + w] = obj[y:y + h, x:x + w]
        return obj_off

    def generate(self, image_file_name, mask_file_name):
        img2 = cv2.imread(image_file_name)
        mask2 = cv2.imread(mask_file_name)
        obj2 = np.bitwise_and(img2, mask2)
        contour2 = self.__find_contours(mask2)

        norm_ctrs2 = contour2[0::int(len(contour2) / (self.ctr_pts + 1))]
        norm_ctrs2 = norm_ctrs2[0:self.ctr_pts]

        # compute the offset for the second object
        obj2_roi = cv2.boundingRect(contour2)
        off2 = (int(mask2.shape[1] / 2 - (obj2_roi[0] + obj2_roi[2] / 2)),
                int(mask2.shape[0] / 2 - (obj2_roi[1] + obj2_roi[3] / 2)))
        norm_ctrs2 += off2

        # offset the object
        obj2_off = self.shift_object(obj2, obj2_roi, off2)

        # find the closest point between the contours
        # right now just finds the closest point for the first point of the src contour
        pt1 = self.norm_ctrs1[0][0]
        d_min = 999999999
        i_min = 0
        for i in range(len(norm_ctrs2)):
            pt2 = norm_ctrs2[i][0]
            d = np.linalg.norm(pt1 - pt2)
            if d_min > d:
                d_min = d
                i_min = i

        # create a list of matches. now i_min points to the point on the second contour
        # that is the closest to the first point of the first contour
        matches = list()
        for i in range(self.ctr_pts):
            matches.append(cv2.DMatch(i, (i_min + i) % self.ctr_pts, 0))

        # estimate the transform
        tps = cv2.createThinPlateSplineShapeTransformer()
        norm_ctrs2_r = norm_ctrs2[:, 0, :].reshape(1, -1, 2).astype(np.float32)
        tps.estimateTransformation(self.norm_ctrs1_r, norm_ctrs2_r, matches)
        obj2_p = tps.warpImage(obj2_off)

        # shift the warped object back, using obj1's shifted roi
        roi_off = (self.obj1_roi[0] + self.off1[0], self.obj1_roi[1] + self.off1[1], self.obj1_roi[2], self.obj1_roi[3])
        obj2_warped = self.shift_object(obj2_p, roi_off, np.negative(self.off1))

        # generate the final image
        img = np.bitwise_or(np.bitwise_and(np.bitwise_not(self.mask1), self.img1), np.bitwise_and(obj2_warped, self.mask1))
        return img
