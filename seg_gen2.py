from typing import Type

import cv2
import numpy as np

from contours import Contour


class SegGen2:
    def __init__(self, image_file_name, mask_file_name):
        self.img1 = cv2.imread(image_file_name)
        self.mask1 = cv2.imread(mask_file_name)
        self.contour = Contour(self.mask1)

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
        contour2 = Contour(mask2, self.contour.num_pts())

        # offset the object
        # todo: maybe centering is not necessary?
        obj2_off = self.shift_object(obj2, contour2.roi(), contour2.offset())

        # matched = self.contour.est_match_points(contour2)
        # reshaped contours, necessary for the transformer
        #norm_ctrs1_r = matched[:, 0, :].reshape(1, -1, 2).astype(np.float32)
        #norm_ctrs2_r = matched[:, 1, :].reshape(1, -1, 2).astype(np.float32)
        norm_ctrs1_r = self.contour.norm_contour[:, 0, :].reshape(1, -1, 2).astype(np.float32)
        norm_ctrs2_r = contour2.norm_contour[:, 0, :].reshape(1, -1, 2).astype(np.float32)

        ######################################################
        cv2.namedWindow("debug1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("debug2", cv2.WINDOW_NORMAL)

        norm_ctrs1 = self.contour.norm_contour[:, 0, :].astype(int)
        n1 = np.empty(tuple([1]) + norm_ctrs1.shape, dtype=int)
        n1[0] = norm_ctrs1
        t1 = tuple(n1)

        norm_ctrs2 = contour2.norm_contour[:, 0, :].astype(int)
        n2 = np.empty(tuple([1]) + norm_ctrs2.shape, dtype=int)
        n2[0] = norm_ctrs2
        t2 = tuple(n2)

        cv2.drawContours(self.img1, t1, 0, (0, 255, 255), 1)
        cv2.drawContours(self.img1, t2, 0, (255, 0, 255), 1)

        for i in range(len(norm_ctrs1)):
            self.img1 = cv2.circle(self.img1, norm_ctrs1[i], 5, color=(0, 255, 255), thickness=-1)
            self.img1 = cv2.circle(self.img1, norm_ctrs2[i], 5, color=(255, 0, 255), thickness=-1)

        cv2.imshow("debug1", self.img1)

        ######################################################

        pt1 = self.contour.norm_contour[0]
        d_min = 999999999
        i_min = 0
        for i in range(contour2.num_pts()):
            pt2 = contour2.norm_contour[i][0]
            d = np.linalg.norm(pt1 - pt2)
            if d_min > d:
                d_min = d
                i_min = i

        matches = list()
        for i in range(self.contour.num_pts()):
            matches.append(cv2.DMatch(i, (i_min + i) % contour2.num_pts(), 0))

        # estimate the transform
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(norm_ctrs1_r, norm_ctrs2_r, matches)
        obj2_p = tps.warpImage(obj2_off)

        #cv2.imshow("debug2", obj2_p)


        # shift the warped object back, using obj1's shifted roi
        obj1_roi = self.contour.roi()
        off1 = self.contour.offset()
        roi_off = (obj1_roi[0] + off1[0], obj1_roi[1] + off1[1], obj1_roi[2], obj1_roi[3])
        obj2_warped = self.shift_object(obj2_p, roi_off, np.negative(off1))

        # generate the final image
        img = np.bitwise_or(np.bitwise_and(np.bitwise_not(self.mask1), self.img1), np.bitwise_and(obj2_warped, self.mask1))
        return img


