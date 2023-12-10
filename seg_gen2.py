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

    def __debug(self, contour2, obj2_off):
        ######################################################
        cv2.namedWindow("debug1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("debug2", cv2.WINDOW_NORMAL)

        norm_ctrs1 = self.contour.contour()[:, 0, :].astype(int)
        n1 = np.empty(tuple([1]) + norm_ctrs1.shape, dtype=int)
        n1[0] = norm_ctrs1
        t1 = tuple(n1)

        norm_ctrs2 = contour2.contour()[:, 0, :].astype(int)
        n2 = np.empty(tuple([1]) + norm_ctrs2.shape, dtype=int)
        n2[0] = norm_ctrs2
        t2 = tuple(n2)

        debug_i2 = np.copy(obj2_off)
        obj1_off = self.shift_object(np.bitwise_and(self.img1, self.contour.norm_mask()),
                                     self.contour.roi(), self.contour.offset())
        debug_i1 = np.copy(obj1_off)

        cv2.drawContours(debug_i1, t1, 0, (0, 255, 255), 1)
        cv2.drawContours(debug_i2, t2, 0, (255, 0, 255), 1)

        for i in range(len(norm_ctrs1)):
            debug_i1 = cv2.circle(debug_i1, norm_ctrs1[i], 5, color=(255 if i == 0 else 0, 255, 255), thickness=-1)
            debug_i2 = cv2.circle(debug_i2, norm_ctrs2[i], 5, color=(255, 255 if i == 0 else 0, 255), thickness=-1)

        cv2.imshow("debug1", debug_i1)
        cv2.imshow("debug2", debug_i2)

    def generate(self, image_file_name, mask_file_name):
        img2 = cv2.imread(image_file_name)
        mask2 = cv2.imread(mask_file_name)
        contour2 = Contour(mask2, self.contour.num_pts())
        obj2 = np.bitwise_and(img2, contour2.norm_mask())

        # offset the object
        obj2_off = self.shift_object(obj2, contour2.roi(), contour2.offset())

        # reshaped contours, necessary for the transformer
        norm_ctrs1_r = self.contour.contour()[:, 0, :].reshape(1, -1, 2).astype(np.float32)
        norm_ctrs2_r = contour2.contour()[:, 0, :].reshape(1, -1, 2).astype(np.float32)

        matches = list()
        for i in range(self.contour.num_pts()):
            matches.append(cv2.DMatch(i, i, 0))

        # estimate the transform
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(norm_ctrs1_r, norm_ctrs2_r, matches)
        obj2_p = tps.warpImage(obj2_off)

        # shift the warped object back, using obj1's shifted roi
        obj1_roi = self.contour.roi()
        off1 = self.contour.offset()
        roi_off = (obj1_roi[0] + off1[0], obj1_roi[1] + off1[1], obj1_roi[2], obj1_roi[3])
        obj2_warped = self.shift_object(obj2_p, roi_off, np.negative(off1))

        # generate the final image
        img = np.bitwise_or(np.bitwise_and(np.bitwise_not(self.contour.norm_mask()), self.img1),
                            np.bitwise_and(obj2_warped, self.contour.norm_mask()))
        return img

