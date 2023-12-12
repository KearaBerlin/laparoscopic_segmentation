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
    def __shift_object(self, obj, obj_roi, off):
        obj_off = np.zeros_like(obj)
        (x, y, w, h) = (obj_roi[0], obj_roi[1], obj_roi[2], obj_roi[3])
        (x_o, y_o) = (obj_roi[0] + off[0], obj_roi[1] + off[1])
        obj_off[y_o:y_o + h, x_o:x_o + w] = obj[y:y + h, x:x + w]
        return obj_off

    def __debug_show(self, img):
        cv2.namedWindow("debug3", cv2.WINDOW_NORMAL)
        cv2.imshow("debug3", img)

    def __draw_contour(self, img, contour, color=(255, 255, 255)):
        ctr_r = contour[:, 0, :].astype(int)
        n1 = np.empty(tuple([1]) + ctr_r.shape, dtype=int)
        n1[0] = ctr_r
        t1 = tuple(n1)
        cv2.drawContours(img, t1, 0, color, 1)

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
        obj1_off = self.__shift_object(np.bitwise_and(self.img1, self.contour.norm_mask()),
                                       self.contour.roi(), self.contour.offset())
        debug_i1 = np.copy(obj1_off)

        cv2.drawContours(debug_i1, t1, 0, (0, 255, 255), 1)
        cv2.drawContours(debug_i2, t2, 0, (255, 0, 255), 1)

        for i in range(len(norm_ctrs1)):
            debug_i1 = cv2.circle(debug_i1, norm_ctrs1[i], 5, color=(255 if i == 0 else 0, 255, 255), thickness=-1)
            debug_i2 = cv2.circle(debug_i2, norm_ctrs2[i], 5, color=(255, 255 if i == 0 else 0, 255), thickness=-1)

        cv2.imshow("debug1", debug_i1)
        cv2.imshow("debug2", debug_i2)

    def __generate_obj2(self, img2, contour2):
        # create the object used for the transformation
        obj2 = np.bitwise_and(img2, contour2.norm_mask())
        obj2_off = self.__shift_object(obj2, contour2.roi(), contour2.offset())
        return obj2_off

    def generate(self, image_file_name, mask_file_name):
        img2 = cv2.imread(image_file_name)
        mask2 = cv2.imread(mask_file_name)
        contour2 = Contour(mask2, self.contour.num_pts())
        obj2_off = self.__generate_obj2(img2, contour2)

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
        obj2_warped = self.__shift_object(obj2_p, roi_off, np.negative(off1))


        # recompute the mask for the warped object
        obj2_warped = np.bitwise_and(obj2_warped, self.mask1)
        obj2_contour = Contour.find_contours(obj2_warped, cv2.CHAIN_APPROX_TC89_L1)
        obj2_contour_img = np.zeros_like(self.mask1)
        self.__draw_contour(obj2_contour_img, obj2_contour)

        # # shrink mask by a couple pixels
        dist = cv2.distanceTransform(np.bitwise_not(cv2.cvtColor(obj2_contour_img, cv2.COLOR_BGR2GRAY)), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ring = cv2.inRange(dist, 3, 4)  # take all pixels at distance between
        contours, h_tree = cv2.findContours(ring, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        for h in range(len(h_tree[0])):
            if h_tree[0][h][2] == -1 and h_tree[0][h][3] != -1:
                break

        obj2_contour2 = contours[h]
        obj2_mask2 = contour2.redraw_mask(obj2_contour2)

        # generate the final image
        img = np.bitwise_or(np.bitwise_and(np.bitwise_not(obj2_mask2), self.img1),
                            np.bitwise_and(obj2_warped, obj2_mask2))
        return img

