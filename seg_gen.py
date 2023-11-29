import cv2
import numpy as np


class Contour:
    def __init__(self, mask):
        self.MAX_CONTOUR_PTS = 60
        self.mask = mask
        contours1, _ = cv2.findContours(cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        self.contour = contours1[0]
        max_pts = min(self.MAX_CONTOUR_PTS, len(self.contour))
        rect = cv2.boundingRect(self.contour)

        # offset is (x_off, y_off)
        self.offset = (int(self.mask.shape[1] / 2 - (rect[0] + rect[2] / 2)),
                       int(self.mask.shape[0] / 2 - (rect[1] + rect[3] / 2)))

        self.corners = cv2.approxPolyDP(self.contour, 10, True)
        self.cnt_off = self.contour + self.offset
        self.corn_off = self.corners + self.offset
        self.cnt_len = cv2.arcLength(self.contour, True)

class ContourIterator:
    def __init__(self, contour: Contour):
        self.contour = contour

    def __iter__(self):
        return self

    def __next__(self):
        pass

class SegGen:
    def __init__(self, image_file_name, mask_file_name):
        self.MAX_CONTOUR_PTS = 60
        self.img1 = cv2.imread(image_file_name)
        self.mask1 = cv2.imread(mask_file_name)
        self.contours1, _ = cv2.findContours(cv2.cvtColor(self.mask1, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
        max_pts = min(self.MAX_CONTOUR_PTS, len(self.contours1[0]))

        # compute the offset of the "center" of the mask relative to image center
        # todo: could use "center of mass" or just center of bounding rectangle
        # M = cv2.moments(contours1[0])
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        # off1 = (mask1.shape[1] / 2 - cX, mask1.shape[0] / 2 - cY)

        # roi is (x, y, w, h)
        self.obj1_roi = cv2.boundingRect(self.contours1[0])
        # offset is (x_off, y_off)
        self.off1 = (int(self.mask1.shape[1] / 2 - (self.obj1_roi[0] + self.obj1_roi[2] / 2)),
                     int(self.mask1.shape[0] / 2 - (self.obj1_roi[1] + self.obj1_roi[3] / 2)))

        # normalize contours
        self.norm_ctrs1 = self.contours1[0][0::int(len(self.contours1[0]) / max_pts)]
        # todo: enhance contours with corners, etc
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
        contours2, _ = cv2.findContours(cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)

        #todo did we get the right number of points?
        norm_ctrs2 = contours2[0][0::int(len(contours2[0]) / self.ctr_pts)]
        norm_ctrs2 = norm_ctrs2[0:self.ctr_pts]

        # compute the offset for the second object
        obj2_roi = cv2.boundingRect(contours2[0])
        off2 = (int(mask2.shape[1] / 2 - (obj2_roi[0] + obj2_roi[2] / 2)),
                int(mask2.shape[0] / 2 - (obj2_roi[1] + obj2_roi[3] / 2)))
        norm_ctrs2 += off2

        # offset the object
        # todo: maybe centering is not necessary?
        obj2_off = self.shift_object(obj2, obj2_roi, off2)

        # find the closest point between the contours
        # right now just finds the closest point for the first point of the src contour
        # todo: maybe could change it to find the actual closest point
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
        norm_ctrs2_r = norm_ctrs2[:, 0, :]
        norm_ctrs2_r = norm_ctrs2_r.reshape(1, -1, 2).astype(np.float32)

        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(self.norm_ctrs1_r, norm_ctrs2_r, matches)
        obj2_p = tps.warpImage(obj2_off)

        # shift the warped object back, using obj1's shifted roi
        roi_off = (self.obj1_roi[0] + self.off1[0], self.obj1_roi[1] + self.off1[1], self.obj1_roi[2], self.obj1_roi[3])
        obj2_warped = self.shift_object(obj2_p, roi_off, np.negative(self.off1))

        # generate the final image
        img = np.bitwise_or(np.bitwise_and(np.bitwise_not(self.mask1), self.img1), np.bitwise_and(obj2_warped, self.mask1))
        return img

    def match_contours(self, cnt1: Contour, cnt2: Contour):

        pass


