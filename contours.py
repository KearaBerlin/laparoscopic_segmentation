import cv2
import numpy as np


class Util:
    @staticmethod
    def closest(pt1, array):
        d_min = 999999999
        j_min = 0
        for j in range(len(array)):
            pt2 = array[j][0]
            d = np.linalg.norm(pt1 - pt2)
            if d_min > d:
                d_min = d
                j_min = j
                if d == 0:
                    break
        return j_min

    @staticmethod
    def closest_iter(pt1, iter):
        d_min = 999999999
        j_min = 0
        j = 0
        for pt2 in iter:
            d = np.linalg.norm(pt1 - pt2)
            if d_min > d:
                d_min = d
                j_min = j
                if d == 0:
                    break
            j += 1
        return j_min


class PointIterator:
    def __init__(self, array, start, count=-1):
        self._array = array
        self._next = start

        if count == -1:
            self._count = len(array)
            self._step = 1
        else:
            self._count = min(count, len(array))
            self._step = int(len(array) / (self._count + 1))
        self._i = 0

    def peek(self):
        return self._array[self._next]

    def count(self):
        return self._count

    def has_next(self):
        return self._i < self._count

    def next_index(self):
        return self._next

    def __iter__(self):
        return self

    def __next__(self):
        if self._i < self._count:
            pt = self._array[self._next]
            self._next = (self._next + self._step) % len(self._array)
            self._i += 1
            return pt
        else:
            raise StopIteration


class ContourIterator:

    def _next(self):
        next_pt = None
        if self._corner_iter.has_next() and self._contour_iter.has_next():
            pt1 = self._contour_iter.peek()
            pt2 = self._corner_iter.peek()
            if np.linalg.norm(pt1 - self._cur) < np.linalg.norm(pt2 - self._cur):
                next_pt = self._contour_iter.__next__()
            else:
                next_pt = self._corner_iter.__next__()
        elif self._corner_iter.has_next():
            next_pt = self._corner_iter.__next__()
        elif self._contour_iter.has_next():
            next_pt = self._contour_iter.__next__()
        else:
            # should not happen
            raise StopIteration
        return next_pt

    def __init__(self, contour: "Contour", start=0, count=-1):
        # number of points sampled along the contour
        self.MAX_CONTOUR_PTS = 42
        self.contour = contour
        self._corner_iter = PointIterator(contour._corn_off, 0)
        start_contour = Util.closest(self._corner_iter.peek(), contour._cnt_off)


        count_contour = count
        if count < 0:
            count_contour = self.MAX_CONTOUR_PTS
        else:
            assert count > self._corner_iter.count()
            # count is total number of points needed, but we want to keep all the corners, so subtract corner count
            # but adjust by one to account for overlap with the first corner point
            count_contour = count - self._corner_iter.count() + 1

        self._contour_iter = PointIterator(contour._cnt_off, start_contour, count_contour)

        # we found the closest point to first corner. since we are starting with first corner, advance to next cnt pt
        self._contour_iter.__next__()
        self._total_points = self._corner_iter.count() + self._contour_iter.count() - 1
        self._i = 0
        self._cur = self._corner_iter.__next__()

        if start > 0:
            assert start < self._contour_iter.count() + self._corner_iter.count()
            for j in range(start - 1):
                self._cur = self._next()
            # reposition iterators
            start_contour = self._contour_iter.next_index()
            start_corner = self._corner_iter.next_index()
            self._contour_iter = PointIterator(contour._cnt_off, start_contour, count_contour)
            self._corner_iter = PointIterator(contour._corn_off, start_corner)

    def count(self):
        return self._total_points

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == 0:
            self._i += 1
            return self._cur

        if self._i < self._total_points:
            self._cur = self._next()
            self._i += 1
            return self._cur
        else:
            raise StopIteration


class Contour:

    def __find_super_corners(self):
        img_corners = [[0, 0], [self._mask.shape[0] / 2, 0], [self._mask.shape[0], 0],
                       [self._mask.shape[0], self._mask.shape[1] / 2], [self._mask.shape[0], self._mask.shape[1]],
                       [self._mask.shape[0] / 2, self._mask.shape[1]], [0, self._mask.shape[1]],
                       [0, self._mask.shape[1] / 2]]

        # looks like contours go in the opposite direction
        img_corners = img_corners[::-1]
        corners = list()

        # find the corner points closest to corners
        l = len(self._contour)
        start = 0
        end = l - 1
        for pt1 in img_corners:
            d_min = 999999999
            i = start
            i_min = 0
            while i != end:
                pt2 = self._contour[i][0]
                d = np.subtract(pt1, pt2)
                d = np.linalg.norm(d)
                if d_min > d:
                    d_min = d
                    i_min = i
                i = (i + 1) % l
            corners.append(self._contour[i_min][0])
            if start == 0 and i_min > 0:
                end = i_min - 1
            start = (i_min + 1) % l
        self._super_corners = np.asarray(corners)

        assert len(self._super_corners) == len(img_corners)

    def __map_corners(self, corners):
        self._corner_map = np.empty(len(corners), dtype=int)
        p_idx = 0
        c_idx = 0

        for c in corners:
            while True:
                p = self._contour[p_idx]
                if np.array_equal(c, p[0]):
                    self._corner_map[c_idx] = p_idx
                    c_idx += 1
                    p_idx = (p_idx + 1) % len(self._contour)
                    break
                p_idx = (p_idx + 1) % len(self._contour)

        assert c_idx == len(corners)

    def __normalize_contour(self, num_pts):
        cnt_len_per_segment = self._cnt_len / (num_pts + 1)
        result = list()
        i = self._corner_map[0]

        for j_idx in range(1, len(self._corner_map)):
            j = self._corner_map[j_idx]
            diff = j - i if j > i else len(self._contour) - i + j
            segments = round(diff / cnt_len_per_segment)
            if segments > 0:
                segment_len = int(diff / segments)
                for s in range(segments):
                    result.append(self._contour[i])
                    i = (i + segment_len) % self._cnt_len
            i = j
        # add last corner
        result.append(self._contour[i])
        self._norm_contour = np.asarray(result)

    def __normalize_contour2(self, num_pts):
        assert num_pts % len(self._corner_map) == 0
        pts_per_corner = int(num_pts / len(self._corner_map))

        result = list()
        i = self._corner_map[0]

        for j_idx in range(1, len(self._corner_map) + 1):
            j = self._corner_map[j_idx % len(self._corner_map)]
            diff = j - i if j > i else len(self._contour) - i + j
            segment_len = int(diff / pts_per_corner)
            for k in range(pts_per_corner):
                result.append(self._contour[i % len(self._contour)])
                i = (i + segment_len) % self._cnt_len
            i = j
        self._norm_contour = np.asarray(result)

        assert len(self._norm_contour) == num_pts

    @staticmethod
    def find_contours(mask):
        contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        longest_contour = []
        for cnt in contours:
            if len(cnt) > len(longest_contour):
                longest_contour = cnt
        return longest_contour

    def redraw_mask(self, contour):
        # redraw the mask based on the chosen contour
        norm_mask = np.zeros_like(self._mask)
        norm_ctr = contour[:, 0, :].astype(int)
        n1 = np.empty(tuple([1]) + norm_ctr.shape, dtype=int)
        n1[0] = norm_ctr
        cv2.drawContours(norm_mask, tuple(n1), -1, color=(255, 255, 255), thickness=cv2.FILLED)
        return norm_mask

    def __init__(self, mask, num_pts=0):
        # the higher the number, the rougher the corner estimation
        self.CORNER_APPROX_EPSILON = 8.0
        self.MAX_CONTOUR_PTS = 96

        self._mask = mask
        self._contour = Contour.find_contours(mask)
        self._corners = np.empty(0)

        self._corners = cv2.approxPolyDP(self._contour, self.CORNER_APPROX_EPSILON, True)

        self._cnt_len = len(self._contour) + 1  # because it's in pixels

        # map the corners back into the contour and normalize
        self.__find_super_corners()
        self.__map_corners(self._super_corners)
        self.__normalize_contour2(num_pts if num_pts > 0 else self.MAX_CONTOUR_PTS)

        # redraw the mask based on the chosen contour
        self._norm_mask = self.redraw_mask(self._contour)

        # offset is (x_off, y_off)
        # self._rect = cv2.boundingRect(self.__contour)
        # self._offset = (int(self.__mask.shape[1] / 2 - (self._rect[0] + self._rect[2] / 2)),
        #                 int(self.__mask.shape[0] / 2 - (self._rect[1] + self._rect[3] / 2)))
        #
        # self.cnt_off = self.__contour + self._offset
        # self.corn_off = self.corners + self._offset

        self._rect = cv2.boundingRect(self._norm_contour)
        self._offset = (int(self._mask.shape[1] / 2 - (self._rect[0] + self._rect[2] / 2)),
                        int(self._mask.shape[0] / 2 - (self._rect[1] + self._rect[3] / 2)))

        self._cnt_off = self._contour + self._offset
        self._corn_off = self._corners + self._offset
        self._norm_contour = self._norm_contour + self._offset

    def mask(self):
        return self._mask

    def norm_mask(self):
        return self._norm_mask

    def contour(self):
        return self._norm_contour

    def num_pts(self):
        return len(self._norm_contour)

    def offset(self):
        return self._offset

    def roi(self):
        return self._rect

    def est_match_points(self, other: "Contour"):
        my_iter = ContourIterator(self)
        other_it = ContourIterator(other, 0, my_iter.count())

        pt1 = my_iter.__next__()
        other_start = Util.closest_iter(pt1, other_it)
        other_it = ContourIterator(other, other_start, my_iter.count())

        assert my_iter.count() == other_it.count()

        matched_points = np.empty((my_iter.count(), 2, 2))
        matched_points[0][0] = pt1
        matched_points[0][1] = other_it.__next__()
        i = 1
        for pt in my_iter:
            matched_points[i][0] = pt
            matched_points[i][1] = other_it.__next__()
            i += 1
        return matched_points
