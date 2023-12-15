import math

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
        result = list()
        self._super_corner_map_i2c = dict()
        self._super_corner_map_c2i = dict()

        # find the contour points closest to corners
        l = len(self._corn_off)
        start = 0
        end = l - 1
        for j in range(len(img_corners)):
            pt1 = img_corners[j]
            d_min = 999999999
            i = start
            i_min = 0
            while i != end:
                pt2 = self._corn_off[i]
                d = np.linalg.norm(np.subtract(pt1, pt2))
                if d_min > d:
                    d_min = d
                    i_min = i
                i = (i + 1) % l

            if self._super_corner_map_i2c.get(j) is None or self._super_corner_map_i2c[j][1] > d_min:
                self._super_corner_map_i2c[j] = (i_min, d_min)
            if self._super_corner_map_c2i.get(i_min) is None or self._super_corner_map_c2i[i_min][1] > d_min:
                self._super_corner_map_c2i[i_min] = (j, d_min)

            if start == 0 and i_min > 0:
                end = i_min - 1
            start = (i_min + 1) % l

    # return image corner coordinate index if the specified corner maps to one
    def is_super_corner(self, c_idx):
        if self._super_corner_map_c2i.get(c_idx) is not None:
            i, _ = self._super_corner_map_c2i[c_idx]
            if self._super_corner_map_i2c.get(i) is not None:
                c, _ = self._super_corner_map_i2c[i]
                if c == c_idx:
                    return i
        return -1

    def get_super_corner(self, i_idx):
        if self._super_corner_map_i2c.get(i_idx) is not None:
            c, _ = self._super_corner_map_i2c[i_idx]
            if self._super_corner_map_c2i.get(c) is not None:
                i, _ = self._super_corner_map_c2i[c]
                if i == i_idx:
                    return c
        return -1

    def __find_super_corners_slow(self):
        img_corners = [[0, 0], [self._mask.shape[0] / 2, 0], [self._mask.shape[0], 0],
                       [self._mask.shape[0], self._mask.shape[1] / 2], [self._mask.shape[0], self._mask.shape[1]],
                       [self._mask.shape[0] / 2, self._mask.shape[1]], [0, self._mask.shape[1]],
                       [0, self._mask.shape[1] / 2]]

        # looks like contours go in the opposite direction
        img_corners = img_corners[::-1]
        corners = list()

        # find the contour points closest to corners
        for pt1 in img_corners:
            d_min = 999999999
            pt_min = []
            for pt2 in self._cnt_off:
                d = np.linalg.norm(np.subtract(pt1, pt2[0]))
                if d_min > d:
                    d_min = d
                    pt_min = pt2[0]
            corners.append(pt_min)

        self._super_corners = np.asarray(corners)
        assert len(self._super_corners) == len(img_corners)

    def __map_corners(self, corners):
        result = list()
        new_corners = list()
        #self._corner_map = np.empty(len(corners), dtype=int)
        segment_len = self._cnt_len / self.MAX_CONTOUR_PTS
        MIN = segment_len * 0.7
        MAX = segment_len * 10
        p_idx = 0
        p_prev = -1

        for i in range(-1, len(corners)):
            c = corners[i]
            while True:
                p = self._contour[p_idx]
                if np.array_equal(c, p):
                    #self._corner_map[c_idx] = p_idx
                    if p_prev != -1:
                        dist = p_idx - p_prev if p_idx > p_prev else self._cnt_len - p_prev + p_idx
                        if dist < MIN:
                            #p_prev = p_idx
                            p_idx = (p_idx + 1) % self._cnt_len
                            break
                        elif dist > MAX:
                            num_pts = math.ceil(dist / MAX)
                            c_dist = dist / num_pts
                            for j in range(1, num_pts):
                                new_idx = int(p_prev + j * c_dist)
                                result.append(new_idx % self._cnt_len)
                                new_corners.append(self._contour[new_idx % self._cnt_len])
                    new_corners.append(c)
                    result.append(p_idx)
                    p_prev = p_idx
                    #c_idx += 1
                    p_idx = (p_idx + 1) % self._cnt_len
                    break
                p_idx = (p_idx + 1) % self._cnt_len

        #assert c_idx == len(corners)
        self._corner_map = np.asarray(result)
        return np.asarray(new_corners)

    def __map_corners2(self):
        result = list()
        new_corners = list()
        segment_len = self._cnt_len / self.MAX_CONTOUR_PTS
        MIN = segment_len * 0.7
        MAX = segment_len * 10
        p_idx = 0
        p_prev = -1

        for i in range(-1, len(self._corn_off)):
            c = self._corn_off[i]
            while True:
                p = self._cnt_off[p_idx]
                if np.array_equal(c, p):
                    if p_prev != -1:
                        dist = p_idx - p_prev if p_idx > p_prev else self._cnt_len - p_prev + p_idx
                        if dist < MIN:
                            p_idx = (p_idx + 1) % self._cnt_len
                            break
                        elif dist > MAX:
                            num_pts = math.ceil(dist / MAX)
                            c_dist = dist / num_pts
                            for j in range(1, num_pts):
                                new_idx = int(p_prev + j * c_dist)
                                result.append(new_idx % self._cnt_len)
                                new_corners.append(self._cnt_off[new_idx % self._cnt_len])
                    new_corners.append(c)
                    result.append(p_idx)
                    p_prev = p_idx
                    p_idx = (p_idx + 1) % self._cnt_len
                    break
                p_idx = (p_idx + 1) % self._cnt_len

        self._corner_map = np.asarray(result)
        return np.asarray(new_corners)

    def __normalize_contour(self, num_pts):
        cnt_len_per_segment = self._cnt_len / (num_pts + 1)
        result = list()
        i = self._corner_map[0]

        for j_idx in range(1, len(self._corner_map) + 1):
            j = self._corner_map[j_idx % len(self._corner_map)]
            diff = j - i if j >= i else self._cnt_len - i + j
            segments = round(diff / cnt_len_per_segment)
            if segments > 0:
                segment_len = diff / segments
                for s in range(segments):
                    k = (i + int(s * segment_len)) % self._cnt_len
                    result.append(self._cnt_off[k])
            #else:
                #add it even if the corners are close
                #result.append(self._contour[i])

            i = j
        self._norm_contour = np.asarray(result)

    def __normalize_contour2(self, num_pts):
        assert num_pts % len(self._corner_map) == 0
        pts_per_corner = int(num_pts / len(self._corner_map))

        result = list()
        i = self._corner_map[0]

        for j_idx in range(1, len(self._corner_map) + 1):
            j = self._corner_map[j_idx % len(self._corner_map)]
            diff = j - i if j > i else self._cnt_len - i + j
            segment_len = int(diff / pts_per_corner)
            for k in range(pts_per_corner):
                result.append(self._contour[i % self._cnt_len])
                i = (i + segment_len) % self._cnt_len
            i = j
        self._norm_contour = np.asarray(result)
        assert len(self._norm_contour) == num_pts

    def match_corners_to_other(self, other: "Contour"):
        # match my corners to the other counter's corners
        # then divide each segment to match the # points for each pair of matched corners
        result = dict()
        #j_max = -1
        other_cont = other.contour()

        for i in range(len(self._corn_off)):
            pt1 = self._corn_off[i]

            # check if this is a super-corner, if yes, then have to match it with other's super corner
            super_corner_id = self.is_super_corner(i)
            if super_corner_id > -1:
                matched_super_corner = other.get_super_corner(super_corner_id)
                if matched_super_corner > -1:
                    pt2 = other._cnt_off[other._corner_map[matched_super_corner]]
                    j = [p for p in range(0, len(other_cont)) if np.array_equal(pt2, other_cont[p])]
                    assert len(j) == 1
                    result[j[0]] = (i, 0.0)
                    continue

            d_min = 999999999
            j_min = 0
            #closest = list()
            #for j in range(len(other._corn_off)):
            for j in range(len(other_cont)):
                #pt2 = other._corn_off[j]
                pt2 = other_cont[j]
                d = np.linalg.norm(np.subtract(pt1, pt2))
                if d_min > d:
                    d_min = d
                    j_min = j
                    #closest.append((j, d_min))

            if result.get(j_min) is not None:
                if result[j_min][1] > d_min:
                    # if there is already a mapping, only update it if we found shorter distance
                    result[j_min] = (i, d_min)
                #j_max = max(j_max, j_min + 1) % len(other_cont)
                continue

            # for new mapping make sure we don't go backwards
            #closest.sort(key=lambda p: p[1])
            j = j_min
            #while j < j_max and len(closest) > 0:
            #    j = closest[0][0]
            #    closest = closest[1:]

            #if j >= j_max:
            result[j] = (i, d_min)
            #j_max = (j + 1) % len(other_cont)

        # result now has for a map from corner (idx) of the other contour to closest corner (idx) on my contour
        # this is an offset (centered) map
        return result

    def __normalize_contour3(self, other: "Contour"):
        # map my corners to other contour's corners
        corner_map = list(self.match_corners_to_other(other).items())
        result = list()
        matches = list()
        #for i in range(self.contour.num_pts()):
        #    matches.append(cv2.DMatch(i, i, 0))

        i_prev = corner_map[-1][1][0]
        j_prev = corner_map[-1][0]

        c_match = 0
        num_pts = other.num_pts()

        # now match up all the points between each pair of corners
        for j, m in corner_map:
            i = m[0]

            # sanity check
            assert np.array_equal(self._corn_off[i_prev], self._cnt_off[self._corner_map[i_prev]])
            #assert np.array_equal(other._corn_off[j_prev], other._cnt_off[other._corner_map[j_prev]])
            assert np.array_equal(self._corn_off[i], self._cnt_off[self._corner_map[i]])
            #assert np.array_equal(other._corn_off[j], other._cnt_off[other._corner_map[j]])

            # find number of normalized contour points between these 2 points
            f = [j_prev] #[p for p in range(0, num_pts) if np.array_equal(other._norm_contour[p], other._corn_off[j_prev])]
            t = [j] #[p for p in range(0, num_pts) if np.array_equal(other._norm_contour[p], other._corn_off[j])]

            assert len(f) == 1
            assert len(t) == 1

            n = t[0] - f[0] if f[0] < t[0] else other.num_pts() - f[0] + t[0]
            segment_len = self._corner_map[i] - self._corner_map[i_prev] if self._corner_map[i] > self._corner_map[i_prev] else self._cnt_len - self._corner_map[i_prev] + self._corner_map[i]
            segment_len = segment_len / n

            for p in range(n):
                cnt_idx = round(self._corner_map[i_prev] + p * segment_len) % self._cnt_len
                result.append(self._cnt_off[cnt_idx])
                matches.append(cv2.DMatch((f[0] + p) % num_pts, c_match, 0))
                c_match = c_match + 1

            i_prev = i
            j_prev = j

        self._norm_contour = np.asarray(result)
        self._matches = matches
        assert len(self._norm_contour) == other.num_pts()

    def __normalize_contour4(self, other: "Contour"):
        # other already has a normalized contour
        other_cont = other.contour()

        f = 0
        t = 0

        for j in range(len(other_cont)):
            # check if this is a super-corner, if yes, then have to match it with my super corner
            super_corner_id = other.is_super_corner(j)
            if super_corner_id > -1:
                matched_super_corner = self.get_super_corner(super_corner_id)
                if matched_super_corner > -1:


                    pt2 = other._cnt_off[other._corner_map[matched_super_corner]]
                    j = [p for p in range(0, len(other_cont)) if np.array_equal(pt2, other_cont[p])]
                    assert len(j) == 1
                    result[j[0]] = (i, 0.0)
                    continue



    @staticmethod
    def find_contours(mask, type=cv2.CHAIN_APPROX_NONE):
        contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, type)
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

    def __init__(self, mask, other: "Contour" = None):
        # the higher the number, the rougher the corner estimation
        self.CORNER_APPROX_EPSILON = 8.0
        self.MAX_CONTOUR_PTS = 96

        self._mask = mask
        self._contour = Contour.find_contours(mask)
        self._cnt_len = len(self._contour)
        self._corners = cv2.approxPolyDP(self._contour, self.CORNER_APPROX_EPSILON, True)

        self._rect = cv2.boundingRect(self._contour)
        self._offset = (int(self._mask.shape[1] / 2 - (self._rect[0] + self._rect[2] / 2)),
                        int(self._mask.shape[0] / 2 - (self._rect[1] + self._rect[3] / 2)))
        self._cnt_off = self._contour + self._offset
        self._corn_off = self._corners + self._offset

        # map the corners back into the contour and enhance
        new_corners = self.__map_corners2()
        self._corn_off = new_corners

        if other is None:
            self.__find_super_corners()
            # just compute my own normalized contour
            self.__normalize_contour(self.MAX_CONTOUR_PTS)
        else:
            self.__find_super_corners_slow()
            # compute a contour that needs to be matched to some existing (offset) contour
            self.__normalize_contour3(other)

        # redraw the mask based on the chosen contour (original or normalized?)
        self._norm_mask = self.redraw_mask(self._contour)

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

    def matches(self):
        return self._matches

    def corners(self):
        return self._corn_off

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
