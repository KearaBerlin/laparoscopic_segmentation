import cv2

from seg_gen import SegGen

gen = SegGen("liver/01/image00.png", "liver/01/mask00.png")
img3 = gen.generate("liver/01/image68.png", "liver/01/mask68.png")
cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
cv2.imshow("img3", img3)
cv2.waitKey(0)
