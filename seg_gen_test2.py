import cv2

from seg_gen import SegGen
from seg_gen2 import SegGen2

#gen = SegGen("Z:/1/dresden/liver/01/image00.png", "Z:/1/dresden/liver/01/mask00.png")
gen = SegGen2("Z:/1/dresden/liver/01/image00.png", "Z:/1/dresden/liver/01/mask00.png")

img3 = gen.generate("Z:/1/dresden/liver/01/image68.png", "Z:/1/dresden/liver/01/mask68.png")
cv2.namedWindow("img3", cv2.WINDOW_NORMAL)
cv2.imshow("img3", img3)
cv2.waitKey(0)
