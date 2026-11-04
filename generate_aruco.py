import cv2

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker = cv2.aruco.generateImageMarker(
    dictionary, 0, 1500
)

# Put the marker on a 2100 x 2970 px white A4 canvas.
page = 255 * __import__("numpy").ones(
    (2970, 2100), dtype="uint8"
)
page[300:1800, 300:1800] = marker

cv2.imwrite("aruco_a4.png", page)