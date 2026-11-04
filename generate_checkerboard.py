import cv2
import numpy as np

# A4 page at 300 DPI
page_width = 2480
page_height = 3508

# Checkerboard: 9 x 6 inner corners = 10 x 7 squares
cols = 10
rows = 7

# Each square is 25 mm at 300 DPI
square_size = round(20 / 25.4 * 300)

board_width = cols * square_size
board_height = rows * square_size

# White A4 canvas
page = np.full(
    (page_height, page_width),
    255,
    dtype=np.uint8
)

# Centre the checkerboard
start_x = (page_width - board_width) // 2
start_y = (page_height - board_height) // 2

# Draw alternating black squares
for row in range(rows):
    for col in range(cols):
        if (row + col) % 2 == 0:
            x = start_x + col * square_size
            y = start_y + row * square_size

            cv2.rectangle(
                page,
                (x, y),
                (x + square_size - 1, y + square_size - 1),
                0,
                -1
            )

cv2.imwrite("checkerboard.png", page)
