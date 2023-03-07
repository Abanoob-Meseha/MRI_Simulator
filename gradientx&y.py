import matplotlib.pyplot as plt
import numpy as np

T_rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
T_ry = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 1]])
T_rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

T = T_rx @ T_ry

img = plt.imread('Phantom512.png')
dimensions = img.shape
print(dimensions)

img_transformed = np.empty((512, 512,3), dtype=np.uint8)

for i, row in enumerate(img):
    for j, col in enumerate(row):
        pixel_data = img[i, j, :]
        input_coords = np.array([i, j, 1])
        i_out, j_out, _ = T_rx @ input_coords
        img_transformed[i_out, j_out, :] = pixel_data





plt.figure(figsize=(5, 5))
plt.imshow(img_transformed)
plt.show()