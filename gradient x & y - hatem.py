import matplotlib.pyplot as plt
import numpy as np


def Rotation_z(theta):
    rotation_z = np.array(
        [[int(np.cos(theta)), int(np.sin(theta)), 0], [int(-np.sin(theta)), int(np.cos(theta)), 0], [0, 0, 1]])
    return rotation_z

Rf_theta =  np.radians(90)
rotation_x = np.array([[1, 0, 0], [0, int(np.cos(Rf_theta)), int(np.sin(Rf_theta))], [0, int(-np.sin(Rf_theta)), int(np.cos(Rf_theta))]])

#rotaion_y = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])



img = plt.imread('letterR.jpg')
dimensions = img.shape




img_transformed = np.empty((512, 512,3), dtype=np.uint8)



for i, row in enumerate(img):
    for j, col in enumerate(row):
        pixel_data = img[i, j, :]
        input_coords = np.array([i, j, 1])
        i_out, j_out, _ = rotation_x @ input_coords
        img_transformed[i_out, j_out, :] = pixel_data



for gradient_y_phase in range(0, int(360/dimensions[0]), 1):
    for i, row in enumerate(img_transformed):
        for j, col in enumerate(row):
            pixel_data = img_transformed[i, j, :]
            input_coords = np.array([i, j, 1])
            i_out, j_out, _ = Rotation_z(np.radians(gradient_y_phase)) @ input_coords
            img_transformed[i_out, j_out, :] = pixel_data
    for gradient_x_phase in range(0, int(360/dimensions[1]), 1):
        for i, row in enumerate(img_transformed):
            for j, col in enumerate(row):
                pixel_data = img_transformed[i, j, :]
                input_coords = np.array([i, j, 1])
                i_out, j_out, _ = Rotation_z(np.radians(gradient_x_phase)) @ input_coords
                img_transformed[i_out, j_out, :] = pixel_data






plt.figure(figsize=(5, 5))
plt.imshow(img_transformed)
plt.show()
