import matplotlib.pyplot as plt
import numpy as np
import cv2


img = plt.imread('letterR.jpg')
dimensions = img.shape
print(dimensions)


width = int(img.shape[1] * 64 / 512)
height = int(img.shape[0] * 64 / 512)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print(resized.shape)

image = resized[:,:,2]
img_transformed = np.zeros((64, 64,3), dtype=np.uint8)
img_transformed[:, :, 2] = image



def Rotation_z(theta):
    rotation_z = np.array(
        [[int(np.cos(theta)), int(-np.sin(theta)), 0], [int(np.sin(theta)), int(np.cos(theta)), 0], [0, 0, 1]])
    return rotation_z

Rf_theta =  np.radians(90)
rotation_x = np.array([[1, 0, 0], [0, int(np.cos(Rf_theta)), int(-np.sin(Rf_theta))], [0, int(np.sin(Rf_theta)), int(np.cos(Rf_theta))]])




def Rotation_x(image):
    for i, row in enumerate(image):
        for j, col in enumerate(row):
            pixel_data = image[i, j, :]
            input_coords = np.array([i, j, 1])
            i_out, j_out, _ = rotation_x @ input_coords
            img_transformed[i_out, j_out, :] = pixel_data


Rotation_x(resized)


"""
for I in range(0, resized.shape[0]):
    for J in range(0, resized.shape[1]):
        step_of_Y = (360 / resized.shape[0]) * I
        step_of_X = (360 / resized.shape[1]) * J
        for i, row in enumerate(img_transformed):
            for j, col in enumerate(row):
                pixel_data = img_transformed[i, j, :]
                input_coords = np.array([i, j, 1])
                phase = step_of_Y * i + step_of_X * j
                i_out, j_out, _ = Rotation_z(np.radians(phase)) @ input_coords
                img_transformed[i_out, j_out, :] = pixel_data
"""


"""
for gradient_y_phase in range(0, 360, 30):
    for i, row in enumerate(img_transformed):
        for j, col in enumerate(row):
            pixel_data = img_transformed[i, j, :]
            input_coords = np.array([i, j, 1])
            i_out, j_out, _ = Rotation_z(np.radians(gradient_y_phase)) @ input_coords
            img_transformed[i_out, j_out, :] = pixel_data
        for gradient_x_phase in range(0, 360, 30):
            for j, col in enumerate(row):
                pixel_data = img_transformed[i, j, :]
                input_coords = np.array([i, j, 1])
                i_out, j_out, _ = Rotation_z(np.radians(gradient_x_phase)) @ input_coords
                img_transformed[i_out, j_out, :] = pixel_data

"""



plt.figure(figsize=(5, 5))
plt.imshow(img_transformed, cmap='gray')
plt.show()
