import matplotlib.pyplot as plt
import numpy as np
import cv2


img = plt.imread('phantom.jpg')
img = img/255
dimensions = img.shape
print(dimensions)
print(np.min(img))
print(np.max(img))



"""
width = int(img.shape[1] * 64 / 512)
height = int(img.shape[0] * 64 / 512)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print(resized.shape)
"""
image = np.zeros((64, 64, 3))
image[:,:,2] = img[:,:,2]
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.show()
"""
img_transformed = np.zeros((64, 64,3), dtype=np.uint8)
img_transformed[:, :, 2] = image
"""
kSpace = np.zeros((64, 64), dtype=np.complex_)

def equ_of_Rotation_z(theta):
    rotation_z = np.array(
        [[int(np.cos(theta)), int(-np.sin(np.radians(theta))), 0], [int(np.sin(np.radians(theta))), int(np.cos(np.radians(theta))), 0], [0, 0, 1]])
    return rotation_z

def equ_of_Rotation_x(theta):
    rotation_x = np.array([[1, 0, 0], [0, int(np.cos(np.radians(theta))), int(-np.sin(np.radians(theta)))], [0, int(np.sin(np.radians(theta))), int(np.cos(np.radians(theta)))]])
    return rotation_x


"""
def Rotation_x(image):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            img_transformed[i,j] = np.dot(equ_of_Rotation_x(90),img_transformed[i,j])

    return img_transformed





for A in range(0, img.shape[0]):
    Rotation_x(img)
    for B in range(0, img.shape[1]):
        step_of_Y = (360 / img.shape[0]) * B
        step_of_X = (360 / img.shape[1]) * A
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                phase = step_of_Y * j + step_of_X * i
                img_transformed[i,j] = np.dot(equ_of_Rotation_z(phase),img_transformed[i,j])
                sumOfRows = np.sum(img_transformed[:, :, 0])
                sumOfColumns = np.sum(img_transformed[:, :, 1])
                complex_value = np.complex(sumOfRows,sumOfColumns)
                kSpace[i,j] = complex_value


kSpace = np.fft.fft2(kSpace)
plt.figure(figsize=(5, 5))
plt.imshow(np.abs(kSpace), cmap='gray')
plt.show()
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




