import matplotlib.pyplot as plt
import numpy as np
import cv2
from phantominator import shepp_logan
from math import sin, cos, pi


def normalize_image(image):
    """
    Normalize an image from 0 to 1.

    Args:
        image (numpy.ndarray): Input image as a numpy array.

    Returns:
        numpy.ndarray: Normalized image as a numpy array.
    """
    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image using the formula (image - min) / (max - min)
    normalized_image = (image - min_val) / (max_val - min_val)

    return normalized_image



phantomImg = shepp_logan(32)

img = normalize_image(phantomImg)
print(np.min(img))
print(np.max(img))




image = np.zeros((32, 32, 3))
image[:,:,2] = img
print(image.shape)
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.show()

kSpace = np.zeros((32, 32), dtype=np.complex_)

def equ_of_Rotation_z(theta):
    rotation_z = np.array(
        [[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0], [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
    return rotation_z

def equ_of_Rotation_x(theta):
    rotation_x = np.array([[1, 0, 0], [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))], [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    return rotation_x


def Rotation_x(Image):
    img_transformed = np.zeros(Image.shape)
    for i in range(0, Image.shape[0]):
        for j in range(0, Image.shape[1]):
            img_transformed[i,j] = np.dot(equ_of_Rotation_x(90),Image[i,j])

    return img_transformed



for A in range(0, image.shape[0]):
    rotated_matrix = Rotation_x(image)
    for B in range(0, image.shape[1]):
        step_of_Y = (360 / image.shape[0]) * B
        step_of_X = (360 / image.shape[1]) * A
        newmatrix = np.zeros(image.shape)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                phase = step_of_Y * j + step_of_X * i
                newmatrix[i, j] = np.dot(equ_of_Rotation_z(phase), rotated_matrix[i, j])
        # gradiented_image = gradient_x_and_y(rotated_matrix,step_of_X,step_of_Y)\
        gradiented_image = newmatrix
        sum_of_x = np.sum(gradiented_image[:, :, 0])
        sum_of_y = np.sum(gradiented_image[:, :, 1])
        complex_value = np.complex(sum_of_x,sum_of_y)
        kSpace[A,B] = complex_value

    image = np.zeros((32, 32, 3))
    image[:, :, 2] = img
    print(A)


kSpace = np.fft.fft2(kSpace)
plt.figure(figsize=(5, 5))
plt.imshow(np.abs(kSpace), cmap='gray')
plt.show()









