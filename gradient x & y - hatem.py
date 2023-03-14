import matplotlib.pyplot as plt
import numpy as np
from phantominator import shepp_logan

phantomImg = shepp_logan(32)


def normalize_image(image):
    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image using the formula (image - min) / (max - min)
    normalized_image = (image - min_val) / (max_val - min_val)

    return normalized_image



def modify_image(Phantom_img):
    normalized_img = normalize_image(Phantom_img)
    final_image = np.zeros((Phantom_img.shape[0], Phantom_img.shape[1], 3))
    final_image[:, :, 2] = normalized_img
    return final_image

def equ_of_Rotation_z(theta):
    rotation_z = np.array(
        [[np.cos(np.radians(theta)), -np.sin(np.radians(theta)), 0], [np.sin(np.radians(theta)), np.cos(np.radians(theta)), 0], [0, 0, 1]])
    return rotation_z

def equ_of_Rotation_x(theta):
    rotation_x = np.array([[1, 0, 0], [0, np.cos(np.radians(theta)), -np.sin(np.radians(theta))], [0, np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    return rotation_x


def Rotation_x(Image,phase_X):
    rotated_image = np.zeros(Image.shape)
    for i in range(0, Image.shape[0]):
        for j in range(0, Image.shape[1]):
            rotated_image[i,j] = np.dot(equ_of_Rotation_x(phase_X),Image[i,j])

    return rotated_image



def reconstruct_image():
    kSpace = np.zeros((32, 32), dtype=np.complex_)
    modified_img = modify_image(phantomImg)
    Phase_of_X = 90
    for A in range(0, modified_img.shape[0]):
        rotated_matrix = Rotation_x(modified_img, Phase_of_X)
        for B in range(0, modified_img.shape[1]):
            step_of_Y = (360 / modified_img.shape[0]) * B
            step_of_X = (360 / modified_img.shape[1]) * A
            Final_matrix = np.zeros(modified_img.shape)
            for i in range(0, modified_img.shape[0]):
                for j in range(0, modified_img.shape[1]):
                    phase = step_of_Y * j + step_of_X * i
                    Final_matrix[i, j] = np.dot(equ_of_Rotation_z(phase), rotated_matrix[i, j])
            gradient_image = Final_matrix
            sum_of_x = np.sum(gradient_image[:, :, 0])
            sum_of_y = np.sum(gradient_image[:, :, 1])
            complex_value = np.complex(sum_of_x, sum_of_y)
            kSpace[A, B] = complex_value


        Final_img = np.zeros((phantomImg.shape[0], phantomImg.shape[1], 3))
        Final_img[:, :, 2] = phantomImg
        plt.imshow(np.abs(kSpace), cmap='gray')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        print(A)

    Reconstructed_image = np.fft.fft2(kSpace)
    plt.imshow(np.abs(Reconstructed_image), cmap='gray')
    plt.show()
    plt.imshow(np.abs(kSpace), cmap='gray')
    plt.show()


reconstruct_image()





