import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# loading Data part
img = nib.load('./data/images/BRATS_002.nii.gz')
imgArray = img.get_fdata()
imgArrayShape = imgArray.shape
print("the image data is :\n" , img)
print("the image Array is :\n" , imgArray)
print("the image Array Shape is: \n" , imgArrayShape)
# the image array shape consists of 4 values 
# (n-voxels in x , n-voxels in y , n-voxels or slices in z , weight values channel or layer <t1 or t2 or pd or>)
#Display scan array's middle slices
fig, axs = plt.subplots(1,3)
fig.suptitle('Scan Array (Middle Slices)')
axs[0].imshow(imgArray[imgArrayShape[0]//2,:,: , 0], cmap='gray')
axs[1].imshow(imgArray[:,imgArrayShape[1]//2,: , 0], cmap='gray')
axs[2].imshow(imgArray[:,:,imgArrayShape[2]//3 , 0], cmap='gray')
fig.tight_layout()
plt.show()
