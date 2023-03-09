# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
 
# # Create 3 layers of data (PD, T1, T2)
# pd = np.random.rand(100, 100, 100) # PD layer
# t1 = np.random.rand(100, 100, 100) # T1 layer
# t2 = np.random.rand(100, 100, 100) # T2 layer
 
# # Combine the 3 layers into a single 3D array 
# mri = np.stack((pd, t1, t2), axis=0)  # axis=0 means along the depth dimension (z-axis) 
 
# # Plot the MRI image in 3D using matplotlib and numpy arrays 
# fig, axs = plt.subplots(1,3)
# fig.suptitle('Scan Array (Middle Slices)')
# axs[0].imshow(mri[2 ,: , : , 90], cmap='gray')
# fig.tight_layout()
# plt.show()

#-------------------------------Working Code------------------------#
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Open the image file
img = Image.open('images/Phantom512.png')

# Get the pixels array as a 2D list
pixel_data = list(img.getdata())

# Print the first pixel
print("image pixels:",pixel_data[0])

# Convert the pixel data to a NumPy array
t1_img_array = np.array(pixel_data)

#converting T1 to T2
img_array = t1_img_array.astype('float64')
img_array /= img_array.max()
te = 80 # ms
alpha = 0.7 # assuming white matter
t2_arr_conv = (img_array / te) ** (1 / alpha)

tmin, tmax = float(t1_img_array.min()), float(t1_img_array.max())
t2_arr_conv *= (tmax - tmin)
t2_arr_conv += tmin
print("T2 is :" , t2_arr_conv)



# Reshape the array to match the image dimensions
width, height = img.size
t1_img_array = t1_img_array.reshape((height, width, 3))
t2_arr_conv = t2_arr_conv.reshape((height, width, 3))
fig, axs = plt.subplots(1,2)
fig.suptitle('T1 and T2')
axs[0].imshow(t1_img_array, cmap='gray')
axs[1].imshow(t2_arr_conv, cmap='gray')
fig.tight_layout()

#onclick function
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()