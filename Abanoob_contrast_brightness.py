from PIL import Image,ImageEnhance
from phantominator import shepp_logan
import matplotlib.pyplot as plt
import numpy as np


phantomImg = shepp_logan(128)
# MR phantom (returns proton density, T1, and T2 maps)
PD, T1, T2 = shepp_logan((128, 128, 20), MR=True)
fig, ax = plt.subplots(2)
plt.imsave('images/tempPhantom.png', phantomImg, cmap='gray')
ax[0].imshow(phantomImg, cmap='gray')

img=Image.open("images/tempPhantom.png")
img_contr_obj=ImageEnhance.Contrast(img)
factor=5
e_img=img_contr_obj.enhance(factor)
arrayImg = np.array(e_img)
ax[1].imshow(arrayImg, cmap='gray')
plt.show()
