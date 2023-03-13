class phantomMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_initial_figure(self):
        # Open the image file
        img = Image.open('images/Phantom512.png')
        # Get the pixels array as a 2D list
        pixel_data = list(img.getdata())
        # Convert the pixel data to a NumPy array
        t1_img_array = np.array(pixel_data)
        # Reshape the array to match the image dimensions
        width, height = img.size
        t1_img_array = t1_img_array.reshape((height, width, 3))
        self.axes.imshow(t1_img_array, cmap='gray')