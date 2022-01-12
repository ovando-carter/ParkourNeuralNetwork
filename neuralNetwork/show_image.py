import cv2
import matplotlib.pyplot as plt

# get image data in colour
#image_data = cv2.imread('prediction_images/tshirt.png', cv2.IMREAD_UNCHANGED)
#plt.imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))


# get image data and change it to grey scale
image_data = cv2.imread('prediction_images/Wide_Brim_Suede .jpg', cv2.IMREAD_GRAYSCALE)

# Resize the plot so that it is the same size as the test data images
image_data = cv2.resize(image_data, (28, 28))

# invert the pixels so that they look like the images in the test data i.e. black background with white clothing.
image_data = 255 - image_data

plt.imshow(image_data, cmap='gray')



plt.show()