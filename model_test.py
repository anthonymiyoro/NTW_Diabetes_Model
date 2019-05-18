# import numpy as np
# import tensorflow as tf
# import cv2
# from PIL import Image

# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="saved_models/converted_model.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Test model on image
# im = Image.open("images/17_left.jpeg")
# np_im = np.array(im)
# print (np_im.shape)
# print (" ")
# print (type(np_im))

# np_im = np_im - 18
# new_im = Image.fromarray(np_im)
# new_im.save("numpy_altered_sample2.png")

# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# print ("input data", np_im)
# print (" ")
# print ("input shape", input_shape) 
# interpreter.set_tensor(input_details[0]['index'], np_im)

# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print("output shape", output_data)


from keras.models import load_model
from keras.preprocessing.image import load_img
# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np

#load saved model file
model = load_model('saved_models/keras_model.h5')

# Load image to be passed to the model
# image = cv2.imread('images/17_left.jpeg')

# dimensions of our images
img_width, img_height = 128, 128

# load the image
img = load_img('images/17_left.jpeg', target_size=(img_width, img_height))

# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)

# show the image
img.show()

# convert image to numpy array
img_array = img_to_array(img)
print(img_array.dtype)
print(img_array.shape)

img_array = np.expand_dims(img_array, axis=0)


classes = model.predict_classes(img_array)

print (classes)