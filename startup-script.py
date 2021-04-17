import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np
# from PIL import Image, ImageOps
# import numpy as np

st.markdown("<h1>Hello!</h1>",  unsafe_allow_html=True)
st.header("How do I identify cricket player name from an image?")
st.write("Solution built using *TeachableMachine* & *streamlit.io*  :sunglasses:")

def teachable_machine_classification(img, file):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = keras.models.load_model(file)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img
    # image = Image.open(img_name).convert('RGB')
    # image = cv2.imread(image)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #print(prediction)
    return np.argmax(prediction)
  
uploaded_file = st.file_uploader("Choose Image", type="jpeg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    #image = Image.open(img_name).convert('RGB')
    st.image(image, caption='Image Uploaded', use_column_width=True)
    st.write("")
    st.write("Identifying the cricketers...")
    label = teachable_machine_classification(image, 'keras_model.h5')
    print(label)
#     if label == 1:
#         st.write("This X ray looks like having pneumonia.It has abnormal opacification.Needs further investigation by a Radiologist/Doctor.")
#     else:
#         st.write("Hooray!! This X ray looks normal.This X ray depicts clear lungs without any areas of abnormal opacification in the image")

# Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# Load the model
# model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# image = Image.open('test_photo.jpg')

# #resize the image to a 224x224 with the same strategy as in TM2:
# #resizing the image to be at least 224x224 and then cropping from the center
# size = (224, 224)
# image = ImageOps.fit(image, size, Image.ANTIALIAS)

# #turn the image into a numpy array
# image_array = np.asarray(image)

# # display the resized image
# image.show()

# # Normalize the image
# normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# # Load the image into the array
# data[0] = normalized_image_array

# # run the inference
# prediction = model.predict(data)
# print(prediction)
