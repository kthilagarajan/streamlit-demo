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
    model = keras.models.load_model(file, compile=False)

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
    
    labels = []
    
    with open("labels.txt", 'r') as File:
        infoFile = File.readlines() #Reading all the lines from File
        for line in infoFile: #Reading line-by-line
            words = line.split() #Splitting lines in words using space character as separator
            labels.append(words[1])
        
    result = prediction[0]
    
    result_map = {}
       
    players = []
    for i in range(len(result)):
        result_map[labels[i]] = float(result[i])
        
    st.write(result_map)
    sorted_players = sorted(result_map, key=result_map.get, reverse=True)
    
    return [result_map, sorted_players]
  
uploaded_file = st.file_uploader("Choose Image", type="jpeg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    #image = Image.open(img_name).convert('RGB')
    st.image(image, caption='Image Uploaded', use_column_width=True)
    output = teachable_machine_classification(image, 'keras_model.h5')
    st.write("Players found with accuracy : ")
    sorted_players = output[1]
    players_score = output[0]
    for i in range(len(sorted_players)):
        st.write(sorted_players[i] + "(" + str(players_score[sorted_players[i]]*100) + "%)")
