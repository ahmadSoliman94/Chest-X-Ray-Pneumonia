import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from efficientnet_model import EfficientNetModel
import torch

import test_model


# Instantiate the EfficientNetModel class
efficientnet_model = EfficientNetModel()
efficientnet_model.load_state_dict(torch.load("./best_model.pth"))

# Create an instance of the ImageClassifier class
predictor = test_moel.ImageClassifier(efficientnet_model)

def main(): 
    
 
    
    
    # Add Streamlit app title
    st.title("Pneumonia Classifier")
    
    # Upload image file
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = plt.imread(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform prediction on the uploaded image
        predicted_class = classifier.predict_single_image(image)

        # Display the predicted class
        st.write("Predicted Class:", predicted_class)


if __name__ == "__main__":
    main()