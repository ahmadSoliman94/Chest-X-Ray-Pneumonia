import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

import create_custom_model as model

# Load the saved model
custom_model = model.CustomModel(num_classes=2)
custom_model.load_state_dict(torch.load('./best_model.pt', map_location=torch.device('cpu')))
custom_model.eval()  # Set the model to evaluation mode

# Define the image transformation pipeline
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB') # Convert the image to RGB (PyTorch expects RGB images)
    image = image_transforms(image).unsqueeze(0)  # Add a batch dimension
    return image


def predict_pneumonia(image):
    with torch.no_grad():
        outputs = custom_model(image)
        predicted_probabilities = F.softmax(outputs, dim=1)
        predicted_class_index = torch.argmax(predicted_probabilities, dim=1)
        return predicted_probabilities[0], predicted_class_index.item()
    

def main():
    st.title("Pneumonia Classification")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Preprocess the uploaded image
        image = preprocess_image(uploaded_file)

        # Make predictions
        probabilities, class_index = predict_pneumonia(image)

        # Display the results
        class_labels = ['Normal', 'Pneumonia']
        st.write(f"Predicted class: {class_labels[class_index]}")
        st.write(f"Probabilities: Normal={probabilities[0]:.4f}, Pneumonia={probabilities[1]:.4f}")

        # Convert tensor back to PIL image
        pil_image = transforms.ToPILImage()(image.squeeze(0))

        # Display the uploaded image
        st.image(pil_image, channels='grayscale')

if __name__ == '__main__':
    main()