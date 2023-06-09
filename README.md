# Pneumonia classification

## Description of the Pneumonia Dataset
### The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). 

<br />

### Becuse the dataset is imbalanced i combined all the images in one folder and split them into train, test and validation sets. After that i used data augmentation to increase the number of images in the training set.

<br />

### The model is trained using transfer learning with the EfficientNetB0 Pytorch-model. Then deploy the model using streamlit.

<br />

### __Results__:
<br />

![1](images/peformance.png)
<br/>
<br/>

![1](images/output.png)

<br />
<br />

![1](images/Classification_report.png)

<br />
<br />

![1](images/streamlit.png)