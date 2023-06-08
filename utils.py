import cv2
import random
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

import csv
import os
from sklearn.model_selection import train_test_split



def show_multiple_images(dataset, num_images):
    # Create a figure and axes for subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 10))

    # Iterate over the number of images
    for i in range(num_images):
        # Choose a random index
        random_idx = random.randint(0, len(dataset) - 1)

        # Retrieve the image and label using the random index
        image, label = dataset[random_idx]

        # Display the image and label in the corresponding subplot
        axes[i].imshow(image)
        axes[i].set_title('Label: ' + label)
        axes[i].axis('off')

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the figure with multiple images
    plt.show()


def show_labels_distribution(dataframe):

    # Assuming you have a dataset named 'df' with a label column 'label'
    label_counts = dataframe['Label'].value_counts()

    # Customizing the pie chart
    plt.pie(label_counts, labels=label_counts.index,autopct='%1.1f%%',shadow=True)

    # Adding a title and displaying the plot
    plt.title('Distribution of Labels')
    plt.show()
    
    
    


def split_csv(csv_file, train_csv, valid_csv, test_csv, test_size=0.2, valid_size=0.25):
    # Open the original CSV file for reading
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read and store the header row

        # Read the remaining rows from the CSV file
        data = [row for row in reader]

    # Split the data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Split the train data into train and validation sets
    train_data, valid_data = train_test_split(train_data, test_size=valid_size, random_state=42)

    # Write the train data to the train CSV file
    with open(train_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header row
        writer.writerows(train_data)  # Write the train data rows

    # Write the validation data to the validation CSV file
    with open(valid_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header row
        writer.writerows(valid_data)  # Write the validation data rows

    # Write the test data to the test CSV file
    with open(test_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header row
        writer.writerows(test_data)  # Write the test data rows

    print("CSV file split into train, validation, and test sets successfully.")
