import csv
import os
import tqdm

def create_csv(dataset_dir, csv_file):
    # Open the CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(["Image path", "Label"])
        
        # Iterate through the folders in the dataset directory
        for folder_name in tqdm.tqdm(os.listdir(dataset_dir)):
            folder_path = os.path.join(dataset_dir, folder_name)
            
            # Get the label from the folder name
            if folder_name == "NORMAL":
                label = "NORMAL"
            else:
                label = "PNEUMONIA"
            
            # Iterate through the images in the folder
            for image_name in tqdm.tqdm(os.listdir(folder_path)):
                image_path = os.path.join(folder_path, image_name)
                
                # Write the image path and label to the CSV file
                writer.writerow([image_path, label])

    print("CSV file created successfully.")
