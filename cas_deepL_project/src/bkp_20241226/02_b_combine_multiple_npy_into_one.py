"""
This is after "02_a_load_preprocess_dask_images_size513.py" 

1.    "01_download_eurosat_selectedcategories.ipynb" -- download eurosat dataset images and save them in "data-selected_categories". use only images from EuroSAT folder which haas RGB images.
Do not use images in folder EuroSATallBands, not for this use case

----------------------------------------------------------------------------------------------------

2. a. "02_a_load_preprocess_dask_images_size513.py" 
                -- convert the image to size 513 using DASK and combine multiple images and save to .npy files, say Annual Crop has 3000 images, here only _149.npy files are created and saved in "data-preprocessed_data_imSize513"
                
   b. "02_b_combine_multiple_npy_into_one.py" --  here a single AnnualCrop.npy is created saved in the folder "data-preprocessed_output_data".

----------------------------------------------------------------------------------------------------

3. "03_split_and_save_TrainValTest_per_category_for_preprocessed_images" -- also seems to be doing the same as below

3. a. "03_a_split_and_save_TrainValTest_per_category_for_preprocessed_images" 
This will split "AnnualCrop.npy" into AnnualCrop_train.npy, AnnualCrop_val.npy and AnnualCrop_test.npy, saved in the folder "data-preprocessed_output_data"

   b. "03_b_test_shape_of_npy_splits" check if the shape of the splits is ok.

------------------------------------------------------------------------------------------------

As MNIST database we do not have correct images and thier lables, which is called ground truth, so we generate the segmentation masks by ourselves using pre-trained "deepLabV3" model.

Prior to this "sandbox_create_masking_test.ipynb" has been used, in notebooks folder, to generate segmentation masks using HSV, however later decided to use pre-tarined "deepLabV3" model.

4. "04_a_generate_segmentation_masks_for_preprocessed_images.py" -- generating segmentation masks for few test images in data-segmentation_masks_test - to verify if the code is working correctly.

    a. 04_a_save_kaggle_deepLabV3Model - download and save the pre-trained deeplLabV3 model from KaggleHub.
    
    b. "04_b_generate_segmentation_masks_for_preprocessed_images" -- generate segmentation masks for all images, save them in data-segmentation_masks in 3 separate folder aka train, val, test per category.
    For example AnnualCrop has 3000 images, so segmented masks are generated for all these 3000 images and distributed over train, val and test folders. The folder saved is "data-segmentation_masks-category-train/val/test" folders

    c. "04_c_test_generated_segmented_masks_quality.ipynb" -- this notebook is used to test the quality of geenrated segmentation masks, using overlay segment on original images, also using following method by generating some stats.

    1. Entropy (Measure of Class Diversity)
    2. Pixel Class Distribution
    3. Compactness (Spatial Consistency)
    4. Region Count

    These stats show its neither great nor bad, so proceed with segmentation projects rather than pivoting to classification. 

5. a. "05_a_prep_to_train_combine_segmentation_masks.ipynb" - to train the model we need to combine the generated segmented masks and split to train, val and test.npy

    The generated segmented masks available in "data-segmentation_masks-category-train/val/test" folders are now combined into a single .npy file as AnnualCrop_train_masks_combined.npy, AnnualCrop_val_masks_combined.npy, and AnnualCrop_test_masks_combined.npy.

    b. "05_b_prep_to_train_test_combined_segmentation_masks.ipynb" - This notebook will check if the original images and segmented masks especially when combined are still in same order.
    This is NOT like MNIST database where inpur digits and labels are at same place. so testing this, by selecting random images for every category for each split manually/visual testing.

------------------------------------------------------------------------------------------------

Files category_train.npy, category_val.npy and category_test.npy are copied from "data-preprocessed_output_data" to "data-trainingdata-split folders for easier organization".
Also the combined masks in "data-trainingdata" are organized in split folder under the same.

"""

import numpy as np
import os

# Define paths
preprocessed_dir = "../data/preprocessed_data_imSize513"
preprocessed_output_dir = "../data/preprocessed_output_data"
os.makedirs(preprocessed_output_dir, exist_ok=True)

def combine_npy_files(category):
    # Get all batch files for the category
    batch_files = sorted(
        [os.path.join(preprocessed_dir, file) 
         for file in os.listdir(preprocessed_dir) 
         if file.startswith(f"{category}_batch_") and file.endswith(".npy")]
    )
    
    # Determine the total number of images
    total_images = 0
    sample_shape = None
    for batch_file in batch_files:
        batch_data = np.load(batch_file, allow_pickle=True)
        total_images += len(batch_data)
        if sample_shape is None:
            sample_shape = batch_data[0].shape  # Get shape of one sample
    
    print(f"Total images for {category}: {total_images}")
    print(f"Sample image shape: {sample_shape}")
    
    # Create a memmapped file
    combined_file_path = os.path.join(preprocessed_output_dir, f"{category}.npy")
    combined_array = np.lib.format.open_memmap(
        combined_file_path,
        mode='w+',
        dtype='float32',
        shape=(total_images, *sample_shape)
    )
    
    # Append batches to the memmapped file
    start_idx = 0
    for batch_file in batch_files:
        print(f"Loading {batch_file}")
        batch_data = np.load(batch_file, allow_pickle=True)
        batch_size = len(batch_data)
        combined_array[start_idx:start_idx+batch_size] = batch_data
        start_idx += batch_size
        print(f"Appended {batch_file} to {combined_file_path}")
    
    print(f"Finished combining files for {category} into {combined_file_path}")

# Categories
categories = ["Forest", "Residential", "Highway", "AnnualCrop", "HerbaceousVegetation", "Industrial"]

# Combine files for each category
for category in categories:
    combine_npy_files(category)
