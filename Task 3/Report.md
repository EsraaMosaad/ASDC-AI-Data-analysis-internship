#Semantic Segmentation for Sports Analytics

**Task Description:**
  Perform pixel-perfect semantic segmentation with 11 classes on a dataset consisting of 100 frames extracted at every 12th frame from 
  the Real Madrid vs. Manchester United match. The frames have been carefully curated, with some blurred frames and outliers replaced.
  The dataset is tailored for training detection models with a focus on sports analytics, particularly soccer.

**Dataset Overview:**
- **Frames:** 100 frames selected at regular intervals from the mentioned soccer match.
- **Classes:** The segmentation task involves 11 classes, providing detailed information about different elements in the frames.

**Dataset Link:** [Semantic Segmentation Dataset](https://www.kaggle.com/datasets/mohammednomer/semantic-segmentation)



## Model Architecture
- Followed the U-Net architecture demonstrated in the video: [229 - Smooth blending of patches for semantic segmentation of large images (using U-Net)](https://www.youtube.com/watch?v=HrGn4uFrMOM&t=7s).
- Implemented a U-Net model with a contraction path and an expansive path for semantic segmentation.


## Training Approaches

### Resizing Input Images

    - **Description:** The input images have been resized to a fixed size of 256x256 pixels.
    - **Advantages:**
      - Simplicity and consistency: Resizing ensures a uniform input size for all images, facilitating neural network training.
      - Potential training speedup: Fixed-size inputs might contribute to faster training times.
    - **Considerations:**
      - Information loss: Depending on the dataset, resizing may result in the loss of important details, especially if there are 
            significant variations in object sizes.

### Patchifying Images

  - **Description:** A patch-based approach has been adopted, involving breaking the images into smaller patches. Each patch is treated as 
                       a separate input during training.
  - **Advantages:**
    - Captures local details: Patch-based training allows the model to focus on specific local features, making it robust to variations in 
                                object sizes.
    - Robust to size variations: Useful for tasks where objects of interest may have varying scales or spatial distributions.
  - **Considerations:**
    - Increased complexity: Handling patches introduces additional complexity in data processing and model training.
    - Correct labeling: Ensuring accurate labeling of patches is crucial for successful training.



## Training
- Trained the U-Net model on the resized images (256x256) and their corresponding masks.
- The dataset was split into training and testing sets for model evaluation.

## Evaluation
- Evaluated the model performance on the test set using metrics like Jaccard coefficient or Intersection over Union (IoU) as suggested in the video.
- These metrics provide insights into the accuracy of the segmentation achieved by the model.




