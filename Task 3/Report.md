#Semantic Segmentation for Sports Analytics

**Task Description:**
  Perform pixel-perfect semantic segmentation with 11 classes on a dataset consisting of 100 frames extracted at every 12th frame from 
  the Real Madrid vs. Manchester United match. The frames have been carefully curated, with some blurred frames and outliers replaced.
  The dataset is tailored for training detection models with a focus on sports analytics, particularly soccer.

**Dataset Overview:**
- **Frames:** 100 frames selected at regular intervals from the mentioned soccer match.
- **Classes:** The segmentation task involves 11 classes, providing detailed information about different elements in the frames.

**Dataset Link:** [Semantic Segmentation Dataset](https://www.kaggle.com/datasets/mohammednomer/semantic-segmentation)


  

***Approach 1: Resizing Images to 256x256***

**Model Architecture:**
- Followed the U-Net architecture demonstrated in the video: [229 - Smooth blending of patches for semantic segmentation of large images (using U-Net)](https://www.youtube.com/watch?v=HrGn4uFrMOM&t=7s).
- Implemented a U-Net model with a contraction path and an expansive path for semantic segmentation.

**Model Modification:**
- Resized input images to 256x256 pixels as recommended in the video.
- Adjusted the input layer of the U-Net model to handle images of size 256x256 pixels.

**Training:**
- Trained the U-Net model on the resized images (256x256) and their corresponding masks.
- The dataset was split into training and testing sets for model evaluation.

**Evaluation:**
- Evaluated the model performance on the test set using metrics like Jaccard coefficient or Intersection over Union (IoU) as suggested in the video.
- These metrics provide insights into the accuracy of the segmentation achieved by the model.

