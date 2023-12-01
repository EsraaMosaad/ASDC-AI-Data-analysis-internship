# ASDC-AI-Data-analysis-internship



Task 1: Perform customer segmentation using any method or tools you prefer.  Analyze the dataset and segment customers based on relevant parameters like Gender ,Age, or any other criteria you find important.
Dataset Link: https://github.com/ELSOUDY2030/Mall-Customers.git

/////////////////////////////////////////////////////
Task 2: 102 Category Flower Dataset

I embarked on the challenging task of classifying images using the well-known "102 Category Flower Dataset." Initially, I opted for the ResNet50 pre-trained model for its powerful feature extraction capabilities. However, to my dismay, the training process resulted in unsatisfactory outcomes, and the situation escalated to the point of Colab crashing. 

In response to these challenges, I decided to switch gears and adopted a batch-wise training approach. This involved breaking down the training process into manageable batches, coupled with the utilization of a generator for data augmentation. The motivation behind this strategy was to mitigate potential RAM issues that could contribute to Colab instability.

In terms of preprocessing, I ensured that the image data was appropriately resized to dimensions of (300, 300, 3). Additionally, label adjustments were made to align with the training requirements. Despite these meticulous efforts, a perplexing issue persists – the model appears to be unresponsive, consistently yielding an accuracy of 0.

As I delved into the intricacies of the problem, I encountered perplexing outcomes during the training epochs, leading me to question the effectiveness of the adopted strategies. At this juncture, I seek valuable insights and guidance to decipher the root cause behind the model's reluctance to learn effectively. Any suggestions or recommendations to enhance the training process and improve the accuracy of the model would be immensely beneficial. Your expertise and assistance in unraveling this puzzle would be highly appreciated.


