# ASDC-AI-Data-analysis-internship



**Task 1: Perform customer segmentation ** 
using any method or tools you prefer.  Analyze the dataset and segment customers based on relevant parameters like Gender ,Age, or any other criteria you find important.
Dataset Link: https://github.com/ELSOUDY2030/Mall-Customers.git
([Task 1/Perform_customer_segmentation.ipynb](https://github.com/EsraaMosaad/ASDC-AI-Data-analysis-internship/blob/main/Task%201/Perform_customer_segmentation.ipynb))

/////////////////////////////////////////////////////

**Task 2: 102 Category Flower Dataset**

I embarked on the challenging task of classifying images using the well-known "102 Category Flower Dataset." Initially, I opted for the ResNet50 pre-trained model for its powerful feature extraction capabilities. However, to my dismay, the training process resulted in unsatisfactory outcomes, and the situation escalated to the point of Colab crashing. 

In response to these challenges, I decided to switch gears and adopted a batch-wise training approach. This involved breaking down the training process into manageable batches, coupled with the utilization of a generator for data augmentation. The motivation behind this strategy was to mitigate potential RAM issues that could contribute to Colab instability.

In terms of preprocessing, I ensured that the image data was appropriately resized to dimensions of (300, 300, 3). Additionally, label adjustments were made to align with the training requirements. Despite these meticulous efforts, a perplexing issue persists – the model appears to be unresponsive, consistently yielding an accuracy of 0.

As I delved into the intricacies of the problem, I encountered perplexing outcomes during the training epochs, leading me to question the effectiveness of the adopted strategies. At this juncture, I seek valuable insights and guidance to decipher the root cause behind the model's reluctance to learn effectively. Any suggestions or recommendations to enhance the training process and improve the accuracy of the model would be immensely beneficial. Your expertise and assistance in unraveling this puzzle would be highly appreciated.

-- update 
I aim to enhance efficiency and address issues by optimizing my approach. I execute code in PyCharm, leveraging TensorFlow-GPU and CUDA to expedite the training process. Additionally, I employ TensorFlow for image loading, ensuring the judicious utilization of resources. I run code for only 10 epoch and the model s learning good  as you see in graph


![accuracy_plot](https://github.com/EsraaMosaad/ASDC-AI-Data-analysis-internship/assets/70305108/5de9a6ec-779e-4482-ad59-0a06ee98f186)




//////////////////////////////////////////

**Task 4: Dental X-ray Image Instance Segmentation**

Objective: The goal of this project is to train a model to recognize tooth numbers and conditions such as implants, root canals, or crowns in dental X-ray images using Instance Segmentation. The project acknowledges the tutorial [link](https://www.youtube.com/watch?v=ytlhMAF6ok0&t=1736s) for guidance on implementing YOLOv8 for instance segmentation.

([Task 4/dental_X_ray_images_using_YOLOv8.ipynb](https://github.com/EsraaMosaad/ASDC-AI-Data-analysis-internship/blob/main/Task%204/dental_X_ray_images_using_YOLOv8.ipynb))

