#Title: Dental X-ray Image Instance Segmentation

Objective: The goal of this project is to train a model to recognize tooth numbers and conditions such as implants, root canals, or crowns in dental X-ray images using Instance Segmentation. The project acknowledges the tutorial [link](https://www.youtube.com/watch?v=ytlhMAF6ok0&t=1736s) for guidance on implementing YOLOv8 for instance segmentation.

Steps:

1. **Dataset Source** The dataset used for training and evaluation is obtained from [Roboflow](https://universe.roboflow.com/bassem-ahmed-ouwsa/dentistry-vbril). The dataset comprises various dental X-ray images that showcase a range of conditions. 

2. **Model Selection:** The YOLOv8 architecture is chosen for this task. YOLO (You Only Look Once) is a real-time object detection system that can be adapted for instance segmentation tasks.

3. **Model Configuration:** The project uses the Ultralytics YOLO library to build and train the model. The configuration is specified in a YAML file (`Dentistry-1/data.yaml`) that defines the number of classes, dataset paths, and other training parameters.

4. **Pre-training:** A pre-trained model is loaded to initialize the weights. The pre-trained weights help the model achieve better performance by leveraging knowledge from a model trained on a larger dataset.

5. **Training:** The model is trained using the provided dental X-ray dataset. Training is performed for 20 epochs, with a batch size of 4 and image size of 800x800 pixels.


6. **Model Evaluation**  Results, including precision, recall, and mAP values 
  *Training Metrics*
              Loss: 0.75
              Precision (P): 0.801
              Recall (R): 0.79
              mAP50: 0.801
              mAP50-95: 0.487

  *Class-wise Performance*

    The model shows good performance across various tooth classes (e.g., 11, 12, ..., 48) with consistent precision and recall values.
    The 'Crown' class achieves high precision (0.895) and recall (0.908).
    'Implant' and 'Root Canal' classes also demonstrate strong performance with high precision and recall.

 *Instance Segmentation*

    The model successfully identifies instances of dental conditions, such as implants, root canals, and crowns, across the dataset.
    The mAP50 (mean Average Precision) values indicate the model's ability to precisely locate and classify instances.


7. **Prediction:** The trained model is then used to make predictions on new dental X-ray images. The predictions include class names, bounding boxes, and instance masks.

8. **Visualization:** The results are visualized using matplotlib. Images with overlaid masks for specific classes, such as 'Root Canal,' demonstrate the model's ability to perform instance segmentation.

9. **Summary:** The project successfully achieves the goal of instance segmentation in dental X-ray images, identifying tooth numbers
and specific conditions. The trained model can be further fine-tuned or used for predictions on new data.





