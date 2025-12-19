# leukemia Classification
## Final project for CSC334 Biomedical Big Data: Using Machine Learning for Bioinformatics. 
Image analysis of acute lymphoblastic leukemia and non-leukemia cells. 
Data source: https://www.kaggle.com/datasets/andrewmvd/leukemia-classification

White blood cell images, even with staining, can be very similar and difficult to tell apart. I attempted to create an image classification model to identify images of cancerous versus non-cancerous white blood cells. The dataset contains cells from patients with acute lymphoblastic leukemia where two-thirds of the training images have been identified as cancerous. 

The program takes in images (downloaded, and in the same folder as the code being run) that are sorted into folders but not labeled. The input folders are training, test, and validation data, but the program also takes a subsection of the training data for validation. 

My solution uses pytorch, tensor, and imagenet. I generated the base code using Smith's Open AI GPT-5-mini. 

I attempted to train the model using 8 epochs, but it had only completed 4 epoches after 24 hours. I was still able to run the later code since the information on the latest iteration of the model is stored as a separate document. The model would have benefitted from further iterations. 

## Packages used: 
os, csv, random, numpy, PIL/pillow, tqdm, torch, torch.nn, torchvision, scikit learn/sklearn, pandas, pyplot, glob, zipfile, Path

## How to run
leukemiaGPT1.ipynb is the main file. Each block can be run sequentially from the top. The Train Head block can be interrupted if taking too long and a model will still be created as long as one epoch is complete. The Fine-Tune block can be skipped if needed for time. If using the default model, training blocks and training helpers can be skipped as the next step is loading back in the "best" model. 

## Files
All files should be in the same folder. 
leukemiaGPT1.py is the original file that I used to generate the model. The constants have since been modified and other steps added, but the setup is basically the same. 
leukemia-classification is the folder containing all of the images (as well as the labels for the validation set). Images are all .bmp but the program is also set up to handle .png .jpg .jpeg .tif and .tiff files. 
Image organization: 
- C-NMC-Leukemia
- - testing_data
  - - C-NMC_test_final_phase_data
  - training_data
  - - fold_0
    - - all
      - hem
    - fold_1
    - - all
      - hem
    - fold_2
    - - all
      - hem
  - validation_data
  - - C-NMC_test_prelim_phase_data
    - C-NMC_test_prelim_phase_data_labels.csv
The program creates a folder called output_pytorch. This contains representations of the models created saved as files best_resnet50.pth and resnet50_final.pth. It also creates predictions_testing_with_gt.csv (which does not actually use ground_truth due to lack of labels but is the modified version of) predictions_testing.csv, predictions_validation_with_gt.csv, and predictions_validation.csv, which are human-readable. 
