# Deblur-wgan-model-on-video
 wgan architecture to perform deblurring task
![img.png](img.png)

![image](https://user-images.githubusercontent.com/40007988/176914601-8b19f42f-b9a1-4d6c-ba2e-905a96c2adae.png)


This Script applies a trained Deblurring GAN model on a "blurry input (video)" to demonstrate the improvement in sharpness

Inputs:
* START_VID_FROM_TIME = 3  # start the video input from from .... [s]
* STOP_AT_TIME = 8.5 # end capturing the video at time.... [s]
* wait_key = 300  # Time in ms between frames
* save_frames = True # flag to decide if to save the output frames or to just show
* IMAGE_DIM_PARAM = 128 # GAN model input shape - in this case the GAN receives 128*128*1 images
* BATCH_SIZE = 20  # Batch size

* Blur_Valid_DS =os.path.join("C:/Users/...","Blur_Valid_DS.npy") # Blurred Validation Data Set
* Sharp_Valid_DS =os.path.join("C:/Users/...","Sharp_Valid_DS.npy") # Sharp Validation Data Set
* hdf_path = os.path.join("C:/Users/...","my_best_model.epoch11-loss47.87.hdf5") # Path to HDF5 file of trained model **weights only**

steps:
1. Run the script and pick any video.
2. select ROI to crop the video to the wanted key region.
3. Apply selection by ENTER/SPACE key and Execute by ESC key.

Output:
* Example Blur / Deblur images for the first frame of the video
* Output Video that contains the deblurring results.

Notes:
* GAN input is NORMALIZED image.
* due to padding = "same" , reconstructed patched image size is bigger than the original and therefore we use CenterCrop at the end of the process
* GAN's generator output is mapped to 0-255 by the function "interval_mapping(image, from_min, from_max, to_min, to_max)"

Showcasing the results:
https://user-images.githubusercontent.com/40007988/176913617-c80387bb-5125-4e44-aeb4-7203d5f71246.MOV


