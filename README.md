# Team6

Instalation:

1. clone/download the repo<br/>
git clone https://github.com/MCV-2022-M1-Project/Team6.git or just download it locally

2. cd Team6

3. create env<br/>
conda create --name myenv (change myenv with any name you want for your enviorment)

4. install dependencies<br/>
conda install -n <env_name> requirements.txt

5. run the code

* Either you run the main.py script - which will cover all the methods (and will take a long time)
    `python main.py`
* Or you can run the week1.py script once for one method: 
    `python week1.py -c rgb -s e -q False -f 1`
      - for RGB as a colorspace, Euclidean distance for comparison, only run on QS1 and save the output in the method1 folder
  
```
-c or --image_colorspace
-- Sets the value for all query image colorspaces.
-- Values: gray, rgb, ycrcb, hsv
-- Default: ycrcb

-g or --grayscale_method
-- Sets the value of the grayscale method used when converting image to grayscale
-- Values: a = average, w = weighted
-- Default: w

-s or --similarity_method
-- Sets the value of the similarity method used when comparing 2 histograms
-- Values: e = euclidean distance, l = L1 distance, x = X square distance
-- Default: x

-1 or --input_folder_q1
-- Sets the value of the input folder for the first image query set
-- Values: path to the input folder - should be relative to current folder, add / at the end
-- Default: ../Data/qsd1_w1/


-2 or --input_folder_q2
-- Sets the value of the input folder for the second image query set
-- Values: path to the input folder - should be relative to current folder, add / at the end
-- Default: ../Data/qsd1_w2/


-d or --input_folder_bd
-- Sets the value of the input folder for the image data base set
-- Values: path to the input folder - should be relative to current folder, add / at the end
-- Default: ../Data/qsd1_w2/


-o or --output
-- Sets the value of the number of results to be returned
-- Values: integers (smaller than image sets)
-- Default: 10


-m or --mask_colorspace
-- Sets the colorspace for the query image before calculating the mask
-- Values: gray, rgb, ycrcb, hsv
-- Default: hsv


-q or --run_qs2
-- If the system should run on the second query set (takes a few minutes)
-- Values: boolean
-- Default: false


-f or --folder_method
-- Number to be appended to the method folder (in the output)
-- Values: integer
-- Default: 1
```
