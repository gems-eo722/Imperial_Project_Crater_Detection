# Moonshot: Automatic Impact Crater Detection on the Moon

<a href="url"><img src="https://drive.google.com/uc?export=view&id=1dJjw6g_S8s5hMsiZ67Sp9f50NrgZvoTm" align="left" height="300" width="300" ></a>

Impact craters are the most ubiquitous surface feature on rocky
planetary bodies. Crater number density can be used to estimate the
age of the surface: the more densely cratered the terrain, the older
the surface. When independent absolute ages for a surface are
available for calibration of crater counts, as is the case for some
lava flows and regions of the Moon, crater density can be used to
estimate an absolute age of the surface.

Crater detection and counting has traditionally been done by laborious
manual interrogation of images of a planetary surface taken by
orbiting spacecraft
([Robbins and Hynek, 2012](https://doi.org/10.1029/2011JE003966);
[Robbins, 2019](https://doi.org/10.1029/2018JE00559)). However,
the size frequency distribution of impact craters is a steep negative
power-law, implying that there are many small craters for each larger
one. For example, for each 1-km crater on Mars, there are more than
a thousand 100-m craters. With the increased fidelity
of cameras on orbiting spacecraft, the number of craters visible in
images of remote surfaces has become so large that manual counting is
unfeasible. Furthermore, manual counting can be time consuming and
subjective
([Robbins et al., 2014](https://doi.org/10.1016/j.icarus.2014.02.022)).
This motivates the need for automated crater detection and counting algorithms ([DeLatte et al., 
2019](https://doi.org/10.1016/j.asr.2019.07.017)).

Recent work has shown that widely used object detection algorithms
from computer vision, such as the YOLO (You Only Look Once) object
detection algorithm ([Redmon et al., 2016](https://doi.org/10.1109/CVPR.2016.91);
[Jocher et al., 2021](https://doi.org/10.5281/zenodo.4418161)), can be effective for crater detection on Mars
([Benedix et al., 2020](https://doi.org/10.1029/2019EA001005); [Lagain et al., 2021](https://doi.org/10.1029/2020EA001598)) and the Moon ([Fairweather
et al., 2022](https://doi.org/10.1029/2021EA002177)).

## Aim
The aim of this project is to develop a software tool for
automatically detecting impact craters in images of planetary surfaces
and deriving from this a crater-size frequency distribution that can
be used for dating.

### Inputs
Your tool should take as input one or more images of the surface of a
planet as well as optional inputs of the planet, planet radius, location and physical size of the
image. We will test your tool for Mars and the Moon.

### Outputs
Your tool should output a list of all the bounding boxes for craters
detected in each image (see Technical Requirements below for more details).

Your tool should have the following additional options:
* Generate a visualisation of the original image with annotated
bounding boxes.
* If a real-world image size and location is provided for the image,
  the tool should also provide physical locations (lat, lon of centre in degrees)
  and size (km) for each crater.
* If ground truth labels are provided, the tool should determine the
  number of True Positive, False Positive and False Negative
  detections and return these values for each image.
* If ground truth labels are provided, the tool should also plot a
  comparison of the ground truth bounding boxes and the model
  detection bounding boxes.

### Testing
At the end of the week, we will provide you with two test sets:

* About 90 small THEMIS images of Mars, very similar to the training
  set provided.
* Two images of parts of the surface of Moon with unknown locations.

For each test set we will ask you to use your tool to return a list of
all the crater bounding boxes in the image, their location and
size. You will not know the locations of these images, so you will not
need to return physical crater locations and sizes for the test. More
details of the metric used to score your results are provided below.

## Description
This task involves three separate subtasks, and we suggest that you divide your group accordingly.

### Crater Detection Model (CDM)
One team should develop a module for automatically locating craters
in images. You are free to base your CDM on any available object
detection model and to investigate multiple models before deciding on your
preferred option.  We suggest that as a starting point you try one of the
YOLO implementations available [here](https://github.com/ultralytics/). 

To develop, train and test your first CDM you have been provided
with a dataset of images of the surface of Mars, taken by the [THEMIS](https://astrogeology.usgs.gov/maps/mars-themis-controlled-mosaics-and-final-smithed-kernels) camera (100-m/px),
together with labels that provide the bounding boxes of any craters in
the image larger than ~1-2 km in diameter. This is a subset of the
training data set used by
([Benedix et al., 2020](https://doi.org/10.1029/2019EA001005)).

You can download the training dataset [here](https://imperiallondon-my.sharepoint.com/:u:/g/personal/gsc_ic_ac_uk/EU_xOXenx_VFheugz7m8ruMBEx5OSzMqOh78ngy9jqbDgw?e=GKSBeX). Note that depending
on your object detection model, you may need to reformat the crater
label data.

The performance of your CDM for detecting craters in THEMIS images of
Mars will be assessed using a test set similar to the training data
set. You will not have access to this test set until *Friday*. 

### A training dateset for the Moon
Your tool should be able to detect craters on the Moon, as well as
Mars. To achieve good results for both the Moon and Mars you will need
to develop a separate CDM for the Moon. This might use the same object
detection algorithm as your Mars CDM but with different network
weights, a different object detection model or an ensemble of
different models. To train your CDM for the Moon requires a
high-quality training dataset for the Moon. Another part of your group
should therefore prepare a labelled lunar crater dataset with which
you can train your Moon CDM.

To develop this dataset you have been provided with four images of
portions of the lunar surface and a csv file containing the location
and size of all manually counted craters on this part of the Moon. The
resources to generate the Moon training data can be downloaded from [here](https://imperiallondon-my.sharepoint.com/:u:/g/personal/gsc_ic_ac_uk/EfId5SSWgkRFlb67n1qbUsEB3zFoybvxQhCem3LLmITllg?e=gGciyy).

The images provided are from a global mosaic of LROC WAC images of the
Moon (100 m/px). You should use ONLY these images to generate your
training dataset. Use of other images of the Moon for training or
testing will result in a substantial penalty mark. However, you are
free to use any augmentation techniques or GANs to enhance your
training data, if you wish.

The four images provided are for the regions:

* A: -180 to -90 longitude, -45 to 0 latitude;
* B: -180 to -90 longitude, 0 to 45
latitude; 
* C: -90 to 0 longitude, -45 to 0 latitude; 
* D: -90 to 0 longitude, 0 to 45 latitude.

The test images will be taken from somewhere in the region
0 to 180 longitude; -45 to 45 latitude. 

The crater database that you can use to generate your training labels
is a subset of the manually derived lunar impact crater
database [Robbins, 2019](https://doi.org/10.1029/2018JE00559). You
should not download the full database for use in your training or
testing. Note that the Robbins crater database is complete for craters larger than
1-2 km; many smaller craters present in the images will be
unlabelled. 

### A tool for analysis of craters
Your final CDMs for the Moon and Mars should be implemented within a
single end-to-end data processing tool that can be used to analyse
craters on either planetary surface.  The inputs and outputs of the
tool are listed above. Further details are provided in the list of
technical requirements below.

To assess your tool you will be given images of Mars and the Moon for which
you must detect and locate all craters. You will be scored on how
accurately your tool locates all the manually identified craters in each image, as
well as the number of true positives and false negatives. It
will also be assessed subjectively on how well it performs at detecting small craters
that have not been manually counted.

The Mars test will only involve small images of the same size as the
training data set. For the Moon, we will provide two test images that
cover a large portion of the Moon. One challenge you must overcome,
therefore, is to develop an approach for detecting craters over a
large size range, from the biggest crater in the image to the smallest
resolvable crater given the image resolution. This could be achieved,
for example, by tiling the image into smaller portions, passing the
tiles into your CDM, and then aggregating the detections over all the
tiles, ensuring that no crater is missed or double counted.

The purpose of this tool is to allow a user to quickly and
automatically identify all craters in the image and from this generate
a size-frequency distribution of the craters for the purpose of dating
the planetary surface. The tool should therefore provide the
functionality to calculate physical, real-world crater sizes and
locations if the image location, size and resolution is provided.

In developing your CDM tool you should assess the tool's accuracy for
crater detection as a function of crater size. You should present your
self-assessed accuracy in the video presentation. Extra credit will be
given to groups that provide an accurate assessment of their model
performance. Groups will lose marks for over-promising and
under-delivering! 

If you have time to spare during the week, extra credit will be given
for tools that provide a User-friendly interface and implement useful
additional features, such as being able to manually edit bounding
boxes or select a portion of the image within which to count
craters.

## Model Performance metric
A challenge that you will face when assessing model performance is the
presence of unlabelled small craters in your training and test
data. These may confuse your model in training and result in a large
number of apparently False Positive detections when testing (your
model correctly thinks a small crater is a crater, but as it isn't
labelled it is counted as a False Positive).

When using crater counts for age dating, the desired result is a
crater size-frequency distribution that is as close to the real one as
possible over as large a diameter range as possible.

Thus, your aim should be to achieve optimal performance for a range
of different crater (bounding box) sizes. To demonstrate this qualitatively, you
should plot the size-frequency distribution of your
detected craters (or bounding boxes) and compare with the ground truth distribution.

To formally score your tool, we will generate a performance metric for
the craters in your file of detected craters by comparing it to
our ground truth data set in the following way:

* We will calculate the
[Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index)
index (IoU) for every crater bounding box in your model detection set
against every crater in our ground truth crater bounding box list
* We will then pair each bounding box $g_i$ in the ground truth list with a
detected crater, $c_i$ in your list, with the pairings chosen to
maximise the sum $$\sum_i \textrm{IoU}(g_i, c_i).$$
* We will calculate a crater recall index using the formula 
$$R=\frac{\textrm{number of crater pairs with IoU>0.5 and area of }g_i>A_R}{\textrm{number of ground truth bounding boxes with area of }g_i>A_R},$$
where $A_R$ is the fractional area of the image that corresponds to a crater size $D_R$.
* We calculate a crater precision index using the formula
$$P=\frac{\textrm{number of crater pairs with IoU>0.5 and area of }c_i>A_P}{\textrm{number of detected bounding boxes with area of }c_i>A_P},$$
where $A_P$ is the fractional area of the image that corresponds to a crater size
$D_P$.
* Finally we will calculate the crater $F1$ score via the usual formula
$$F1 =\frac{2}{\frac{1}{P}+\frac{1}{R}}. $$

For the Mars test set, we will calculate a single $R$, $P$ and
$F1$-score, using $D_R \approx 2$ km. For the Moon test, we will calculate
three $R$, $P$ and $F1$-scores, using $D_R \approx$  1, 10 and 100 km, respectively,
to probe the performance of your model for three different crater
sizes. In all cases, we will use $D_P = 1.2D_R$ to allow for some
uncertainty in your bounding box sizes when calculating precision.

To score highly on these measures, your model needs to do a good job at
detecting craters of different sizes and not suggest that craters exist where we do not expect them. 

## Technical requirements
The deadline for submission of the software tool is *12:00 pm (noon),
Friday 3rd February, 2023*.

### Input images
You can assume that the images of the planet surface will use simple
cylindrical projection and a spherical planet.

Your tool should:
* Accept a User-specified input folder location. The input folder
should contain a subdirectory `images/` that contains a single image
or multiple images, which should be treated independently.  The input
folder location should also contain an optional subdirectory
  `labels/` containing a `.csv` file associated with each image file
  that provides a list of all the ground truth bounding boxes for
  craters in the image.
* Accept images in any sensible format (e.g., `.jpg`,
`.tif`, `.png`) and any size (width and height in pixels).
* Allow the User to specify the location of the image centre in
latitude and longitude; the image width and image height in degrees.
* Allow the User to specify the image resolution in metres per pixel
(m/px).
* Allow the User to specify the radius of the target planet.

An example of the format of the input directory structure, image files
and label files is provided in the Mars THEMIS training data set.

### Output images, bounding boxes, etc.
Your tool should create an output directory with a User-specified
name. The output directory should contain three subdirectories. A
subdirectory `detections/` should contain a `.csv` file for each input
image that contains a list of all the bounding boxes for craters in
the image as detected by your tool. A subdirectory `images/` should
contain a `.png` file for each input image that shows the bounding
boxes of the craters detected by the CDM in one colour and (if ground
truth labels are provided) the ground truth bounding boxes in a
different color.  A subdirectory `statistics/` that contains a `.csv`
file for each input image that summarises the True Positive, False
Positive and False Negative detections in the image (if ground truth
labels exist).

The format of all bounding boxes files (both input and output) should be:
x, y, w, h, where x, y are the horizontal and vertical
locations, respectively, of the centre of the bounding box; w is the width of the bounding box and h
is the height of the bounding box. The units of x and w are fractional
image width; the units of y and h are fractional image height.

If the User provides information that allows the crater size, latitude
and longitude to be determined, this data should also be
provided in the output csv file for each detected crater. The units of
crater size should be km; the units of latitude and longitude of the
crater centre should be in degrees. 

### Visualisation
Your tool should allow the User to visualise the following:
* The original input image without annotations
* The original input image with bounding boxes for craters detected by
the CDM
* The original input image with bounding boxes for both the results of
the CDM and for the ground truth bounding boxes, if available
* A separate plot of the cumulative crater size-frequency distribution
  of detected craters, if information to calculate crater size is
  provided
* If ground truth data is available, performance statistics including
  the number of True Positive, False Negative and False Positive detections.

### User Interface
Your interface should be designed to make it as easy as possible for users to utilize your code. You will not be assessed on its beauty, but on the functionality it exposes and its accessibility or "user friendliness" to an intermediate user. Here are some examples of possible interface schemes (remember, you don't need to generate more than one. if you do, then you should share code via Python modules, rather than re-writing it many times. You will be marked only on the one which has the most functionality):
 - A command line interface (CLI), for example a Python script file. When run, the script should as a minimum process an image or images and output a .csv file containing detected crater information in the specified format. When run without arguments, a CLI tool should explain how to use itself, in a manner suitable for giving to a new MSc student who has just completed the ESE Deep Learning module. Possible optional arguments to your script might include:
    - image file name(s) or directory path(s) to take as input or output to.
    - a label indicating if data is for Mars or the Moon.
    - (optional) longitude/latitude labels for the target image.
    - additional parameters useful to your pretrained model (image sizes, IoU thresholds, etc.)
 -  A Jupyter notebook with code cells which when run generate a pandas DataFrame and (optionally) visualizations of your detected craters. The notebook should contain Markdown cells explaining the workflow as apprpriate, and should import and call your packaged code rather than repeating code blocks from your modelling notebooks or python files.
 - A local GUI, exposing similar options to the CLI interface described above. This is not required to work on all operating systems, but it is desirable to do so. Options should be explained in a help page, via tool tips or in some other accessible manner.
- A web interface, for example using the Flask or Django python frameworks, exposing similar options to the CLI interface described above. Options should be explained in a help page, via tool tips or in some other accessible manner.

### Resources
Each team will be provided with a **Colab Pro + license**. A member of staff will contact you on Monday and help you set up your account. Colab Pro + provides:

- 500 compute units per month: Compute units expire after 90 days. Purchase more as you need them.
- Faster GPUs: Priority access to upgrade to more powerful premium GPUs.
- More memory: Access our higher memory machines.
- Background execution: Upgrade your notebooks to keep executing for up to 24 hours even if you close your browser.
- Terminal: Ability to use a terminal with the connected VM.

Use your account wisely, resource management is critical. We suggest you nominate one person on each team to be the resource manager, responsible for managing and administrating the compute units during the week. Although it's not essential, it is a good idea to reserve a few compute units (10 to 20) to run your models on the test sets released on Friday morning.

### Video Presentation
In addition to the software, your group must submit a short (15 minute) video
presentation. The deadline for submission of the presentation is *5 pm
on Friday 3rd February, 2023*.

The presentation must include the following:

* Describe and justify your choice of Crater Detection Models for Mars
  and the Moon. In this section you should explain which object detection algorithm(s) your CDMs
  employ and why you chose them. If you made any modifications to the
  underlying algorithm or network architecture, describe it here.
* Describe and justify your training data set of lunar impact
  craters. Explain how you subdivided the images of the Moon provided;
  how you generated the ground truth labels; and how you selected
  training and validation images. If you applied any image
  augmentation techniques as part of pre-processing, describe them
  here. Provide a justification for all the choices you made.
* Describe and justify how you trained your crater detection
  model. Explain how you trained your CDM on both the Mars and Moon
  training images. Include a discussion of how you tuned the
  hyperparameters of your model and any image augmentation techniques
  that were used inside the CDM.
* If appropriate, describe how your tool uses the CDM to detect craters on large
  images that cannot be input into the CDM in one go. Explain how the
  image is subdivided into tiles and how detections from each tile are
  aggregated together. 
* Demonstrate the performance of your crater detection model using
  your own ground truth data. Using the training dataset that you
  generated for the Moon, critically assess the performance of your
  model for detecting craters of different sizes. Your presentation
  should state what size range of craters your model can detect (smallest and largest) and
  the expected accuracy for LROC WAC images of the Moon (100 m/px).
* Demonstrate how a User would interact with your model to detect
  craters in an image and, if possible, to derive a crater
  size-frequency distribution from the detections. Demonstrate any
  additional features of your tool that you have had time to implement.
* Show the results of your tool for two images of the Moon provided on Friday.


