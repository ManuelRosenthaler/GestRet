# Hand Gesture Retrieval in Art Image Collections
## General informations/Disclaimer
This github directory contains the work of Manuel Rosenthaler. He did this work in the context of his master project.
He used an already preexisting software named CDCL that you can find here:
https://github.com/kevinlin311tw/CDCL-human-part-segmentation. The whole work is built on the already existing CDCL.
The programming work done by Manuel Rosenthaler is limited to the two files "human_seg_gt" and "inference_15parts".
The image database, which is located in the input, was collected by Manuel Rosenthaler and is not located in the original CDCL.
***All other files inside the folder CDCL belong to the authors of the CDCL***.

## Requirements
**Docker** is the only requirement to run the installation file. Go to https://www.docker.com/ for more informations. 

## Installation process
To install the gesture retrieval program you have to do the following steps:
1. Clone this Git directory with: `git clone https://github.com/ManuelRosenthaler/GestRet.git`
2. Change to the directory "GestRet": `cd GestRet`
3. Use the installation bash file to get everything running: `./install.sh`
4. You can run the Code with the command: `python3 inference_15parts.py --scale=1`
5. The program will ask you at several instances for User input.
    1. First it will ask you, if you would like to do a similarity search. This means, that you can provide the Program with an example image
     that contains a gesture you would like to search the dataset for. If you answer the first question with 'yes', you will need to give the
     path to your input file as input. A working example path to an image is e.g. /CDCL/input/01christ.jpg.
     2. First, the program wants to know how many similar images to return. For example, if you choose 10, it will return 10
     similar images - the example image that the user provides and for which the similarity search is made is not among them.
     However, if you specify an image as Similarity picture, which is already in the database, it will be returned as one of
     the 10 similar images.

        Secondly, the program wants to know how many clusters to create. The program uses k-means clustering to create a
        number of groups, each containing similar pictures. As an example, you could also give 10. Then you would get 10
        different groups, which all contain similar images.
