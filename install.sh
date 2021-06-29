#!/bin/bash
#==================================
# Installation of Gesture Retrieval
#----------------------------------
echo ""
echo "Cloning CDCL..."
git clone https://github.com/kevinlin311tw/CDCL-human-part-segmentation.git
#
#
echo ""
echo "Moving modified Python files to CDCL..."
cp inference_15parts.py CDCL-human-part-segmentation/
cp pascal_voc_human_seg_gt.py CDCL-human-part-segmentation/human_seg/
#
#
echo "Creating folders for process chain..."
mkdir CDCL-human-part-segmentation/similarity_search_results
mkdir CDCL-human-part-segmentation/features
#
#
echo ""
echo "Building Docker image..."
sudo docker build -t cdcl:v1 .
#
#
echo "Running Docker..."
sudo docker run --runtime=nvidia -v '/tank/Manuel/CDCL-human-part-segmentation':'/CDCL'   -it cdcl:v1 bash
