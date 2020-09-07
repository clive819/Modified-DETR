Modified **DE⫶TR**: End-to-End Object Detection with Transformers
========

The pytorch re-implement of the official [DETR](https://github.com/facebookresearch/detr), original paper link: <https://arxiv.org/pdf/2005.12872.pdf>

# Quickstart
```
python3 train.py --dataDir "path_to_your_training_data" --numClass "number_of_classes" --numQuery "number_of_queries"
```


## Modification

* DenseNet backbone
* add support for YOLO dataset
* modifiable number of classes
* add support for negative sample (no object) training


## Data preparation

We expect the directory structure to be the following:
```
path/to/data/
	xxx.jpg
	xxx.txt
	123.jpg
	123.txt
```
As in each jpg has a corresponding txt file in the format of 
```
classIndex CenterX CenterY Width Height
```
for each line.



