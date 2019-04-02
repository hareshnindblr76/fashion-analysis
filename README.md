This is the code to scrape images from the given urls, match images,
train CNN to classify these images.
Run the code as 
python extract_data.py to download all the images from the given json file

python train_val_split.py to form train/val splits which maintains the data
distribution

python train.py which by default fine-tunes a Resnet50 based classifier, 
with block4 replaced by a single conv layer.

