# !/bin/sh

# create train, test and valid folders
mkdir -p train
mkdir -p test
mkdir -p valid

# unzip the folder
unzip -o data.zip

# move the files into their respective folders
mv train.csv train/
mv valid.csv valid/
mv *_TEST_*.csv test/