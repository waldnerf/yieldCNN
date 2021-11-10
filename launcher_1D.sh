#!/bin/bash

echo activate conda environment
eval "$(conda shell.bash hook)"
conda activate tensorflow2_latest_p37

# check if data are already there, if not download from s3 bucket
DIR="/home/ec2-user/leanyf"
DIRCode="/home/ec2-user/yieldCNN"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "${DIR} exists"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "${DIR} not found. Pulling it from S3 bucket."
  aws s3 cp s3://ml4cast/leanyf $DIR --recursive
fi


echo Preprocess data
python preprocess_inputs.py --D 1

echo Run Deep learning part
# echo Delete old nohup
# rm nohup.out

# Option 1: 1DCNN_SISO

# Option 2: 1DCNN_MISO
#python optimise_so_1D_architectures_v3.py --Xshift --Xnoise --Ynoise >> python.log
python optimise_so_1D_architectures_v3.py --Xshift --Xnoise --Ynoise 2>>&1 | tee python.log

echo Copy log files
cp $DIRCode/python.log $DIR/
cp $DIRCode/launcher_2D_out.log $DIR/

echo Syncing on S3
aws s3 sync $DIR s3://ml4cast/leanyf
#aws s3 cp $DIR s3://ml4cast/leanyf --recursive

echo Shutting down machine
sudo shutdown -h now

#
#echo "PID: $process_id"
#wait $process_id
##echo "Exit status: $?"
#
#echo Syncing on S3
#aws s3 cp $DIR s3://ml4cast/leanyf --recursive
#
#echo shutting down machine
#sudo shutdown -h now
