#!/bin/bash

echo activate conda environment
eval "$(conda shell.bash hook)"
conda activate tensorflow2_latest_p37

# check if data are already there, if not download from s3 bucket
DIR="/home/ec2-user/leanyf"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "${DIR} exists"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "${DIR} not found. Pulling it from S3 bucket."
  aws s3 cp s3://ml4cast/leanyf $DIR --recursive
fi

echo Execute script

# Option 1: 1DCNN_SISO
nohup python optimise_so_1D_architectures.py --model 1DCNN_SISO & process_id=$!
# Option 2: 1DCNN_MISO
#nohup python optimise_so_1D_architectures.py --model 1DCNN_MISO & process_id=$!

echo "PID: $process_id"
wait $process_id
#echo "Exit status: $?"

echo Syncing on S3
aws s3 cp $DIR s3://ml4cast/leanyf --recursive

echo shutting down machine
sudo shutdown -h now
