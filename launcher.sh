#!/bin/bash

echo activate conda environment
eval "$(conda shell.bash hook)"
conda activate leanyf

# check if data are already there, if not download from s3 bucket
DIR="/home/ubuntu/leanyf"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "${DIR} exists"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "${DIR} not found. Pulling it from S3 bucket."
  aws s3 cp s3://ml4cast/leanyf $DIR --recursive
fi

echo Execute script

# Option 1: Norm by province, OHE, Yield
python optimise_so_2D_archictectures.py --normalisation True --model 2DCNN_MISO --target yield & process_id=$!
# Option 2: Norm by province, no OHE, Yield
#python optimise_so_2D_archictectures.py --normalisation True --model 2DCNN_SISO --target yield & process_id=$!
# Option 3: Norm by province, OHE, Yield
#python optimise_so_2D_archictectures.py --normalisation False --model 2DCNN_MISO --target yield & process_id=$!
# Option 4: Not norm by province, no OHE, Yield
#python optimise_so_2D_archictectures.py --normalisation False --model 2DCNN_SISO --target yield & process_id=$!
# Option 5: Norm by province, OHE, Area
#python optimise_so_2D_archictectures.py --normalisation True --model 2DCNN_MISO --target area & process_id=$!
# Option 6: Norm by province, no OHE, Area
#python optimise_so_2D_archictectures.py --normalisation True --model 2DCNN_SISO --target area & process_id=$!
# Option 7: Not norm by province, OHE, Area
#python optimise_so_2D_archictectures.py --normalisation False --model 2DCNN_MISO --target area & process_id=$!
# Option 8: Not norm by province, no OHE, Area
#python optimise_so_2D_archictectures.py --normalisation False --model 2DCNN_SISO --target area & process_id=$!

echo "PID: $process_id"
wait $process_id
echo "Exit status: $?"

echo Syncing on S3
aws s3 cp s3://ml4cast/leanyf $DIR --recursive

echo shutting down machine
sudo shutdown -h now