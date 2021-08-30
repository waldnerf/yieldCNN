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


echo Preprocess data
python preprocess_2D_inputs.py

echo Run Deep learning part
echo Delete old nohup
rm nohup.out

# Option 1: Norm by image, OHE, Yield
nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_MISO --target yield & process_id=$!
# Option 2: Norm by image, no OHE, Yield
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_SISO --target yield & process_id=$!
# Option 3: Not Norm by image, OHE, Yield
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_MISO --target yield & process_id=$!
# Option 4: Not norm by image, no OHE, Yield
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_SISO --target yield & process_id=$!
# Option 5: Norm by image, OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_MISO --target area & process_id=$!
# Option 6: Norm by image, no OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_SISO --target area & process_id=$!
# Option 7: Not norm by image, OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_MISO  --target area & process_id=$!
# Option 8: Not norm by image, no OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_SISO --target area & process_id=$!

echo "PID: $process_id"
wait $process_id
#echo "Exit status: $?"

echo Syncing on S3
aws s3 cp $DIR s3://ml4cast/leanyf --recursive

echo Shutting down machine
sudo shutdown -h now
