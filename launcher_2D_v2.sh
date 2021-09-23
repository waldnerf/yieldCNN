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
#python preprocess_2D_inputs.py
python preprocess_inputs.py --D 2

echo Run Deep learning part
# echo Delete old nohup
# rm nohup.out

#Michele tests 23 Sep 2021 update (fixed agparser bugs) * use -wandb to activate wandb loggin
# Option 1: Norm by image, OHE, Yield
python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_MISO --target yield >> python.log
# Option 2: Norm by image, no OHE, Yield
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_SISO --target yield & process_id=$!
# Option 3: Norm by image, OHE, Yield, X aumentation
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_MISO --target yield --Xshift --Xnoise & process_id=$!
# Option 4: Norm by image, OHE, Yield, X aumentation, Y augmentation
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_SISO --target yield --Xshift --Xnoise --Ynoise & process_id=$!


# Franz's tests:
# Option 1: Norm by image, OHE, Yield
# nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_MISO --target yield & process_id=$!
# Option 2: Norm by image, no OHE, Yield
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_SISO --target yield & process_id=$!
# Option 3: Not Norm by image, OHE, Yield
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_MISO --target yield & process_id=$!
# Option 4: Not norm by image, no OHE, Yield
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_SISO --target yield & process_id=$!

#MM: stop year for the moment, enough..
#-----------------------------------------------------------------------------------------------------------------

# Option 5: Norm by image, OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_MISO --target area & process_id=$!
# Option 6: Norm by image, no OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation norm --model 2DCNN_SISO --target area & process_id=$!
# Option 7: Not norm by image, OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_MISO  --target area & process_id=$!
# Option 8: Not norm by image, no OHE, Area
#nohup python optimise_so_2D_architectures.py --normalisation raw --model 2DCNN_SISO --target area & process_id=$!

#echo "PID: $process_id"
#wait $process_id
#echo "Exit status: $?"

echo Copy log files
cp $DIRCode/python.log $DIR/
cp $DIRCode/launcher_2D_out.log $DIR/

echo Syncing on S3
aws s3 sync $DIR s3://ml4cast/leanyf
#aws s3 cp $DIR s3://ml4cast/leanyf --recursive

echo Shutting down machine
sudo shutdown -h now
