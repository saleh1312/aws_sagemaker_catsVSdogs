
## 1- downloading dataset

Link : https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

## 2- dataset sampling

2_1 - after i downloaded it 
2_2 - extract zip file inside data folder
2_3 - run sample.py to sampled 1000 image from cats and dogs to train faster

## 3- edit .env file with correct paths ( if needed )

## 3- run train.py

run train.py and test if it works succfully

## 4- dockerize and test the docker image

docker build -t cats_vs_dogs .


docker run -di --name t_cont -v "E:\github\aws_sagemaker_catsVSdogs\data\input_data:/opt/ml/input/data" -v "E:\github\aws_sagemaker_catsVSdogs\outs:/opt/ml/output" -v "E:\github\aws_sagemaker_catsVSdogs\hyperparams.json:/opt/ml/input/config/hyperparams.json" -e INPUT_DATA_PATH="/opt/ml/input/data" -e OUTPUT_DATA_PATH="/opt/ml/output" -e HYPERPARAMS_PATH="/opt/ml/input/config/hyperparams.json" -e RUN_ID="1" cats_vs_dogs

docker exec -it t_cont bash

## 5- push to ECR repository

5_1 - create ecr repo in aws and push to it

## 6- upload the sampled dataset to s3 bucket

## 7- edit create_sage_maker_training_job.py file and run it