##
i followed the official asw sage maker docs to build this project :
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html


## 1- downloading dataset

Link : https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset

## 2- dataset sampling

2_1 - after i downloaded it 
2_2 - extract zip file inside data folder
2_3 - run sample.py to sampled 1000 image from cats and dogs to train faster

## 3- edit .env file with correct paths ( if your data is not in the same dir )

## 3- run train.py

run train.py and test if it works succfully

## 4- dockerize and test the docker image

docker build -t cats_vs_dogs .


docker run -di --name t_cont -v "E:\github\aws_sagemaker_catsVSdogs\data\input_data:/opt/ml/input/data" -v "E:\github\aws_sagemaker_catsVSdogs\outs:/opt/ml/output" -v "E:\github\aws_sagemaker_catsVSdogs\hyperparams.json:/opt/ml/input/config/hyperparams.json" -e INPUT_DATA_PATH="/opt/ml/input/data" -e OUTPUT_DATA_PATH="/opt/ml/output" -e HYPERPARAMS_PATH="/opt/ml/input/config/hyperparams.json" -e RUN_ID="1" cats_vs_dogs

docker exec -it t_cont bash

## 5- push to ECR repository

5_1 - create ecr repo in aws and push to it
5_2 - because the images is > 10G and my internet is so slow , i created ec2 instance in aws 
and cloned the repo in the ec2 and configure it with iam-role and finally pushed the image to the ecr repo

MORE DETAILS HERE :
https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-docker.html
https://www.youtube.com/watch?v=li3dAYo761c&list=LL&index=1

## 6- upload the sampled dataset to s3 bucket

i created a bucket and this folder structer inside it 

BUCKET
- dataset ( to upload the sampled images inside this folder )
    - Cat
    - Dog

- outs ( to catch the sagemaker training job artifacts )

## 7- edit create_sage_maker_training_job.py file and run it