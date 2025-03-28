import boto3
import time


sagemaker_client = boto3.client(
    "sagemaker",
    region_name="us-east-1",
)

# aws params
training_job_name = 'salo7ty-'+time.strftime('%Y-%m-%d-%H-%M-%S')

BUCKET_NAME = None # put your bucket name
input_s3_uri = f's3://{BUCKET_NAME}/dataset/'
output_bucket = f's3://{BUCKET_NAME}/outs/'
train_mounted_folder = "train"

YOUR_AWS_ACCOUNT_ID = # put your account id
ecr_container_url = f'{YOUR_AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/cats_vs_dogs:latest'
sagemaker_role = f'arn:aws:iam::{YOUR_AWS_ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20241025T201675'

instance_type = 'ml.m5.large'
instance_count = 1
memory_volume = 20


# container params
environment_variables = {
    'INPUT_DATA_PATH': f'/opt/ml/input/data/{train_mounted_folder}',
    'OUTPUT_DATA_PATH': "/opt/ml/output/data",
    'HYPERPARAMS_PATH': '/opt/ml/input/config/hyperparameters.json',
    'RUN_ID': '1'
}



_ = sagemaker_client.create_training_job(

    # a custom name for training job
    TrainingJobName=training_job_name,

    # hyperparameters in as python dictionary 
    AlgorithmSpecification={
        # our container url from step 4
        'TrainingImage': ecr_container_url,
        "ContainerEntrypoint": ["python"],
        "ContainerArguments": ["train.py"],
        'TrainingInputMode': 'File'
    },

    HyperParameters={
        "learning_rate": "0.001",
        "batch_size": "32",
        "num_epochs": "100"
    },

    # a IAM role with proper permissions so that sagemaker can run
    RoleArn=sagemaker_role,


    InputDataConfig=[{
        'ChannelName': train_mounted_folder,
        'DataSource': {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": input_s3_uri,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        'CompressionType': 'None'
    }],


    OutputDataConfig={"S3OutputPath": output_bucket},

    ResourceConfig={
        "InstanceType": instance_type,
        "InstanceCount": instance_count,
        "VolumeSizeInGB": memory_volume
    },

    Environment=environment_variables,

    StoppingCondition={'MaxRuntimeInSeconds': 3600}
)
