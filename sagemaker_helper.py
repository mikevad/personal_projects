# Libraries
import boto3
import re
import sagemaker
from sagemaker import get_execution_role

# SageMaker SDK Documentation: http://sagemaker.readthedocs.io/en/latest/estimators.html

bucket_name = 'blah'
training_file_key = 'iris/iris_train.csv'
validation_file_key = 'iris/iris_validation.csv'

s3_model_output_location = r's3://{0}/iris/model'.format(bucket_name)
s3_training_file_location = r's3://{0}/{1}'.format(bucket_name,training_file_key)
s3_validation_file_location = r's3://{0}/{1}'.format(bucket_name,validation_file_key)

# Write and Reading from S3 is just as easy
# files are referred as objects in S3.  
# file name is referred as key name in S3
# Files stored in S3 are automatically replicated across 3 different availability zones 
# in the region where the bucket was created.

# http://boto3.readthedocs.io/en/latest/guide/s3.html
def write_to_s3(filename, bucket, key):
    with open(filename,'rb') as f: # Read in binary mode
        return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(f)

# S3 write example
write_to_s3('iris_train.csv',bucket_name,training_file_key)
write_to_s3('iris_validation.csv',bucket_name,validation_file_key)

# listing contents of s3 buckets
s3 = boto3.resource('s3')

for bucket in s3.buckets.all():
    bucket.name

conn = boto3.client('s3')
for key in conn.list_objects(Bucket = 'djk-ml-sagemaker')['Contents']:
    key['Key']

# s3 read example
role = get_execution_role()
bucket='djk-ml-sagemaker'
data_key = 'music_lyrics/cleaned_lemmatized_unstopped_df.csv'
data_location = 's3://{}/{}'.format(bucket, data_key)

final_df = pd.read_csv(data_location)

# ---------------------------------------

# sagemaker building model example

role = get_execution_role()
sess = sagemaker.Session()

container = sagemaker.amazon.amazon_estimator.get_image_uri(
    sess.boto_region_name,
    "xgboost", 
    "latest")

estimator = sagemaker.estimator.Estimator(containers[boto3.Session().region_name], # or container
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.m4.xlarge',
                                       output_path=s3_model_output_location,
                                       sagemaker_session=sess,
                                       base_job_name ='xgboost-iris-v1')

estimator.set_hyperparameters(max_depth=5,
                              objective="multi:softmax",
                              num_class=3,
                              num_round=50,
                              early_stopping_rounds=10)

training_input_config = sagemaker.session.s3_input(s3_data=s3_training_file_location,content_type="csv")
validation_input_config = sagemaker.session.s3_input(s3_data=s3_validation_file_location,content_type="csv")

estimator.fit({'train':training_input_config, 'validation':validation_input_config})

# deploy model

predictor = estimator.deploy(initial_instance_count=1,
                             instance_type='ml.m4.xlarge',
                             endpoint_name = 'xgboost-iris-v1')

# --------------------------------------

# three methods:
## train model with sagemaker resources
## train in notebook, build model endpoint with sagemaker resources
## preprocess w/ .py file, train 

# ----------------------------------

# resources

### https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/mxnet_mnist/mxnet_mnist.ipynb
### https://docs.aws.amazon.com/sagemaker/latest/dg/sklearn.html
### entry_point example: https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/scikit_learn_iris/scikit_learn_iris.py

# predictions from sagemaker endoint

# Acquire a realtime endpoint
endpoint_name = 'xgboost-bikerental-v1'
predictor = sagemaker.predictor.RealTimePredictor(endpoint=endpoint_name)

from sagemaker.predictor import csv_serializer, json_deserializer

predictor.content_type = 'text/csv'
predictor.serializer = csv_serializer
predictor.deserializer = None

predictor.predict()

# ----------------------------------------------

# IAM

S3FullAccess
AmazonMachineLearningRealTimePredictionOnlyAccess
AdministratorAccess

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": "sagemaker:InvokeEndpoint",
            "Resource": "*"
        }
    ]
}