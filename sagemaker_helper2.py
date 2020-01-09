# Reference:
# https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/
# https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-output-format


import boto3
import csv
import math
import dateutil
import json
import os
from time import time
from io import StringIO


# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
client = boto3.client(service_name='sagemaker-runtime')


# create csv
def create_csv(feature_list, inference_list):
    firstline = 'feature1, feature2, feature3, feature4, inference'
    inference_list = inference_list.split(',')
    feature_list = [string + f',{i}' for string, i in zip(feature_list, inference_list)]
    feature_list.insert(0, firstline)
    return '\n'.join(feature_list)


# writes to s3 file
def write_to_s3(df:str, bucket):
    unix_now = int(time())
    encoded_df = df.encode('utf-8')
    lambda_path = f'/tmp/{unix_now}_predictions'
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).put_object(Key = f'x/{unix_now}_predictions.csv', Body = encoded_df)


# transforms features into sagemaker desired format
def transform_data(data):
    try:
        features = [data[0], data[1], data[2], (data[3])]
        return ','.join(str(s) for s in features)
        
    except Exception as err:
        print('Error when transforming: {0},{1}'.format(data,err))
        raise Exception('Error when transforming: {0},{1}'.format(data,err))


# calls the sagemaker endpoints
def lambda_handler(event, context):
    
    # for single inferences
    if len(event['instances']) == 1:
        try:
            features = transform_data(event['instances'][0]['features'])
            features_list = [features]
            response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                              Body=features.encode('utf-8'),
                              ContentType='text/csv')
            
            inference = response['Body'].read().decode('utf-8')
            inference_list = inference
            
            # write to s3
            save_data = create_csv(features_list, inference_list)
            print(f'writing inferences to s3://gemini-ml/x/{time()}_predictions.csv')
            write_to_s3(save_data, 'gemini-ml')
                              
            return {
                'statusCode':200,
                'body':inference
            }
            
        except Exception as e:
            return {
                'statusCode':400,
                'body':f'Call Failed - {e}'
            }
            
    # for multiple inferences
    if len(event['instances']) > 1:
        try:
            features_list = [transform_data(f['features']) for f in event['instances']]
            # print(features_list)
            
            response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                       Body=('\n'.join(features_list).encode('utf-8')),
                       ContentType='text/csv')
            
            inference_list = response['Body'].read().decode('utf-8')
            save_data = create_csv(features_list, inference_list)
            
            print(f'writing inferences to s3://gemini-ml/x/{time()}_predictions.csv')
            write_to_s3(save_data, 'gemini-ml')
            
            return {
                'statusCode':200,
                'body':inference_list
            }
            
        except Exception as e:
            return {
                'statusCode':400,
                'body':f'Call Failed - {e}'
            }