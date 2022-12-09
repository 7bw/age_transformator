import os
import os.path
import sys
import cv2
import json
import boto3
import time
import numpy as np

#import logging
#logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

from scripts import inference, load_model
from utils.common import json2im, im2json
from botocore.exceptions import ClientError

s3_role_credential = {}
S3_BUCKET = 'cse291-virtualization'

local_dir = '/tmp/'
walltime0 = time.time()

def download_file(s3_client, bucket, object_name, file_name=None):
    if file_name is None:
        file_name = os.path.basename(object_name)

    print(bucket)
    print(object_name)
    print(file_name)
    # Upload the file
    try:
        s3_client.download_file(bucket, object_name, file_name)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_file(s3_client, bucket, file_name, object_name=None):
    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print(e)
        return False
    return True

def lambda_handler(event, context):
    '''
    Lambda handler that takes input images and ages, performs the prediction and returns
    the resulting images.

    Args
    ----
        event : json
            event['image'] : base64 encoded image of a person
            event['age'] : List of ages to predict. These values must be provided as int between 1 and 100.
            event['only_load_model'] : If true, the model is loaded but no inference is executed.
    Returns
    -------
        json 
            age : image (base64)
    '''

    only_load_model = event['only_load_model'] 

    if only_load_model == True:
        _ = load_model()
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "message": 'Model is loaded'
            })
        }

    else:   
        # Perform the age transformation
        
        img_path = event['image']
        ages = event['age']

        s3_client = boto3.client(
            's3'
        )

        # parse input image path
        path = img_path[5:]
        # path: bucket-name-format/folder1/folder2/myfile.jpg
        uri_parts = path.split('/')
        # uri_parts: ["bucket-name", "folder1", "folder2", ... , "filename"]
        bucket_name = uri_parts[0]
        prefix = '/'.join(uri_parts[1:-1]) + '/'
        file_name = uri_parts[-1]
        key = prefix + file_name

        # download image from s3
        download_file(s3_client, bucket_name, key, '/tmp/' + file_name)

        walltime1 = time.time()

        # convert image to cv2
        cv_image = cv2.imread('/tmp/'+ file_name)
        
        print("cv image:")
        print(cv_image)

        results = inference.predict_age(cv_image, ages)

        walltime2 = time.time()

        # Encode the result
        encoded_result = {}
        out_names = []
        for i in range(len(results)):
            result = results[i]
            cv_image = np.array(result['img'], dtype=np.uint8)
            image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            out_name = f"{file_name}-{ages[i]}.jpg"
            out_names.append(out_name)
            cv2.imwrite('/tmp/' + out_name, image)
            #encoded_result[result['age']] = im2json(image)
        #print(encoded_result)
        object_name = "ageoutput/" + out_name
        
        # upload to s3
        for out_name in out_names:
            upload_file(s3_client, S3_BUCKET, '/tmp/'+out_name, object_name)

        walltime3 = time.time()

        return {
            "P1": str(walltime1 - walltime0),
            "P2": str(walltime2 - walltime1),
            "P3": str(walltime3 - walltime2)
        }