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

def assume_role():
    global s3_role_credential
    sts_client = boto3.client('sts')
    assumed_role_object = sts_client.assume_role(
        RoleArn="arn:aws:iam::009491271470:role/S3Role",
        RoleSessionName="AssumeRoleSession1"
    )

    s3_role_credential = assumed_role_object['Credentials']

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

# def lambda_handler(event, context):
def main():
    '''
    Main function that takes input images and ages, performs the prediction and returns
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
    walltime0 = time.time()

    only_load_model = os.environ['only_load_model'] 
    in_dir = 'input'
    out_dir = "ageoutput"
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if only_load_model == True:
        #logging.debug('only_load_model flag is true')
        #logging.debug('Start loading model')
        _ = load_model()
        #logging.debug('Model loaded')
        print("Model is loaded")

    else:   
        # Perform the age transformation
        #logging.debug('only_load_model flag is false')
        
        #logging.debug('Casting input parameters')


        #img = json2im(os.environ['image'])
        ages = os.environ['age']
        img_path = os.environ['image']
        # img_path: s3://bucket-name-format/folder1/folder2/myfile.csv.gz

        # initialize s3 client
        assume_role()

        s3_client = boto3.client(
            's3',
            aws_access_key_id=s3_role_credential['AccessKeyId'],
            aws_secret_access_key=s3_role_credential['SecretAccessKey'],
            aws_session_token=s3_role_credential['SessionToken']
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
        print(bucket_name)
        print(key)

        # download image from s3
        download_file(s3_client, bucket_name, key, key)

        walltime1 = time.time()

        # convert image to cv2
        cv_image = cv2.imread(key)
        
        print("cv image:")
        print(cv_image)
        img = json2im(im2json(cv_image))
        print(img)

        # base64 encode image then decode?
        # img = im2json(cv_image)
        
        # perform inference
        #logging.debug('Starting inference')
        results = inference.predict_age(cv_image, ages)

        walltime2 = time.time()

        # save the result images
        #logging.debug('Encode Results')
        encoded_result = {}
        out_names = []
        for i in range(len(results)):
            result = results[i]
            cv_image = np.array(result['img'], dtype=np.uint8)
            image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            out_name = f"{out_dir}/{file_name}-{ages[i]}.jpg"
            out_names.append(out_name)
            cv2.imwrite(out_name, image)
            #encoded_result[result['age']] = im2json(image)
        #print(encoded_result)

        # upload to s3
        for out_name in out_names:
            upload_file(s3_client, S3_BUCKET, out_name, out_name)

        walltime3 = time.time()

        print(f'P1: {walltime1 - walltime0}')
        print(f'P2: {walltime2 - walltime1}')
        print(f'P3: {walltime3 - walltime2}')

if __name__ == '__main__':
    main()