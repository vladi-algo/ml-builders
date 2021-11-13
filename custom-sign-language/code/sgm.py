
from sagemaker.tensorflow import TensorFlow
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from sagemaker import get_execution_role, Session
import boto3

sagemaker_session = sagemaker.Session()
role = get_execution_role()
print('SageMaker version: ' + sagemaker.__version__)
print('role ' + role)
region = sagemaker_session.boto_session.region_name
print("Region:" + region)

training_data_uri = "s3://ml-builders-training-data/train/"
model_dir = "s3://ml-builders-model/"


estimator2 = TensorFlow(
    entry_point="sm_trainCNN_refactor.py",
    role=role,
    instance_count=1,
    instance_type= "ml.p3.2xlarge",#"ml.p2.xlarge", # "ml.p3.xlarge", #"ml.c4.2xlarge",
    framework_version="2.2.0",
    py_version="py37",
    distribution={"parameter_server": {"enabled": True}},
    output_path=model_dir,
    dependencies=["./requirements.txt"])

estimator2.fit({'train':'s3://ml-builders-training-data/train/',
              'test':'s3://ml-builders-training-data/test/'})