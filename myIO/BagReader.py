'''
Created on Nov 14, 2015

@author: xin
'''
import os
import numpy as np
from DataType import BagMIL
from DataType import TrainingSample
from myTools import reader
from myTools import vector
import collections
import json
def readIndividualFeatureFile(feature_path):
    features = reader.file2list(feature_path)
    return [float(feature) for feature in features]

def readIndividualFeatureExample(example, bias):
    example_info = example.split()
    filename = example_info[0]
    label = int(example_info[1])
    feature_num = int(example_info[2])
    features_path = example_info[3:]
    
    assert feature_num == len(features_path)
    
    features = np.array([readIndividualFeatureFile(feature_path) for feature_path in features_path])
    features = vector.L2norm(features)
    feature_rownum = features.shape[0]
    if bias :
        features = np.concatenate((features, np.ones((feature_rownum,1))), axis=1)
    return TrainingSample.TrainingSample(BagMIL.BagMIL(filename, label, features), label)
    
def readIndividualBagMIL(example_filepath, dim, bias, dataSource):
    example_list = []
    if not os.path.exists(example_filepath):
        print "%s not found"%example_filepath
        raise IOError
    else:
        print ' '.join(["reading bag:",example_filepath,"\t dimension: ",str(dim)])
        with open(example_filepath) as ef:
            example_number = int(ef.readline().strip())
            examples = ef.readlines()
            assert example_number == len(examples)
            for example in examples:
                example_list.append(readIndividualFeatureExample(example.strip(), bias))
    return example_list


def readBatchFeatureExample(example, batch_features, bias):
    example_info = example.split()
    filename = example_info[0]
    label = int(example_info[1])
    
    features = np.array(batch_features[filename].values())
    features = vector.L2norm(features)
    feature_rownum = features.shape[0]
    if bias :
        features = np.concatenate((features, np.ones((feature_rownum,1))), axis=1)
    return TrainingSample.TrainingSample(BagMIL.BagMIL(filename, label, features), label)

def readBatchBagMIL(example_filepath, batch_feature_json_fp, dim, bias, dataSource):
    example_list = []
    if not os.path.exists(example_filepath):
        print "%s not found"%example_filepath
        raise IOError
    else:
        print ' '.join(["reading bag:",example_filepath,"\t dimension: ",str(dim)])
        with open(batch_feature_json_fp) as batch_feature_json:
            batch_features = json.load(batch_feature_json)
        with open(example_filepath) as ef:
            example_number = int(ef.readline().strip())
            examples = ef.readlines()
            assert example_number == len(examples)
            for example in examples:
                example_list.append(readBatchFeatureExample(example.strip(), batch_features, bias))
    return example_list
