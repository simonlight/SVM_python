'''
Created on Nov 14, 2015

@author: xin
'''
import json
import os
import numpy as np
from DataType import BagMIL
from DataType import TrainingSample
from myTools import vector
import collections
from myTools import converter



def file2list(filepath):
    with open(filepath) as fp:
        return [line.strip() for line in fp.readlines()]

def file2FloatList(filepath):
    with open(filepath) as fp:
        return [float(line.strip()) for line in fp.readlines()]
    
def readFeatureJson(batch_feature_json_fp):
    with open(batch_feature_json_fp) as batch_feature_json:
        return json.load(batch_feature_json)    

def readIndividualFeatureFile(feature_path):
    features = file2list(feature_path)
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

def readBatchFeatureExampleijson(example_labels, batch_features, bias, scale):
    for el in example_labels:
        example_info = el.split()
        filename = example_info[0]
        label = int(example_info[1])
    

def readBatchFeatureExample(example, batch_features, bias, scale):
    example_info = example.split()
    filename = example_info[0]
    label = int(example_info[1])
    
    
    #normalization
    #sorted by key
    
    bag_features = batch_features[filename]    
    
    features = np.array([bag_features[str(k)] for k in xrange(converter.scale2RowNumber(scale)**2)])
    features = vector.L2norm(features)
    feature_rownum = features.shape[0]
    if bias :
        features = np.concatenate((features, np.ones((feature_rownum,1))), axis=1)

    return TrainingSample.TrainingSample(BagMIL.BagMIL(filename, label, features), label)

def readBatchBagMILijson(example_filepath, batch_features, bias,  scale):
    print example_filepath
    if not os.path.exists(example_filepath):
        print "%s not found"%example_filepath
        raise IOError
    else:
        print ' '.join(["reading bag:",example_filepath])
        with open(example_filepath) as ef:
            example_labels = [example.strip() for example in ef]
        example_list = readBatchFeatureExampleijson(example_labels, batch_features, bias, scale)
    return example_list

def readBatchBagMIL(example_filepath, batch_features, bias,  scale):
    if not os.path.exists(example_filepath):
        print "%s not found"%example_filepath
        raise IOError
    else:
        example_list=[]
        print ' '.join(["reading bag:",example_filepath])
        with open(example_filepath) as ef:
            for example in ef:
                example_list.append(readBatchFeatureExample(example.strip(), batch_features, bias, scale))
    return example_list

def combineFeatureJsonIntoOneFile(batch_feature_mainfolders,scales):
    """combine seperate jsons together, use only once!!!!"""
    for batch_feature_mainfolder in batch_feature_mainfolders:
        for scale in scales:
            print batch_feature_mainfolder,scale
            final_json = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
            batch_feature_folder = os.path.join(batch_feature_mainfolder,str(scale))
            for cnt,feature_fp in enumerate(os.listdir(batch_feature_folder)):
                print cnt
                feature_json = readFeatureJson(os.path.join(batch_feature_folder, feature_fp))
                for k, v in feature_json.items():
                    for k2, v2 in v.items():
                        final_json[k][k2] = v2
            
            json.dump(final_json,open(os.path.join(batch_feature_folder,'all.json'),'w'))

def combineFeatureJson(batch_feature_folder):
    """combine seperate jsons together, use only once!!!!"""
    final_json = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
    for cnt,feature_fp in enumerate(os.listdir(batch_feature_folder)):
        feature_json = readFeatureJson(os.path.join(batch_feature_folder, feature_fp))
        for k, v in feature_json.items():
            for k2, v2 in v.items():
                final_json[k][k2] = v2
    
    return final_json

if __name__ == '__main__':
#     import sys
#     sys.path.append("")
    #combineFeatureJsonIntoOneFile(["/local/wangxin/Data/ferrari_gaze/m_2048_test_batch_feature/", "/local/wangxin/Data/ferrari_gaze/m_2048_trainval_batch_feature/"], [90])

    
    pass