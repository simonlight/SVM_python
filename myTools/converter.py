'''
Created on Nov 15, 2015

@author: xin
'''
import os
import json
from collections import defaultdict
import myIO.basic
import reader
def scale2RowNumber(scale):
    return 1+(100-scale)/10

def getJsonFeature(scale, filename, feature_root_folder):
    json_feature = defaultdict(lambda:None)
    for d1 in range(scale2RowNumber(scale)):
        for d2 in range(scale2RowNumber(scale)):
            if scale != 100:
                with open(os.path.join(feature_root_folder, '_'.join([filename, str(d1), str(d2)+'.txt']))) as feature_file:
                    feature = reader.file2FloatList(feature_file)
                    json_feature[d1*scale2RowNumber(scale)+d2] = feature
            else:
                with open(os.path.join(feature_root_folder, filename)) as feature_file:
                    feature = reader.file2FloatList(feature_file)
                    json_feature[d1*scale2RowNumber(scale)+d2] = feature
    return json_feature

def jsonGazeCategoryIndependent(txt_feature_folder, json_feature_folder, scales):
    
    myIO.basic.check_folder(json_feature_folder)
        
    for scale in scales: 
        feature_root_folder = os.path.join(txt_feature_folder, str(scale2RowNumber(scale)**2))
        filenames = ['_'.join([filename.split('_')[0],filename.split('_')[1]]) for filename in os.listdir(feature_root_folder)]
        total_file_num = len(filenames)
        #scale_feature['30']['2011_4428']['4'] = 0.2
        scale_feature=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:None)))
        for cnt, filename in enumerate(filenames):
            if cnt % 1000 == 0:
                print "scale: %d, %d / %d finished"%(scale, cnt, total_file_num)
            scale_feature[scale][filename]=getJsonFeature(scale, filename, feature_root_folder)
        with open(os.path.join(json_feature_folder, 'ETLoss+_'+str(scale)+'.json'),'w') as json_feature_file:
            json.dump(scale_feature, json_feature_file)

            
def jsonGazeCategoryDependent(txt_feature_folder, json_feature_folder, scales, categories):
    
    myIO.basic.check_folder(json_feature_folder)
    for scale in scales: 
        #scale_feature['30']['2011_4428']['cat']['4'] = 0.2
        scale_feature=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:None))))
        for category in categories:
            feature_root_folder = os.path.join(txt_feature_folder, category, str(scale2RowNumber(scale)**2))
            filenames = ['_'.join([filename.split('_')[0],filename.split('_')[1],filename.split('_')[2]]) for filename in os.listdir(feature_root_folder)]
            for cnt, filename in enumerate(filenames):
                scale_feature[scale]['_'.join([filename.split('_')[1],filename.split('_')[2]])][category]=getJsonFeature(scale, filename, feature_root_folder) 
        with open(os.path.join(json_feature_folder, 'ETLoss+_'+str(scale)+'.json'),'w') as all_feature_file:
            json.dump(scale_feature, all_feature_file)
        
# jsonGazeCategoryDependent('./ETLoss_ratio','./ETLoss_json', [50], ["dog", "cat", "motorbike", "boat", "aeroplane", "horse", "cow", "sofa" ,"diningtable" ,"bicycle"])
def jsonFileFeatures(txt_feature_folder, json_feature_folder, scales, batch_size):
    for scale in scales:
        myIO.basic.check_folder(os.path.join(json_feature_folder, str(scale)))
    
    for scale in scales: 
        scale_feature_json = defaultdict(lambda: defaultdict(lambda: None))
        txt_feature_folder_of_scale = os.path.join(txt_feature_folder,str(scale))
        feature_fps = os.listdir(txt_feature_folder_of_scale)
        batch_feature_num = batch_size * scale2RowNumber(scale)**2
        for cnt, feature_fp in enumerate(feature_fps):
            if cnt != 0 and cnt % (batch_feature_num)== 0:
                with open(os.path.join(json_feature_folder, str(scale), str(cnt / (batch_feature_num)-1) + '.json'),'w') as batch_feature_file:
                    json.dump(scale_feature_json, batch_feature_file)
                    scale_feature_json = defaultdict(lambda: defaultdict(lambda: None))
                    print "%d / %d finished and saved"%(cnt, len(feature_fps))
            if scale != 100:
                year, imid, d1, d2 = feature_fp.split('.')[0].split('_')
                filename = '_'.join([year,imid])
                index = int(d1)*scale2RowNumber(scale)+int(d2)
                scale_feature_json[filename][index] = reader.file2FloatList(os.path.join(txt_feature_folder_of_scale, feature_fp))
            else:
                filename= feature_fp.split('.')[0]
                index = 0
                scale_feature_json[filename][index] = reader.file2FloatList(os.path.join(txt_feature_folder_of_scale, feature_fp))
        
if __name__ == "__main__":
    import sys
    jsonFileFeatures("/local/wangxin/Data/full_stefan_gaze/m2048_trainval_features", "/local/wangxin/Data/full_stefan_gaze/m2048_trainval_batch_feature", [int(sys.argv[1])], 100)
    