'''
Created on Nov 14, 2015

@author: xin
'''


def getTestResults(lssvm, examples,typ, resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True):
    if recording:
        detection_folder = os.path.join(resDir,detailFolder)
        myIO.basic.check_folder(detection_folder)
        detection_fp = os.path.join(detection_folder,'_'.join(["metric_"+typ, str(tradeoff), str(scale), str(epsilon), str(lbd), category+".txt"])) 
    ap = lssvm.testAPRegion(examples,detection_fp)
    return ap

def writeResultScore(lssvm, examples, typ, resDir,detailFolder, tradeoff, scale, epsilon, lbd, category):
    score_folder = os.path.join(resDir,detailFolder)
    myIO.basic.check_folder(score_folder)
    score_fp = os.path.join(score_folder,'_'.join(["score_"+typ, str(tradeoff), str(scale), str(epsilon), str(lbd), category+".txt"]))    
    lssvm.getTestScore(examples, score_fp)
    print "write scores to file:%s"%score_fp

def print_exp_detail(categories, lambdaCV, epsilonCV, scaleCV, tradeoffCV,
                     initializedType, hnorm, numWords,
                     optim, epochsLatentMax, epochsLatentMin, cpmax, cpmin, splitCV, exp_type):
    print "Experiment detail:"
    print "categories \t%s"%str(categories),
    print "\tlambda \t\t%s"%str(lambdaCV)
    print "epsilon \t%s"%str(epsilonCV),
    print "\tscale \t\t%s"%str(scaleCV)
    print "tradeoff \t%s"%str(tradeoffCV),
    print "\t\tinitializedType\t%s"%initializedType
    print "numWords \t%d"%numWords,
    print "\t\thnorm \t\t%s"%str(hnorm)
    print "epochsLatentMax\t%d"%epochsLatentMax,
    print "\t\toptim \t\t%d"%optim
    print "cpmax \t\t%s"%cpmax,
    print "\t\tepochsLatentMin\t%d"%epochsLatentMin
    print "splitCV \t%s"%str(splitCV),
    print "\t\tcpmin \t\t%s"%cpmin
    print "exp_type \t%s"%str(exp_type),

def writeAP(lssvm, result_file_fp, test_ap, train_ap):
    result_file = open(result_file_fp, 'a+')
    result_file.write(' '.join([lssvm.category, str(lssvm.tradeoff), str(lssvm.scale),
                                str(lssvm.lbd), str(lssvm.epsilon), str(test_ap), str(train_ap)]))
    result_file.write('\n')
    result_file.close()
    
def writeDetectionResult(lssvm, examples, example_typ, detection_folder):
    detection_fp = os.path.join(detection_folder,'_'.join(["metric_"+example_typ, str(lssvm.tradeoff),
                                str(lssvm.scale), str(lssvm.epsilon), str(lssvm.lbd), lssvm.category+".txt"])) 
    ap = lssvm.writeDetectionResult(examples,detection_fp)
    return ap

def writeScore(lssvm, examples, example_typ, score_folder):
    score_fp = os.path.join(score_folder,'_'.join(["score_"+example_typ, str(lssvm.tradeoff), str(lssvm.scale), 
                            str(lssvm.epsilon), str(lssvm.lbd), lssvm.category+".txt"]))    
    lssvm.writeScore(examples, score_fp)
    print "write scores to file:%s"%score_fp

def get_LSSVM_name(category, scale, lbd, epsilon, tradeoff,
                   initializedType, hnorm, numWords,
                   optim, epochsLatentMax, epochsLatentMin,
                   cpmax, cpmin, split, exp_type):
    return '_'.join([str(ele) for ele in [category, scale, lbd, epsilon, tradeoff,
                   initializedType, hnorm, numWords,
                   optim, epochsLatentMax, epochsLatentMin,
                   cpmax, cpmin, split, exp_type,'.lssvm']])
    
def get_thibaut_examplefile_fp(sourceDir,scale, category, example_typ, test_suffix):
    return os.path.join(sourceDir, "example_files", str(scale),'_'.join([category,example_typ, 'scale',str(scale),'matconvnet_m_2048_layer_20.txt'+test_suffix]))

def get_VOC_examplefile_fp(example_root_folder, category, example_typ):
    return os.path.join(example_root_folder, '_'.join([category, example_typ+'.txt']))
    
def pickle_LSSVM(classifier_fp):
    with open(classifier_fp) as lssvm:
        return pickle.load(lssvm)

def getExample(category, scale, example_root_folder, batch_features, example_type):
    example_file_fp = get_VOC_examplefile_fp(example_root_folder, category, example_type)
    listExample = reader.readBatchBagMIL(example_file_fp, batch_features, True, scale)
    example = STrainingList(listExample)
    return example

def generate_examples(category, scale, example_root_folder, train_batch_features, exp_type):
    if exp_type == "validation":
        example_train = getExample(category, scale, example_root_folder, train_batch_features, "train")
        test_batch_features = train_batch_features
        example_test = getExample(category, scale, example_root_folder, test_batch_features, "val")

    elif exp_type == "trainval_valtest":
        example_train = getExample(category, scale, example_root_folder, train_batch_features, "train")
        test_batch_features = train_batch_features
        example_test = getExample(category, scale, example_root_folder, test_batch_features, "val_val")
    
    elif exp_type == "fulltest":
        example_train = getExample(category, scale, example_root_folder, train_batch_features, "tarinval")
        #TODO
#         test_batch_features = reader.combineFeatureJson(test_batch_json_main_folder, False)
        example_test = getExample(category, scale, example_root_folder, test_batch_features, "test")
    
    else:
        raise NotImplementedError
    return example_train, example_test


def evaluation_phase(lssvm, example_train, example_test, result_file_fp):
    #Training ap results
    train_ap = lssvm.getAP(example_train)
    
    #Test results
    test_ap = lssvm.getAP(example_test)
    
    #Result Summary
    writeAP(lssvm, result_file_fp, test_ap, train_ap)
     
    print "train ap: %f"%train_ap
    print "test ap: %f"%test_ap
    print "***************************************************"


def train_phase(resDir, classifier_folder,\
                category, scale, lbd, epsilon, tradeoff,\
                initializedType, hnorm, numWords,\
                optim, epochsLatentMax, epochsLatentMin,\
                cpmax, cpmin, split,exp_type,\
                load_classifier, example_train, gazeType, lossPath, save_classifier):
    
    lssvm_name = get_LSSVM_name(category, scale, lbd, epsilon, tradeoff,\
                                initializedType, hnorm, numWords,\
                                optim, epochsLatentMax, epochsLatentMin,\
                                cpmax, cpmin, split,exp_type)
    classifier_fp = os.path.join(classifier_folder, lssvm_name)

    print "***************************************************"
    
    if load_classifier and os.path.exists(classifier_fp):
        print "loading classifier:%s"%classifier_fp
        lssvm = pickle_LSSVM(classifier_fp)
    else:
        print "training classifier:%s"%classifier_fp
        
        lssvm = LSSVMMulticlassFastBagMILET()
        lssvm.setOptim(optim);
        lssvm.setEpochsLatentMax(epochsLatentMax);
        lssvm.setEpochsLatentMin(epochsLatentMin);
        lssvm.setCpmax(cpmax);
        lssvm.setCpmin(cpmin);
        lssvm.setLambda(lbd);
        lssvm.setEpsilon(epsilon);
        lssvm.setGazeType(gazeType);
        lssvm.setLossDict(lossPath+"ETLoss+_"+str(scale)+".json");
        lssvm.setTradeoff(tradeoff);
        lssvm.setScale(scale)
        lssvm.setHnorm(hnorm)
        lssvm.setCategory(category);
        
        lssvm.train(example_train)
        
        if save_classifier:
            print "saving lssvm:%s"%classifier_fp 
            with  open(classifier_fp, 'w') as lssvm_path:
                pickle.dump(lssvm,lssvm_path)
    
    return lssvm

def main():
    
    # big    stefan
#     sourceDir = "/home/wangxin/Data/gaze_voc_actions_stefan/"
#     simDir = "/home/wangxin/results/ferrari_gaze/std_et/"
#     gazeType = "stefan"
    
    # local stefan
#     sourceDir = "/local/wangxin/Data/gaze_voc_actions_stefan/"
#     simDir = "/local/wangxin/results/stefan_gaze/std_et/"
#     gazeType = "stefan"

#     # big ferrari
    sourceDir = "/home/wangxin/Data/ferrari_gaze/"
    resDir = "/home/wangxin/results/ferrari_gaze/std_et/"
    gazeType = "ferrari";
        
#     local ferrari
#     sourceDir = "/local/wangxin/Data/ferrari_gaze/";
#     resDir = "/local/wangxin/results/ferrari_gaze/std_et/";
#     gazeType = "ferrari"
    #validation, fulltest, trainval_valtest
    exp_type = "trainval_valtest"
    
    # local test laptop
#     sourceDir='/home/xin/'
#     resDir = "/home/xin/results/ferrari_gaze/std_et/";
#     gazeType = "ferrari"
        
    #Content in the source directory 
    lossPath = sourceDir+"ETLoss_json/"
    trainval_batch_json_main_folder = os.path.join(sourceDir, 'm_2048_trainval_batch_feature')
    test_batch_json_main_folder = os.path.join(sourceDir, 'm_2048_test_batch_feature')
    example_root_folder = os.path.join(sourceDir, "voc_example_file_10_categories")
    trainval_single_json_folder = os.path.join(sourceDir, "m_2048_trainval_batch_feature","single_json")
    test_single_json_folder = os.path.join(sourceDir, "m_2048_test_batch_feature","single_json")
    
    exp_description = "cv_fulltrain_halfval"
    result_file_fp = os.path.join(resDir,exp_description, "ap_summary.txt")
    detection_folder= os.path.join(resDir,exp_description,"metric")
    myIO.basic.check_folder(detection_folder)
    score_folder= os.path.join(resDir,exp_description,"score")
    myIO.basic.check_folder(score_folder)
    classifier_folder = os.path.join(resDir,exp_description, "classifier")
    myIO.basic.check_folder(classifier_folder)
    
    #parameters
    lambdaCV = [1e-4]
    epsilonCV = [1e-3]
#     categories = ["dog", "cat", "motorbike"]
#     categories = ["boat","aeroplane","horse"]
#     categories = ["cow","sofa","diningtable","bicycle"]
    categories = [sys.argv[1]]
#     scaleCV = [30]    
    scaleCV = [int(sys.argv[2])]    
#     tradeoffCV = [0.1]
#     tradeoffCV = [float(sys.argv[3])]
    tradeoffCV = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    initializedType = "noInit"
    hnorm = False;
    
    numWords = 2048
    optim = 1;
    epochsLatentMax = 500;
    epochsLatentMin = 2;
    cpmax = 5000;
    cpmin = 2;
    splitCV = [1];
    
    load_classifier = True
    save_classifier = True
    
    print_exp_detail(categories, lambdaCV, epsilonCV, scaleCV, tradeoffCV,\
                     initializedType, hnorm, numWords,\
                     optim, epochsLatentMax, epochsLatentMin, cpmax, cpmin, splitCV, exp_type)
    
    for scale in scaleCV:
        #batch feature folder
#         trainval_batch_feature_mainfolder = os.path.join(trainval_batch_json_main_folder, str(scale))
#         test_batch_feature_mainfolder = os.path.join(test_batch_json_main_folder, str(scale))

        for category in categories:
            for split in scaleCV:
                # save memory
                train_batch_features = json.load(open(os.path.join(trainval_single_json_folder,str(scale)+".json")))
    
                example_train, example_test = generate_examples(category, scale, example_root_folder, train_batch_features,exp_type)
                               
                                        
                for epsilon in epsilonCV:
                    for lbd in lambdaCV:
                        for tradeoff in tradeoffCV:
                            lssvm = train_phase(resDir, classifier_folder,\
                                                category, scale, lbd, epsilon, tradeoff,\
                                                initializedType, hnorm, numWords,\
                                                optim, epochsLatentMax, epochsLatentMin,\
                                                cpmax, cpmin, split,exp_type,\
                                                load_classifier, example_train, gazeType, lossPath, save_classifier)
                            
                            evaluation_phase(lssvm, example_train, example_test, result_file_fp)
                                                       
                                    
                                        
if __name__ == "__main__":
    import sys
    #for big
    sys.path.append("/home/wangxin/lib/lib/python2.7/site-packages")
    sys.path.append("/home/wangxin/code/SVM_python")
    
    import os
    import json
    from myTools import reader 
    import myIO.basic 
    import pickle
    from myTools.STrainingList import STrainingList 
    from LSSVMMulticlassFastBagMILET import LSSVMMulticlassFastBagMILET
    
#     import cProfile
#     cProfile.run('main()')

    main()