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
                     initializedType, test_suffix, hnorm, numWords,
                     optim, epochsLatentMax, epochsLatentMin, cpmax, cpmin, splitCV, exp_type):
    print "Experiment detail:"
    print "categories \t%s"%str(categories),
    print "\tlambda \t\t%s"%str(lambdaCV)
    print "epsilon \t%s"%str(epsilonCV),
    print "\tscale \t\t%s"%str(scaleCV)
    print "tradeoff \t%s"%str(tradeoffCV),
    print "\t\tinitializedType\t%s"%initializedType
    print "test_suffix \t%s"%test_suffix,
    print "\t\thnorm \t\t%s"%str(hnorm)
    print "numWords \t%d"%numWords,
    print "\t\toptim \t\t%d"%optim
    print "epochsLatentMax\t%d"%epochsLatentMax,
    print "\t\tepochsLatentMin\t%d"%epochsLatentMin
    print "cpmax \t\t%s"%cpmax,
    print "\t\tcpmin \t\t%s"%cpmin
    print "splitCV \t%s"%str(splitCV),
    print "\t exp_type \t%s"%str(exp_type)

def get_LSSVM_name(category, scale, lbd, epsilon, tradeoff,
                   initializedType, test_suffix, hnorm, numWords,
                   optim, epochsLatentMax, epochsLatentMin,
                   cpmax, cpmin, split, exp_type):
    return '_'.join([str(ele) for ele in [category, scale, lbd, epsilon, tradeoff,
                   initializedType, test_suffix, hnorm, numWords,
                   optim, epochsLatentMax, epochsLatentMin,
                   cpmax, cpmin, split, exp_type,'.lssvm']])
    
def get_thibaut_examplefile_fp(sourceDir,scale, category, example_typ, test_suffix):
    return os.path.join(sourceDir, "example_files", str(scale),'_'.join([category,example_typ, 'scale',str(scale),'matconvnet_m_2048_layer_20.txt'+test_suffix]))

def get_VOC_examplefile_fp(example_root_folder, category, example_typ, test_suffix):
    return os.path.join(example_root_folder, '_'.join([category, example_typ+'.txt'+test_suffix]))
    
def pickle_LSSVM(classifier_fp):
    with open(classifier_fp) as lssvm:
        return pickle.load(lssvm)



def evaluation_phase(lssvm,example_train, exp_type,\
                     resDir,detailFolder, tradeoff, scale, epsilon, lbd, category):
    #Training results
    train_ap = getTestResults(lssvm, example_train, "train", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)
    
    if exp_type == "validation":
        test_batch_features = train_batch_features
    elif exp_type == "fulltest":
        print 
    elif exp_type == "trainval_valtest":
        test_batch_features = train_batch_features
    #Prepare Test examples
                
    test_example_file_fp = get_VOC_examplefile_fp(example_root_folder, category, "val",test_suffix)
    listTest = reader.readBatchBagMIL(test_example_file_fp,test_batch_features, numWords, True, dataSource, scale)
    example_test = STrainingList(listTest)
    
    #Test results
    test_ap = getTestResults(lssvm, example_test, "val", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)

    #Recording score for each example
    writeResultScore(lssvm, example_test, "val", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category)
    
    #Result Summary
    result_file = open(os.path.join(resDir, resultFileName),'w+')
    result_file.write(' '.join([category, str(tradeoff), str(scale), str(lbd), str(epsilon), str(test_ap), str(train_ap)]))
    result_file.close()
     
    print "train ap: %f"%train_ap
    print "test ap: %f"%test_ap
    print "***************************************************"


def train_phase(resDir, classifier_folder_name,\
                category, scale, lbd, epsilon, tradeoff,\
                initializedType, test_suffix, hnorm, numWords,\
                optim, epochsLatentMax, epochsLatentMin,\
                cpmax, cpmin, split,exp_type,\
                load_classifier, listTrain, gazeType, lossPath, save_classifier):
    
    lssvm_name = get_LSSVM_name(category, scale, lbd, epsilon, tradeoff,\
                                                        initializedType, test_suffix, hnorm, numWords,\
                                                        optim, epochsLatentMax, epochsLatentMin,\
                                                        cpmax, cpmin, split,exp_type)
                            
    classifier_folder = os.path.join(resDir, classifier_folder_name)
    myIO.basic.check_folder(classifier_folder)
    classifier_fp = os.path.join(classifier_folder, lssvm_name)
    
    print "***************************************************"

    if load_classifier and os.path.exists(classifier_fp):
        print "loading classifier:%s"%classifier_fp
        lssvm = pickle_LSSVM(classifier_fp)
    else:
        example_train =  STrainingList(listTrain)
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
        lssvm.setClassName(category);
        
        lssvm.train(example_train)
        
        if save_classifier:
            print "saving lssvm:%s"%classifier_fp 
            with  open(classifier_fp, 'w') as lssvm_path:
                pickle.dump(lssvm,lssvm_path)
    
    return lssvm, example_train
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
#     sourceDir = "/home/wangxin/Data/ferrari_gaze/"
#     resDir = "/home/wangxin/results/ferrari_gaze/std_et/"
#     gazeType = "ferrari";
        
#     local ferrari
    sourceDir = "/local/wangxin/Data/ferrari_gaze/";
    resDir = "/local/wangxin/results/ferrari_gaze/std_et/";
    gazeType = "ferrari"
    
    # local test laptop
#     sourceDir='/home/xin/'
#     resDir = "/home/xin/results/ferrari_gaze/std_et/";
#     gazeType = "ferrari"
        
    #local or other things
    
    dataSource= "local";
    lossPath = sourceDir+"ETLoss_json/"
    trainval_batch_json_main_folder = os.path.join(sourceDir, 'm_2048_trainval_batch_feature')
    test_batch_json_main_folder = os.path.join(sourceDir, 'm_2048_test_batch_feature')
    example_root_folder = os.path.join(sourceDir, "voc_example_file_10_categories")
    
    #validation, fulltest, trainval_valtest
    exp_type = "trainval_valtest"
#     lossPath = '/home/xin/ETLoss_dict/'
#     resultFileName = "bestgamma_on_test.txt"
#     detailFolder= "bestgamma_on_test"
#     classifier_folder_name = "bestgamma_on_test"
    resultFileName = "train_valval_valtest.txt"
    detailFolder= "train_valval_valtest_metric"
    classifier_folder_name = "train_valval_valtest_classifier"
    
    lambdaCV = [1e-4]
    epsilonCV = [1e-3]
    categories = [sys.argv[1]]
    scaleCV = [int(sys.argv[2])]    
#     tradeoffCV = [float(sys.argv[3])]
    tradeoffCV = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    initializedType = "noInit"
    test_suffix="";
    hnorm = False;
    #add sys params
    
    numWords = 2048
    optim = 1;
    epochsLatentMax = 500;
    epochsLatentMin = 2;
    cpmax = 5000;
    cpmin = 2;
    splitCV = [1];
    
    load_classifier = False
    save_classifier = True
    
    print_exp_detail(categories, lambdaCV, epsilonCV, scaleCV, tradeoffCV,\
                     initializedType, test_suffix, hnorm, numWords,\
                     optim, epochsLatentMax, epochsLatentMin, cpmax, cpmin, splitCV, exp_type)
    
    for scale in scaleCV:
        
        trainval_batch_feature_mainfolder = os.path.join(trainval_batch_json_main_folder, str(scale))
        train_batch_features = json.load(open("/home/wangxin/Data/ferrari_gaze/m_2048_trainval_batch_feature/single_json/"+str(scale)+".json"))

#         test_batch_feature_mainfolder = os.path.join(test_batch_json_main_folder, str(scale))
        if exp_type == "fulltest":
            pass
#             train_batch_features = reader.combineFeatureJson(trainval_batch_feature_mainfolder)
#             test_batch_features = reader.combineFeatureJson(test_batch_feature_mainfolder)
        elif exp_type == "validation":
            pass
#             train_batch_features = json.load(open("/local/wangxin/Data/ferrari_gaze/m_2048_trainval_batch_feature/single_json/90.json"))
#             test_batch_features = train_batch_features
#             train_batch_features = reader.combineFeatureJson(trainval_batch_feature_mainfolder)
#             test_batch_features = train_batch_features            
#             train_batch_features = reader.combineFeatureJson(trainval_batch_json_folder)
#             test_batch_features = train_batch_features
        elif exp_type == "trainval_valtest": 
            pass
        else:
            raise NotImplementedError    
            
        for category in categories:
            for split in scaleCV:
#                 listTrain = BagReader.readIndividualBagMIL(get_example_file_fp(sourceDir, scale, category, "train",test_suffix), numWords, True, dataSource)
#                 listVal = BagReader.readIndividualBagMIL(get_example_file_fp(sourceDir, scale, category, "val",test_suffix), numWords, True, dataSource)
                if exp_type == "fulltest":
                    train_example_file_fp = get_VOC_examplefile_fp(example_root_folder, category, "trainval",test_suffix)
#                     test_example_file_fp = get_VOC_examplefile_fp(example_root_folder, category, "test",test_suffix)

                elif exp_type == "validation" or exp_type == "trainval_valtest":
                    train_example_file_fp = get_VOC_examplefile_fp(example_root_folder, category, "train",test_suffix)
                
                listTrain = reader.readBatchBagMIL(train_example_file_fp,train_batch_features, numWords, True, dataSource, scale)
                
                for epsilon in epsilonCV:
                    for lbd in lambdaCV:
                        for tradeoff in tradeoffCV:
                            
                            
                            lssvm, example_train = train_phase(resDir, classifier_folder_name,\
                                                category, scale, lbd, epsilon, tradeoff,\
                                                initializedType, test_suffix, hnorm, numWords,\
                                                optim, epochsLatentMax, epochsLatentMin,\
                                                cpmax, cpmin, split,exp_type,\
                                                load_classifier, listTrain, gazeType, lossPath, save_classifier)
                           
                            
                            if exp_type == "validation":
                                
                                #Training results
                                train_ap = getTestResults(lssvm, example_train, "train", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)
                                
                                #Prepare Test examples
                                test_batch_features = train_batch_features            
                                test_example_file_fp = get_VOC_examplefile_fp(example_root_folder, category, "val",test_suffix)
                                listTest = reader.readBatchBagMIL(test_example_file_fp,test_batch_features, numWords, True, dataSource, scale)
                                example_test = STrainingList(listTest)
                                
                                #Test results
                                test_ap = getTestResults(lssvm, example_test, "val", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)

                                #Recording score for each example
                                writeResultScore(lssvm, example_test, "val", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category)
                                
                                #Result Summary
                                result_file = open(os.path.join(resDir, resultFileName),'w+')
                                result_file.write(' '.join([category, str(tradeoff), str(scale), str(lbd), str(epsilon), str(test_ap), str(train_ap)]))
                                result_file.close()
                                 
                                print "train ap: %f"%train_ap
                                print "test ap: %f"%test_ap
                                print "***************************************************"
                            elif exp_type == "fulltest":
                                #Training results
                                train_ap = getTestResults(lssvm, example_train, "train", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)
                                
                                
                                test_batch_features = reader.combineFeatureJson(test_batch_feature_mainfolder, False)
                                test_example_file_fp = get_VOC_examplefile_fp(example_root_folder, category, "val_val",test_suffix)
                                listTest = reader.readBatchBagMIL(test_example_file_fp,test_batch_features, numWords, True, dataSource, scale)
                                example_test = STrainingList(listTest)

                                train_ap = getTestResults(lsvm, example_train, "trainval", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)
                                result_file = open(os.path.join(resDir, resultFileName),'a+')
                                result_file.write(' '.join([category, str(tradeoff), str(scale), str(lbd), str(epsilon), str(train_ap)]))
                                result_file.close()
#                                 test_ap = writeResultScore(lsvm, example_test, "test", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)
                            
                            elif exp_type == "trainval_valtest":
                                
                           
                                    
                                        
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