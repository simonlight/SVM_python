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
    
def print_exp_detail(categories, lambdaCV, epsilonCV, scaleCV, tradeoffCV,
                     initializedType, test_suffix, hnorm, numWords,
                     optim, epochsLatentMax, epochsLatentMin, cpmax, cpmin, splitCV):
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
    print "splitCV \t%s"%str(splitCV)

def get_LSSVM_name(category, scale, lbd, epsilon, tradeoff,
                   initializedType, test_suffix, hnorm, numWords,
                   optim, epochsLatentMax, epochsLatentMin,
                   cpmax, cpmin, split):
    return '_'.join([str(ele) for ele in [category, scale, lbd, epsilon, tradeoff,
                   initializedType, test_suffix, hnorm, numWords,
                   optim, epochsLatentMax, epochsLatentMin,
                   cpmax, cpmin, split, '.lssvm']])

def get_example_file_fp(sourceDir,scale, category, exp_type, test_suffix):
    return os.path.join(sourceDir, "example_files", str(scale),'_'.join([category,exp_type, 'scale',str(scale),'matconvnet_m_2048_layer_20.txt'+test_suffix]))

def pickle_LSSVM(classifier_fp):
    with open(classifier_fp) as lssvm:
        return pickle.load(lssvm)

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
#     sourceDir = "/local/wangxin/Data/ferrari_gaze/";
#     resDir = "/home/wangxin/results/ferrari_gaze/std_et/";
#     gazeType = "ferrari"
    
    # local test laptop
    sourceDir='/home/xin/'
    resDir = "/home/xin/results/ferrari_gaze/std_et/";
    gazeType = "ferrari"
        
    #local or other things
    
    dataSource= "local";
    lossPath = sourceDir+"ETLoss_json/"
#     lossPath = '/home/xin/ETLoss_dict/'
    testResultFileName = "debug_w.txt"
    detailFolder= "debug_w/"

    lambdaCV = [1e-4]
    epsilonCV = [1e-3]
    scaleCV = [int(sys.argv[2])]    
    categories = [sys.argv[1]]
    tradeoffCV = [0]
    
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
    save_classifier = False
    
    print_exp_detail(categories, lambdaCV, epsilonCV, scaleCV, tradeoffCV,\
                     initializedType, test_suffix, hnorm, numWords,\
                     optim, epochsLatentMax, epochsLatentMin, cpmax, cpmin, splitCV)
    
    for scale in scaleCV:
        batch_feature_json_fp =os.path.join(sourceDir, "m_2048_train_batch_feature",str(scale)+'.json')
        with open(batch_feature_json_fp) as batch_feature_json:
            batch_features = json.load(batch_feature_json)
        for category in categories:
            for split in scaleCV:
#                 listTrain = BagReader.readIndividualBagMIL(get_example_file_fp(sourceDir, scale, category, "train",test_suffix), numWords, True, dataSource)
#                 listVal = BagReader.readIndividualBagMIL(get_example_file_fp(sourceDir, scale, category, "val",test_suffix), numWords, True, dataSource)
                train_example_file_fp = get_example_file_fp(sourceDir, scale, category, "train",test_suffix)
                listTrain = BagReader.readBatchBagMIL(train_example_file_fp,batch_features, numWords, True, dataSource)
                val_example_file_fp = get_example_file_fp(sourceDir, scale, category, "val",test_suffix)
                listVal = BagReader.readBatchBagMIL(val_example_file_fp,batch_features, numWords, True, dataSource)
#                 
                for epsilon in epsilonCV:
                    for lbd in lambdaCV:
                        for tradeoff in tradeoffCV:
                            
                            example_train =  STrainingList(listTrain)
                            example_val = STrainingList(listVal)
                            #Initialization
                            
                            ###############
                            lssvm_name = get_LSSVM_name(category, scale, lbd, epsilon, tradeoff,\
                                                        initializedType, test_suffix, hnorm, numWords,\
                                                        optim, epochsLatentMax, epochsLatentMin,\
                                                        cpmax, cpmin, split)
                            classifier_folder = os.path.join(resDir, 'classifier/')
                            myIO.basic.check_folder(classifier_folder)
                            classifier_fp = os.path.join(classifier_folder, lssvm_name)
                            
                            print "***************************************************"
                            if load_classifier and os.path.exists(classifier_fp):
                                print "loading classifier:%s"%classifier_fp
                                lsvm = pickle_LSSVM(classifier_fp)
                            else:
                                print "training classifier:%s"%classifier_fp
                                lsvm = LSSVMMulticlassFastBagMILET()
                                lsvm.setOptim(optim);
                                lsvm.setEpochsLatentMax(epochsLatentMax);
                                lsvm.setEpochsLatentMin(epochsLatentMin);
                                lsvm.setCpmax(cpmax);
                                lsvm.setCpmin(cpmin);
                                lsvm.setLambda(lbd);
                                lsvm.setEpsilon(epsilon);
                                lsvm.setGazeType(gazeType);
                                lsvm.setLossDict(lossPath+"ETLoss+_"+str(scale)+".json");
                                lsvm.setTradeoff(tradeoff);
                                lsvm.setScale(scale)
                                lsvm.setHnorm(hnorm)
                                lsvm.setClassName(category);
                                
                                lsvm.train(example_train)
                                if save_classifier:
                                    print "saving lssvm:%s"%classifier_fp 
                                    with  open(classifier_fp, 'w') as lssvm_path:
                                        pickle.dump(lsvm,lssvm_path)
                            train_ap = getTestResults(lsvm, example_train, "train", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)
                            val_ap = getTestResults(lsvm, example_val, "val", resDir,detailFolder, tradeoff, scale, epsilon, lbd, category, recording = True)
                            print "train ap: %f"%train_ap
                            print "val ap: %f"%val_ap
                            print "***************************************************"
                                    
                                        
if __name__ == "__main__":
    import sys
    #for big
    sys.path.append("/home/wangxin/lib/lib/python2.7/site-packages")
    sys.path.append("/home/wangxin/code/SVM_python")
    
    import os
    import json
    import myIO.basic 
    import pickle
    from myIO import BagReader  
    from myTools.STrainingList import STrainingList 
    from LSSVMMulticlassFastBagMILET import LSSVMMulticlassFastBagMILET
    
    import cProfile
    cProfile.run('main()')