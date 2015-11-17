'''
Created on Nov 15, 2015

@author: xin
'''
from DataType.LatentRepresentation import LatentRepresentation
from DataType.StructuralTrainingSample import StructuralTrainingSample
def STrainingList(listTrain):
    example_train = []
    for cnt, train_bag in enumerate(listTrain):
        x = LatentRepresentation(train_bag.sample, 0)
        label = train_bag.label
        example_train.append(StructuralTrainingSample(x, label))
#         if cnt == 30:
#             break
    return example_train