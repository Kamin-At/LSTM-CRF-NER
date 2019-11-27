def my_eval(target: '(list[list[int]])', pred: '(list[list[int]])', \
    tags: '(dict[int:str])'):

# True Positive(TP): entities that are recognized by NER and \
#     match ground truth
# False Positive(FP): entities that are recognized by NER but \
#     do not match ground truth
# False Negative(FN): entities annotated in the ground truth \
#     that are not recognized by NER
    if len(target) != len(pred):
        print('error 1')
        raise('WTF len(target) != len(pred)')
    cnt_TP = 0
    cnt_FP = 0
    cnt_FN = 0
    for i in range(len(target)):
        if len(target[i]) != len(pred[i]):
            print('error 2')
            raise('len(target[i]) != len(pred[i])')
        for j in range(len(target[i])):
            if target[i] == 1:#1 means B-tag
                
    return Recall, Precision, F1