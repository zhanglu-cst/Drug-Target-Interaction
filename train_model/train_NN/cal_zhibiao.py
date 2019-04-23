import json
from matplotlib import pyplot as plt

with open('all_out.json', 'r') as f:
    L = json.load(f)


def get_acc(out_label):
    threshold = 0.5
    # ACC
    acc_cnt = 0
    for item in out_label:
        out, label = item
        if (out > threshold):
            tmp = 1
        else:
            tmp = 0
        if(tmp==label):
            acc_cnt+=1
    ACC = acc_cnt / len(out_label)
    return ACC


def cal_indicators(out_label):
    def cmp(tmp):
        return tmp[0]

    out_label = sorted(out_label, key = cmp, reverse = True)
    ACC = get_acc(out_label)
    print('ACC:{}'.format(ACC))

    threshold = 0.5
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for item in out_label:
        out,label = item
        if(out>threshold):
            yuce = 1
        else:
            yuce = 0
        if(yuce==1 and label ==1):
            TP+=1
        elif(yuce==1 and label==0):
            FP+=1
        elif(yuce==0 and label==1):
            FN+=1
        elif(yuce==0 and label==0):
            TN+=1
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print('precision:{}'.format(precision))
    print('recall:{}'.format(recall))
    TPR = TP/(TP+FN)
    FPR = FP/(TN+FP)
    TNR = TN/(FP+TN)
    print('TPR:{}'.format(TPR))
    print('FPR:{}'.format(FPR))
    print('TNR:{}'.format(TNR))


    #ROC
    ROC_TPRs = []
    ROC_FPRs = []
    yuzhi_list = []
    yuzhi_list.append(1)
    for item in out_label:
        out,label = item
        yuzhi_list.append(out)
    yuzhi_list.append(0)
    for yuzhi in yuzhi_list:
        cur_TP = 0
        cur_FN = 0
        cur_TN = 0
        cur_FP = 0
        for item in out_label:
            out,label = item
            if(out>=yuzhi):
                yuce = 1
            else:
                yuce = 0

            if(yuce==1 and label==1):
                cur_TP+=1
            elif(yuce==1 and label==0):
                cur_FP+=1
            elif(yuce==0 and label==1):
                cur_FN+=1
            elif(yuce==0 and label==0):
                cur_TN+=1
        cur_TPR = cur_TP/(cur_TP+cur_FN)
        cur_FPR = cur_FP/(cur_TN+cur_FP)
        ROC_TPRs.append(cur_TPR)
        ROC_FPRs.append(cur_FPR)
    plt.plot(ROC_FPRs,ROC_TPRs)


    AUC = 0
    for i in range(len(ROC_TPRs)-1):
        tmp = (ROC_FPRs[i+1]-ROC_FPRs[i])*(ROC_TPRs[i]+ROC_TPRs[i+1])
        AUC+=tmp
    AUC = AUC/2
    print('AUC:{}'.format(AUC))

    plt.show()












cal_indicators(L)
