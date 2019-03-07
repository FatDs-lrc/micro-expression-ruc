import torch
import os
import time, datetime
import numpy as np
from data.single_img_dataset import SingleImgDataLoader
from model.baseline_model import BaselineModel
from opt import opt_parse
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


def eval_on_metric(pred, label, isTrain=True):
    acc = accuracy_score(pred, label)
    recal = recall_score(pred, label, average='macro')
    f1 = f1_score(pred, label, average='macro')
    print("acc: {:.4f}, recal: {:.4f}, f1_score:{:.4f}".format(acc, recal, f1))
    if not isTrain:
        # C_ij => i(label), j(pred)
        print("confusion matrix:\n {}".format(confusion_matrix(pred, label)))
    # print('\n')
    return acc, f1

def flatmap(data):
    return data

def validate(model, dataloader):
    print("Eval on valid set:")
    model.eval()
    total_pred = []
    total_label = []
    for data in dataloader:
        pred = model.run_one_batch(data, False)
        label = data['label']
        total_pred.append(pred.cpu().numpy())
        total_label.append(label.numpy())

    total_pred = np.hstack(total_pred)
    total_label = np.hstack(total_label)
    acc, f1 = eval_on_metric(total_pred, total_label, False)
    return acc, f1


if __name__ == '__main__':
    opt = opt_parse()
    # build PATH
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # init model and dataloader
    model = BaselineModel(opt)

    # for eval
    tst_dataloader = SingleImgDataLoader(opt, False)
    if opt.evaluate:
        validate(model, tst_dataloader)
        exit()

    trn_dataloader = SingleImgDataLoader(opt, True)


    best_acc, best_acc_epoch = 0.0, 0
    best_f1, best_f1_epoch = 0.0, 0

    for epoch in range(opt.start_epoch, opt.epochs):
        model.scheduler.step()
        model.train()
        print("Epoch {} Start".format(epoch))
        total_pred = []
        total_label = []
        total_loss = []
        step = 0
        cur_step = 0

        epoch_start = time.time()
        for data in trn_dataloader:
            # forward one batch
            pred = model.run_one_batch(data)
            total_loss.append(model.get_current_loss())
            cur_step += opt.train_batch
            step += opt.train_batch
            # record preds
            label = data['label']
            total_pred.append(pred.cpu().numpy())
            total_label.append(label.numpy())
            # print current ans
            if cur_step > 200:
                cur_step -= 200
                localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                loss = model.get_current_loss()
                print('{} epoch {} step {} current loss in one batch: {}'.format(localtime, epoch, step, loss))

        print("Avg loss in this epoch:{}".format(sum(total_loss)/len(total_loss)))
        print("Evaluate training on epoch:{}, current learning rate: {} ".format(epoch, model.get_lr()))
        total_pred = np.hstack(total_pred)
        total_label = np.hstack(total_label)
        eval_on_metric(total_pred, total_label)
        if epoch % 1 == 0:
            acc, f1 = validate(model, tst_dataloader)
            if best_acc < acc:
                best_acc, best_acc_epoch = acc, epoch
            if best_f1 < f1:
                best_f1, best_f1_epoch = f1, epoch
        epoch_end = time.time()
        time_span = str(datetime.timedelta(seconds=int(epoch_end-epoch_start)))
        print('Using {} to finish epoch\n\n'.format(time_span, epoch))
        model.save(epoch)

    print('best acc:{}   epoch:{}\n'.format(best_acc, best_acc_epoch))
    print('best f1:{}   epoch:{}\n'.format(best_f1, best_f1_epoch))
    with open('checkpoint/record.txt', 'w') as f:
        f.write('best acc:{}\t\t\tepoch:{}\n'.format(best_acc, best_acc_epoch))
        f.write('best f1:{}\t\t\tepoch:{}\n'.format(best_f1, best_f1_epoch))
