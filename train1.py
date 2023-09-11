import os
from copy import deepcopy

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import wandb
import csv

from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import metrics
from losses import Proxy_Anchor
from utils import l2_norm, pred_proxy_knn, pred_classwise_avg, pred_nearest_proxy
from dataset import LABEL_DICT
from torchmetrics.functional import precision, recall

class Trainer:
    def __init__(self, model, optimizer, loss_function, scheduler, config):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.config = config
        pass

    def train_step(self, data, epoch, device):
        train_accuracy = 0.
        train_loss = 0.
        train_f1 = 0.
        count = 0
        target_list = []
        pred_list = []

        with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:

            for step, batch in enumerate(data):

                self.model.train()
                self.model.zero_grad()

                x, y = batch

                count += 1
                x = x.to(device)
                y = y.to(device)

                x_emb = self.model(x)
                loss, proxy = self.loss_function(x_emb, y)

                #print(proxy)
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clips)
                torch.nn.utils.clip_grad_norm_(self.loss_function.parameters(), self.config.clips)

                self.optimizer.step()

                x_emb = l2_norm(x_emb)
                proxy = l2_norm(proxy)

                y_pred = pred_nearest_proxy(x_emb, proxy, device)

                target_list.append(y.tolist())
                pred_list.append(y_pred.tolist())

                train_accuracy += metrics.accuracy(self.config.n_classes, y_pred, y)
                train_f1 += metrics.f1_score(self.config.n_classes, y_pred, y)

                train_loss += loss.detach().cpu()

                if self.scheduler:
                    self.scheduler.step()

                learning_rate = self.scheduler.get_last_lr()[0]

                pbar.update(1)
                pbar.set_postfix_str(
                    f"Train acc: {(train_accuracy / count):1.4f} Train f1: {(train_f1 / count):1.4f} Loss: {loss.item():.4f} ({train_loss / count:.4f})"
                )

            pred_list = sum(pred_list, [])
            target_list = sum(target_list, [])

            cm = confusion_matrix(target_list, pred_list)
            print('\n' + str(cm))

        return train_accuracy / count, train_f1 / count, train_loss / count, learning_rate

    def validation_step(self, train_data, data, epoch, device, flag=False):
        validation_accuracy = 0.
        validation_f1 = 0.
        validation_loss = 0.
        validation_count = 0
        target_list=[]
        pred_list=[]
        validation_re = 0.
        validation_pr = 0.

        self.model.eval()
        self.loss_function.eval()

        with torch.no_grad():
            train_x_emb = []
            train_y_emb = []

            for step, (x, y) in enumerate(train_data):

                x = x.to(device)
                y = y.to(device)

                x_emb = self.model(x)

                train_x_emb.append(x_emb)
                train_y_emb.append(y)

            train_x_emb = torch.cat(train_x_emb, dim=0)
            train_y_emb = torch.cat(train_y_emb, dim=0)

            with tqdm(total=len(data), desc=f"EPOCH - {epoch} ") as pbar:
                if flag == True:
                    x_file = open(self.config.name+'x_embedding.tsv', 'w', newline='\n')
                    wr = csv.writer(x_file, delimiter='\t')
                    y_file = open(self.config.name+'y_true.tsv', 'w', newline='\n')
                    wr_y = csv.writer(y_file, delimiter='\t')
                    wr_y.writerow(['label'])
                for step, (x, y) in enumerate(data):

                    validation_count += 1

                    x = x.to(device)
                    y = y.to(device)

                    x_emb = self.model(x)
                    loss, proxy = self.loss_function(x_emb, y)

                    x_emb = l2_norm(x_emb)
                    proxy = l2_norm(proxy)

                    if self.config.eval_method == 'pred_nearest_proxy':
                        y_pred = pred_nearest_proxy(x_emb, proxy, device)
                    elif self.config.eval_method == 'pred_proxy_1nn':
                        y_pred = pred_proxy_knn(x_emb, train_x_emb, train_y_emb, proxy, 1, self.config.data, device)
                    elif self.config.eval_method == 'pred_proxy_50nn':
                        y_pred = pred_proxy_knn(x_emb, train_x_emb, train_y_emb, proxy, 50, self.config.data, device)
                    elif self.config.eval_method == 'pred_proxy_100nn':
                        y_pred = pred_proxy_knn(x_emb, train_x_emb, train_y_emb, proxy, 100, self.config.data, device)
                    elif self.config.eval_method == 'pred_proxy_200nn':
                        y_pred = pred_proxy_knn(x_emb, train_x_emb, train_y_emb, proxy, 200, self.config.data, device)
                    elif self.config.eval_method == 'pred_classwise_avg':
                        y_pred = pred_classwise_avg(x_emb, train_x_emb, train_y_emb, proxy, self.config.data, device)

                    target_list.append(y.tolist())
                    pred_list.append(y_pred.tolist())

                    validation_loss += loss.detach().cpu()
                    validation_accuracy += metrics.accuracy(self.config.n_classes, y_pred, y)
                    validation_f1 += metrics.f1_score(self.config.n_classes, y_pred, y)
                    validation_re += metrics.recall(self.config.n_classes, y_pred, y)
                    validation_pr += metrics.precision(self.config.n_classes, y_pred, y)


                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"valid acc: {validation_accuracy / validation_count:1.4f} valid f1: {(validation_f1 / validation_count):1.4f} Loss: {loss.item():.3f} ({validation_loss / validation_count:.3f})"
                    )
                    if flag == True:
                        for x in x_emb:
                            wr.writerow(x.tolist())

                        for y_true in y:
                            wr_y.writerow([y_true.item()])


                pred_list = sum(pred_list, [])
                target_list = sum(target_list, [])

                cm = confusion_matrix(target_list, pred_list)
                print('\n' + str(cm))

                report = classification_report(
                    target_list, pred_list,
                    labels=list(range(len(LABEL_DICT[self.config.data]))),
                    target_names=list(LABEL_DICT[self.config.data].keys()),
                    output_dict=True
                )
                f1 = report['macro avg']['f1-score']
                prec = report['macro avg']['precision']
                rec = report['macro avg']['recall']

                log_template = "{}\tF1: {:.5f}\tPREC: {:.5f}\tREC: {:.5f}"
                print(log_template.format("TOTAL", f1, prec, rec))
                for key, value in report.items():
                    if key in LABEL_DICT[self.config.data]:
                        cur_f1 = value['f1-score']
                        cur_prec = value['precision']
                        cur_rec = value['recall']
                        print(log_template.format(key, cur_f1, cur_prec, cur_rec))

                acc_score = accuracy_score(target_list, pred_list)
                f1_ = f1_score(target_list, pred_list, average='weighted')
                pre_score = precision_score(target_list, pred_list, average='weighted')
                rec_score = recall_score(target_list, pred_list, average='weighted')

                print('acc:', acc_score, 'f1:', f1_, 'pre:', pre_score, ', recall: ', rec_score)

            if flag == True:
                x_file.close()
                y_file.close()

        return acc_score, f1_, validation_loss/validation_count, rec_score, pre_score
        #return validation_accuracy/validation_count, validation_f1 / validation_count, validation_loss/validation_count, validation_re/validation_count, validation_pr/validation_count

    def fit(self, train_dataloader, test_dataloader, device):
        best_model = None
        best_score = 0.

        self.model.zero_grad()
        self.optimizer.zero_grad()
        for epoch in range(self.config.n_epochs):

            if self.config.warm > 0:
                if epoch == 0:
                    for param in list(set(self.model.parameters())):
                        param.requires_grad = False
                if epoch == self.config.warm:
                    for param in list(set(self.model.parameters())):
                        param.requires_grad = True

            train_accuracy, train_f1, train_loss, learning_rate = self.train_step(train_dataloader, epoch, device)
            valid_acc, valid_f1, valid_loss, valid_re, valid_pr = self.validation_step(train_dataloader, test_dataloader, epoch, device)

            # save model
            if not os.path.exists(os.path.join(self.config.model_dir, self.config.data)):
                os.makedirs(os.path.join(self.config.model_dir, self.config.data))

            if valid_f1 >= best_score:
                best_model = deepcopy(self.model.state_dict())
                print(f"SAVE! Epoch: {epoch + 1}/{self.config.n_epochs}")
                best_score = valid_f1

                model_name = f"{self.config.name}.pt"
                model_path = os.path.join(self.config.model_dir, self.config.data, model_name)
                torch.save({
                    'model': self.model.state_dict(),
                    'config': self.config
                }, model_path)

            if self.config.wan >= 1:
                wandb.log({
                    "Train loss": train_loss,
                    "Train accuracy": train_accuracy,
                    "Train F1-score": train_f1,
                    "Learning rate": learning_rate,
                    "Validation loss": valid_loss,
                    "Validation accuracy": valid_acc,
                    "Validation F1-score": valid_f1
                })



            print("END")

        self.model.load_state_dict(best_model)
