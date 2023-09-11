import os
from copy import deepcopy

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import wandb
import csv
from sklearn.metrics import classification_report, confusion_matrix
import metrics
from losses import Proxy_Anchor
from utils import l2_norm

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

                y_pred = self.model(x)
                loss = self.loss_function(y_pred, y)

                #print(proxy)
                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clips)
                self.optimizer.step()

                pred_list += y_pred.max(dim=1)[1].tolist()
                target_list += y.tolist()

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

            cm = confusion_matrix(target_list, pred_list)
            print('\n' + str(cm))

        return train_accuracy / count, train_f1 / count, train_loss / count, learning_rate



    def validation_step(self, data, epoch, device, flag=False):
        validation_accuracy = 0.
        validation_f1 = 0.
        validation_loss = 0.
        validation_count = 0
        target_list = []
        pred_list = []
        validation_re = 0.
        validation_pr = 0.

        self.model.eval()

        with torch.no_grad():
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

                    y_pred = self.model(x)
                    loss = self.loss_function(y_pred, y)

                    pred_list += y_pred.max(dim=1)[1].tolist()
                    target_list += y.tolist()

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
                        x_emb = self.model.model(**x).pooler_output
                        for x in x_emb:
                            wr.writerow(x.tolist())

                        for y_true in y:
                            wr_y.writerow([y_true.item()])

                cm = confusion_matrix(target_list, pred_list)
                print('\n' + str(cm))

            if flag == True:
                x_file.close()
                y_file.close()

        return validation_accuracy/validation_count, \
               validation_f1 / validation_count, \
               validation_loss/validation_count, \
               validation_re/validation_count, \
               validation_pr/validation_count


    def fit(self, train_dataloader, test_dataloader, device):
        best_model = None
        best_score = 0.

        self.model.zero_grad()
        self.optimizer.zero_grad()
        for epoch in range(self.config.n_epochs):
            train_accuracy, train_f1, train_loss, learning_rate = self.train_step(train_dataloader, epoch, device)
            valid_acc, valid_f1, valid_loss, valid_re, valid_pr = self.validation_step(test_dataloader, epoch, device)

            if not os.path.exists(os.path.join(self.config.model_dir,self.config.data)):
                os.makedirs(os.path.join(self.config.model_dir,self.config.data))

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
