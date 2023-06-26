import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from tqdm import tqdm
from utils import FocalLoss


class Trainer:
    def __init__(
        self,
        option,
        model,
        train_dataset,
        valid_dataset,
        test_dataset=None,
        weight=[[1.0, 1.0]],
        tasks_num=17,
    ):
        # Most important variable
        self.option = option
        self.device = torch.device(
            "cuda:{}".format(option["gpu"][0]) if torch.cuda.is_available() else "cpu"  # noqa: E501
        )
        self.model = (
            DataParallel(model).to(self.device)
            if option["parallel"]
            else model.to(self.device)
        )

        # Setting the train valid and test data loader
        if self.option["parallel"]:
            self.train_dataloader = DataListLoader(
                train_dataset, batch_size=self.option["batch_size"],
                shuffle=True
            )
            self.valid_dataloader = DataListLoader(
                valid_dataset, batch_size=self.option["batch_size"]
            )
            if test_dataset:
                self.test_dataloader = DataListLoader(
                    test_dataset, batch_size=self.option["batch_size"]
                )
        else:
            self.train_dataloader = DataLoader(
                train_dataset, batch_size=self.option["batch_size"],
                shuffle=True
            )
            self.valid_dataloader = DataLoader(
                valid_dataset, batch_size=self.option["batch_size"]
            )
            if test_dataset:
                self.test_dataloader = DataLoader(
                    test_dataset, batch_size=self.option["batch_size"]
                )
        self.save_path = self.option["exp_path"]
        # Setting the Adam optimizer with hyper-param
        if option["focalloss"]:
            self.log("Using FocalLoss")
            self.criterion = [FocalLoss(alpha=1 / w[0]) for w in weight]  # noqa: E501
        else:
            self.criterion = [
                torch.nn.CrossEntropyLoss(
                    torch.Tensor(w).to(self.device), reduction="mean"
                )
                for w in weight
            ]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.option["lr"],
            weight_decay=option["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.7,
            patience=self.option["lr_scheduler_patience"],
            min_lr=1e-6,
        )

        # other
        self.start = time.time()
        self.tasks_num = tasks_num

        self.records = {
            "trn_record": [],
            "val_record": [],
            "val_losses": [],
            "best_ckpt": None,
            "val_roc": [],
            "val_prc": [],
        }
        self.log(
            msgs=["\t{}:{}\n".format(k, v) for k, v in self.option.items()],
            show=False
        )
        self.log(
            "train set num:{}    valid set num:{}    test set num: {}".format(
                len(train_dataset), len(valid_dataset), len(test_dataset)
            )
        )
        self.log(
            "total parameters:"
            + str(sum([p.nelement() for p in self.model.parameters()]))
        )
        self.log(msgs=str(model).split("\n"), show=False)

    def train_iterations(self):
        self.model.train()
        losses = []
        y_pred_list = {}
        y_label_list = {}
        for data in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()

            data = data.to(self.device)
            output = self.model(data)
            loss = 0

            for i in range(self.tasks_num):
                y_pred = output[:, i * 2: (i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                validId = np.where(
                    (y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1)
                )[0]

                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(self.device)]
                y_label = y_label[torch.tensor(validId).to(self.device)]

                loss += self.criterion[i](y_pred, y_label)
                y_pred = F.softmax(y_pred.detach().cpu(),
                                   dim=-1)[:, 1].view(-1).numpy()
                try:
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)
                except Exception:
                    y_label_list[i] = []
                    y_pred_list[i] = []
                    y_label_list[i].extend(y_label.cpu().numpy())
                    y_pred_list[i].extend(y_pred)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        trn_roc = [
            metrics.roc_auc_score(y_label_list[i], y_pred_list[i])
            for i in range(self.tasks_num)
        ]
        trn_prc = [
            metrics.auc(
                precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                precision_recall_curve(y_label_list[i], y_pred_list[i])[0],
            )
            for i in range(self.tasks_num)
        ]
        trn_loss = np.array(losses).mean()

        return trn_loss, np.array(trn_roc).mean(), np.array(trn_prc).mean()

    def valid_iterations(self, mode="valid"):
        self.model.eval()
        if mode == "test" or mode == "eval":
            dataloader = self.test_dataloader
        if mode == "valid":
            dataloader = self.valid_dataloader
        losses = []
        y_pred_list = {}
        y_label_list = {}
        with torch.no_grad():
            for data in tqdm(dataloader):
                data = data.to(self.device)
                output = self.model(data)
                loss = 0
                for i in range(self.tasks_num):
                    y_pred = output[:, i * 2: (i + 1) * 2]
                    y_label = data.y[:, i].squeeze()
                    validId = np.where(
                        (y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1)  # noqa: E501
                    )[0]
                    if len(validId) == 0:
                        continue
                    if y_label.dim() == 0:
                        y_label = y_label.unsqueeze(0)
                    # print(len(validId))
                    # try:
                    #     print(len(y_label))
                    # except:
                    #     print(y_label)
                    #     print(data.y[:,i])
                    #     print(data.y)

                    y_pred = y_pred[torch.tensor(validId).to(self.device)]
                    y_label = y_label[torch.tensor(validId).to(self.device)]

                    loss = self.criterion[i](y_pred, y_label)

                    y_pred = (
                        F.softmax(y_pred.detach().cpu(),
                                  dim=-1)[:, 1].view(-1).numpy()
                    )

                    try:
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    except Exception:
                        y_label_list[i] = []
                        y_pred_list[i] = []
                        y_label_list[i].extend(y_label.cpu().numpy())
                        y_pred_list[i].extend(y_pred)
                    losses.append(loss.item())

        val_roc = [
            metrics.roc_auc_score(y_label_list[i], y_pred_list[i])
            for i in range(self.tasks_num)
        ]
        val_prc = [
            metrics.auc(
                precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                precision_recall_curve(y_label_list[i], y_pred_list[i])[0],
            )
            for i in range(self.tasks_num)
        ]
        val_loss = np.array(losses).mean()
        if mode == "eval":
            self.log(
                "SEED {} DATASET {}  The best test_loss:{:.3f} test_roc:{:.3f} test_prc:{:.3f}.".  # noqa: E501
                format(
                    self.option["seed"],
                    self.option["dataset"],
                    val_loss,
                    np.array(val_roc).mean(),
                    np.array(val_prc).mean(),
                )
            )

        return val_loss, np.array(val_roc).mean(), np.array(val_prc).mean()

    def train(self):
        self.log("Training start...")
        early_stop_cnt = 0
        for epoch in range(self.option["epochs"]):
            trn_loss, trn_roc, trn_prc = self.train_iterations()
            val_loss, val_roc, val_prc = self.valid_iterations()
            test_loss, test_roc, test_prc = self.valid_iterations(mode="test")

            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]["lr"]

            self.log(
                "Epoch:{} {} trn_loss:{:.3f} trn_roc:{:.3f} trn_prc:{:.3f} lr_cur:{:.5f}".  # noqa: E501
                format(
                    epoch, self.option["dataset"], trn_loss, trn_roc,
                    trn_prc, lr_cur
                ),
                with_time=True,
            )
            self.log(
                "Epoch:{} {} val_loss:{:.3f} val_roc:{:.3f} val_prc:{:.3f} lr_cur:{:.5f}".format(  # noqa: E501
                    epoch, self.option["dataset"], val_loss, val_roc,
                    val_prc, lr_cur
                ),
                with_time=True,
            )
            self.log(
                "Epoch:{} {} test_loss:{:.3f} test_roc:{:.3f} test_prc:{:.3f} lr_cur:{:.5f}".format(  # noqa: E501
                    epoch, self.option["dataset"], test_loss, test_roc,
                    test_prc, lr_cur
                ),
                with_time=True,
            )

            self.records["val_roc"].append(val_roc)
            self.records["val_prc"].append(val_prc)
            self.records["val_record"].append(
                [epoch, val_loss, val_roc, val_prc, lr_cur]
            )
            self.records["trn_record"].append(
                [epoch, trn_loss, trn_roc, trn_prc, lr_cur]
            )
            if (
                val_roc == np.array(self.records["val_roc"]).max()
                or val_prc == np.array(self.records["val_prc"]).max()
            ):
                self.save_model_and_records(epoch)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if (
                self.option["early_stop_patience"] > 0
                and early_stop_cnt > self.option["early_stop_patience"]
            ):
                self.log("Early stop hitted!")
                break
        self.save_model_and_records(epoch, final_save=True)

    def save_model_and_records(self, epoch, final_save=False):
        if final_save:
            self.save_loss_records()
            file_name = "best_model.ckpt"
        else:
            file_name = "best_model.ckpt"
            self.records["best_ckpt"] = file_name

        with open(os.path.join(self.save_path, file_name), "wb") as f:
            torch.save(
                {
                    "option": self.option,
                    "records": self.records,
                    "model_state_dict": self.model.state_dict(),
                },
                f,
            )
        self.log("Model saved at epoch {}".format(epoch))

    def save_loss_records(self):
        trn_record = pd.DataFrame(
            self.records["trn_record"],
            columns=["epoch", "trn_loss", "trn_auc", "trn_acc", "lr"],
        )
        val_record = pd.DataFrame(
            self.records["val_record"],
            columns=["epoch", "val_loss", "val_auc", "val_acc", "lr"],
        )
        ret = pd.DataFrame(
            {
                "epoch": trn_record["epoch"],
                "trn_loss": trn_record["trn_loss"],
                "val_loss": val_record["val_loss"],
                "trn_auc": trn_record["trn_auc"],
                "val_auc": val_record["val_auc"],
                "trn_lr": trn_record["lr"],
                "val_lr": val_record["lr"],
            }
        )
        ret.to_csv(self.save_path + "/record.csv")
        return ret

    def load_best_ckpt(self):
        ckpt_path = self.save_path + "/" + self.records["best_ckpt"]
        self.log("The best ckpt is {}".format(ckpt_path))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.log("Ckpt loading: {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.option = ckpt["option"]
        self.records = ckpt["records"]
        self.model.load_state_dict(ckpt["model_state_dict"])

    def log(self, msg=None, msgs=None, with_time=False, show=True):
        if with_time:
            msg = msg + " time elapsed {:.2f} hrs ({:.1f} mins)".format(
                (time.time() - self.start) / 3600.0,
                (time.time() - self.start) / 60.0
            )
        with open(self.save_path + "/log.txt", "a+") as f:
            if msgs:
                self.log("#" * 80)
                if "\n" not in msgs[0]:
                    msgs = [m + "\n" for m in msgs]
                f.writelines(msgs)
                if show:
                    for x in msgs:
                        print(x, end="")
                self.log("#" * 80)
            if msg:
                f.write(msg + "\n")
                if show:
                    print(msg)
