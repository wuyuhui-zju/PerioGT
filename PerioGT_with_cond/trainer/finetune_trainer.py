import os.path
import torch
import numpy as np


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator, result_tracker,
                 device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.final_evaluator = final_evaluator
        self.result_tracker = result_tracker
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank

    def _forward_epoch(self, model, batched_data):
        (smiles, g, fp, md, gs, labels) = batched_data
        fp =fp.to(self.device)
        md = md.to(self.device)
        gs = gs.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune(g, fp, md, gs)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean) / self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()

    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result, best_test_result, best_train_result = self.result_tracker.init(), self.result_tracker.init(), self.result_tracker.init()
        best_epoch = 0
        test_r2 = 0
        for epoch in range(1, self.args.n_epochs + 1):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                val_result = self.eval(model, val_loader)
                test_result = self.eval(model, test_loader)
                train_result = self.eval(model, train_loader)
                print(f"Epoch: {epoch}\ttrain: {train_result}\tval: {val_result}")
                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                    print("saved!")
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_epoch = epoch
                    test_r2 = self.eval(model, test_loader, is_test=True)
                    if self.args.save:
                        save_path = os.path.join("../checkpoints", self.args.dataset)
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join("../checkpoints", self.args.dataset, f"best_model_{self.args.save_suffix}.pth"))
                if epoch - best_epoch >= 20:
                    break
        return best_train_result, best_val_result, best_test_result, test_r2

    def eval(self, model, dataloader, is_test=False):
        model.eval()
        predictions_all = []
        labels_all = []

        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        if is_test:
            result = self.final_evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        else:
            result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        return result

