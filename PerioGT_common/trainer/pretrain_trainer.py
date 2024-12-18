import torch


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, nce_loss_fn, reg_evaluator, clf_evaluator, result_tracker, device, ddp=False, local_rank=1):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reg_loss_fn = reg_loss_fn
        self.clf_loss_fn = clf_loss_fn
        self.sl_loss_fn = sl_loss_fn
        self.nce_loss_fn = nce_loss_fn
        self.reg_evaluator = reg_evaluator
        self.clf_evaluator = clf_evaluator
        self.result_tracker = result_tracker
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0

    def _forward_epoch(self, model, batched_data):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        sl_labels = sl_labels.to(self.device)
        disturbed_fps = disturbed_fps.to(self.device)
        disturbed_mds = disturbed_mds.to(self.device)
        sl_predictions, fp_predictions, md_predictions, cl_projections = model(batched_graph, disturbed_fps, disturbed_mds)
        mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask']>=1].cpu().numpy()
        return mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds, cl_projections
    
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            try:
                self.optimizer.zero_grad()
                mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps, disturbed_fps, md_predictions, mds, cl_projections = self._forward_epoch(model, batched_data)
                sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()  # reconstruction loss of PNs
                fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()  # reconstruction loss of binary features in VNs
                md_loss = self.reg_loss_fn(md_predictions, mds).mean()  # reconstruction loss of numerical features in VNs
                cl_loss = self.nce_loss_fn(cl_projections, temperature=0.2, dist=True)  # PA-based contrastive learning loss
                loss = ((sl_loss + fp_loss + md_loss)/3) + cl_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                self.optimizer.step()
                print(f"n_step: {self.n_updates}")
                self.n_updates += 1
                self.lr_scheduler.step()

                if self.n_updates == self.args.n_steps:
                    if self.local_rank == 0:
                        self.save_model(model)
                    break

            except Exception as e:
                print(e)
            else:
                continue

    def fit(self, model, train_loader):
        for epoch in range(1, 1001):
            model.train()
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.n_updates >= self.args.n_steps:
                break

    def save_model(self, model):
        torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config}.pth")

    
