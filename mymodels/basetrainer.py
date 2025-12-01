import  timm
import  torch
import torch.nn as nn
#
# class BaseTrainer(nn.Module):
class BaseTrainer(object):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()


        self.trainable_layer = "self.backbone"

    def train_step(self, batch, **kwargs):
        loss =  self.forward_step(batch,train=True)

        return loss, {"Total_Loss": loss.item()}
    def forward_step(self, batch, train=False):
        return self.backbone(batch["image"])
    def eval_step(self, batch, **kwargs):
        return self.forward_step(batch,train=False)
    def get_models(self):
        for m in eval(self.trainable_layer):
            m.train()
        return (self.trainable_layer.split(","),
                *eval(self.trainable_layer))
