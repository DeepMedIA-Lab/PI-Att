"""
epoch: the current epoch of training
warmup_length: the determined warm-up period for training
loss_func: piatt_loss(nn.Module)
lr_pers_weight: the learning rate of the persistence image weight parameter
args.weight: weight parameter in the persistence image calculation
"""

if  epoch >= warmup_length:
    with torch.no_grad():
          args.weight -= args.lr_pers_weight * loss_func.beta.grad
          loss_func.beta.grad.zero_()