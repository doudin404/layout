import torch.nn.functional as F
import os
import math
import logging
import wandb
from dataset import JSONLayout, sample
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import torch

from torch.utils.data.dataloader import DataLoader


from model import BLT
import argparse
logger = logging.getLogger(__name__)


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))  # 强制修改当前目录
    train_dataset = JSONLayout("train.json")
    Test_dataset = JSONLayout("val.json")
    #loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=3)
    print(test_masking(train_dataset[:4], train_dataset.mask_token,
          train_dataset.eos_token, train_dataset.pad_token))


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.98)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_iters = 5000
    final_iters = 200000  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def check_for_nan_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).sum() > 0:
                print(f"{name} has NaN gradients")
                return True
    return False


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.iters = 0
        self.optimizer = model.configure_optimizers(config)
        print("Using wandb")
        wandb.init(project='layout_BLT', name=args.exp)
        wandb.config.update(args)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if args.load:
            self.load_checkpoint()

    def load_checkpoint(self):
        print(f"load checkpoint {self.config.load_name}")
        ckpt_path = f"save/{self.config.load_name}.pth"
        logger.info("loading %s", ckpt_path)
        checkpoint = torch.load(ckpt_path)
        # 本来代码里的raw_model是为了处理使用DataParallel的情况,下面的代码是等价的
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

    def save_checkpoint(self,id=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, f'checkpoint{f"No.{id}" if id is not None else ""}.pth')
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def save_combined_image(self, layouts, name):
        # 拼接四张图片
        combined_image = Image.new('RGB', (512, 512))
        combined_image.paste(layouts[0], (0, 0))
        combined_image.paste(layouts[1], (256, 0))
        combined_image.paste(layouts[2], (0, 256))
        combined_image.paste(layouts[3], (256, 256))

        # 保存拼接后的图片到同名文件
        filename = f'{name}.jpg'
        combined_image.save(filename)

    def run_epoch_(self, epoch, split, optimizer):
        is_train = split == 'train'
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)
        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)
                    ) if is_train else enumerate(loader)
        for it, y in pbar:
             # place data on the correct device
            y = y.to(self.device)

            x, masks = BLT_masking(
                y, self.train_dataset.mask_token, self.train_dataset.eos_token, self.train_dataset.pad_token, is_train)

            # place data on the correct device

            # forward the model
            with torch.set_grad_enabled(is_train):
                if is_train:
                    self.model.train()
                else:
                    self.model.eval()
                # import ipdb; ipdb.set_trace()
                _, loss = self.model(x, targets=y, masks=masks)
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

            if is_train:

                # backprop and update the parameters
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(  # 这个函数用来限制梯度的范围
                    self.model.parameters(), self.config.grad_norm_clip)

                if check_for_nan_gradients(self.model):
                    print("NaN gradients detected. Aborting training.")
                    return

                self.optimizer.step()
                self.iters += 1
                # decay the learning rate based on our progress
                if self.config.lr_decay:
                    # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    if self.iters < self.config.warmup_iters:
                        # linear warmup
                        lr_mult = float(self.iters) / float(max(1, self.config.warmup_iters))
                    else:
                        # cosine learning rate decay
                        progress = min(1,float(self.iters - self.config.warmup_iters) / float(
                                max(1, self.config.final_iters - self.config.warmup_iters)))
                        lr_mult = max(
                            0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = self.config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = self.config.learning_rate

                # report progress
                wandb.log({
                    'train loss': loss.item(),
                    'lr': lr, 'epoch': epoch
                }, step=self.iters)
                pbar.set_description(
                    f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

        if not is_train:
            test_loss = float(np.mean(losses))
            logger.info("test loss: %f", test_loss)
            wandb.log({'test loss': test_loss}, step=self.iters)
            return test_loss
        # def run_epoch结束

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        best_loss = float('inf')
        epoch = 0
        while True:
            epoch += 1
            if epoch > config.max_epochs:
                print("运行结束,输入正整数继续训练:\n")
                temp = input()
                try:
                    temp = int(temp)
                    if temp > 0:
                        config.max_epochs += temp
                        epoch -= 1
                        continue
                    else:
                        print("Please enter a positive number.")
                        
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
                    break

            self.run_epoch_(epoch, 'train', optimizer=optimizer)
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = self.run_epoch_(
                        epoch, 'test', optimizer=optimizer)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                self.test(epoch+1)
    
    def test(self, epoch=0):
        model = self.model
        data = self.test_dataset
        test_batch = data[:16].to(self.device)

        data = self.test_dataset
        test_batch = data[:4].to(self.device)

        layouts = test_batch.detach().cpu().numpy()
        input_layouts = [self.train_dataset.render(
            layout) for layout in layouts]
        # self.save_combined_image(input_layouts,"input_layouts")

        masked_batch, masks = test_masking(
            test_batch, data.mask_token, data.eos_token, data.pad_token)

        layouts = sample(model, masked_batch, masks=masks, steps=0,
                            temperature=1.0, sample=False, top_k=None, y=test_batch).detach().cpu().numpy()
        recon_layouts = [self.train_dataset.render(
            layout) for layout in layouts]
        # self.save_combined_image(recon_layouts,"recon_layouts")

        layouts = sample(model, masked_batch, masks=masks, steps=10,
                            temperature=1.0, sample=True, top_k=5, y=test_batch).detach().cpu().numpy()
        sample_random_layouts = [
            self.train_dataset.render(layout) for layout in layouts]
        # self.save_combined_image(sample_random_layouts,"sample_random_layouts")

        layouts = sample(model, masked_batch, masks=masks, steps=10,
                            temperature=1.0, sample=False, top_k=None, y=test_batch).detach().cpu().numpy()
        sample_det_layouts = [self.train_dataset.render(
            layout) for layout in layouts]
        # self.save_combined_image(sample_det_layouts,"sample_det_layouts")

        wandb.log({
            "input_layouts": [wandb.Image(pil, caption=f'input_{epoch:02d}_{i:02d}.png')
                                for i, pil in enumerate(input_layouts)],
            "recon_layouts": [wandb.Image(pil, caption=f'recon_{epoch:02d}_{i:02d}.png')
                                for i, pil in enumerate(recon_layouts)],
            "sample_random_layouts": [wandb.Image(pil, caption=f'sample_random_{epoch:02d}_{i:02d}.png')
                                        for i, pil in enumerate(sample_random_layouts)],
            "sample_det_layouts": [wandb.Image(pil, caption=f'sample_det_{epoch:02d}_{i:02d}.png')
                                    for i, pil in enumerate(sample_det_layouts)], }, step=self.iters)


def BLT_masking(inputs, mask_token, eos_token, pad_token, training=True):
    """
    原版代码,这里的某些操作我也不能完全理解.
    """


    if not training:
        original_seed = torch.initial_seed()
        torch.manual_seed(0)  # 设置固定的随机种子

    total_dim = 5
    layout_dim = 2

    #rng = np.random.default_rng(np.sum(inputs, dtype="int32"))

    is_pad = (inputs == pad_token).to(torch.bool)
    position_ids = torch.arange(inputs.shape[-1])[None, :].to(inputs.device)
    is_asset = (position_ids % total_dim == 0)

    is_size = ((position_ids % total_dim >= 1) & (position_ids %
               total_dim < layout_dim + 1)).to(torch.bool)
    is_position = ((position_ids % total_dim >= layout_dim + 1)
                   & (position_ids % total_dim < total_dim)).to(torch.bool)

    rand = torch.rand(inputs.shape[0], 1, device=inputs.device)

    should_mask = (~is_pad) & is_asset
    should_mask = torch.where(
        (rand >= 0.2) & (rand < 0.4),
        (is_asset | is_size) & (~is_pad), should_mask)
    should_mask = torch.where(rand >= 0.4, ~is_pad, should_mask)

    fullmask = torch.full_like(inputs, mask_token)
    fullmask = torch.where(is_pad, pad_token, fullmask)

    pre_masked_inputs = torch.where(should_mask, inputs, fullmask)
    weights = is_asset & (~is_pad)
    weights = torch.where(
        (rand >= 0.2) & (rand < 0.4), is_size & (~is_pad), weights)
    weights = torch.where(
        (rand >= 0.4) & (rand < 0.6), is_position & (~is_pad),
        weights)
    weights = torch.where(
        (rand >= 0.6) & (rand < 0.8), is_size & (~is_pad), weights)
    weights = torch.where(rand >= 0.8, is_asset & (~is_pad), weights)

    lens = torch.sum(weights, axis=-1)
    mask_rate = 1 - torch.rand(lens.shape, device=inputs.device)

    mask_lens = torch.ceil(
        lens * mask_rate).clone().detach().requires_grad_(True).to(torch.int64)

    should_mask = torch.rand(inputs.shape, device=inputs.device)
    should_mask = torch.where(weights, should_mask, 2.)

    sorted_should_mask, _ = torch.sort(should_mask, dim=-1)
    cut_off = sorted_should_mask.gather(-1, (mask_lens-1).unsqueeze(1))

    should_mask = torch.where(should_mask <= cut_off, True, False)

    fullmask = torch.full_like(inputs, mask_token)

    masked_inputs = torch.where(should_mask, fullmask, pre_masked_inputs)
    weights = torch.where(is_pad, 0, should_mask)
    if not training:
        torch.manual_seed(original_seed)  # 还原随机种子
    return masked_inputs, weights

def test_masking(inputs, mask_token, eos_token, pad_token):
    device = inputs.device
    batch_size = len(inputs)
    max_len = max(len(s) for s in inputs)

    masked_inputs = torch.ones(
        batch_size, max_len, dtype=torch.long, device=device) * pad_token
    should_mask = torch.zeros(
        batch_size, max_len, dtype=torch.bool, device=device)

    for i, seq in enumerate(inputs):
        eos_index = (seq == eos_token).nonzero(as_tuple=True)[
            0].item() if eos_token in seq else len(seq)
        seq_len = eos_index  # exclude eos_token and padding
        masked_inputs[i, :seq_len + 1] = seq[:seq_len+1].clone().detach()

        if i == 0:
            mask_indices = torch.arange(seq_len, device=device)[torch.isin(
                torch.arange(seq_len, device=device) % 5, torch.tensor([3, 4], device=device))]
        elif i == 1:
            mask_indices = torch.arange(seq_len, device=device)[torch.isin(
                torch.arange(seq_len, device=device) % 5, torch.tensor([1, 2], device=device))]
        elif i == 2:
            mask_indices = torch.arange(seq_len, device=device)[(
                torch.arange(seq_len, device=device) // 5) >= (seq_len // 5) // 2]
        else:
            mask_indices = torch.arange(seq_len, device=device)[(
                torch.arange(seq_len, device=device) // 5) % 2 == 1]

        should_mask[i, mask_indices] = True
        masked_inputs[i, mask_indices] = mask_token

    return masked_inputs, should_mask


if __name__ == '__main__':
    main()
