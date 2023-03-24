import numpy as np
import torch

inf = float("inf")

n = torch.tensor([3, 1, 4])
probs = torch.tensor([[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
[[1.0,0.0,0.0],[0.0,1.0,0.0],[1.0,1.1,1.2],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]]) # your sorted confidence values

flat_probs = probs.reshape(-1, probs.shape[-1])
flat_result = torch.multinomial(flat_probs, 1)
result = flat_result.reshape(-1, probs.shape[-2])

confidence = probs.gather(-1, result.unsqueeze(-1)).squeeze(-1)
print(confidence)

            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                layouts = self.fixed_x.detach().cpu().numpy()
                input_layouts = [self.train_dataset.render(
                    layout) for layout in layouts]
                x_cond = self.fixed_x.to(self.device)
                logits, _ = model(x_cond)
                probs = F.softmax(logits, dim=-1)
                _, y = torch.topk(probs, k=1, dim=-1)
                layouts = torch.cat(
                    (x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
                recon_layouts = [self.train_dataset.render(
                    layout) for layout in layouts]
                layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
                sample_random_layouts = [
                    self.train_dataset.render(layout) for layout in layouts]
                layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
                sample_det_layouts = [self.train_dataset.render(
                    layout) for layout in layouts]
                wandb.log({
                    "input_layouts": [wandb.Image(pil, caption=f'input_{epoch:02d}_{i:02d}.png')
                                      for i, pil in enumerate(input_layouts)],
                    "recon_layouts": [wandb.Image(pil, caption=f'recon_{epoch:02d}_{i:02d}.png')
                                      for i, pil in enumerate(recon_layouts)],
                    "sample_random_layouts": [wandb.Image(pil, caption=f'sample_random_{epoch:02d}_{i:02d}.png')
                                              for i, pil in enumerate(sample_random_layouts)],
                    "sample_det_layouts": [wandb.Image(pil, caption=f'sample_det_{epoch:02d}_{i:02d}.png')
                                           for i, pil in enumerate(sample_det_layouts)],}, step=self.iters)
