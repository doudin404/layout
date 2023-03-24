import os
import argparse
import torch
import numpy as np
import random
from dataset import JSONLayout
from model import BLT, BLTConfig
from trainer import Trainer, TrainerConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))  # 强制修改当前目录
    parser = argparse.ArgumentParser('Layout BLT')
    parser.add_argument("--load_name", default="明日之星", help="读取现有模型进行测试")
    parser.add_argument("--load", type=bool, default=False, help="是否读取现有模型")

    parser.add_argument("--exp", default="layout", help="experiment name")
    parser.add_argument("--log_dir", default="./logs",
                        help="/path/to/logs/dir")

    # COCO/PubLayNet options
    parser.add_argument(
        "--train_json", default="../PubLayNet/train.json", help="/path/to/train/json")
    parser.add_argument(
        "--val_json", default="../PubLayNet/val.json", help="/path/to/val/json")

    # Layout options
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument('--precision', default=8, type=int)
    # parser.add_argument('--element_order', default='raster')
    # parser.add_argument('--attribute_order', default='cxywh')

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of epochs")
    parser.add_argument("--batch_size", type=int,
                        default=64, help="batch size")
    parser.add_argument("--lr", type=float,
                        default=4.5e-06, help="learning rate")

    parser.add_argument('--n_layer', default=4, type=int)#4
    parser.add_argument('--n_head', default=8, type=int)#8
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument('--attention_probs_dropout_prob',
                        default=0.1, type=float)
    parser.add_argument('--max_position_embeddings', default=150, type=int)
    parser.add_argument('--embed_size', default=512, type=int)
    parser.add_argument('--intermediate_size', default=2048, type=int)
    parser.add_argument('--data_cut_percentage', default=0.99, type=float)

    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--lr_decay', type=bool, default=True,
                        help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int,
                        default=5000, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int,
                        default=25000, help="cosine lr final iters")

    parser.add_argument('--sample_every', type=int,
                        default=1, help="sample every epoch")

    args = parser.parse_args()
    print(f"{args.batch_size=}")
    log_dir = os.path.join(args.log_dir, args.exp)
    samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_dataset = JSONLayout(
        args.train_json, percentage=args.data_cut_percentage)
    valid_dataset = JSONLayout(
        args.val_json, max_length=train_dataset.max_length)

    mconf = BLTConfig(train_dataset.vocab_size, train_dataset.max_length, pad_token=train_dataset.pad_token,
                      n_layer=args.n_layer,
                      n_head=args.n_head,
                      embed_size=args.embed_size,
                      intermediate_size=args.intermediate_size,
                      hidden_dropout_prob=args.hidden_dropout_prob,
                      attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                      max_position_embeddings=args.max_position_embeddings
                      )

    model = BLT(mconf)

    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate=args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=ckpt_dir,
                          load_name=args.load_name,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)

    if args.load:    
        trainer.test()
    else:
        trainer.train()