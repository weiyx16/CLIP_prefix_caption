import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from tf import GPT2LMHeadModel
from tqdm import tqdm
import io
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from clip1 import clip
from clip1.clip import _transform
import math
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import skimage.io as io1
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

def setup_for_distributed(is_master):

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = torch.cat((torch.tensor(self.tokenizer.encode('<|endoftext|>')), self.captions_tokens[item]), dim=0)
        gt = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            gt = torch.cat((gt, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            gt = gt[:self.max_seq_len]
        mask = tokens.ge(0)
        mask1 = gt.ge(0) # mask is zero where we out of sequence
        assert mask.equal(mask1)
        tokens[~mask] = 0
        gt[~mask1] = 0
        mask = mask.float()
        # mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask, gt

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, gt = self.pad_tokens(item)
        img_id = self.image_ids[item]
        filename = f"/zzx_vlexp/VQ-Diffusion-my2/MSCOCO_Caption/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"/zzx_vlexp/VQ-Diffusion-my2/MSCOCO_Caption/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        # image_encoding = self.clip_model.encode_image(image.unsqueeze(0))
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, image, gt

    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        self.preprocess = _transform(224)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        batch_size = embedding_text.size()[0]
        bos_toekn_embedding = self.bos_embedding.unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0)
        embedding_text = torch.cat((bos_toekn_embedding.unsqueeze(dim=1), embedding_text[:, 1:, :]), dim=1)
        # prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        prefix_projections = self.clip_project(self.image_encode(prefix).to(dtype=torch.float32))
        # embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_text, labels=labels, attention_mask=mask, encoder_hidden_states=prefix_projections)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.bos_embedding = nn.Parameter(torch.randn(768))
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2').train()
        self.clip_model, _ = clip.load("ViT-B/32", jit=False)
        self.clip_model.requires_grad_(False)
        self.image_encode = self.clip_model.encode_image
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            # self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
            #                          self.gpt_embedding_size * prefix_length))
            self.clip_project = MLP((768, self.gpt_embedding_size // 2,
                                     self.gpt_embedding_size ))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 1e-4, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # device = torch.device('cuda:0')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    # train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    # )
    # save_config(args)
    for epoch in range(epochs):
        train_dataloader.sampler.set_epoch(epoch)
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix, gt) in enumerate(train_dataloader):
            if epoch + 1 == 10:
                for p in optimizer.param_groups:
                    p['lr'] = 0.0002
            if epoch + 1 == 20:
                for p in optimizer.param_groups:
                    p['lr'] = 0.00005
            model.zero_grad()
            tokens, mask, prefix, gt = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32), gt.to(device)
            outputs = model(tokens, prefix, mask)
            # logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            logits = outputs.logits
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item(), 'lr': optimizer.param_groups[0]['lr']})
            if dist.get_rank() == 0:
                wandb.log({'train_loss': loss.item(),
                           'lr': optimizer.param_groups[0]['lr']})
            progress.update()
            if (idx + 1) % 100 == 0 and dist.get_rank() == 0:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            if dist.get_rank() == 0:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
                )
    return model


def main(local_rank, world_size):
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    torch.distributed.barrier()
    setup_for_distributed(local_rank == 0) ##### HERE
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--tag', default='wo_pre_linearlr_49token',
                        help='tag of job, used for wandb and output')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument("--local_rank", default=-1)
    args = parser.parse_args()
    args.local_rank = local_rank
    if dist.get_rank() == 0:
        wandb.login(key='49222ad51163763788e59460ea91552f32605e38')
        run = wandb.init(
            id=args.tag,
            name=args.tag,
            entity='buxiangzhiren',
            project='baseline',
            job_type='train_model',
            config=args,
        )
    prefix_length = args.prefix_length
    dataset = ClipCocoDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters')
    train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    world_size = 8
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)

