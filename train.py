import argparse
import json
import numpy as np
import os
import pickle
import sys
import wandb
from enum import Enum
from PIL import ImageFile
from tqdm import tqdm
from typing import Tuple, Optional, Union

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from clip1 import clip
from lr_scheduler import build_scheduler
from misc import generate2, generate_beam, evaluate_on_pure_caption, evaluate_on_coco_caption
from tf import GPT2LMHeadModel

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
        # notice the endoftext in tokens is just used as placehold, will be replaced by a learnable bos embedding
        tokens = torch.cat((torch.tensor(self.tokenizer.encode('<|endoftext|>')), self.captions_tokens[item]), dim=0)
        gt = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            gt = torch.cat((gt, torch.zeros(padding, dtype=torch.int64) - 1))  # we set target == -1 as ignore target
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            gt = gt[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        return tokens, mask, gt

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res + torch.zeros(broadcast_shape, device=timesteps.device)

    def preprocess_x0(self, x0):
        x0 = x0 / self.text_embedding_all_var
        # return x0
        weights = np.cos(np.linspace(0, np.pi/2, self.num_timesteps))
        t = torch.multinomial(torch.tensor(weights), x0.size(0), replacement=True)
        noise = torch.randn_like(x0)
        x0_noise = (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )
        # print(t, x0 / x0.norm(p=2) @ (x0_noise / x0_noise.norm(p=2)).t())
        return x0_noise

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, gt = self.pad_tokens(item)
        # this is for pre-computed clip feature
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        # this is for text embedding
        token_feature = self.preprocess_x0(self.text_embedding_all[item])
        return tokens, mask, token_feature, gt

        # # this is for text embedding
        # token_feature = self.preprocess_x0(self.text_embedding_all[item])
        # image_id = self.image_ids[item]
        # return token_feature, image_id

    def __init__(self, data_root: str, data_path: str,  gpt2_type: str = "gpt2", normalize_prefix=False):
        self.data_root = data_root
        self.tokenizer = clip._tokenizer
        self.normalize_prefix = normalize_prefix
        self.text_embedding_all_var = 0.12
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_clip_tokens.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            print(" inference the clip text tokens ")
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in tqdm(captions_raw):
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])  # just index
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        # notice current one is 40
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

        # about clip-text embedding
        if os.path.isfile(f"{data_path[:-4]}_clip_tokens_embed.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens_embed.pkl", 'rb') as f:
                self.text_embedding_all = pickle.load(f)
        else:
            print(" inference the clip text embedding ")
            self.clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False)
            self.clip_model.requires_grad_(False)
            self.text_encode = self.clip_model.encode_text
            self.clip_model.eval()
            self.text_embedding_all = []
            for eci in tqdm(range(len(self.captions_tokens))):
                tokens = torch.cat((self.captions_tokens[eci], torch.tensor(self.tokenizer.encode('<|endoftext|>'))),
                                   dim=0)
                tokens = torch.cat((torch.tensor(self.tokenizer.encode('<|startoftext|>')), tokens), dim=0)
                with torch.no_grad():
                    text_embedding = self.text_encode(tokens.unsqueeze(0))
                self.text_embedding_all.append(text_embedding)
            if dist.get_rank() == 0:
                with open(f"{data_path[:-4]}_clip_tokens_embed.pkl", 'wb') as f:
                    pickle.dump(self.text_embedding_all, f)
            torch.distributed.barrier()
        if not torch.is_tensor(self.text_embedding_all):
            self.text_embedding_all = torch.stack(self.text_embedding_all)

        # simulate noise term in diffusion:
        self.num_timesteps = 1000
        betas = np.linspace(0.0001, 0.02, self.num_timesteps, dtype=np.float64)

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)


class ClipCocoTrainDataset(Dataset):

    def __len__(self) -> int:
        return len(self.order_ids)

    def preprocess_x0(self, x0):
        x0 = x0 / self.text_embedding_all_var
        return x0

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # this is for text embedding
        token_feature = self.preprocess_x0(self.text_embedding_all[item])
        return token_feature, self.order_ids[item]

    def __init__(self, data_root: str, data_path: str):
        self.tokenizer = clip._tokenizer
        self.text_embedding_all_var = 1.0
        self.data_root = data_root
        with open('captioneval/coco_train_110k.txt') as f:
            self.files = f.read().splitlines()

        # load text:
        annos = json.load(open(os.path.join(data_root, 'annotations', 'captions_train2014_110k.json')))['annotations']
        self.captions_all = dict()
        for anno in annos:
            if anno["image_id"] in self.captions_all:
                self.captions_all[anno["image_id"]].append(anno["caption"])
            else:
                self.captions_all[anno["image_id"]] = [anno["caption"]]

        self.order_ids = list(range(len(self.files)))
        # for image_file in self.files:
        #     image_id = int(image_file.split('_')[-1].split('.')[0])
        #     captions = self.captions_all[image_id]
        #     captions.sort()
        #     self.order_ids.extend(range(len(self.order_ids), len(self.order_ids)+len(captions)))

        self.order_ids = self.order_ids[:len(self.order_ids) // dist.get_world_size() * dist.get_world_size()]

        # about clip-text embedding
        if os.path.isfile(f"{data_path[:-4]}_clip_tokens_embed.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens_embed.pkl", 'rb') as f:
                self.text_embedding_all = pickle.load(f)
        else:
            raise NotImplementedError

        if not torch.is_tensor(self.text_embedding_all):
            self.text_embedding_all = torch.stack(self.text_embedding_all)


class ClipCocoValDataset(Dataset):

    def __len__(self) -> int:
        return len(self.order_ids)

    def preprocess_x0(self, x0):
        x0 = x0 / self.text_embedding_all_var
        return x0

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # this is for text embedding
        token_feature = self.preprocess_x0(self.text_embedding_all[item])
        return token_feature, self.order_ids[item]

    def __init__(self, data_root: str, data_path: str):
        self.tokenizer = clip._tokenizer
        self.text_embedding_all_var = 0.12
        self.data_root = data_root
        with open('captioneval/coco_val.txt') as f:
            self.files = f.read().splitlines()

        # load text:
        annos = json.load(open(os.path.join(data_root, 'annotations', 'captions_val2014.json')))['annotations']
        self.captions_all = dict()
        for anno in annos:
            if anno["image_id"] in self.captions_all:
                self.captions_all[anno["image_id"]].append(anno["caption"])
            else:
                self.captions_all[anno["image_id"]] = [anno["caption"]]

        self.order_ids = []
        for image_file in self.files:
            image_id = int(image_file.split('_')[-1].split('.')[0])
            captions = self.captions_all[image_id]
            captions.sort()
            self.order_ids.extend(range(len(self.order_ids), len(self.order_ids)+len(captions)))

        self.order_ids = self.order_ids[:len(self.order_ids) // dist.get_world_size() * dist.get_world_size()]

        # about clip-text embedding
        if os.path.isfile(f"{data_path[:-4]}_clip_tokens_embed_val.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens_embed_val.pkl", 'rb') as f:
                self.text_embedding_all = pickle.load(f)
        else:
            # torch.Size([25010, 1, 512]) tensor(0.1273)
            print(" inference the clip text embedding on val set ")
            self.clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False)
            self.clip_model.requires_grad_(False)
            self.text_encode = self.clip_model.encode_text
            self.clip_model.eval()
            self.text_embedding_all = []

            for image_file in tqdm(self.files):
                image_id = int(image_file.split('_')[-1].split('.')[0])
                captions = self.captions_all[image_id]
                captions.sort()
                for caption in captions:
                    tokens = torch.tensor(
                        self.tokenizer.encode('<|startoftext|> '+caption+' <|endoftext|>'),
                        dtype=torch.int64
                    )
                    tokens_pad = torch.zeros(1, 77, dtype=torch.long)
                    tokens_pad[0, :len(tokens)] = tokens
                    with torch.no_grad():
                        text_embedding = self.text_encode(tokens_pad)
                    self.text_embedding_all.append(text_embedding)
            if dist.get_rank() == 0:
                with open(f"{data_path[:-4]}_clip_tokens_embed_val.pkl", 'wb') as f:
                    pickle.dump(self.text_embedding_all, f)
            torch.distributed.barrier()

        self.text_embedding_all = torch.stack(self.text_embedding_all)


class ClipCocoValStage2Dataset(Dataset):
    """
        Don't take gt emb as input, but take predicted prior as input, for testing prior network
    """

    def __len__(self) -> int:
        return len(self.files)

    def preprocess_x0(self, x0):
        x0 = x0 / self.text_embedding_all_var
        return x0

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # this is for text embedding
        token_feature = self.preprocess_x0(self.text_embedding_all[item])
        return token_feature, self.files[item]

    def __init__(self, data_root: str, data_path: str):
        self.tokenizer = clip._tokenizer
        self.text_embedding_all_var = 1.0
        self.data_root = data_root
        with open('captioneval/coco_val.txt') as f:
            self.files = f.read().splitlines()

        # about pred_clip-text embedding
        with open(f"{data_path[:-4]}_clip_tokens_embed_val_predbyprior.pkl", 'rb') as f:
            self.text_embedding_all = pickle.load(f)
        self.text_embedding_all_dict = dict()
        for item in self.text_embedding_all:
            self.text_embedding_all_dict[item['image_id']] = item['feature']
        self.text_embedding_all = [self.text_embedding_all_dict[_file] for _file in self.files]
        self.text_embedding_all = torch.stack(self.text_embedding_all, dim=0).unsqueeze(1)  # 5000 * 1 * 512


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
            if i % 2 == 0 and self.enc_dec:  # cross
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
                layers.append(
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer)
                )
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
        embedding_text = torch.cat((self.clip_project(prefix), embedding_text[:, 1:, :]), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_text, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        configuration = GPT2Config.from_pretrained('gpt2')
        configuration = configuration.__dict__.copy()
        configuration.update({'scale_attn_by_inverse_layer_idx': False})
        configuration.update({'reorder_and_upcast_attn': False})
        configuration.update({'n_embd': prefix_size})
        configuration.update({'n_head': prefix_size//64})
        configuration.update({'add_cross_attention': False})
        configuration.update({'vocab_size': 49152+256})  # adapt to clip tokenizer
        configuration = GPT2Config(**configuration)
        self.gpt = GPT2LMHeadModel(configuration)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size,
                                     self.gpt_embedding_size // 2,
                                     self.gpt_embedding_size))
        else:
            self.clip_project = TransformerMapper(prefix_size,
                                                  self.gpt_embedding_size,
                                                  prefix_length,
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
    out_path = os.path.join(args.out_dir, f"{args.tag}-args.json")
    if args.local_rank == 0:
        with open(out_path, 'w') as outfile:
            json.dump(config, outfile)


def load_model(model, args, epoch_or_latest: Union[str, int] = '_latest'):
    # with open(config_path) as f:
    #     config = json.load(f)
    # parser = argparse.ArgumentParser()
    # parser.set_defaults(**config)
    # args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.tag}{epoch_or_latest}.pt")
    if os.path.isfile(model_path):
        msg = model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["model"])
        print(f"loading model from {model_path}, with msg: {msg}")
    else:
        print(f"{model_path} is not exist")
    return model


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    print(f'No decay params: {no_decay_name}')
    print(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


@torch.no_grad()
def val_prior(model, val_dataloader, args):
    model.eval()
    tokenizer = clip._tokenizer
    print(">>> Evaling Prior Model")
    sys.stdout.flush()
    progress = tqdm(total=len(val_dataloader), desc=args.tag)
    result_all = []
    for idx, (prefix, image_path) in enumerate(val_dataloader):
        prefix = prefix.cuda(non_blocking=True)

        prefix_embed = model.module.clip_project(prefix)
        if args.use_beam_search:
            assert False, "Not check beam search for now"
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        torch.cuda.synchronize()
        progress.update()

        r = [
            {'image_id': _image_path, 'result': str(_text)}
            for _image_path, _text in zip(image_path, generated_text_prefix)
        ]
        result_all.extend(r)
    progress.close()
    os.makedirs('.cache', exist_ok=True)
    json.dump(result_all, open(f".cache/tmp-results-{dist.get_rank()}.json", "w"))
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        result_all = []
        for i in range(dist.get_world_size()):
            part_result = json.load(open(f".cache/tmp-results-{i}.json"))
            result_all.extend(part_result)
        result = evaluate_on_coco_caption(
            result_all,
            os.path.join(args.out_dir, f"{args.tag}-priormodel-results.json"),
            os.path.join(args.data_root, 'annotations/captions_val2014.json')
            # os.path.join(args.data_root, 'annotations/captions_train_110k.json')
        )
    else:
        result = None
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        wandb.log(
            {'Overall_BLEU_4': result['Bleu_4'],
             'Overall_METEOR': result['METEOR'],
             'Overall_ROUGE_L': result['ROUGE_L'],
             'Overall_CIDEr': result['CIDEr'],
             'Overall_SPICE': result['SPICE']}
        )
    return result


@torch.no_grad()
def val(model, epoch, val_dataloader, args):
    model.eval()
    tokenizer = clip._tokenizer
    print(f">>> Evaling epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(val_dataloader), desc=args.tag)
    result_all = []
    for idx, (prefix, text_id) in enumerate(val_dataloader):
        prefix = prefix.cuda(non_blocking=True)

        prefix_embed = model.module.clip_project(prefix)
        if args.use_beam_search:
            assert False, "Not check beam search for now"
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

        torch.cuda.synchronize()
        progress.update()

        r = [
            {'text_id': _text_id.item(), 'result': str(_text)}
            for _text_id, _text in zip(text_id, generated_text_prefix)
        ]
        result_all.extend(r)
    progress.close()
    os.makedirs('.cache', exist_ok=True)
    json.dump(result_all, open(f".cache/tmp-results-{dist.get_rank()}.json", "w"))
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        result_all = []
        for i in range(dist.get_world_size()):
            part_result = json.load(open(f".cache/tmp-results-{i}.json"))
            result_all.extend(part_result)
        result = evaluate_on_pure_caption(
            result_all,
            os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}-results.json"),
            os.path.join(args.data_root, 'annotations/captions_val2014_puretext.json')
            # os.path.join(args.data_root, 'annotations/captions_train2014_110k_puretext.json')
        )
    else:
        result = None
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        wandb.log(
            {'BLEU_4': result['Bleu_4'],
             'METEOR': result['METEOR'],
             'ROUGE_L': result['ROUGE_L'],
             'CIDEr': result['CIDEr'],
             'SPICE': result['SPICE']}
        )
    return result


def train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args,
          output_dir: str = ".", output_prefix: str = ""):
    model.train()
    num_steps = len(train_dataloader)

    train_dataloader.sampler.set_epoch(epoch)
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(train_dataloader), desc=output_prefix)
    for idx, (tokens, mask, prefix, gt) in enumerate(train_dataloader):
        # prefix is raw images
        tokens = tokens.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        prefix = prefix.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        with amp.autocast(enabled=args.enable_amp):
            outputs = model(tokens, prefix, mask)
        logits = outputs.logits
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt.flatten(), ignore_index=-1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()
        progress.set_postfix({"loss": loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        if dist.get_rank() == 0:
            wandb.log({'train_loss': loss.item(),
                       'lr': optimizer.param_groups[0]['lr']})
        progress.update()
    progress.close()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--data_root', default='./data/MSCOCO_CAPTION', help='raw coco training image path')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--tag', default='wo_pre_linearlr_49token',
                        help='tag of job, used for wandb and output')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--use_beam_search', action='store_true')
    parser.add_argument('--eval-prior', action='store_true', help='eval prior model only')
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--disable-amp', action='store_false', dest='enable_amp')
    parser.set_defaults(enable_amp=True)

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(args.out_dir, exist_ok=True)
    save_config(args)
    return args


def main(args):
    if dist.get_rank() == 0:
        wandb.login(key='c26712d8885e3e6742ffd9c311e10870a46a197f')
        wandb.init(
            id=args.tag,
            name=args.tag,
            entity='msravcg',
            project='diffuseIC',
            job_type='coco',
            config=args,
        )
    prefix_dim = 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(
            args.prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            mapping_type=args.mapping_type
        )
        print("Train both prefix and GPT")
        sys.stdout.flush()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params:,} total parameters')

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    model = model.cuda()

    parameters = get_pretrain_param_groups(model)
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    scaler = amp.GradScaler()
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    if args.eval_prior:
        # load model
        model_without_ddp = model.module
        _ = load_model(model_without_ddp, args, epoch_or_latest=args.epochs-1)
        # build Dataset
        val_dataset = ClipCocoValStage2Dataset(args.data_root, args.data)
        # val_dataset = ClipCocoDataset(args.data_root, args.data, normalize_prefix=args.normalize_prefix)
        # val_dataset = ClipCocoValDataset(args.data_root, args.data)
        # val_dataset = ClipCocoTrainDataset(args.data_root, args.data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.bs,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=False
        )
        result = val_prior(model, val_dataloader, args)
        # result = val(model, args.epochs, val_dataloader, args)
        return

    dataset = ClipCocoDataset(args.data_root, args.data, normalize_prefix=args.normalize_prefix)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = ClipCocoValDataset(args.data_root, args.data)
    # val_dataset = ClipCocoValStage2Dataset(args.data_root, args.data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    lr_args = {
        "LR_SCHEDULER_NAME": "cosine",
        "EPOCHS": args.epochs,
        "WARMUP_EPOCHS": 5,
        "MIN_LR": 1e-6,
        "WARMUP_LR": 1e-7
    }
    lr_scheduler = build_scheduler(lr_args, optimizer, len(train_dataloader))

    best_cider = 0
    for epoch in range(args.epochs):
        _ = train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args,
                  output_dir=args.out_dir, output_prefix=args.tag)
        result = val(model, epoch, val_dataloader, args)
        # result = val_prior(model, val_dataloader, args)
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if dist.get_rank() == 0:
                torch.save(
                    {'model': model.module.state_dict(),
                     'lr_scheduler': lr_scheduler.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict()},
                    os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}.pt"),
                )
        if dist.get_rank() == 0 and result['CIDEr'] > best_cider:
            best_cider = result['CIDEr']
            torch.save(
                {'model': model.module.state_dict()},
                os.path.join(args.out_dir, f"{args.tag}-best.pt"),
            )


if __name__ == '__main__':
    # command:  python -m torch.distributed.launch --nproc_per_node 4 train.py \
    #           --data ./oscar_split_ViT-B_32_train_512.pkl --out_dir ./output --bs 32
    args = parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    dist.init_process_group("nccl", init_method='env://', rank=args.local_rank, world_size=world_size)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)  # HERE

    seed = dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args)
