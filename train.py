import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from tf import GPT2LMHeadModel
from tqdm import tqdm
from collections import defaultdict
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
import numpy as np
from lr_scheduler import build_scheduler
from misc import generate2, generate_beam, evaluate_on_coco_caption

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

    def preprocess_x0(self, x0):
        x0 = x0 / self.text_embedding_all_var
        return x0

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens_feature = self.preprocess_x0(self.text_embedding_all[item])
        img_id = self.image_ids[item]
        # train+restval
        filename = f"{self.data_root}/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        try:
            image = io1.imread(filename)
        except:
            filename = f"{self.data_root}/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        ## this is for pre-computed clip feature
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return image, tokens_feature

    def __init__(self, data_root: str, data_path: str,  gpt2_type: str = "gpt2", normalize_prefix=False):
        self.data_root = data_root
        self.tokenizer = clip._tokenizer
        self.normalize_prefix = normalize_prefix
        self.preprocess = _transform(224)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        # sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_clip_tokens.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            print(" inference the clip text tokenizer ")
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in tqdm(captions_raw):
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"]) # just index
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            if dist.get_rank() == 0:
                with open(f"{data_path[:-4]}_clip_tokens.pkl", 'wb') as f:
                    pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
            torch.distributed.barrier()
        print("tokens are initialized")
        # all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        ## notice current one is 40
        # self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
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
        print("text embeddings are initialized")
        self.text_embedding_all = torch.stack(self.text_embedding_all)
        self.text_embedding_all_var = 0.12


class ClipCocoValDataset(Dataset):

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int):
        _filename = self.files[item]
        filename = f"{self.data_root}/val2014/{_filename}"
        image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        return image, _filename, self.id2captions[int(_filename.split('.')[0].split('_')[-1])]

    def __init__(self, data_root: str):
        self.data_root = data_root
        with open('captioneval/coco_val.txt') as f:
            self.files = f.read().splitlines()
        with open(os.path.join(data_root, 'annotations/captions_val2014.json'), "r") as f:
            self.annotations = json.load(f)["annotations"]
        self.id2captions = defaultdict(list)
        for annos in self.annotations:
            if len(self.id2captions[int(annos["image_id"])]) < 5:
                self.id2captions[int(annos["image_id"])].append(annos["caption"])
        self.preprocess = _transform(224)


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
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
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

    def forward(self, prefix: torch.Tensor, tokens_features: torch.Tensor, t: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        self.clip_model.eval()
        mask, labels = None, None
        embedding_text = tokens_features
        noise_feature = self.q_sample(embedding_text, t)
        batch_size = embedding_text.size()[0]
        bos_token_embedding = self.mask_embedding.unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0)
        with torch.no_grad():
            prefix_projections = self.image_encode(prefix).unsqueeze(1)
        # prefix_projections = self.clip_project(prefix)
        t_embedding = self.t_embedding(t).unsqueeze(1)
        all_embedding = torch.cat([prefix_projections,t_embedding,noise_feature,bos_token_embedding], dim=1)
        out = self.gpt(inputs_embeds=all_embedding, labels=labels, attention_mask=mask)
        return out, embedding_text.squeeze(dim=1)

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, Timestep: int = 1000):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.prefix_size = prefix_size
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        configuration = GPT2Config.from_pretrained('gpt2')
        configuration = configuration.__dict__.copy()
        configuration.update({'scale_attn_by_inverse_layer_idx': False})
        configuration.update({'reorder_and_upcast_attn': False})
        configuration.update({'add_cross_attention': False})
        configuration.update({'n_embd': self.prefix_size})
        configuration.update({'n_head': self.prefix_size // 64})
        configuration.update({'vocab_size': self.prefix_size})
        configuration.update({'tie_word_embeddings': False})
        configuration = GPT2Config(**configuration)
        self.gpt = GPT2LMHeadModel(configuration)
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False)
        self.clip_model.requires_grad_(False)
        self.image_encode = self.clip_model.encode_image
        self.text_encode = self.clip_model.encode_text
        self.time_step = Timestep

        self.t_embedding = nn.Embedding(self.time_step, self.prefix_size)
        self.t_embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.mask_embedding = nn.Parameter(torch.randn(self.prefix_size))

        betas = get_beta_schedule(args.beta_sche, num_diffusion_timesteps=Timestep)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x.size(0)
        assert t.shape == (B,)
        model_output = model(inputs_embeds=x)


        # The model_var_values is [-1, 1] for [min_var, max_var].

        pred_xstart = model_output.logits[:,-1,:]
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x[:,-2,:], t=t)

        assert model_mean.shape == pred_xstart.shape == x[:,-2,:].shape
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_xstart": pred_xstart,
        }


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def get_beta_schedule(beta_schedule="linear", *, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000):
    """
    This is the deprecated API for creating beta schedules.

    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def _extract_into_tensor(arr, timesteps, broadcast_shape):
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

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["model"])
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
def val(model, epoch, val_dataloader, args):
    model.eval()
    tokenizer = clip._tokenizer
    print(f">>> Evaling epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(val_dataloader), desc=args.tag)
    result_all = []
    cos_pre = torch.tensor(0.0).cuda()  # generate text emb with gt image emb
    cos_pre_self = torch.tensor(0.0).cuda()  # generate text emb with 5 gt text emb mean cosine sim
    for idx, (image, image_path, captions) in enumerate(val_dataloader):
        # 
        image = image.cuda(non_blocking=True)
        prefix_embed = model.module.image_encode(image)
        # prefix_embed = model.module.clip_project(prefix)
        if args.use_beam_search:
            assert False, "Not check beam search for now"
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
        generated_text_prefix_cpu = generated_text_prefix.cpu()
        torch.cuda.synchronize()
        progress.update()
        r = [{'image_id': _image_path, 'feature': _generated_text_prefix} for _image_path, _generated_text_prefix in zip(image_path, generated_text_prefix_cpu)]
        result_all.extend(r)
        # valiation metrics
        for sample_idx, img_captions in tqdm(enumerate(captions)):
            per_text_embedding = []
            for caption in img_captions:
                tokens = torch.cat((torch.tensor(tokenizer.encode('<|startoftext|>')), 
                                    torch.tensor(tokenizer.encode(caption)), 
                                    torch.tensor(tokenizer.encode('<|endoftext|>'))),
                                    dim=0)
                text_embedding = model.module.text_encode(tokens.unsqueeze(0))
                per_text_embedding.append(text_embedding)
            per_text_embedding = torch.stack(per_text_embedding, dim=0) # 5*512
            
            cos_pre_self += nnf.cosine_similarity(per_text_embedding, generated_text_prefix[sample_idx].unsqueeze(0).repeat(per_text_embedding.size(0), 1), dim=1, eps=1e-6).mean() / generated_text_prefix.size(0)
        cos_pre += nnf.cosine_similarity(prefix_embed, generated_text_prefix, dim=1, eps=1e-6).mean()   
    progress.close()
    os.makedirs('.cache', exist_ok=True)
    pickle.dump(result_all, open(f".cache/tmp-results-{dist.get_rank()}.pkl", "wb"))
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        result_all = []
        for i in range(dist.get_world_size()):
            part_result = pickle.load(open(f".cache/tmp-results-{i}.pkl", "rb"))
            result_all.extend(part_result)
        pickle.dump(result_all, open(os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}-results.pkl"), "wb"))
    if dist.get_rank() == 0:
        wandb.log({'cos_pre': cos_pre.item()/ len(val_dataloader), 'cos_pre_self': cos_pre_self.item()/ len(val_dataloader)})
    return result_all


def sample_time(b, num_timesteps, device):
    t = torch.randint(0, num_timesteps, (b,), device=device).long()
    pt = torch.ones_like(t).float() / num_timesteps
    return t, pt


def train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args,
          output_dir: str = ".", output_prefix: str = ""):
    model.train()
    num_steps = len(train_dataloader)
    loss_fn = torch.nn.MSELoss()

    train_dataloader.sampler.set_epoch(epoch)
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(train_dataloader), desc=output_prefix)
    for idx, (prefix, tokens_feature) in enumerate(train_dataloader):
        # prefix is raw images
        prefix = prefix.cuda(non_blocking=True)
        tokens_feature = tokens_feature.cuda(non_blocking=True)
        b, device = prefix.size()[0], prefix.device
        t, pt = sample_time(b, torch.tensor(args.time_step).to(torch.int), device)
        with amp.autocast(enabled=args.enable_amp):
            outputs, gt_feature = model(prefix, tokens_feature, t=t)
        logits = outputs.logits[:,-1,:]
        loss = loss_fn(logits, gt_feature.to(dtype=logits.dtype))
        cosine = nnf.cosine_similarity(logits, gt_feature, dim=1, eps=1e-6).mean()  
        optimizer.zero_grad()
        scaler.scale(loss).backward() #loss.backward()
        scaler.step(optimizer) #optimizer.step()
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()
        progress.set_postfix({"loss": loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        if dist.get_rank() == 0:
            wandb.log({'train_loss': loss.item(),
                       'train_cosine': cosine.item(),
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
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--time_step', type=int, default=1000)
    parser.add_argument('--beta_sche', type=str, default='linear', help='the schedule on beta')
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
        run = wandb.init(
            id=args.tag,
            name=args.tag,
            entity='msravcg',
            project='diffuseIC',
            job_type='coco',
            config=args,
        )
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, Timestep=args.time_step)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type, Timestep=args.time_step)
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

    dataset = ClipCocoDataset(args.data_root, args.data, normalize_prefix=args.normalize_prefix)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=args.bs, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)

    val_dataset = ClipCocoValDataset(args.data_root)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=False)

    lr_args = {"LR_SCHEDULER_NAME": "cosine", "EPOCHS": args.epochs, "WARMUP_EPOCHS": 5, "MIN_LR": 1e-6, "WARMUP_LR": 1e-7}
    lr_scheduler = build_scheduler(lr_args, optimizer, len(train_dataloader))
    
    for epoch in range(args.epochs):
        _ = train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args, output_dir=args.out_dir, output_prefix=args.tag)
        result = val(model, epoch, val_dataloader, args)
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if dist.get_rank() == 0:
                torch.save(
                    {'model':model.module.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()},
                    os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}.pt"),
                )

if __name__ == '__main__':
    # command:  python -m torch.distributed.launch --nproc_per_node 4 train.py --data ./oscar_split_ViT-B_32_train_512.pkl --out_dir ./output --bs 32
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
    setup_for_distributed(args.local_rank == 0) ##### HERE

    seed = dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args)
