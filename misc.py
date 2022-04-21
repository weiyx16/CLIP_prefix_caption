import importlib
import random
import numpy as np
import torch
import warnings
import os
from tqdm import tqdm, trange
import torch.nn.functional as nnf

def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def merge_opts_to_config(config, opts):
    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            # print(nl)
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2):
            name = opts[2*i]
            value = opts[2*i+1]
            config = modify_dict(config, name.split('.'), value)
    return config 

def modify_config_for_debug(config):
    config['dataloader']['num_workers'] = 0
    config['dataloader']['batch_size'] = 1
    return config



def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']
        
        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    # format the numbers
    def format_number(num):
        K = 2**10
        M = 2**20
        G = 2**30
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 2)
        else:
            uint = ''
        
        return '{}{}'.format(num, uint)
    
    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)
    
    format_dict(parameters)
    return parameters


def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)
        else:
           ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
 
    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls

def get_all_file(dir, end_with='.h5'):
    if isinstance(end_with, str):
        end_with = [end_with]
    filenames = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            for ew in end_with:
                if f.endswith(ew):
                    filenames.append(os.path.join(root, f))
                    break
    return filenames


def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]
    return sub_dirs


def get_model_buffer(model):
    state_dict = model.state_dict()
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]
    return buffers_


def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '<|endoftext|>',
):
    # embed: image feature, we only need to inference once
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in range(entry_count):
            generated = model.module.bos_embedding.unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=embed.size(0), dim=0) # bs, 1, dim
            stop_signal = torch.zeros(embed.size(0)).to(torch.bool).to(device)
            for i in range(entry_length):
                outputs = model.module.gpt(inputs_embeds=generated, adaln=embed)
                logits = outputs.logits
                # we only need the last (newest) token, so using -1 will be ok.
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0) # bs, vocab
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0 # incase of one word with > top_p prob

                # indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # logits[:, indices_to_remove] = filter_value
                for ele in range(logits.size(0)):
                    logits[ele, sorted_indices[ele, sorted_indices_to_remove[ele]]] = filter_value
                next_token = torch.argmax(logits, -1)
                next_token_embed = model.module.gpt.transformer.wte(next_token).unsqueeze(1) # bs, 1, dim
                next_token = next_token.unsqueeze(1) # bs, 1
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                # stop in parallel one...
                # one case: if all sentences appear eos, then we stop
                stop_signal = stop_signal | (stop_token_index == next_token).squeeze()
                if torch.all(stop_signal):
                    break
            output_list = list(tokens.cpu().numpy()) # bs, entry_length
            output_text =[tokenizer.decode(_output_list) for _output_list in output_list]
            generated_list.append(output_text)

    return generated_list[0]


# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
# from: https://github.com/microsoft/Oscar/blob/4788a7425cd0f9861ea80fed79528abbb72eb169/oscar/utils/caption_evaluate.py

from collections import OrderedDict, defaultdict
import json
import numpy as np
import os.path as op
from pprint import pprint
import torch
import re
import subprocess
import tempfile
import time
from typing import Dict, Optional
import torch.distributed as dist
from captioneval.coco_caption.pycocotools.coco import COCO
from captioneval.coco_caption.pycocoevalcap.eval import COCOEvalCap
from captioneval.cider.pyciderevalcap.ciderD.ciderD import CiderD


def evaluate_on_nocaps(split, predict_file, data_dir='data/nocaps/', evaluate_file=None):
    '''
    NOTE: Put the auth file in folder ~/.evalai/
    '''
    if not evaluate_file:
        evaluate_file = op.splitext(predict_file)[0] + '.eval.json'
    if op.isfile(evaluate_file):
        print('{} already exists'.format(evaluate_file))
        with open(evaluate_file, 'r') as fp:
            metrics = json.load(fp)
        return metrics

    image_info_file = op.join(data_dir,
            'nocaps_{}_image_info.json'.format(split))
    image_info = json.load(open(image_info_file))
    open_image_id2id = {}
    for it in image_info['images']:
        open_image_id2id[it['open_images_id']] = it['id']
    predictions = []
    cap_id = 0
    with open(predict_file, 'r') as fp:
        for line in fp:
            p = line.strip().split('\t')
            predictions.append(
                    {'image_id': open_image_id2id[p[0]],
                    'caption': json.loads(p[1])[0]['caption'],
                    'id': cap_id})
            cap_id += 1
    if split == 'test':
        print('Are you sure to submit test split result at: {}'.format(predict_file))
        import ipdb;ipdb.set_trace()
    nocapseval = NocapsEvaluator(phase=split)
    metrics = nocapseval.evaluate(predictions)
    pprint(metrics)
    with open(evaluate_file, 'w') as fp:
        json.dump(metrics, fp)
    return metrics


def evaluate_on_coco_caption(results, res_file, label_file, outfile=None):
    """
        # ref to fake example: https://github.com/LuoweiZhou/coco-caption/tree/de6f385503ac9a4305a1dcdc39c02312f9fa13fc
    """
    assert label_file.endswith('.json')
    parsed_res = []
    for result in results:
        id = int(result['image_id'].split('.')[0].split('_')[2])
        # cap = result['result'].replace('<|startoftext|>', '').replace('<|endoftext|>', '').replace('!', '').replace(' .', '.').strip() #result["ground truth"][0]
        cap = str(result['result'].split('<|endoftext|>')[0].strip())
        parsed_res.append({"image_id": id, "caption": cap})
    if ((dist.is_initialized() or dist.is_available()) and int(dist.get_rank()) == 0) or not dist.is_available():
        json.dump(parsed_res, open(res_file, 'w'))

    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file)
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    result = {k: v*100 for k,v in result.items()}
    if not outfile:
        print(result)
    else:
        if ((dist.is_initialized() or dist.is_available()) and int(dist.get_rank()) == 0) or not dist.is_available():
            with open(outfile, 'w') as fp:
                json.dump(result, fp, indent=4)
    return result


def convert_tsv_to_coco_format(res_tsv, outfile,
        sep='\t', key_col=0, cap_col=1):
    results = []
    with open(res_tsv) as fp:
        for line in fp:
            parts = line.strip().split(sep)
            key = parts[key_col]
            if cap_col < len(parts):
                caps = json.loads(parts[cap_col])
                assert len(caps) == 1, 'cannot evaluate multiple captions per image'
                cap = caps[0].get('caption', '')
            else:
                # empty caption generated
                cap = ""
            results.append(
                    {'image_id': key,
                    'caption': cap}
                    )
    with open(outfile, 'w') as fp:
        json.dump(results, fp)


class ScstRewardCriterion(torch.nn.Module):
    CIDER_REWARD_WEIGHT = 1

    def __init__(self, cider_cached_tokens='corpus', baseline_type='greedy'):
        self.CiderD_scorer = CiderD(df=cider_cached_tokens)
        assert baseline_type in ['greedy', 'sample']
        self.baseline_type = baseline_type
        self._cur_score = None
        super().__init__()

    def forward(self, gt_res, greedy_res, sample_res, sample_logprobs):
        batch_size = len(gt_res)
        sample_res_size = len(sample_res)
        seq_per_img = sample_res_size // batch_size

        gen_res = []
        gen_res.extend(sample_res)
        gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        if self.baseline_type == 'greedy':
            assert len(greedy_res) == batch_size
            gen_res.extend(greedy_res)
            gt_idx.extend([i for i in range(batch_size)])

        scores = self._calculate_eval_scores(gen_res, gt_idx, gt_res)

        if self.baseline_type == 'greedy':
            baseline = scores[-batch_size:][:, np.newaxis]
        else:
            sc_ = scores.reshape(batch_size, seq_per_img)
            baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        # sample - baseline
        reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        self._cur_score = reward.mean()
        reward = reward - baseline
        reward = reward.reshape(sample_res_size)

        reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        loss = - sample_logprobs * reward
        loss = loss.mean()
        return loss

    def get_score(self):
        return self._cur_score

    def _calculate_eval_scores(self, gen_res, gt_idx, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [self._wrap_sentence(gen_res[i])]

        gts = OrderedDict()
        gt_res_ = [
            [self._wrap_sentence(gt_res[i][j]) for j in range(len(gt_res[i]))]
                for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[gt_idx[i]]

        res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
        _, batch_cider_scores = self.CiderD_scorer.compute_score(gts, res_)
        scores = self.CIDER_REWARD_WEIGHT * batch_cider_scores
        return scores

    @classmethod
    def _wrap_sentence(self, s):
        # ensure the sentence ends with <eos> token
        # in order to keep consisitent with cider_cached_tokens
        r = s.strip()
        if r.endswith('.'):
            r = r[:-1]
        r += ' <eos>'
        return r


class NocapsEvaluator(object):
    r"""
    Code from https://github.com/nocaps-org/updown-baseline/blob/master/updown/utils/evalai.py
    A utility class to submit model predictions on nocaps splits to EvalAI, and retrieve model
    performance based on captioning metrics (such as CIDEr, SPICE).
    Extended Summary
    ----------------
    This class and the training script together serve as a working example for "EvalAI in the
    loop", showing how evaluation can be done remotely on privately held splits. Annotations
    (captions) and evaluation-specific tools (e.g. `coco-caption <https://www.github.com/tylin/coco-caption>`_)
    are not required locally. This enables users to select best checkpoint, perform early
    stopping, learning rate scheduling based on a metric, etc. without actually doing evaluation.
    Parameters
    ----------
    phase: str, optional (default = "val")
        Which phase to evaluate on. One of "val" or "test".
    Notes
    -----
    This class can be used for retrieving metrics on both, val and test splits. However, we
    recommend to avoid using it for test split (at least during training). Number of allowed
    submissions to test split on EvalAI are very less, and can exhaust in a few iterations! However,
    the number of submissions to val split are practically infinite.
    """

    def __init__(self, phase: str = "val"):

        # Constants specific to EvalAI.
        self._challenge_id = 355
        self._phase_id = 742 if phase == "val" else 743

    def evaluate(
        self, predictions, iteration: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        r"""
        Take the model predictions (in COCO format), submit them to EvalAI, and retrieve model
        performance based on captioning metrics.
        Parameters
        ----------
        predictions: List[Prediction]
            Model predictions in COCO format. They are a list of dicts with keys
            ``{"image_id": int, "caption": str}``.
        iteration: int, optional (default = None)
            Training iteration where the checkpoint was evaluated.
        Returns
        -------
        Dict[str, Dict[str, float]]
            Model performance based on all captioning metrics. Nested dict structure::
                {
                    "B1": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-1
                    "B2": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-2
                    "B3": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-3
                    "B4": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-4
                    "METEOR": {"in-domain", "near-domain", "out-domain", "entire"},
                    "ROUGE-L": {"in-domain", "near-domain", "out-domain", "entire"},
                    "CIDEr": {"in-domain", "near-domain", "out-domain", "entire"},
                    "SPICE": {"in-domain", "near-domain", "out-domain", "entire"},
                }
        """
        # Save predictions as a json file first.
        _, predictions_filename = tempfile.mkstemp(suffix=".json", text=True)
        with open(predictions_filename, "w") as f:
            json.dump(predictions, f)

        submission_command = (
            f"evalai challenge {self._challenge_id} phase {self._phase_id} "
            f"submit --file {predictions_filename}"
        )

        submission_command_subprocess = subprocess.Popen(
            submission_command.split(),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # This terminal output will have submission ID we need to check.
        submission_command_stdout = submission_command_subprocess.communicate(input=b"N\n")[
            0
        ].decode("utf-8")

        submission_id_regex = re.search("evalai submission ([0-9]+)", submission_command_stdout)
        try:
            # Get an integer submission ID (as a string).
            submission_id = submission_id_regex.group(0).split()[-1]  # type: ignore
        except:
            # Very unlikely, but submission may fail because of some glitch. Retry for that.
            return self.evaluate(predictions)

        if iteration is not None:
            print(f"Submitted predictions for iteration {iteration}, submission id: {submission_id}.")
        else:
            print(f"Submitted predictions, submission_id: {submission_id}")

        # Placeholder stdout for a pending submission.
        result_stdout: str = "The Submission is yet to be evaluated."
        num_tries: int = 0

        # Query every 10 seconds for result until it appears.
        while "CIDEr" not in result_stdout:

            time.sleep(10)
            result_stdout = subprocess.check_output(
                ["evalai", "submission", submission_id, "result"]
            ).decode("utf-8")
            num_tries += 1

            # Raise error if it takes more than 5 minutes.
            if num_tries == 30:
                raise ConnectionError("Unable to get results from EvalAI within 5 minutes!")

        # Convert result to json.
        metrics = json.loads(result_stdout, encoding="utf-8")

        # keys: {"in-domain", "near-domain", "out-domain", "entire"}
        # In each of these, keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        metrics = {
            "in-domain": metrics[0]["in-domain"],
            "near-domain": metrics[1]["near-domain"],
            "out-domain": metrics[2]["out-domain"],
            "entire": metrics[3]["entire"],
        }

        # Restructure the metrics dict for better tensorboard logging.
        # keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        # In each of these, keys: keys: {"in-domain", "near-domain", "out-domain", "entire"}
        flipped_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        for key, val in metrics.items():
            for subkey, subval in val.items():
                flipped_metrics[subkey][key] = subval

        return flipped_metrics
