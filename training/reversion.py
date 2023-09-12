import math
import random

import torch
import torch.nn.functional as F

from environment.src.reversion.templates.relation_words import relation_words
from environment.src.reversion.templates.stop_words import stop_words


def calculate_steer_loss(token_embedding,
                         input_ids,
                         placeholder_token_id,
                         stop_ids,
                         special_ids,
                         positive_ids,
                         temperature=0.07):
    """L_steer"""
    # compute input embeddings
    inputs_embeds = token_embedding(input_ids)  # (bs, 77, 768)
    positive_embeds = token_embedding(positive_ids)

    with torch.no_grad(
    ):  # no gradients from positive and negative embeds, only from <R>
        # compute entity embeds
        stop_mask = torch.isin(
            input_ids,
            torch.tensor(stop_ids + special_ids +
                         [placeholder_token_id]).cuda())  # (bs, 77)
        negative_embds = inputs_embeds[~stop_mask]  # (num_stop_tokens, 768)

        # remove bos and eos in positive embeddings
        stop_mask = torch.isin(positive_ids,
                               torch.tensor(special_ids).cuda())  # (bs, 77)
        positive_embeds = positive_embeds[
            ~stop_mask]  # (num_positive_tokens, 768), where num_positive_tokens = num_positives * bs

        # stack positives and negatives as a pn_block
        pn_embeds = torch.cat([positive_embeds, negative_embds], dim=0)
        pn_embeds_normalized = F.normalize(
            pn_embeds, p=2,
            dim=1)  # (num_positive_tokens+num_negative_tokens, 768)

    # compute relation embeds <R>
    relation_mask = (input_ids[0] == placeholder_token_id)  # (77)
    relation_embeds = inputs_embeds[0][relation_mask]  # (1, 768)
    relation_embeds_normalized = F.normalize(relation_embeds, p=2, dim=1)

    # compute Multi-Instance InfoNCE loss
    logits = torch.einsum('nc,mc->nm',
                          [relation_embeds_normalized, pn_embeds_normalized
                           ])  # (1, num_positive_tokens+num_negative_tokens)

    logits /= temperature
    nominator = torch.logsumexp(logits[:, :positive_embeds.shape[0]], dim=1)
    denominator = torch.logsumexp(logits, dim=1)

    return torch.mean(denominator - nominator)


def get_importance_sampling_probs(num_train_timesteps, scaled_cosine_alpha=0.5):
    print("Using Relation-Focal Importance Sampling")
    list_of_candidates = [x for x in range(num_train_timesteps)]
    prob_dist = [
        importance_sampling_fn(x, num_train_timesteps, scaled_cosine_alpha)
        for x in list_of_candidates
    ]
    prob_sum = 0
    # normalize the prob_list so that sum of prob is 1
    for i in prob_dist:
        prob_sum += i
    prob_dist = [x / prob_sum for x in prob_dist]
    
    return list_of_candidates, prob_dist


def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 - alpha * math.cos(math.pi * t / max_t))


def get_stop_special_ids(tokenizer):
    # stop words id
    expanded_stop_words = stop_words + relation_words  # add relation words to stop_words
    stop_ids = tokenizer(
        " ".join(expanded_stop_words),
        truncation=False,
        return_tensors="pt",
    ).input_ids[0].detach().tolist()

    # stop_ids = stop_ids + [tokenizer.bos_token_id, tokenizer.eos_token_id] # add special token ids to stop ids
    special_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id]

    return stop_ids, special_ids


def sample_positive_ids(tokenizer, num_positives):
    if num_positives > 0:
        positive_words = random.sample(relation_words, k=num_positives)
        positive_words_string = " ".join(positive_words)
        
        positive_ids = tokenizer(
            positive_words_string,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        
        return positive_ids

    raise ValueError("num_positives must be greater than 0")