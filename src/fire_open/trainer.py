from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .datasets import CIRREvalDataset, FashionIQEvalDataset, GenericGalleryDataset, GenericQueryDataset, GenericTrainPairDataset
from .losses import retrieval_loss



def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def make_train_collator(model_wrapper, prompt_cfg: Dict, max_length: int):
    tokenizer = model_wrapper.tokenizer

    def collate(batch: List[Dict]):
        ref_images = [item['reference_images'] for item in batch]
        ref_sizes = [item['reference_sizes'] for item in batch]
        tgt_images = [item['target_images'] for item in batch]
        tgt_sizes = [item['target_sizes'] for item in batch]

        query_texts = []
        target_texts = []
        for item in batch:
            reference_caption = item['reference_caption'].strip() or 'the reference image'
            target_caption = item['target_caption'].strip() or 'the target image'
            query_texts.append(
                prompt_cfg['train_query']
                .replace('{reference_caption}', reference_caption)
                .replace('{modification}', item['modification'].strip())
            )
            target_texts.append(prompt_cfg['train_target'].replace('{target_caption}', target_caption))

        query_tokens = tokenizer(query_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        target_tokens = tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        return {
            'reference_images': ref_images,
            'reference_sizes': ref_sizes,
            'target_images': tgt_images,
            'target_sizes': tgt_sizes,
            'query_tokens': query_tokens,
            'target_tokens': target_tokens,
            'target_ids': [item['target_id'] for item in batch],
        }

    return collate



def make_gallery_collator(model_wrapper, gallery_prompt: str, max_length: int):
    tokenizer = model_wrapper.tokenizer

    def collate(batch):
        images, image_sizes, image_ids = zip(*batch)
        prompts = [gallery_prompt for _ in image_ids]
        tokens = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        return list(images), list(image_sizes), list(image_ids), tokens

    return collate



def make_query_collator(model_wrapper, query_prompt: str, max_length: int):
    tokenizer = model_wrapper.tokenizer

    def collate(batch: List[Dict]):
        ref_images = [item['reference_images'] for item in batch]
        ref_sizes = [item['reference_sizes'] for item in batch]
        prompts = [query_prompt.replace('{modification}', item['modification'].strip()) for item in batch]
        query_tokens = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        return {
            'reference_images': ref_images,
            'reference_sizes': ref_sizes,
            'query_tokens': query_tokens,
            'target_ids': [item['target_id'] for item in batch],
            'reference_ids': [item['reference_id'] for item in batch],
            'exclude_ids': [item.get('exclude_ids', []) for item in batch],
            'query_ids': [item.get('query_id') for item in batch],
        }

    return collate



def train(cfg: Dict, model_wrapper) -> Path:
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_wrapper.to(device)
    model_wrapper.train()

    dataset = GenericTrainPairDataset(
        metadata_path=cfg['data']['train_metadata'],
        image_root=cfg['data']['image_root'],
        preprocessor=model_wrapper.image_processor,
        max_short_edge=cfg['data']['max_short_edge'],
    )
    collate_fn = make_train_collator(model_wrapper, cfg['prompts'], cfg['training']['max_length'])
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['training']['per_device_train_batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        collate_fn=collate_fn,
    )

    trainable_params = [p for p in model_wrapper.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )
    total_steps = math.ceil(len(dataloader) / cfg['training']['gradient_accumulation_steps']) * cfg['training']['num_train_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg['training']['warmup_steps'],
        num_training_steps=max(total_steps, 1),
    )

    output_dir = Path(cfg['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(cfg['training']['num_train_epochs']):
        progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{cfg["training"]["num_train_epochs"]}')
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress):
            query_features = model_wrapper.encode_query(batch['query_tokens'], batch['reference_images'], batch['reference_sizes'])
            target_features = model_wrapper.encode_target(batch['target_tokens'], batch['target_images'], batch['target_sizes'])
            loss = retrieval_loss(
                query_features,
                target_features,
                scale=cfg['training']['loss_scale'],
                recall_weight_at_1=cfg['training']['recall_loss_weight_at_1'],
                recall_weight_at_5=cfg['training']['recall_loss_weight_at_5'],
            )
            loss = loss / cfg['training']['gradient_accumulation_steps']
            loss.backward()

            if (step + 1) % cfg['training']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % cfg['training']['logging_steps'] == 0:
                    progress.set_postfix({'loss': float(loss.detach().cpu())})
                if cfg['training']['save_every_steps'] and global_step % cfg['training']['save_every_steps'] == 0:
                    ckpt_dir = output_dir / f'checkpoint-{global_step}'
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model_wrapper.save(str(ckpt_dir))

    model_wrapper.save(str(output_dir))
    with (output_dir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return output_dir



def build_eval_datasets(cfg: Dict, model_wrapper):
    task = cfg['data']['task']
    if task == 'fashioniq':
        query_dataset = FashionIQEvalDataset(
            data_root=cfg['data']['image_root'],
            dress_type=cfg['data']['dress_type'],
            split=cfg['data']['split'],
            preprocessor=model_wrapper.image_processor,
            max_short_edge=cfg['data']['max_short_edge'],
        )
        gallery_dataset = query_dataset.gallery_dataset()
    elif task == 'cirr':
        query_dataset = CIRREvalDataset(
            data_root=cfg['data']['image_root'],
            split=cfg['data']['split'],
            preprocessor=model_wrapper.image_processor,
            max_short_edge=cfg['data']['max_short_edge'],
        )
        gallery_dataset = query_dataset.gallery_dataset()
    else:
        query_dataset = GenericQueryDataset(
            metadata_path=cfg['data']['query_metadata'],
            image_root=cfg['data']['image_root'],
            preprocessor=model_wrapper.image_processor,
            max_short_edge=cfg['data']['max_short_edge'],
        )
        gallery_dataset = GenericGalleryDataset(
            metadata_path=cfg['data']['gallery_metadata'],
            image_root=cfg['data']['image_root'],
            preprocessor=model_wrapper.image_processor,
            max_short_edge=cfg['data']['max_short_edge'],
        )
    return query_dataset, gallery_dataset



def compute_recall_at_k(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    target_ids: List,
    gallery_ids: List,
    ks: List[int],
    exclude_ids_per_query: List[List] | None = None,
) -> Dict[str, float]:
    sims = query_features @ gallery_features.T
    gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
    recalls = {k: 0 for k in ks}
    num_queries = len(target_ids)
    exclude_ids_per_query = exclude_ids_per_query or [[] for _ in range(num_queries)]

    for i, target_id in enumerate(target_ids):
        row = sims[i].clone()
        for excluded in exclude_ids_per_query[i]:
            if excluded in gallery_index:
                row[gallery_index[excluded]] = float('-inf')
        ranking = row.argsort(descending=True)
        gt_index = gallery_index.get(target_id, None)
        if gt_index is None:
            continue
        for k in ks:
            if gt_index in ranking[:k]:
                recalls[k] += 1

    return {f'Recall@{k}': recalls[k] / max(num_queries, 1) for k in ks}



def evaluate(cfg: Dict, model_wrapper) -> Dict[str, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_wrapper.to(device)
    model_wrapper.eval()

    query_dataset, gallery_dataset = build_eval_datasets(cfg, model_wrapper)
    gallery_collator = make_gallery_collator(model_wrapper, cfg['prompts']['gallery'], cfg['eval']['max_length'])
    query_collator = make_query_collator(model_wrapper, cfg['prompts']['eval_query'], cfg['eval']['max_length'])

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg['eval']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        collate_fn=gallery_collator,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg['eval']['batch_size'],
        shuffle=False,
        num_workers=cfg['data']['num_workers'],
        collate_fn=query_collator,
    )

    gallery_features = []
    gallery_ids = []
    with torch.no_grad():
        for images, image_sizes, image_ids, tokens in tqdm(gallery_loader, desc='Encoding gallery'):
            feats = model_wrapper.encode_gallery(tokens, images, image_sizes)
            gallery_features.append(feats.cpu())
            gallery_ids.extend(image_ids)
    gallery_features = torch.cat(gallery_features, dim=0)

    query_features = []
    target_ids = []
    exclude_ids = []
    with torch.no_grad():
        for batch in tqdm(query_loader, desc='Encoding queries'):
            feats = model_wrapper.encode_query(batch['query_tokens'], batch['reference_images'], batch['reference_sizes'])
            query_features.append(feats.cpu())
            target_ids.extend(batch['target_ids'])
            exclude_ids.extend(batch['exclude_ids'])
    query_features = torch.cat(query_features, dim=0)

    metrics = compute_recall_at_k(
        query_features=query_features,
        gallery_features=gallery_features,
        target_ids=target_ids,
        gallery_ids=gallery_ids,
        ks=cfg['eval']['ks'],
        exclude_ids_per_query=exclude_ids if cfg['eval'].get('exclude_reference', True) else None,
    )
    return metrics
