from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


Record = Dict[str, Any]



def _load_json_or_jsonl(path: str | Path) -> List[Record]:
    path = Path(path)
    if path.suffix == '.jsonl':
        records: List[Record] = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if 'annotations' in data and isinstance(data['annotations'], list):
            return data['annotations']
        if 'data' in data and isinstance(data['data'], list):
            return data['data']
        return [{"id": k, **v} if isinstance(v, dict) else {"id": k, "value": v} for k, v in data.items()]
    raise TypeError(f'Unsupported metadata structure for {path}')



def _resize_if_needed(image: Image.Image, max_short_edge: int) -> Image.Image:
    width, height = image.size
    min_edge = min(width, height)
    if min_edge <= max_short_edge:
        return image
    scale = max_short_edge / float(min_edge)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)



def _resolve_image_path(image_root: str | Path | None, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    if image_root is None:
        return path
    return Path(image_root) / image_path



def load_preprocessed_image(image_path: str | Path, preprocessor, max_short_edge: int = 380):
    image = Image.open(image_path).convert('RGB')
    image = _resize_if_needed(image, max_short_edge=max_short_edge)
    image_size = [torch.tensor(image.size)]
    pixel_values = preprocessor([image], image_aspect_ratio='anyres')["pixel_values"]
    return [pixel_values], image_size


class GenericTrainPairDataset(Dataset):
    """Generic training metadata for FiRE-style pairwise training.

    Required fields per record:
      - reference_image
      - target_image
      - modification

    Optional fields:
      - reference_caption
      - target_caption
      - sample_id
    """

    def __init__(self, metadata_path: str | Path, image_root: str | Path | None, preprocessor, max_short_edge: int = 380):
        self.records = _load_json_or_jsonl(metadata_path)
        self.image_root = image_root
        self.preprocessor = preprocessor
        self.max_short_edge = max_short_edge

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        ref_path = _resolve_image_path(self.image_root, record['reference_image'])
        tgt_path = _resolve_image_path(self.image_root, record['target_image'])
        ref_images, ref_sizes = load_preprocessed_image(ref_path, self.preprocessor, self.max_short_edge)
        tgt_images, tgt_sizes = load_preprocessed_image(tgt_path, self.preprocessor, self.max_short_edge)
        return {
            'reference_images': ref_images,
            'reference_sizes': ref_sizes,
            'target_images': tgt_images,
            'target_sizes': tgt_sizes,
            'modification': record['modification'],
            'reference_caption': record.get('reference_caption', ''),
            'target_caption': record.get('target_caption', ''),
            'reference_id': record.get('reference_id', record['reference_image']),
            'target_id': record.get('target_id', record['target_image']),
            'sample_id': record.get('sample_id', idx),
        }


class GenericGalleryDataset(Dataset):
    """Gallery metadata with fields image_id and image_path."""

    def __init__(self, metadata_path: str | Path, image_root: str | Path | None, preprocessor, max_short_edge: int = 380):
        self.records = _load_json_or_jsonl(metadata_path)
        self.image_root = image_root
        self.preprocessor = preprocessor
        self.max_short_edge = max_short_edge

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image_path = record.get('image_path') or record.get('image') or record.get('path')
        image_id = record.get('image_id') or record.get('id') or image_path
        image_path = _resolve_image_path(self.image_root, image_path)
        images, image_sizes = load_preprocessed_image(image_path, self.preprocessor, self.max_short_edge)
        return images, image_sizes, image_id


class GenericQueryDataset(Dataset):
    """Query metadata for fair evaluation.

    Required fields per record:
      - reference_image
      - modification
      - target_id

    Optional fields:
      - query_id
      - exclude_ids (list)
    """

    def __init__(self, metadata_path: str | Path, image_root: str | Path | None, preprocessor, max_short_edge: int = 380):
        self.records = _load_json_or_jsonl(metadata_path)
        self.image_root = image_root
        self.preprocessor = preprocessor
        self.max_short_edge = max_short_edge

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        ref_path = _resolve_image_path(self.image_root, record['reference_image'])
        ref_images, ref_sizes = load_preprocessed_image(ref_path, self.preprocessor, self.max_short_edge)
        return {
            'reference_images': ref_images,
            'reference_sizes': ref_sizes,
            'modification': record['modification'],
            'target_id': record['target_id'],
            'reference_id': record.get('reference_id', record['reference_image']),
            'exclude_ids': record.get('exclude_ids', []),
            'query_id': record.get('query_id', idx),
        }


class FashionIQEvalDataset(Dataset):
    def __init__(self, data_root: str | Path, dress_type: str, split: str, preprocessor, max_short_edge: int = 380):
        self.data_root = Path(data_root)
        self.dress_type = dress_type
        self.split = split
        self.preprocessor = preprocessor
        self.max_short_edge = max_short_edge
        self.caption_records = _load_json_or_jsonl(self.data_root / 'captions' / f'cap.{dress_type}.{split}.json')
        split_name = 'val' if split in {'val', 'test'} else split
        with (self.data_root / 'image_splits' / f'split.{dress_type}.{split_name}.json').open('r', encoding='utf-8') as f:
            self.gallery_ids = json.load(f)

    def __len__(self) -> int:
        return len(self.caption_records)

    def __getitem__(self, idx: int):
        item = self.caption_records[idx]
        ref_id = item['candidate']
        target_id = item['target']
        modification = f"{item['captions'][0].strip('.?, ')} and {item['captions'][1].strip('.?, ')}"
        ref_path = self.data_root / self.dress_type / f'{ref_id}.jpg'
        ref_images, ref_sizes = load_preprocessed_image(ref_path, self.preprocessor, self.max_short_edge)
        return {
            'reference_images': ref_images,
            'reference_sizes': ref_sizes,
            'modification': modification,
            'target_id': target_id,
            'reference_id': ref_id,
            'exclude_ids': [ref_id],
            'query_id': idx,
        }

    def gallery_dataset(self) -> Dataset:
        records = [{'image_id': image_id, 'image_path': str(Path(self.dress_type) / f'{image_id}.jpg')} for image_id in self.gallery_ids]
        tmp_path = self.data_root / f'_tmp_gallery_{self.dress_type}_{self.split}.json'
        with tmp_path.open('w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False)
        return GenericGalleryDataset(tmp_path, self.data_root, self.preprocessor, self.max_short_edge)


class CIRREvalDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, preprocessor, max_short_edge: int = 380):
        self.data_root = Path(data_root)
        self.split = split
        self.preprocessor = preprocessor
        self.max_short_edge = max_short_edge
        captions_path = self.data_root / 'cirr' / 'captions' / f'cap.rc2.{split}.json'
        split_path = self.data_root / 'cirr' / 'image_splits' / f'split.rc2.{split}.json'
        self.records = _load_json_or_jsonl(captions_path)
        with split_path.open('r', encoding='utf-8') as f:
            self.path_dict = json.load(f)
        self.gallery_ids = list(self.path_dict.keys())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        item = self.records[idx]
        ref_id = item['reference']
        target_id = item.get('target_hard')
        modification = item['caption'].strip('.?, ')
        ref_path = self.data_root / self.path_dict[ref_id]
        ref_images, ref_sizes = load_preprocessed_image(ref_path, self.preprocessor, self.max_short_edge)
        exclude_ids = [ref_id]
        if 'img_set' in item and isinstance(item['img_set'], dict):
            members = item['img_set'].get('members', [])
            exclude_ids = list({*exclude_ids, *members})
        return {
            'reference_images': ref_images,
            'reference_sizes': ref_sizes,
            'modification': modification,
            'target_id': target_id,
            'reference_id': ref_id,
            'exclude_ids': exclude_ids,
            'query_id': item.get('pairid', idx),
        }

    def gallery_dataset(self) -> Dataset:
        records = [{'image_id': image_id, 'image_path': self.path_dict[image_id]} for image_id in self.gallery_ids]
        tmp_path = self.data_root / f'_tmp_gallery_cirr_{self.split}.json'
        with tmp_path.open('w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False)
        return GenericGalleryDataset(tmp_path, self.data_root, self.preprocessor, self.max_short_edge)
