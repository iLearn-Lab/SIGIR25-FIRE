from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer


@dataclass
class BatchEncodingWithImages:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor


class FiREModelWrapper(torch.nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        model_name = cfg['model']['base_model']
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.eos_token = '<|end|>'

        adapter_path = cfg['model'].get('adapter_path')
        lora_cfg = cfg['model']['lora']
        if adapter_path:
            self.model.vlm.lang_model = PeftModel.from_pretrained(self.model.vlm.lang_model, adapter_path)
        elif lora_cfg.get('enabled', True):
            peft_cfg = LoraConfig(
                r=lora_cfg['r'],
                lora_alpha=lora_cfg['alpha'],
                lora_dropout=lora_cfg['dropout'],
                target_modules=lora_cfg['target_modules'],
                task_type=TaskType.CAUSAL_LM,
            )
            self.model.vlm.lang_model = get_peft_model(self.model.vlm.lang_model, peft_cfg)

        if cfg['model'].get('freeze_base_model', True):
            for name, param in self.model.named_parameters():
                param.requires_grad = 'lora' in name.lower()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, output_dir: str):
        if hasattr(self.model.vlm.lang_model, 'save_pretrained'):
            self.model.vlm.lang_model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def tokenize(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    def _prepare_inputs_for_forward(
        self,
        vision_tokens: List[List[torch.Tensor]],
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        lang_model = self.model.vlm.lang_model
        media_token_id = self.model.vlm.media_token_id
        image_aspect_ratio = self.model.vlm.image_aspect_ratio
        pad_token_id = self.model.vlm.pad_token_id
        num_tokens_per_vis = self.model.vlm.num_tokens_per_vis

        lang_embeds = lang_model.get_input_embeddings()(lang_x)
        batch_size = lang_x.shape[0]
        has_labels = labels is not None
        multimodal_embeds = []
        multimodal_attention_mask = []
        multimodal_labels = [] if has_labels else None

        for i in range(batch_size):
            image_token_idxs = torch.where(lang_x[i] == media_token_id)[0]
            if len(image_token_idxs) == 0:
                multimodal_embeds.append(lang_embeds[i].clone())
                multimodal_attention_mask.append(attention_mask[i].clone())
                if has_labels:
                    multimodal_labels.append(labels[i].clone())
                continue

            new_embed = lang_embeds[i].clone()
            new_attention_mask = attention_mask[i].clone()
            if has_labels:
                new_label = labels[i].clone()

            offset = 0
            for img_num, img_idx in enumerate(image_token_idxs):
                if image_aspect_ratio == 'anyres':
                    num_vis_tokens = vision_tokens[i][img_num].shape[0]
                    vis_attention_mask = torch.ones(num_vis_tokens, dtype=torch.long, device=attention_mask.device)
                    if vision_attention_mask is not None:
                        vis_attention_mask = vision_attention_mask[i]
                else:
                    num_vis_tokens = num_tokens_per_vis
                    vis_attention_mask = torch.ones(num_vis_tokens, dtype=torch.long, device=attention_mask.device)

                insert_at = img_idx + offset
                pre = new_embed[:insert_at]
                post = new_embed[insert_at + 1 :]
                vis = vision_tokens[i][img_num].to(pre.device)
                new_embed = torch.cat([pre, vis, post], dim=0)

                mask_pre = new_attention_mask[:insert_at]
                mask_post = new_attention_mask[insert_at + 1 :]
                new_attention_mask = torch.cat([mask_pre, vis_attention_mask, mask_post], dim=0)

                if has_labels:
                    label_pre = new_label[:insert_at]
                    label_post = new_label[insert_at + 1 :]
                    ignore = torch.full((num_vis_tokens,), -100, dtype=new_label.dtype, device=new_label.device)
                    new_label = torch.cat([label_pre, ignore, label_post], dim=0)
                offset += num_vis_tokens - 1

            multimodal_embeds.append(new_embed)
            multimodal_attention_mask.append(new_attention_mask)
            if has_labels:
                multimodal_labels.append(new_label)

        max_len = max(x.shape[0] for x in multimodal_embeds)
        padded_embeds = []
        padded_masks = []
        padded_labels = [] if has_labels else None
        for i in range(batch_size):
            embed = multimodal_embeds[i]
            mask = multimodal_attention_mask[i]
            pad_len = max_len - embed.shape[0]
            if pad_len > 0:
                pad_embed = torch.zeros((pad_len, embed.shape[1]), dtype=embed.dtype, device=embed.device)
                pad_mask = torch.zeros((pad_len,), dtype=mask.dtype, device=mask.device)
                embed = torch.cat([pad_embed, embed], dim=0)
                mask = torch.cat([pad_mask, mask], dim=0)
                if has_labels:
                    pad_label = torch.full((pad_len,), -100, dtype=multimodal_labels[i].dtype, device=multimodal_labels[i].device)
                    multimodal_labels[i] = torch.cat([pad_label, multimodal_labels[i]], dim=0)
            padded_embeds.append(embed)
            padded_masks.append(mask)
            if has_labels:
                padded_labels.append(multimodal_labels[i])

        outputs = {
            'inputs_embeds': torch.stack(padded_embeds, dim=0),
            'attention_mask': torch.stack(padded_masks, dim=0),
        }
        if has_labels:
            outputs['labels'] = torch.stack(padded_labels, dim=0)
        return outputs

    def _encode_multimodal(self, tokenized: Dict[str, torch.Tensor], images, image_sizes, last_n_tokens: int = 10) -> torch.Tensor:
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            vision_tokens = self.model.vlm.get_vision_tokens(images, image_sizes, self.device)
        for token_group in vision_tokens:
            for j, token in enumerate(token_group):
                token_group[j] = token.to(self.device)
        packed_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
        )
        outputs = self.model.vlm.lang_model(**packed_inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -last_n_tokens:, :].mean(dim=1)
        return torch.nn.functional.normalize(hidden, p=2, dim=-1)

    def encode_gallery(self, tokenized: Dict[str, torch.Tensor], images, image_sizes) -> torch.Tensor:
        return self._encode_multimodal(tokenized, images, image_sizes, last_n_tokens=10)

    def encode_query(self, tokenized: Dict[str, torch.Tensor], images, image_sizes) -> torch.Tensor:
        return self._encode_multimodal(tokenized, images, image_sizes, last_n_tokens=10)

    def encode_target(self, tokenized: Dict[str, torch.Tensor], images, image_sizes) -> torch.Tensor:
        return self._encode_multimodal(tokenized, images, image_sizes, last_n_tokens=10)
