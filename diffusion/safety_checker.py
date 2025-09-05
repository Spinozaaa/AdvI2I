# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
import torchvision
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())

def get_safety_checker_loss(pipe, image, device, dtype, safety_checker):
    image = (image / 2 + 0.5).clamp(0, 1)
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    re = torchvision.transforms.Resize([224, 224])
    normlize = torchvision.transforms.Normalize(clip_mean, clip_std)
    image_nat_re = re(image)
    clip_input = normlize(image_nat_re)
    sc_loss = safety_checker(clip_input=clip_input.to(device, dtype))
    return sc_loss

class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig, orig_sc):
        super().__init__(config)

        self.vision_model = orig_sc.vision_model
        self.visual_projection = orig_sc.visual_projection

        self.concept_embeds = orig_sc.concept_embeds
        self.special_care_embeds = orig_sc.special_care_embeds

        self.concept_embeds_weights = orig_sc.concept_embeds_weights
        self.special_care_embeds_weights = orig_sc.special_care_embeds_weights

    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        cos_loss = (torch.sum(torch.clamp(cos_dist, min=0), dim=1)
                    + torch.sum(torch.clamp(special_cos_dist, min=0), dim=1))/ (cos_dist.shape[1] + special_cos_dist.shape[1])
        return torch.mean(cos_loss, dim=0)

    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # special_scores = special_scores.round(decimals=3)
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        images[has_nsfw_concepts] = 0.0  # black image

        return images, has_nsfw_concepts
