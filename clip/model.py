"""
reference: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""

from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from clip.model_utils import ModifiedResNet, LayerNorm
from clip.vision_transformer_vifi import VisionTransformer
from clip.vision_transformer_tc import TCVisionTransformer
from clip.transformer import Transformer


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length    # text context length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            vision_model = design_details['vision_model']
            if vision_model == "VisionTransformer":
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif vision_model == "TCVisionTransformer":
                self.visual = TCVisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    num_frames=design_details['temporal_length'],
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            else:
                raise NotImplementedError

        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # added for custom attention module
        for name in ["q_proj", "k_proj", "v_proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                attr.weight.data = attr.weight.data.half()
                if attr.bias is not None:
                    attr.bias.data = attr.bias.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = design_details["context_length"]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64 # 8
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # vision positional embedding
    if design_details["positional_embedding_type"] == "joint":
        print(f"Expanding CLIP positional embedding for {design_details['temporal_length']} times")  # [197, 768] -> [T, 197, 768]
        positional_embedding_param = state_dict.pop("visual.positional_embedding")
        expanded_positional_embedding = positional_embedding_param.unsqueeze(0)
        expanded_positional_embedding = expanded_positional_embedding.repeat(design_details["temporal_length"], 1, 1)
        state_dict["visual.positional_embedding"] = expanded_positional_embedding

    # In case of using separate qkv projection weight
    if design_details["use_custom_attention"]:
        new_state_dict = state_dict.copy()
        for name in state_dict.keys():
            if name.startswith("visual.transformer.resblocks") and "attn" in name:
                if name.endswith("in_proj_weight"):
                    param = new_state_dict.pop(name)
                    new_state_dict[name.replace("in_proj_weight", "q_proj.weight")] = param[:768]
                    new_state_dict[name.replace("in_proj_weight", "k_proj.weight")] = param[768:768*2]
                    new_state_dict[name.replace("in_proj_weight", "v_proj.weight")] = param[768*2:]
                elif name.endswith("in_proj_bias"):
                    param = new_state_dict.pop(name)
                    new_state_dict[name.replace("in_proj_bias", "q_proj.bias")] = param[:768]
                    new_state_dict[name.replace("in_proj_bias", "k_proj.bias")] = param[768:768*2]
                    new_state_dict[name.replace("in_proj_bias", "v_proj.bias")] = param[768*2:]
        state_dict = new_state_dict

    convert_weights(model)

    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)

    return model.eval()
