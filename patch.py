
import torch
import math
from typing import Type, Dict, Any, Tuple, Callable
from typing import Optional
import torch.nn as nn
import types
from . import merge
from .utils import isinstance_str, init_generator
from .cache import SimilarityCache

block_call_count=0
block_list=[]
def track_blocks_hook(module, input, output):
    global block_call_count, block_list
    block_call_count += 1
    block_list.append(module)

block_counter = 0

def wrap_attn_with_block_label(attn_module, block_label: str):
    orig_forward = attn_module.forward
    def wrapped_forward(*args, **kwargs):
        merge.CURRENT_BLOCK_NAME = block_label
        return orig_forward(*args, **kwargs)
    attn_module.forward = wrapped_forward


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any], timestep: int) -> Tuple[Callable, ...]:
    if "block_counter" not in tome_info:
        tome_info["block_counter"] = 0  

    tome_info["block_counter"] += 1

    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    if tome_info["block_counter"] % 48 < 16:  # 앞 16번
        sx, sy = 2,2
    elif 16 <= tome_info["block_counter"] % 48 < 32:  # 중간 16번
        sx, sy = 16,16
    else:  # 뒤 16번
        sx, sy = 2,2

    a = tome_info["block_counter"]
    if downsample <= args["max_downsample"]:
    
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        
        r = int(x.shape[1] * args["ratio"])

        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])

        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]

        cache_state = tome_info.setdefault("cache_state", {"tick": 0, "last_recompute_tick": -1})
        cache_state["tick"] += 1
        tick = cache_state["tick"]

        # adaptive schedule: 초반 p_min, 후반 p_max
        p_min = tome_info["args"].get("cache_p_min", 3)
        p_max = tome_info["args"].get("cache_p_max", 20)
        grow = tome_info["args"].get("cache_grow_steps", 200)
        alpha = min(1.0, tick / float(max(1, grow)))
        period = int(round(p_min + (p_max - p_min) * alpha))

        # 재계산 여부
        need_recompute = (tick - cache_state["last_recompute_tick"] >= period)
        block_label = "block" + str(tome_info["block_counter"])

        m, u = merge.bipartite_soft_matching_max_cosine2d(
        x, w, h, sx, sy, r,
        text_embedding=args["text_embedding"],
        text_weight=args["text_weight"],
        final_layer_norm=args.get("final_layer_norm", None),
        q_proj=args["q_proj"], k_proj=args["k_proj"],
        block_name=f"block{tome_info['block_counter']}",
        step=(tick if need_recompute else cache_state["last_recompute_tick"])
)

        
        if need_recompute:
            cache_state["last_recompute_tick"] = tick
        # m, u = merge.bipartite_soft_matching_random2d(x, w, h, sx, sy, r,
        #                                                generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good

def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock

def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info, timestep)
            # if timestep is None:
            #     timestep = self._tome_info.get("current_timestep", 159)
            # print(f"after timestep: {timestep}")
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)
              
            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        if len(args) > 1 and isinstance(args[1], torch.Tensor):
            module._tome_info["current_timestep"] = args[1].item()
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))

def _wrap_attn_with_block_label(attn_module, block_label: str):
    if not hasattr(attn_module, "_orig_forward"):
        attn_module._orig_forward = attn_module.forward

        def _wrapped_forward(*args, **kwargs):
            merge.CURRENT_BLOCK_NAME = block_label
            return attn_module._orig_forward(*args, **kwargs)
        attn_module.forward = _wrapped_forward



def _label_block_attentions(block, prefix: str):
    """
    diffusers 버전별로 서로 다른 필드를 대응:
    - SD1.x 계열: block.attentions (list of Attention)
    - SDXL/최근: block.transformer_blocks[i].attn1/attn2
    """
    if hasattr(block, "attentions"):
        for i, attn in enumerate(block.attentions):
            _wrap_attn_with_block_label(attn, f"{prefix}.attn{i}")

    if hasattr(block, "transformer_blocks"):
        for i, tblk in enumerate(block.transformer_blocks):
            if hasattr(tblk, "attn1"):
                _wrap_attn_with_block_label(tblk.attn1, f"{prefix}.attn{i}")
            if hasattr(tblk, "attn2"):
                _wrap_attn_with_block_label(tblk.attn2, f"{prefix}.cross{i}")


def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        text_embedding: torch.Tensor = None,
        final_layer_norm :torch.nn.LayerNorm = None, 
        text_weight: float = 0.7,
        q_proj: Optional[nn.Linear] = None,
        k_proj: Optional[nn.Linear] = None
        ):

    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model
    

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "text_embedding": text_embedding,
            "final_layer_norm":final_layer_norm,
            "text_weight": text_weight,
            "q_proj": q_proj,
            "k_proj": k_proj
        }
    }
    hook_tome_model(diffusion_model)
 
    for name, module in diffusion_model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module._tome_block_name = name
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model
