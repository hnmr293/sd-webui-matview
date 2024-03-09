from typing import Any, Union, List, Dict, Callable

import torch
from torch import Tensor
from tqdm import tqdm
import gradio as gr

from modules import sd_models

from scripts.matviewlib.layer import Layer, LayerType, match_op, match_any
from scripts.matviewlib.lora import lora_path, lora2sd
from scripts.matviewlib.utils import try_to_int

def reload_models():
    sd_models.list_models()
    models = list_models()
    return [gr.update(choices=models), gr.update(choices=models)]

def load_model(
    model_name: Union[str,None],
    lora: bool = False,
) -> Dict[str,Tensor]:
    
    if model_name is None or len(model_name) == 0:
        raise ValueError('model was not found.')

    if not lora:
        model_info = sd_models.get_closet_checkpoint_match(model_name)
        if model_info is None:
            raise ValueError(f'model was not found: {model_name}')
        filename = model_info.filename
    else:
        filename = lora_path(model_name)

    return sd_models.read_state_dict(filename, map_location='cpu') # type: ignore

def list_models():
    return [''] + sd_models.checkpoint_tiles()

def retrieve_weights(
    state_dict: Dict[str,Tensor],
    lora_matmul: bool = False,
    filter: Union[Callable[[Layer],bool],None] = None
):
    target_layers: List[Layer] = []
    
    alphas: Dict[str,float] = dict()
    lora_up: Dict[str,Layer] = dict()
    lora_down: Dict[str, Layer] = dict()
    
    LT = LayerType
    
    for o_key in state_dict.keys(): # type: ignore
        key = o_key
        tensor: Tensor = state_dict[key] # type: ignore
        
        if key.startswith('lora_'):
            key = lora2sd(key)
        
        layers = key.split('.') # type: ignore
        
        layer_type = LT.NONE
        
        if key.endswith('.bias'):
            layer_type |= LT.Bias
        if key.endswith('.weight'):
            layer_type |= LT.Weight
            dim = tensor.dim()
            if dim == 1:
                # normalization (really?)
                layer_type |= LT.Norm
            elif dim == 2:
                # linear
                layer_type |= LT.MLP
            else:
                # conv
                layer_type |= LT.Conv
                
        
        if layers[0] == 'cond_stage_model':
            # text encoder
            layer_type |= LT.TextEncoder
            short_name = '.'.join(['TE'] + layers[3:])
            
            if 'transformer' not in layers:
                continue
            if '.embeddings.' in key:
                continue
            
            if match_any('attn', layers):
                layer_type |= LT.SAttn
                layer_type |= match_op({
                    'q_proj': LT.AttnQ,
                    'k_proj': LT.AttnK,
                    'v_proj': LT.AttnV,
                    'out_proj': LT.AttnOut,
                }, layers)
            
        elif layers[0] == 'first_stage_model':
            # VAE
            layer_type |= LT.VAE
            short_name = '.'.join(['VAE'] + layers[1:])
            
            if match_any('attn', layers):
                layer_type |= LT.SAttn
                if 'q' in layers:
                    layer_type |= LT.AttnQ
                if 'k' in layers:
                    layer_type |= LT.AttnK
                if 'v' in layers:
                    layer_type |= LT.AttnV
                if 'proj_out' in layers:
                    layer_type |= LT.AttnOut
        
        elif key.startswith('model.diffusion_model.'):
            # U-Net
            layer_type |= LT.UNet
            
            abbr = {
                'input_blocks': 'IN',
                'output_blocks': 'OUT',
                'middle_block': 'M',
            }
            short_name = '.'.join(['UNet'] + [ abbr.get(v, v) for v in layers[2:] ])
            
            if 'attn1' in layers:
                layer_type |= LT.SAttn
            if 'attn2' in layers:
                layer_type |= LT.XAttn
            if 'to_q' in layers:
                layer_type |= LT.AttnQ
            if 'to_k' in layers:
                layer_type |= LT.AttnK
            if 'to_v' in layers:
                layer_type |= LT.AttnV
            if 'to_out' in layers:
                layer_type |= LT.AttnOut
        
        else:
            short_name = key
        
        if 'lora_up' in layers:
            layer_type |= LT.LoraUp
        if 'lora_down' in layers:
            layer_type |= LT.LoraDown
        
        if short_name.endswith('.weight'):
            short_name = short_name[:-len('.weight')]
        if short_name.endswith('.bias'):
            short_name = short_name[:-len('.bias')]
        
        if short_name.endswith('.alpha'):
            assert tuple(tensor.shape) == (), tuple(tensor.shape)
            alphas[short_name[:-len('.alpha')]] = tensor.item()
            continue
        
        layer = Layer(key, short_name, o_key, layers, layer_type, tensor, None)
        
        if 'lora_up' in layers:
            lora_up[short_name[:-len('.lora_up')]] = layer
        if 'lora_down' in layers:
            lora_down[short_name[:-len('.lora_down')]] = layer
        
        if filter is None or filter(layer):
            target_layers.append(layer)
    
    for layer in target_layers:
        if layer.short_name.endswith('.lora_up'):
            s = '.lora_up'
            dim = layer.value.shape[1]
        if layer.short_name.endswith('.lora_down'):
            s = '.lora_down'
            dim = layer.value.shape[0]
        else:
            continue
        
        name = layer.short_name[:-len(s)]
        
        if len(alphas) == 0:
            # old version LoRA
            alpha = dim
        else:
            assert name in alphas, f'{name} not found in {list(alphas.keys())}'
            alpha = alphas[name]
        
        layer.lora_alpha = alpha
        
    if lora_matmul:
        for lora in lora_up.keys():
            up = lora_up[lora]
            down = lora_down[lora]
            up_v = up.value.squeeze().float()
            down_v = down.value.squeeze().float()
            if up_v.dim() != 2 or down_v.dim() != 2:
                continue
            dw_tensor = up_v.float() @ down_v.float()
            layer = Layer(
                lora + '.lora_matmul',
                up.short_name.replace('.lora_up', '.lora_matmul'),
                up.original_name.replace('.lora_up', '.lora_matmul'),
                up.names[:-1] + ['lora_matmul'],
                (up.type ^ LT.LoraUp) | LT.LoraMatMul,
                dw_tensor,
                up.lora_alpha,
            )
            if filter is None or filter(layer):
                target_layers.append(layer)
    
    target_layers = sorted(target_layers, key=lambda x: tuple(map(try_to_int, x.names)))
    
    #for l in target_layers:
    #    print(l.name)
    
    return target_layers

def repr_values(
    state_dict: Dict[str,Tensor],
    lora_matmul: bool = False,
    filter: Union[Callable[[Layer],bool],None] = None,
    **kwargs
) -> Dict[str,Dict[str,Any]]:
    
    result: Dict[str,Dict[str,Any]] = dict()
    
    layers = retrieve_weights(state_dict, lora_matmul=lora_matmul, filter=filter)
    
    with tqdm(layers) as tq:
        for layer in tq:
            tq.set_postfix_str(layer.short_name)
            tensor = layer.value.float()
            
            out = dict()
            
            out['layer'] = layer
            out['n'] = torch.numel(tensor)
            
            if 'values' in kwargs:
                out['values'] = tensor.flatten().to('cpu')
            if 'fro' in kwargs:
                out['fro'] = torch.norm(tensor, p='fro').item()
            if 'p' in kwargs:
                out['p'] = torch.norm(tensor, p=kwargs['p']).item() # type: ignore
            if 'mean' in kwargs or 'std' in kwargs:
                std, mean = torch.std_mean(tensor)
                if 'mean' in kwargs:
                    out['mean'] = mean.item()
                if 'std' in kwargs:
                    out['std'] = std.item()
            if 'max' in kwargs:
                out['max'] = torch.max(tensor).item()
            if 'min' in kwargs:
                out['min'] = torch.min(tensor).item()
            
            result[layer.name] = out
    
    return result


def retrieve_weights2(
    model_name: Union[str,None],
    is_lora: bool,
    wb: List[str],
    network: List[str],
    layer: List[str],
    attn: List[str],
    lora: List[str],
    value: List[str]
):
    # retrieve tensor statistics
    LT = LayerType
    
    kwargs = dict()
    
    wt = LT.NONE
    if 'Weight' in wb: wt |= LT.Weight
    if 'Bias' in wb:   wt |= LT.Bias
    
    nt = LT.NONE
    if 'Text encoder' in network: nt |= LT.TextEncoder
    if 'VAE' in network:          nt |= LT.VAE
    if 'U-Net' in network:        nt |= LT.UNet
    
    lt = LT.NONE
    if 'Linear' in layer: lt |= LT.MLP
    if 'Conv' in layer:   lt |= LT.Conv
    if 'SAttn' in layer:  lt |= LT.SAttn
    if 'XAttn' in layer:  lt |= LT.XAttn
    if 'Norm' in layer:   lt |= LT.Norm
    
    at = LT.NONE
    if 'Q' in attn:   at |= LT.AttnQ
    if 'K' in attn:   at |= LT.AttnK
    if 'V' in attn:   at |= LT.AttnV
    if 'Out' in attn: at |= LT.AttnOut
    
    rt = LT.NONE
    if 'up' in lora:   rt |= LT.LoraUp
    if 'down' in lora: rt |= LT.LoraDown
    if 'ΔW' in lora:   rt |= LT.LoraMatMul
    
    def filter(layer: Layer) -> bool:
        #if layer.type & LT.UNet:
        #    import pdb; pdb.set_trace()
        t = layer.type
        x = ((t & wt) and (t & nt) and (t & lt))
        if at != LT.NONE and ((x & LT.SAttn) or (x & LT.XAttn)):
            x = x and (t & at)
        if is_lora and rt != LT.NONE:
            x = x and (t & rt)
        return x != LayerType.NONE
    
    if 'Mean' in value: kwargs['mean'] = True
    if 'Frobenius' in value: kwargs['fro'] = True
    if 'Histogram' in value: kwargs['values'] = True
    
    result = repr_values(
        load_model(model_name, lora=is_lora),
        lora_matmul='ΔW' in lora,
        filter=filter,
        **kwargs
    )
    
    return result
