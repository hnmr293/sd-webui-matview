from typing import Any, Union, List, Dict, Tuple, Iterable, Callable
from dataclasses import dataclass
from enum import Flag, auto
import colorsys
from math import isinf

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import gradio as gr

from modules import script_callbacks, sd_models
from modules.ui_components import ToolButton

NAME = 'MatView'

# =======================================================================================

def ensure_install(module_name: str, lib_name: Union[str,None] = None):
    import sys, traceback
    from importlib.util import find_spec
    
    if lib_name is None:
        lib_name = module_name
    
    if find_spec(module_name) is None:
        import subprocess
        try:
            print('-' * 80, file=sys.stderr)
            print(f'| installing {lib_name} ...', file=sys.stderr)
            print('-' * 80, file=sys.stderr)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", lib_name],
                stdout=sys.stdout,
                stderr=sys.stderr
            )
        except Exception as e:
            msg = ''.join(traceback.TracebackException.from_exception(e).format())
            print(msg, file=sys.stderr)
            print('-' * 80, file=sys.stderr)
            print(f'| failed to install {lib_name}. exit...', file=sys.stderr)
            print('-' * 80, file=sys.stderr)
            sys.exit(1)

# ---------------------------------------------------------------------------------------

ensure_install('plotly')
ensure_install('pandas')

# =======================================================================================

class LayerType(Flag):
    NONE = 0
    
    Weight = 1
    Bias = auto()
    
    # Normalization Layer
    Norm = auto()
    
    # Linear Layer
    MLP = auto()
    
    # ConvND Layer
    Conv = auto()
    
    # Attentions
    SAttn = auto()
    XAttn = auto()
    AttnQ = auto()
    AttnK = auto()
    AttnV = auto()
    AttnOut = auto()
    
    # currently not used
    Other = auto()
    
    TextEncoder = auto()
    VAE = auto()
    UNet = auto()

@dataclass
class Layer:
    name: str
    short_name: str
    names: List[str]
    type: LayerType
    value: Tensor

def match(piece: str, list: List[str]):
    return any(piece in s for s in list)

def match_any(pieces: Iterable[str], list: List[str]):
    return any(match(p, list) for p in pieces)

def match_op(table: Dict[Union[str,Tuple[str,...]],LayerType], layers):
    ty = LayerType.NONE
    
    for k, v in table.items():
        if isinstance(k, str):
            if match(k, layers):
                ty |= v
        elif isinstance(k, tuple):
            if match_any(k, layers):
                ty |= v
    
    return ty
    
    
def id(s: str):
    return f'{NAME.lower()}-{s}'

def reload_models():
    sd_models.list_models()
    return gr.update(choices=sd_models.checkpoint_tiles())

def retrieve_weights(
    model_name: Union[str,None],
    filter: Union[Callable[[Layer],bool],None] = None
):
    if model_name is None or len(model_name) == 0:
        raise ValueError('model was not found.')
    
    model_info = sd_models.get_closet_checkpoint_match(model_name)
    if model_info is None:
        raise ValueError(f'model was not found: {model_name}')
    
    sd = sd_models.read_state_dict(model_info.filename, map_location='cpu')
    target_layers: List[Layer] = []
    
    LT = LayerType
    
    for key in sd.keys(): # type: ignore
        layers = key.split('.') # type: ignore
        tensor: Tensor = sd[key] # type: ignore
        
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
        
        if short_name.endswith('.weight'):
            short_name = short_name[:-len('.weight')]
        if short_name.endswith('.bias'):
            short_name = short_name[:-len('.bias')]
        
        layer = Layer(key, short_name, layers, layer_type, tensor)
        if filter is None or filter(layer):
            target_layers.append(layer)
    
    target_layers = sorted(target_layers, key=lambda x: tuple(map(try_to_int, x.names)))
    
    #for l in target_layers:
    #    print(l.name)
    
    return target_layers

def try_to_int(v: str) -> Union[str,int]:
    try:
        return int(v)
    except ValueError:
        return v

def repr_values(
    model_name: Union[str,None],
    filter: Union[Callable[[Layer],bool],None] = None,
    **kwargs
) -> Dict[str,Dict[str,Any]]:
    result: Dict[str,Dict[str,Any]] = dict()
    
    layers = retrieve_weights(model_name, filter=filter)
    with tqdm(layers) as tq:
        for layer in tq:
            tq.set_postfix_str(layer.short_name)
            tensor = layer.value
            
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

def histogram(model_name: Union[str,None]):
    pass


def show(
    model_name,
    width: float,
    height: float,
    hmin: Union[str,float],
    hmax: Union[str,float],
    wb: List[str],
    network: List[str],
    layer: List[str],
    attn: List[str],
    value: List[str]
):
    if len(hmin) == 0: # type: ignore
        hmin = float('inf')
    else:
        hmin = float(hmin)
    if len(hmax) == 0: # type: ignore
        hmax = -float('inf')
    else:
        hmax = float(hmax)
    
    # 1. retrieve tensor statistics
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
    
    def filter(layer: Layer) -> bool:
        t = layer.type
        x = ((t & wt) and (t & nt) and (t & lt))
        if at != LT.NONE:
            x = x and (t & at)
        return x != LayerType.NONE
    
    if 'Mean' in value: kwargs['mean'] = True
    if 'Frobenius' in value: kwargs['fro'] = True
    if 'Histogram' in value: kwargs['values'] = True
    
    result = repr_values(model_name, filter=filter, **kwargs)
    
    # 2. build graph
    import pandas as pd
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    fro_is_right = 'Mean' in value or 'Histogram' in value
    
    if 'Mean' in value:
        x = list(range(len(result)))
        y = [v['mean'] for v in result.values()]
        
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='lines+markers', name='Mean',
                yaxis='y',
                marker=dict(
                    size=6,
                    symbol='circle',
                    color='rgba(0,0,0,0)',
                    line=dict(
                        color='rgba(255,128,128,1)',
                        width=1,
                    ),
                ),
                line=dict(
                    color='rgba(255,128,128,0.5)',
                    width=2,
                ),
                hoverlabel=dict(bgcolor='rgba(255,214,214,0.8)'),
            )
        )
    
    if 'Histogram' in value:
        # retrieve min/max value
        if isinf(hmin) or isinf(hmax):
            c_min = float('inf')
            c_max = -float('inf')
            for vs in (v['values'] for v in result.values()):
                v_min = torch.min(vs).item()
                v_max = torch.max(vs).item()
                if v_min < c_min: c_min = v_min
                if c_max < v_max: c_max = v_max
            if isinf(hmin): hmin = c_min
            if isinf(hmax): hmax = c_max
        
        hmin, hmax = min(hmin, hmax), max(hmin, hmax)
        RANGE = (hmin, hmax)
        BINS = 500
        HEIGHT = 2.0
        for x0, rs in enumerate(result.values()):
            vs: Tensor = rs['values']
            n = torch.numel(vs)
            hist, edges = torch.histogram(vs.float(), BINS, range=RANGE, density=False)
            #small = torch.sum(vs < RANGE[0]) / n
            #large = torch.sum(RANGE[1] < vs) / n
            yvals = F.avg_pool1d(edges.unsqueeze(0), kernel_size=2, stride=1).squeeze()
            assert tuple(yvals.shape) == (BINS,), tuple(yvals.shape)
            xvals = x0 + hist / torch.max(hist) * HEIGHT
            
            r, g, b = colorsys.hls_to_rgb(x0/len(result)/-3, 0.5, 1.0)
            r, g, b = int(r*255), int(g*255), int(b*255)
            fig.add_trace(
                go.Scatter(
                    x=xvals, y=yvals, mode='lines', name='Hist.',
                    yaxis='y', showlegend=False,
                    line=dict(
                        color=f'rgba({r},{g},{b},0.25)',
                        width=1,
                    ),
                    hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)'),
                    hovertemplate=f'{rs["layer"].short_name}<br>%{{y:.3e}}<br>%{{customdata[0]:.3f}}',
                    customdata=(hist / torch.sum(hist)).unsqueeze(1),
                )
            )
    
    if 'Frobenius' in value:
        x = list(range(len(result)))
        y = [v['fro'] for v in result.values()]
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='lines+markers', name='Frobenius',
                yaxis=['y', 'y2'][fro_is_right],
                marker=dict(
                    size=6,
                    symbol='circle',
                    color='rgba(0,0,0,0)',
                    line=dict(
                        color='rgba(128,128,255,1)',
                        width=1,
                    ),
                ),
                line=dict(
                    color='rgba(128,128,255,0.5)',
                    width=2,
                ),
                hoverlabel=dict(bgcolor='rgba(214,214,255,0.8)'),
            )
        )
    
    if fro_is_right:
        y1_title = 'value'
        y2_title = 'frobenius norm'
    else:
        y1_title = 'frobenius norm'
        y2_title = ''
    
    x_title = 'layer'
    x_ticks = [v['layer'].short_name for v in result.values()]
    
    fig.update_layout(
        autosize=False,
        width=int(width),
        height=int(height),
        hoverlabel=dict(font_family='monospace'),
        xaxis=dict(
            title=x_title,
            tickmode='array',
            tickvals=list(range(len(result))),
            ticktext=x_ticks,
        ),
        yaxis=dict(
            title=y1_title,
            side='left',
            tickformat='.3e' if 'Mean' in value else '~g',
        ),
        yaxis2=dict(
            title=y2_title,
            side='right',
            tickformat='~g',
            overlaying='y',
        ),
    )
    
    return fig
    
    x = []
    y = []
    for name, obj in result.items():
        x.extend([name] * obj['n'])
        y.extend(obj['values'])
    
    return pd.DataFrame({
        'Layer': x,
        'Value': y,
    })


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    models = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id=id('models'), label='Model')
                    refresh = ToolButton(value='\U0001f504', elem_id=id('reload_model'))
                with gr.Row():
                    width = gr.Slider(minimum=256, maximum=4096, value=1366, step=1, label='Width')
                    height = gr.Slider(minimum=256, maximum=4096, value=768, step=1, label='Height')
                with gr.Row():
                    min = gr.Textbox(value='-0.5', label='Hist. Min')
                    max = gr.Textbox(value='0.5', label='Hist. Max')
                run = gr.Button('Show')
            with gr.Column():
                with gr.Row():
                    wb = gr.CheckboxGroup(choices=['Weight', 'Bias'], value=['Weight'], label='Weight and/or Bias')
                    network = gr.CheckboxGroup(choices=['Text encoder', 'VAE', 'U-Net'], value=['U-Net'], label='Network')
                    layer_type = gr.CheckboxGroup(choices=['Linear', 'Conv', 'SAttn', 'XAttn', 'Norm'], value=['SAttn', 'XAttn'], label='Layer Type')
                    attn_type = gr.CheckboxGroup(choices=['Q', 'K', 'V', 'Out'], value=['Q', 'K', 'V'], label='Attentions')
                value_type = gr.CheckboxGroup(choices=['Mean', 'Frobenius', 'Histogram'], value=['Mean'], label='Value')
        
        plot = gr.Plot()
        
        with gr.Group(visible=False):
            pass
    
        run.click(fn=show, inputs=[models, width, height, min, max, wb, network, layer_type, attn_type, value_type], outputs=[plot])
        refresh.click(fn=reload_models, inputs=[], outputs=[models])
    
    return [(ui, NAME, NAME.lower())]


script_callbacks.on_ui_tabs(add_tab)
