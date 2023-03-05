# =======================================================================================
from scripts.matviewlib.utils import ensure_install

ensure_install('plotly')
#ensure_install('pandas')
# =======================================================================================

from typing import Union, List, Dict, Any
import colorsys
from math import isinf

import torch
from torch import Tensor
import torch.nn.functional as F
import gradio as gr
import plotly.graph_objects as go

from modules import script_callbacks, sd_models
from modules.ui_components import ToolButton

#from scripts.tempcsv import csv_write
from scripts.matviewlib.lora import available_loras, reload_loras
from scripts.matviewlib.model import reload_models, retrieve_weights2, list_models

# =======================================================================================

NAME = 'MatView'

def id(s: str):
    return f'{NAME.lower()}-{s}'


def build_graph(
    fig: go.Figure,
    result: Dict[str,Dict[str,Any]],
    width: float,
    height: float,
    hmin: float,
    hmax: float,
    value: List[str]
):
    
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
            
            # fill
            fig.add_trace(
                go.Scatter(
                    x=[x0] * len(xvals), y=yvals, mode='lines',
                    yaxis='y', showlegend=False,
                    fill='tonextx', fillcolor=f'rgba({r},{g},{b},0.125)',
                    line=dict(width=0),
                    hoverinfo='none',
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


def show(
    model_name: Union[str,None],
    lora_name: Union[str,None],
    width: float,
    height: float,
    hmin: Union[str,float],
    hmax: Union[str,float],
    wb: List[str],
    network: List[str],
    layer: List[str],
    attn: List[str],
    lora: List[str],
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
    
    if model_name is not None and len(model_name) != 0:
        result = retrieve_weights2(model_name, False, wb, network, layer, attn, lora, value)
    else:
        result = None
    
    if lora_name is not None and len(lora_name) != 0:
        result2 = retrieve_weights2(lora_name, True, wb, network, layer, attn, lora, value)
    else:
        result2 = None
    
    # 2. build graph
    
    fig = go.Figure()
    
    if result is not None:
        build_graph(fig, result, width, height, hmin, hmax, value)
    
    if result2 is not None:
        build_graph(fig, result2, width, height, hmin, hmax, value)
    
    return fig


#def save_csv(
#    model_name,
#    wb: List[str],
#    network: List[str],
#    layer: List[str],
#    attn: List[str],
#):
#    # 1. retrieve tensor statistics
#    result = retrieve_weights2(model_name, wb, network, layer, attn, ['Histogram'])
#    
#    # 2. save data
#    with csv_write(close=False) as (csv, io):
#        pass
#    

def add_tab():
    def wrap(fn):
        def f(*args, **kwargs):
            v, e = None, ''
            try:
                v = fn(*args, **kwargs)
            except Exception as ex:
                e = str(ex)
            return [v, e]
        return f
    
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    models = gr.Dropdown(list_models(), elem_id=id('models'), label='Model')
                    refresh = ToolButton(value='\U0001f504', elem_id=id('reload_model'))
                with gr.Row():
                    loras = gr.Dropdown(available_loras(), label='Lora')
                    refresh_loras = ToolButton(value='\U0001f504')
                with gr.Accordion('Graph Settings', open=False):
                    with gr.Row():
                        width = gr.Slider(minimum=256, maximum=4096, value=1366, step=1, label='Fig. Width')
                        height = gr.Slider(minimum=256, maximum=4096, value=768, step=1, label='Fig. Height')
                    with gr.Row():
                        min = gr.Textbox(value='-0.5', label='Hist. Min')
                        max = gr.Textbox(value='0.5', label='Hist. Max')
                run = gr.Button('Show', variant="primary")
            with gr.Column():
                with gr.Row():
                    wb = gr.CheckboxGroup(choices=['Weight', 'Bias'], value=['Weight'], label='Weight and/or Bias')
                    network = gr.CheckboxGroup(choices=['Text encoder', 'VAE', 'U-Net'], value=['U-Net'], label='Network')
                    layer_type = gr.CheckboxGroup(choices=['Linear', 'Conv', 'SAttn', 'XAttn', 'Norm'], value=['SAttn', 'XAttn'], label='Layer Type')
                    attn_type = gr.CheckboxGroup(choices=['Q', 'K', 'V', 'Out'], value=['Q', 'K', 'V'], label='Attentions')
                    lora_type = gr.CheckboxGroup(choices=['up', 'down'], value=['up', 'down'], label='LoRA')
                value_type = gr.CheckboxGroup(choices=['Mean', 'Frobenius', 'Histogram'], value=['Mean'], label='Value')
                #with gr.Row():
                #    csv = gr.Button('Download CSV')
                #    out = gr.File()
        
        err = gr.HTML(elem_id='matview-error')
        
        plot = gr.Plot()
        
        with gr.Group(visible=False):
            pass
    
        refresh.click(fn=wrap(reload_models), inputs=[], outputs=[models, err])
        refresh_loras.click(fn=wrap(reload_loras), inputs=[], outputs=[loras, err])
        run.click(fn=wrap(show), inputs=[models, loras, width, height, min, max, wb, network, layer_type, attn_type, lora_type, value_type], outputs=[plot, err])
        #csv.click(fn=wrap(save_csv), inputs=[models, wb, network, layer_type, attn_type], outputs=[out, err])
    
    return [(ui, NAME, NAME.lower())]


script_callbacks.on_ui_tabs(add_tab)
