# =======================================================================================
from scripts.matviewlib.utils import ensure_install

ensure_install('plotly')
#ensure_install('pandas')
# =======================================================================================

import sys
import traceback
from typing import Union, List, Dict, Any

import gradio as gr
import plotly.graph_objects as go

from modules import script_callbacks
from modules.ui_components import ToolButton

#from scripts.tempcsv import csv_write
from scripts.matviewlib.lora import available_loras, reload_loras
from scripts.matviewlib.model import reload_models, retrieve_weights2, list_models
from scripts.matviewlib.weights import Weights

# =======================================================================================

NAME = 'MatView'

def id(s: str):
    return f'{NAME.lower()}-{s}'


def build_graph(
    fig: go.Figure,
    weights: Weights,
    width: float,
    height: float,
    hmin: float,
    hmax: float,
    value: List[str],
    **kwargs
):
    
    fro_is_right = 'Mean' in value or 'Histogram' in value
    
    if 'Mean' in value:
        weights.draw_mean(
            fig,
            yaxis='y',
        )
    
    if 'Histogram' in value:
        weights.draw_hist(
            fig, hmin=hmin, hmax=hmax,
            yaxis='y',
        )
        
    if 'Frobenius' in value:
        weights.draw_fro(
            fig,
            yaxis=['y', 'y2'][fro_is_right],
        )
    
    if fro_is_right:
        y1_title = 'value'
        y2_title = 'frobenius norm'
    else:
        y1_title = 'frobenius norm'
        y2_title = ''
    
    x_title = 'layer'
    x_ticks = [v['layer'].short_name for v in weights.values()]
    
    fig.update_layout(
        autosize=False,
        width=int(width),
        height=int(height),
        hoverlabel=dict(font_family='monospace'),
        xaxis=dict(
            title=x_title,
            tickmode='array',
            tickvals=list(range(len(weights))),
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
        v = retrieve_weights2(model_name, False, wb, network, layer, attn, lora, value)
        result = Weights(v, False)
    else:
        result = None
    
    if lora_name is not None and len(lora_name) != 0:
        v = retrieve_weights2(lora_name, True, wb, network, layer, attn, lora, value)
        result_lora = Weights(v, True)
    else:
        result_lora = None
    
    # 2. build graph
    
    fig = go.Figure()
    
    if result is not None:
        build_graph(fig, result, width, height, hmin, hmax, value)
    
    if result_lora is not None:
        build_graph(fig, result_lora, width, height, hmin, hmax, value)
    
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
                e = traceback.format_exc()
                print(e, file=sys.stderr)
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
                    lora_type = gr.CheckboxGroup(choices=['up', 'down', 'ΔW'], value=['up', 'down'], label='LoRA')
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
