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
    hist_height: float,
    value: List[str],
):
    
    fro_is_right = 'Mean' in value or 'Histogram' in value
    
    if 'Mean' in value:
        weights.draw_mean(
            fig,
            yaxis='y',
            hsv=(0.0, 0.5, 1.0),
        )
    
    if 'Histogram' in value:
        hsv_0 = (0.0, 0.5, 1.0) if not weights.is_lora else (0.1, 0.5, 1.0)
        hsv_1 = (hsv_0[0] - 1/3, hsv_0[1], hsv_0[2])
        weights.draw_hist(
            fig, hmin=hmin, hmax=hmax, height=hist_height,
            yaxis='y',
            hsv_0=hsv_0,
            hsv_1=hsv_1,
        )
        
    if 'Frobenius' in value:
        weights.draw_fro(
            fig,
            yaxis=['y', 'y2'][fro_is_right],
            hsv=(2/3, 0.5, 1.0),
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
    model_A: Union[str,None],
    model_B: Union[str,None],
    lora_A: Union[str,None],
    lora_B: Union[str,None],
    width: float,
    height: float,
    hmin: Union[str,float],
    hmax: Union[str,float],
    hist_height: Union[int,float],
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
    
    hist_height = float(hist_height)
    
    fig = go.Figure()
    
    def draw(model: Union[str,None], is_lora: bool, *args, **kwargs):
        if model is not None and len(model) != 0:
            # 1. retrieve tensor statistics
            v = retrieve_weights2(model, is_lora, *args, **kwargs)
            w = Weights(v, is_lora)
            # 2. build graph
            build_graph(fig, w, width, height, hmin, hmax, hist_height, value)
    
    draw(model_A, False, wb, network, layer, attn, lora, value)
    draw(lora_A, True, wb, network, layer, attn, lora, value)
    
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
            if isinstance(v, list):
                return [*v, e]
            else:
                return [v, e]
        return f
    
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    all_models = list_models()
                    model_A = gr.Dropdown(all_models, label='Model A')
                    model_B = gr.Dropdown(all_models, label='Model B', visible=False)
                    refresh = ToolButton(value='\U0001f504', elem_id=id('reload_model'))
                with gr.Row():
                    all_loras = available_loras()
                    lora_A = gr.Dropdown(all_loras, label='Lora A')
                    lora_B = gr.Dropdown(all_loras, label='Lora B', visible=False)
                    refresh_loras = ToolButton(value='\U0001f504')
                with gr.Accordion('Graph Settings', open=False):
                    with gr.Row():
                        width = gr.Slider(minimum=256, maximum=4096, value=1366, step=1, label='Fig. Width')
                        height = gr.Slider(minimum=256, maximum=4096, value=768, step=1, label='Fig. Height')
                    with gr.Row():
                        min = gr.Textbox(value='-0.5', label='Hist. Min')
                        max = gr.Textbox(value='0.5', label='Hist. Max')
                        hist_height = gr.Slider(minimum=0.05, maximum=5, value=1, step=0.05, label='Hist. Height')
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
    
        refresh.click(fn=wrap(reload_models), inputs=[], outputs=[model_A, model_B, err])
        refresh_loras.click(fn=wrap(reload_loras), inputs=[], outputs=[lora_A, lora_B, err])
        run.click(fn=wrap(show), inputs=[model_A, model_B, lora_A, lora_B, width, height, min, max, hist_height, wb, network, layer_type, attn_type, lora_type, value_type], outputs=[plot, err])
        #csv.click(fn=wrap(save_csv), inputs=[models, wb, network, layer_type, attn_type], outputs=[out, err])
    
    return [(ui, NAME, NAME.lower())]


script_callbacks.on_ui_tabs(add_tab)
