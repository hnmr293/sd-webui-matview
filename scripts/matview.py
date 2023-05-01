# =======================================================================================
from scripts.matviewlib.utils import ensure_install

ensure_install('plotly')
#ensure_install('pandas')
# =======================================================================================

import os
import sys
import traceback
from dataclasses import dataclass
from typing import Union, List, Dict, Any, Tuple

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


def get_dir_path():
    if '__file__' in globals():
        return os.path.dirname(__file__)
    else:
        # cf. https://stackoverflow.com/a/53293924
        import inspect
        return os.path.dirname(inspect.getfile(lambda: None))

@dataclass
class GraphColors:
    mean:       Tuple[float,float,float] = (0.0, 0.5, 1.0)
    hist_start: Tuple[float,float,float] = (0.0, 0.5, 1.0)
    hist_end:   Tuple[float,float,float] = (-1/3, 0.5, 1.0)
    fro:        Tuple[float,float,float] = (2/3, 0.5, 1.0)
    
    def shift(self, h_shift_mean: float, h_shift_hist: float, h_shift_fro: float):
        self.shift_mean(h_shift_mean)
        self.shift_hist(h_shift_hist)
        self.shift_fro(h_shift_fro)
        return self
    
    def shift_all(self, h_shift: float):
        return self.shift(h_shift, h_shift, h_shift)
    
    def shift_mean(self, h_shift):
        self.mean = (self.mean[0]+h_shift, self.mean[1], self.mean[2])
        return self
    
    def shift_hist(self, h_shift):
        self.hist_start = (self.hist_start[0]+h_shift, self.hist_start[1], self.hist_start[2])
        self.hist_end   = (self.hist_end[0]+h_shift,   self.hist_end[1],   self.hist_end[2])
        return self

    def shift_fro(self, h_shift):
        self.fro = (self.fro[0]+h_shift, self.fro[1], self.fro[2])
        return self
    

def build_graph(
    fig: go.Figure,
    keys: Dict[str,List[Weights]],
    weights: Weights,
    width: float,
    height: float,
    hmin: float,
    hmax: float,
    hist_height: float,
    value: List[str],
    colors: GraphColors,
):
    
    fro_is_right = 'Mean' in value or 'Histogram' in value
    
    if 'Mean' in value:
        weights.draw_mean(
            fig,
            keys,
            yaxis='y',
            hsv=colors.mean,
        )
    
    if 'Histogram' in value:
        weights.draw_hist(
            fig,
            keys,
            hmin=hmin, hmax=hmax, height=hist_height,
            yaxis='y',
            hsv_0=colors.hist_start,
            hsv_1=colors.hist_end,
        )
        
    if 'Frobenius' in value:
        weights.draw_fro(
            fig,
            keys,
            yaxis=['y', 'y2'][fro_is_right],
            hsv=colors.fro,
        )
    
    if fro_is_right:
        y1_title = 'value'
        y2_title = 'frobenius norm'
    else:
        y1_title = 'frobenius norm'
        y2_title = ''
    
    x_title = 'layer'
    x_ticks = [key for key in keys.keys()]
    
    fig.update_layout(
        autosize=False,
        width=int(width),
        height=int(height),
        hoverlabel=dict(font_family='monospace'),
        xaxis=dict(
            title=x_title,
            tickmode='array',
            tickvals=list(range(len(x_ticks))),
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
    
    models: List[Tuple[str,Union[str,None],bool,GraphColors]] = [
        ('Model A', model_A, False, GraphColors()),
        ('Model B', model_B, False, GraphColors().shift(-0.05, 0.0, -0.05)),
        ('LoRA A', lora_A, True, GraphColors().shift_hist(0.1)),
        ('LoRA B', lora_B, True, GraphColors().shift(0.1, 0.2, 0.1))
    ]
    
    # 1. retrieve sd keys
    keys: Dict[str,List[Weights]] = dict() # actually I need "OrderedDefaultDict" here
    weights: List[Tuple[Weights,GraphColors]] = []
    for name, model, is_lora, colors in models:
        if model is not None and len(model) != 0:
            # 2. retrieve tensor statistics
            v = retrieve_weights2(model, is_lora, wb, network, layer, attn, lora, value)
            w = Weights(name, v, is_lora)
            
            for val in v.values():
                k = val['layer'].short_name
                if k not in keys:
                    keys[k] = []
                keys[k].append(w)
            
            weights.append((w, colors))
    
    # 2. build graph
    for w, colors in weights:
        build_graph(fig, keys, w, width, height, hmin, hmax, hist_height, value, colors)
    
    return fig


def export_csv(
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
    from csv import writer as csv_writer
    import tempfile
    
    io = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    csv = csv_writer(io)
    
    models: List[Tuple[str,Union[str,None],bool]] = [
        ('Model A', model_A, False),
        ('Model B', model_B, False),
        ('LoRA A', lora_A, True),
        ('LoRA B', lora_B, True)
    ]
    
    # 1. retrieve sd keys
    keys: Dict[str,List[Dict[str,Dict[str,Any]]]] = dict() # actually I need "OrderedDefaultDict" here
    weights: Dict[str,Dict[str,Dict[str,Any]]] = dict()
    for name, model, is_lora in models:
        if model is not None and len(model) != 0:
            # 2. retrieve tensor statistics
            v = retrieve_weights2(model, is_lora, wb, network, layer, attn, lora, value)
            
            for val in v.values():
                k = val['layer'].short_name
                if k not in keys:
                    keys[k] = []
                keys[k].append(v)
            
            weights[model] = v
    
    # header
    csv.writerow(['#', 'model name', 'layer name', 'value type', 'value'])
    
    def get_value(key: str, weight: Dict[str,Any], value_type: str):
        for val in weight.values():
            if key == val['layer'].short_name:
                return val[value_type]
    
    def write(model: str, is_lora: bool, value_type: str):
        # retrieve tensor statistics
        v = weights[model]
        # 3. build csv
        for idx, key in enumerate(keys.keys()):
            val = get_value(key, v, value_type)
            if val is not None:
                csv.writerow([idx, model, key, value_type, val])
    
    def write_value_type(value_type: str):
        if model_A is not None and model_A != '':
            write(model_A, False, value_type)
        if model_B is not None and model_B != '':
            write(model_B, False, value_type)
        if lora_A is not None and lora_A != '':
            write(lora_A, True, value_type)
        if lora_B is not None and lora_B != '':
            write(lora_B, True, value_type)
    
    if 'Mean' in value:
        write_value_type('mean')
    if 'Frobenius' in value:
        write_value_type('fro')
    
    io.flush()
    return io.name


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
                    model_B = gr.Dropdown(all_models, label='Model B')
                    refresh = ToolButton(value='\U0001f504', elem_id=id('reload_model'))
                with gr.Row():
                    all_loras = available_loras()
                    lora_A = gr.Dropdown(all_loras, label='Lora A')
                    lora_B = gr.Dropdown(all_loras, label='Lora B')
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
                
                with gr.Row():
                    csv = gr.Button('Export CSV')
                    # If `value` is empty, `gradio.File` will occupy a verrrrry large region.
                    # And if `value` is not an existed file, `gradio.File` throws error! >_<
                    dummy = os.path.join(get_dir_path(), '..', 'dummy.csv')
                    # I want to use `__file__`, but some platform such as colab does not define `__file__`.
                    if not os.path.exists(dummy):
                        # Okay, I give up.
                        dummy = None
                    exported_file = gr.File(label='CSV', interactive=False, value=dummy)
            
            with gr.Column():
                with gr.Row():
                    wb = gr.CheckboxGroup(choices=['Weight', 'Bias'], value=['Weight'], label='Weight and/or Bias')
                    network = gr.CheckboxGroup(choices=['Text encoder', 'VAE', 'U-Net'], value=['U-Net'], label='Network')
                    layer_type = gr.CheckboxGroup(choices=['Linear', 'Conv', 'SAttn', 'XAttn', 'Norm'], value=['SAttn', 'XAttn'], label='Layer Type')
                    attn_type = gr.CheckboxGroup(choices=['Q', 'K', 'V', 'Out'], value=['Q', 'K', 'V'], label='Attentions')
                    lora_type = gr.CheckboxGroup(choices=['up', 'down', 'ΔW'], value=['up', 'down'], label='LoRA')
                value_type = gr.CheckboxGroup(choices=['Mean', 'Frobenius', 'Histogram'], value=['Mean'], label='Value')
        
        err = gr.HTML(elem_id='matview-error')
        
        plot = gr.Plot()
        
        with gr.Group(visible=False):
            pass
    
        refresh.click(fn=wrap(reload_models), inputs=[], outputs=[model_A, model_B, err])
        refresh_loras.click(fn=wrap(reload_loras), inputs=[], outputs=[lora_A, lora_B, err])
        run.click(fn=wrap(show), inputs=[model_A, model_B, lora_A, lora_B, width, height, min, max, hist_height, wb, network, layer_type, attn_type, lora_type, value_type], outputs=[plot, err])
        csv.click(fn=wrap(export_csv), inputs=[model_A, model_B, lora_A, lora_B, width, height, min, max, hist_height, wb, network, layer_type, attn_type, lora_type, value_type], outputs=[exported_file, err])
    
    return [(ui, NAME, NAME.lower())]


script_callbacks.on_ui_tabs(add_tab)
