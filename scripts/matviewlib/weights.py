from typing import Union, List, Dict, Any, Tuple, Iterable
import colorsys
from math import isinf

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import plotly.graph_objects as go

class DrawContext(dict):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def update_context(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict) and k in self:
                old = self[k]
                old.update(v)
            else:
                self[k] = v
    

class Weights:
    
    def __init__(self, model_name: str, weights: Dict[str, Dict[str, Any]], is_lora: bool):
        self.name = model_name
        self.weights = weights
        self.is_lora = is_lora
        if self.is_lora:
            self.up, self.down, self.matmul = self.split_lora(self.weights)
        else:
            self.up = self.down = self.matmul =  self.weights
    
    def keys(self):
        return self.weights.keys()
    
    def values(self):
        return self.weights.values()
    
    def items(self):
        return self.weights.items()
    
    def __len__(self):
        return len(self.weights)
    
    def draw_mean(self, fig: go.Figure, hsv: Tuple[float,float,float], **kwargs):
        if self.is_lora:
            h, s, v = hsv
            draw_mean(self.name, fig, self.up, name='Mean (lora_up)',     hsv=((h-0.05)%1,s,v), marker=dict(symbol='triangle-up'), **kwargs)
            draw_mean(self.name, fig, self.down, name='Mean (lora_down)', hsv=((h-0.10)%1,s,v), marker=dict(symbol='triangle-down'), **kwargs)
            draw_mean(self.name, fig, self.matmul, name='Mean (lora_ΔW)', hsv=((h-0.15)%1,s,v), marker=dict(symbol='square'), **kwargs)
        else:
            draw_mean(self.name, fig, self.weights, name=f'Mean ({self.name})',  hsv=hsv, marker=dict(symbol='circle'), **kwargs)
    
    def draw_hist(self, fig: go.Figure, hmin: float, hmax: float, height: float, hsv_0: Tuple[float,float,float], hsv_1: Tuple[float,float,float], **kwargs):
        if self.is_lora:
            draw_hist(self.name, fig, self.up, hmin, hmax, name='Hist. (lora_up)',     height=height, hsv_0=hsv_0, hsv_1=hsv_1, **kwargs)
            draw_hist(self.name, fig, self.down, hmin, hmax, name='Hist. (lora_down)', height=height, hsv_0=hsv_0, hsv_1=hsv_1, **kwargs)
            draw_hist(self.name, fig, self.matmul, hmin, hmax, name='Hist. (lora_ΔW)', height=height, hsv_0=hsv_0, hsv_1=hsv_1, **kwargs)
        else:
            draw_hist(self.name, fig, self.weights, hmin, hmax, name=f'Hist. ({self.name})',  height=height, hsv_0=hsv_0, hsv_1=hsv_1, **kwargs)
    
    def draw_fro(self, fig: go.Figure, hsv: Tuple[float,float,float], **kwargs):
        if self.is_lora:
            h, s, v = hsv
            draw_fro(self.name, fig, self.up, name='Frobenius (lora_up)',     hsv=(h+0.05,s,v), marker=dict(symbol='triangle-up'), **kwargs)
            draw_fro(self.name, fig, self.down, name='Frobenius (lora_down)', hsv=(h+0.10,s,v), marker=dict(symbol='triangle-down'), **kwargs)
            draw_fro(self.name, fig, self.matmul, name='Frobenius (lora_ΔW)', hsv=(h+0.15,s,v), marker=dict(symbol='square'), **kwargs)
        else:
            draw_fro(self.name, fig, self.weights, name=f'Frobenius ({self.name})',  hsv=hsv, marker=dict(symbol='circle'), **kwargs)
    
    def split_lora(self, weights: Dict[str, Dict[str, Any]]):
        up: Dict[str, Dict[str, Any]] = dict()
        down: Dict[str, Dict[str, Any]] = dict()
        matmul: Dict[str, Dict[str, Any]] = dict()
        
        for longname, obj in weights.items():
            layer = obj['layer']
            if layer.short_name.endswith('.lora_up'):
                layer.short_name = layer.short_name[:-len('.lora_up')]
                up[longname] = obj
            elif layer.short_name.endswith('.lora_down'):
                layer.short_name = layer.short_name[:-len('.lora_down')]
                down[longname] = obj
            elif layer.short_name.endswith('.lora_matmul'):
                layer.short_name = layer.short_name[:-len('.lora_matmul')]
                matmul[longname] = obj
        
        return up, down, matmul

    
def draw_mean(
    series_name: str,
    fig: go.Figure,
    weights: Dict[str, Dict[str, Any]],
    hsv: Tuple[float,float,float] = (0.0,0.5,1.0),
    **kwargs
):
    x = list(range(len(weights)))
    y = [v['mean'] for v in weights.values()]
    
    color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(*hsv))
    label_color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(hsv[0], hsv[1]/2, hsv[2]))
    
    args = DrawContext(
        x=x,
        y=y,
        mode='lines+markers',
        name='Mean',
        marker=dict(
            size=6,
            symbol='circle',
            color='rgba(0,0,0,0)',
            line=dict(
                color=f'rgba({color},1)',
                width=1,
            ),
        ),
        line=dict(
            color=f'rgba({color},0.5)',
            width=2,
        ),
        hoverlabel=dict(bgcolor=f'rgba({label_color},0.8)'),
        hovertemplate=f'{series_name}<br>%{{x}}<br>%{{y:.3e}}',
    )
    
    args.update_context(**kwargs)
    
    fig.add_trace(go.Scatter(**args))


def draw_hist(
    series_name: str,
    fig: go.Figure,
    weights: Dict[str, Dict[str, Any]],
    hmin: float,
    hmax: float,
    height: float,
    hsv_0: Tuple[float,float,float],
    hsv_1: Tuple[float,float,float],
    **kwargs
):
    # retrieve min/max value
    if isinf(hmin) or isinf(hmax):
        c_min = float('inf')
        c_max = -float('inf')
        for vs in (v['values'] for v in weights.values()):
            v_min = torch.min(vs).item()
            v_max = torch.max(vs).item()
            if v_min < c_min: c_min = v_min
            if c_max < v_max: c_max = v_max
        if isinf(hmin): hmin = c_min
        if isinf(hmax): hmax = c_max
    
    hmin, hmax = min(hmin, hmax), max(hmin, hmax)
    RANGE = (hmin, hmax)
    BINS = 500
    HEIGHT = height
    
    def lerp(v0: Union[np.ndarray,Iterable[float]], v1: Union[np.ndarray,Iterable[float]], t: float):
        if not isinstance(v0, np.ndarray):
            v0 = np.array(v0, dtype=float)
        if not isinstance(v1, np.ndarray):
            v1 = np.array(v1, dtype=float)
        return v0 + t * (v1 - v0) # type: ignore
    
    for x0, rs in enumerate(weights.values()):
        vs: Tensor = rs['values']
        #n = torch.numel(vs)
        hist, edges = torch.histogram(vs.float(), BINS, range=RANGE, density=False)
        #small = torch.sum(vs < RANGE[0]) / n
        #large = torch.sum(RANGE[1] < vs) / n
        yvals = F.avg_pool1d(edges.unsqueeze(0), kernel_size=2, stride=1).squeeze()
        assert tuple(yvals.shape) == (BINS,), tuple(yvals.shape)
        xvals = x0 + hist / torch.max(hist) * HEIGHT
        
        h, s, v = lerp(hsv_0, hsv_1, x0/len(weights))
        # Actually we do not use HSV, but HLS.
        r, g, b = colorsys.hls_to_rgb(h, s, v)
        r, g, b = int(r*255), int(g*255), int(b*255)
        
        args = DrawContext(
            x=xvals,
            y=yvals,
            mode='lines',
            name='Hist.',
            showlegend=False,
            line=dict(
                color=f'rgba({r},{g},{b},0.25)',
                width=1,
            ),
            hoverlabel=dict(bgcolor='rgba(0,0,0,0.8)'),
            hovertemplate=f'{series_name}<br>{rs["layer"].short_name}<br>%{{y:.3e}}<br>%{{customdata[0]:.3f}}',
            customdata=(hist / torch.sum(hist)).unsqueeze(1),
        )
        
        if 'name' in kwargs:
            args['hovertemplate'] = kwargs['name'] + '<br>' + args['hovertemplate']
        
        args.update_context(**kwargs)
        
        fig.add_trace(go.Scatter(**args))
        
        # fill
        
        fill_args = DrawContext(
            x=[x0] * len(xvals),
            y=yvals,
            mode='lines',
            showlegend=False,
            fill='tonextx',
            fillcolor=f'rgba({r},{g},{b},0.125)',
            line=dict(width=0),
            hoverinfo='none',
        )
        
        fill_args.update_context(**kwargs)
        
        fig.add_trace(go.Scatter(**fill_args))


def draw_fro(
    series_name: str,
    fig: go.Figure,
    weights: Dict[str, Dict[str, Any]],
    hsv: Tuple[float,float,float] = (2/3,0.5,1.0),
    **kwargs
):
    x = list(range(len(weights)))
    y = [v['fro'] for v in weights.values()]
    
    color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(*hsv))
    label_color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(hsv[0], hsv[1]/2, hsv[2]))
    
    args = DrawContext(
        x=x,
        y=y,
        mode='lines+markers',
        name='Frobenius',
        marker=dict(
            size=6,
            symbol='circle',
            color='rgba(0,0,0,0)',
            line=dict(
                color=f'rgba({color},1)',
                width=1,
            ),
        ),
        line=dict(
            color=f'rgba({color},0.5)',
            width=2,
        ),
        hoverlabel=dict(bgcolor=f'rgba({label_color},0.8)'),
        hovertemplate=f'{series_name}<br>%{{x}}<br>%{{y:.3e}}',
    )
    
    args.update_context(**kwargs)
    
    fig.add_trace(go.Scatter(**args))
