from typing import Union, List, Dict, Any, Tuple
import colorsys
from math import isinf

import torch
from torch import Tensor
import torch.nn.functional as F
import plotly.graph_objects as go

class Weights:
    
    def __init__(self, weights: Dict[str, Dict[str, Any]], is_lora: bool):
        self.weights = weights
        self.is_lora = is_lora
        if self.is_lora:
            self.up, self.down = self.split_lora(self.weights)
        else:
            self.up = self.down = self.weights
    
    def keys(self):
        return self.weights.keys()
    
    def values(self):
        return self.weights.values()
    
    def items(self):
        return self.weights.items()
    
    def __len__(self):
        return len(self.weights)
    
    def draw_mean(self, fig: go.Figure, **kwargs):
        if self.is_lora:
            draw_mean(fig, self.up, name='Mean (lora_up)', hsv=(0.95,0.5,1.0), mark='triangle-up', **kwargs)
            draw_mean(fig, self.down, name='Mean (lora_down)', hsv=(0.9,0.5,1.0), mark='triangle-down', **kwargs)
        else:
            draw_mean(fig, self.weights, hsv=(0.0,0.5,1.0), **kwargs)
    
    def draw_hist(self, fig: go.Figure, hmin: float, hmax: float, **kwargs):
        if self.is_lora:
            draw_hist(fig, self.up, hmin, hmax, name='Hist. (lora_up)', **kwargs)
            draw_hist(fig, self.down, hmin, hmax, name='Hist. (lora_down)', **kwargs)
        else:
            draw_hist(fig, self.weights, hmin, hmax, **kwargs)
    
    def draw_fro(self, fig: go.Figure, **kwargs):
        if self.is_lora:
            draw_fro(fig, self.up, name='Frobenius (lora_up)', hsv=(2/3+0.05,0.5,1.0), mark='triangle-up', **kwargs)
            draw_fro(fig, self.down, name='Frobenius (lora_down)', hsv=(2/3+0.1,0.5,1.0), mark='triangle-down', **kwargs)
        else:
            draw_fro(fig, self.weights, hsv=(2/3,0.5,1.0), **kwargs)
    
    def split_lora(self, weights: Dict[str, Dict[str, Any]]):
        up: Dict[str, Dict[str, Any]] = dict()
        down: Dict[str, Dict[str, Any]] = dict()
        
        for longname, obj in weights.items():
            layer = obj['layer']
            if layer.short_name.endswith('.lora_up'):
                layer.short_name = layer.short_name[:-len('.lora_up')]
                up[longname] = obj
            elif layer.short_name.endswith('.lora_down'):
                layer.short_name = layer.short_name[:-len('.lora_down')]
                down[longname] = obj
        
        return up, down

    
def draw_mean(
    fig: go.Figure,
    weights: Dict[str, Dict[str, Any]],
    hsv: Tuple[float,float,float] = (0.0,0.5,1.0),
    mark: str = 'circle',
    **kwargs
):
    x = list(range(len(weights)))
    y = [v['mean'] for v in weights.values()]
    
    color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(*hsv))
    label_color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(hsv[0], hsv[1]/2, hsv[2]))
    
    default_args = dict(
        x=x,
        y=y,
        mode='lines+markers',
        name='Mean',
        marker=dict(
            size=6,
            symbol=mark,
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
    )
    
    args = { **default_args, **kwargs }
    
    fig.add_trace(go.Scatter(**args))


def draw_hist(fig: go.Figure, weights: Dict[str, Dict[str, Any]], hmin: float, hmax: float, **kwargs):
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
    HEIGHT = 2.0
    for x0, rs in enumerate(weights.values()):
        vs: Tensor = rs['values']
        #n = torch.numel(vs)
        hist, edges = torch.histogram(vs.float(), BINS, range=RANGE, density=False)
        #small = torch.sum(vs < RANGE[0]) / n
        #large = torch.sum(RANGE[1] < vs) / n
        yvals = F.avg_pool1d(edges.unsqueeze(0), kernel_size=2, stride=1).squeeze()
        assert tuple(yvals.shape) == (BINS,), tuple(yvals.shape)
        xvals = x0 + hist / torch.max(hist) * HEIGHT
        
        r, g, b = colorsys.hls_to_rgb(x0/len(weights)/-3, 0.5, 1.0)
        r, g, b = int(r*255), int(g*255), int(b*255)
        
        default_args = dict(
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
            hovertemplate=f'{rs["layer"].short_name}<br>%{{y:.3e}}<br>%{{customdata[0]:.3f}}',
            customdata=(hist / torch.sum(hist)).unsqueeze(1),
        )
        
        args = { **default_args, **kwargs }
        
        fig.add_trace(go.Scatter(**args))
        
        # fill
        
        fill_default_args = dict(
            x=[x0] * len(xvals),
            y=yvals,
            mode='lines',
            showlegend=False,
            fill='tonextx',
            fillcolor=f'rgba({r},{g},{b},0.125)',
            line=dict(width=0),
            hoverinfo='none',
        )
        
        fill_args = { **fill_default_args, **kwargs }
        
        fig.add_trace(go.Scatter(**fill_args))


def draw_fro(
    fig: go.Figure,
    weights: Dict[str, Dict[str, Any]],
    hsv: Tuple[float,float,float] = (2/3,0.5,1.0),
    mark: str = 'circle',
    **kwargs
):
    x = list(range(len(weights)))
    y = [v['fro'] for v in weights.values()]
    
    color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(*hsv))
    label_color = ','.join(str(int(v*255)) for v in colorsys.hsv_to_rgb(hsv[0], hsv[1]/2, hsv[2]))
    
    default_args = dict(
        x=x,
        y=y,
        mode='lines+markers',
        name='Frobenius',
        marker=dict(
            size=6,
            symbol=mark,
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
    )
    
    args = { **default_args, **kwargs }
    
    fig.add_trace(go.Scatter(**args))


