from typing import Union, List, Dict, Tuple, Iterable
from dataclasses import dataclass
from enum import Flag, auto
from torch import Tensor

from scripts.matviewlib.utils import match, match_any

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
    
    # LoRA
    LoraUp = auto()
    LoraDown = auto()
    
    # currently not used
    Other = auto()
    
    TextEncoder = auto()
    VAE = auto()
    UNet = auto()

@dataclass
class Layer:
    name: str
    short_name: str
    original_name: str
    names: List[str]
    type: LayerType
    value: Tensor
    lora_alpha: Union[float,None]

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
