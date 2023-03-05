import os
import re
from modules import extensions, script_loading

try:
    _lora_ext = next(ex for ex in extensions.active() if ex.name == 'Lora')
    _lora_mod = script_loading.load_module(os.path.join(_lora_ext.path, 'lora.py'))
except:
    print('[WARN] MatView: Lora is not activated!')
    _lora_mod = None


def reload_loras():
    if _lora_mod is None:
        raise ValueError('Lora is inactive. See `Extensions` tab.')
    _lora_mod.list_available_loras()
    return list(_lora_mod.available_loras.keys())

def available_loras():
    return (
        ([''] + list(_lora_mod.available_loras.keys()))
        if _lora_mod is not None
        else ['']
    )

def lora_path(name: str):
    if _lora_mod is None:
        raise ValueError('Lora is inactive. See `Extensions` tab.')
    filename = _lora_mod.available_loras[name].filename
    return filename

re_digits = re.compile(r"\d+")
re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")
re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")

re_prenum = re.compile(r"_(?=\d)")
re_postnum = re.compile(r"(?<=\d)_")
re_proj = re.compile(r"_(?=(q|k|v|out)_proj)")

def lora2sd(key: str):
    # cf. extensions-builtin/Lora/lora.py
    
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True
    
    def fix(s: str):
        s = str(s)
        s = re.sub(re_prenum, '.', s)
        s = re.sub(re_postnum, '.', s)
        s = re.sub(re_proj, '.', s)
        return s

    m = []

    if match(m, re_unet_down_blocks):
        return f"model.diffusion_model.input_blocks.{1 + m[0] * 3 + m[1]}.1.{fix(m[2])}"

    if match(m, re_unet_mid_blocks):
        return f"model.diffusion_model.middle_block.1.{fix(m[1])}"

    if match(m, re_unet_up_blocks):
        return f"model.diffusion_model.output_blocks.{m[0] * 3 + m[1]}.1.{fix(m[2])}"

    if match(m, re_text_block):
        return f"cond_stage_model.transformer.text_model.encoder.layers.{fix(m[0])}.{fix(m[1])}"

    return key
