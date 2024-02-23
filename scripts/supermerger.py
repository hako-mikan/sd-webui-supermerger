import gc
import os
import os.path
import re
import json
import shutil
from tqdm import tqdm
import torch
from statistics import mean
import csv
import torch.nn as nn
import torch.nn.functional as F
from importlib import reload
from pprint import pprint
import gradio as gr
from modules import (script_callbacks, sd_models,sd_vae, shared)
from modules.scripts import basedir
from modules.sd_models import checkpoints_loaded, load_model,unload_model_weights
from modules.shared import opts
from modules.sd_samplers import samplers
from modules.ui import create_output_panel, create_refresh_button
import scripts.mergers.mergers
import scripts.mergers.pluslora
import scripts.mergers.xyplot
import scripts.mergers.components as components
from importlib import reload
reload(scripts.mergers.mergers)
reload(scripts.mergers.xyplot)
reload(scripts.mergers.pluslora)
import csv
import scripts.mergers.pluslora as pluslora
from scripts.mergers.mergers import (TYPESEG,EXCLUDE_CHOICES, freezemtime, rwmergelog, blockfromkey, clearcache, getcachelist)
from scripts.mergers.xyplot import freezetime, nulister
from scripts.mergers.model_util import filenamecutter, savemodel

path_root = basedir()
xyzpath = os.path.join(path_root,"xyzpresets.json")

CALCMODES  = ["normal", "cosineA", "cosineB","trainDifference","smoothAdd","smoothAdd MT","extract","tensor","tensor2","self","plus random"]

class ResizeHandleRow(gr.Row):
    """Same as gr.Row but fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.elem_classes.append("resize-handle-row")

    def get_block_name(self):
        return "row"

from typing import Union
def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None
    

def fix_network_reset_cached_weight():
    try:
        import networks as net
        net.network_reset_cached_weight = network_reset_cached_weight
    except:
        pass

def on_ui_tabs():
    fix_network_reset_cached_weight()

    weights_presets=""
    userfilepath = os.path.join(path_root, "scripts","mbwpresets.txt")
    
    if os.path.isfile(userfilepath):
        try:
            with open(userfilepath) as f:
                weights_presets = f.read()
                filepath = userfilepath
        except OSError as e:
                pass
    else:
        filepath = os.path.join(path_root, "scripts","mbwpresets_master.txt")
        try:
            with open(filepath) as f:
                weights_presets = f.read()
                shutil.copyfile(filepath, userfilepath)
        except OSError as e:
                pass

    if "ALLR" not in weights_presets: weights_presets += ADDRAND

    with gr.Blocks() as supermergerui:
        with gr.Tab("Merge"):
            with ResizeHandleRow(equal_height=False):
                with gr.Column(variant="compact"):
                    gr.HTML(value="<p>Merge models and load it for generation</p>")

                    with gr.Row():
                        s_reverse= gr.Button(value="Load settings from:",elem_classes=["compact_button"],variant='primary')
                        mergeid = gr.Textbox(label="merged model ID (-1 for last)", elem_id="model_converter_custom_name",value = "-1")
                        mclearcache= gr.Button(value="Clear Cache",elem_classes=["compact_button"],variant='primary')

                    with gr.Row(variant="compact"):
                        model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model A",interactive=True)
                        create_refresh_button(model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                        model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model B",interactive=True)
                        create_refresh_button(model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                        model_c = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model C",interactive=True)
                        create_refresh_button(model_c, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                    mode = gr.Radio(label = "Merge Mode",choices = ["Weight sum", "Add difference", "Triple sum", "sum Twice"], value="Weight sum", info="A*(1-alpha)+B*alpha")
                    calcmode = gr.Radio(label = "Calculation Mode",choices = CALCMODES, value = "normal") 
                    with gr.Row(variant="compact"):
                        with gr.Column(scale = 1):
                            useblocks =  gr.Checkbox(label="use MBW", info="use Merge Block Weights")
                        with gr.Column(scale = 3), gr.Group() as alpha_group:
                            with gr.Row():
                                base_alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                                base_beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=0.25, interactive=False)
                        #weights = gr.Textbox(label="weights,base alpha,IN00,IN02,...IN11,M00,OUT00,...,OUT11",lines=2,value="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")

                    with gr.Accordion("Options", open=False):
                        with gr.Row(variant="compact"):
                            save_sets = gr.CheckboxGroup(["use cuda","save model", "overwrite","safetensors","fp16","save metadata","copy config","prune","Reset CLIP ids","use old calc method","debug"], value=["safetensors"], show_label=False, label="save settings")
                        with gr.Row():
                            components.id_sets = gr.CheckboxGroup(["image", "PNG info"], label="save merged model ID to")
                            opt_value = gr.Slider(label="option(gamma) ", minimum=-1.0, maximum=20, step=0.1, value=0.3, interactive=True)
                        with gr.Row(variant="compact"):
                            with gr.Column(min_width = 50):
                                with gr.Row():
                                    custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="model_converter_custom_name")

                            with gr.Column():
                                with gr.Row():
                                    bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="Bake in VAE", elem_id="modelmerger_bake_in_vae")
                                    create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "modelmerger_refresh_bake_in_vae")

                        with gr.Row(variant="compact"):
                            savecurrent = gr.Button(elem_id="savecurrent", elem_classes=["compact_button"], value="Save current merge(fp16 only)")

                    with gr.Row():
                        components.merge = gr.Button(elem_id="model_merger_merge", elem_classes=["compact_button"], value="Merge!",variant='primary')
                        components.mergeandgen = gr.Button(elem_id="model_merger_merge", elem_classes=["compact_button"], value="Merge&Gen",variant='primary')
                        components.gen = gr.Button(elem_id="model_merger_merge", elem_classes=["compact_button"], value="Gen",variant='primary')
                        stopmerge = gr.Button(elem_id="stopmerge", elem_classes=["compact_button"], value="Stop")


                    with gr.Accordion("Merging Block Weights", open=False):
                        with gr.Row():
                            isxl = gr.Radio(label = "Block Type",choices = ["1.X or 2.X", "XL"], value = "1.X or 2.X", type="index")

                        with gr.Tab("Weights Setting"):
                            with gr.Group(), gr.Tabs():
                                with gr.Tab("Weights for alpha"):
                                    with gr.Row(variant="compact"):
                                        weights_a = gr.Textbox(label="BASE,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5", show_copy_button=True)
                                    with gr.Row(scale=2):
                                        setalpha = gr.Button(elem_id="copytogen", value="↑ Set alpha",variant='primary', scale=3)
                                        readalpha = gr.Button(elem_id="copytogen", value="↓ Read alpha",variant='primary', scale=3)
                                        setx = gr.Button(elem_id="copytogen", value="↑ Set X", min_width="80px", scale=1)
                                with gr.Tab("beta"):
                                    with gr.Row(variant="compact"):
                                        weights_b = gr.Textbox(label="BASE,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2", show_copy_button=True)
                                    with gr.Row(scale=2):
                                        setbeta = gr.Button(elem_id="copytogen", value="↑ Set beta",variant='primary', scale=3)
                                        readbeta = gr.Button(elem_id="copytogen", value="↓ Read beta",variant='primary', scale=3)
                                        sety = gr.Button(elem_id="copytogen", value="↑ Set Y", min_width="80px", scale=1)

                            with gr.Group(), gr.Tabs():
                                with gr.Tab("Preset"):
                                    with gr.Row():
                                        dd_preset_weight = gr.Dropdown(label="Select preset", choices=preset_name_list(weights_presets), interactive=True, elem_id="refresh_presets")
                                        preset_refresh = gr.Button(value='\U0001f504', elem_classes=["tool"])

                                with gr.Tab("Random Preset"):
                                    with gr.Row():
                                        dd_preset_weight_r = gr.Dropdown(label="Load Romdom preset", choices=preset_name_list(weights_presets,True), interactive=True, elem_id="refresh_presets")
                                        preset_refresh_r = gr.Button(value='\U0001f504', elem_classes=["tool"])
                                        luckab = gr.Radio(label = "for",choices = ["none", "alpha", "beta"], value = "none", type="value")

                                with gr.Tab("Helper"):
                                    with gr.Column():
                                        resetval = gr.Slider(label="Value", show_label=False, info="Value to set/add/mul", minimum=0, maximum=2, step=0.0001, value=0)
                                        resetopt = gr.Radio(label="Pre defined", show_label=False, choices = ["0", "0.25", "0.5", "0.75", "1"], value = "0", type="value")
                                    with gr.Column():
                                        resetblockopt = gr.CheckboxGroup(["BASE","INP*","MID","OUT*"], value=["INP*","OUT*"], label="Blocks", show_label=False, info="Select blocks to change")
                                    with gr.Column():
                                        with gr.Row():
                                            resetweight = gr.Button(elem_classes=["reset"], value="Set")
                                            addweight = gr.Button(elem_classes=["reset"], value="Add")
                                            mulweight = gr.Button(elem_classes=["reset"], value="Mul")
                                        with gr.Row():
                                            lower = gr.Slider(label="Slider Lower Limit", minimum=-2, maximum=3, step=0.1, value=0)
                                            upper = gr.Slider(label="Slider Upper Limit", minimum=-2, maximum=3, step=0.1, value=1)

                            with gr.Row():
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)
                                with gr.Column(scale=2, min_width=200):
                                    base = gr.Slider(label="Base", minimum=0, maximum=1, step=0.0001, value=0.5)
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)
                            with gr.Row():
                                with gr.Column(scale=2, min_width=200):
                                    in00 = gr.Slider(label="IN00", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in01 = gr.Slider(label="IN01", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in02 = gr.Slider(label="IN02", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in03 = gr.Slider(label="IN03", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in04 = gr.Slider(label="IN04", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in05 = gr.Slider(label="IN05", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in06 = gr.Slider(label="IN06", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in07 = gr.Slider(label="IN07", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in08 = gr.Slider(label="IN08", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in09 = gr.Slider(label="IN09", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in10 = gr.Slider(label="IN10", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    in11 = gr.Slider(label="IN11", minimum=0, maximum=1, step=0.0001, value=0.5)
                                with gr.Column(scale=2, min_width=200):
                                    ou11 = gr.Slider(label="OUT11", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou10 = gr.Slider(label="OUT10", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou09 = gr.Slider(label="OUT09", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou08 = gr.Slider(label="OUT08", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou07 = gr.Slider(label="OUT07", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou06 = gr.Slider(label="OUT06", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou05 = gr.Slider(label="OUT05", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou04 = gr.Slider(label="OUT04", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou03 = gr.Slider(label="OUT03", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou02 = gr.Slider(label="OUT02", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou01 = gr.Slider(label="OUT01", minimum=0, maximum=1, step=0.0001, value=0.5)
                                    ou00 = gr.Slider(label="OUT00", minimum=0, maximum=1, step=0.0001, value=0.5)
                            with gr.Row():
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)
                                with gr.Column(scale=2, min_width=200):
                                    mi00 = gr.Slider(label="M00", minimum=0, maximum=1, step=0.0001, value=0.5)
                                with gr.Column(scale=1, min_width=100):
                                    gr.Slider(visible=False)

                        with gr.Tab("Weights Presets"):
                            with gr.Row():
                                s_reloadtext = gr.Button(value="Reload Presets",variant='primary')
                                s_reloadtags = gr.Button(value="Reload Tags",variant='primary')
                                s_savetext = gr.Button(value="Save Presets",variant='primary')
                                s_openeditor = gr.Button(value="Open TextEditor",variant='primary')
                            weightstags= gr.Textbox(label="available",lines = 2,value=tagdicter(weights_presets),visible =True,interactive =True)
                            wpresets= gr.TextArea(label="",value=(weights_presets+ADDRAND),visible =True,interactive = True)

                    with gr.Accordion("XYZ Plot", open=False):
                        with gr.Row():
                            x_type = gr.Dropdown(label="X type", choices=[x for x in TYPESEG], value="alpha", type="index")
                            x_randseednum = gr.Number(value=3, label="number of -1", interactive=True, visible = True)
                        xgrid = gr.Textbox(label="X Values",lines=3,value="0.25,0.5,0.75")
                        y_type = gr.Dropdown(label="Y type", choices=[y for y in TYPESEG], value="none", type="index")
                        ygrid = gr.Textbox(label="Y Values (Disabled if blank)",lines=3,value="",visible =False)
                        z_type = gr.Dropdown(label="Z type", choices=[y for y in TYPESEG], value="none", type="index")
                        zgrid = gr.Textbox(label="Z Values (Disabled if blank)",lines=3,value="",visible =False)
                        esettings = gr.CheckboxGroup(label = "XYZ plot settings",choices=["swap XY","save model","save csv","save anime gif","not save grid","print change","0 stock"],type="value",interactive=True)

                        with gr.Row():
                            components.gengrid = gr.Button(elem_id="model_merger_merge", value="Run XYZ Plot",variant='primary')
                            stopgrid = gr.Button(elem_id="model_merger_merge", value="Stop XYZ Plot")
                            components.s_reserve1 = gr.Button(value="Reserve XYZ Plot",variant='primary')
                        
                        with gr.Accordion("XYZ presets",open = True):
                            with gr.Row():
                                xyzpresets = gr.Dropdown(label="Preset name",allow_custom_value=True,choices=get_xyzpreset_keylist(),scale=10)
                                refreshxyzpresets_b = gr.Button(value='\U0001f504', elem_classes=["tool"],scale=1)
                                savexyzpreset_overwrite = gr.CheckboxGroup(label = " ",choices=["Overwrite"],type="index",interactive=True,scale=1)
                            with gr.Row():
                                loadxyzpreset_b = gr.Button(value="Load preset",variant='primary')
                                savexyzpreset_b = gr.Button(value="Save current plot as preset",variant='primary')
                                deletexyzpreset_b = gr.Button(value="Delete preset",variant='primary')
                                openxyzpreset = gr.Button(value="Open XYZ Preset file")

                                openxyzpreset.click(fn=lambda:subprocess.Popen(['start', xyzpath], shell=True))
                                
                        with gr.Column(visible = False, variant="compact") as row_inputers:
                            with gr.Row(variant="compact"):
                                inputer = gr.Textbox(label="Selected", lines=1, value="", show_copy_button=True)
                            with gr.Row(variant="compact"):
                                addtox = gr.Button(value="↑ Add to X Values")
                                addtoy = gr.Button(value="↑ Add to Y Values")
                                addtoz = gr.Button(value="↑ Add to Z Values")
                        with gr.Row(visible = False) as row_blockids:
                            blockids = gr.CheckboxGroup(label = "block IDs",choices=BLOCKID[:-1],type="value",interactive=True)
                        with gr.Row(visible = False) as row_calcmode:
                            calcmodes = gr.CheckboxGroup(label = "calcmode",choices=CALCMODES,type="value",interactive=True)
                        with gr.Row(visible = False) as row_checkpoints:
                            checkpoints = gr.CheckboxGroup(label = "checkpoints",choices=[x.model_name for x in sd_models.checkpoints_list.values()],type="value",interactive=True)
                            create_refresh_button(checkpoints, sd_models.list_models, lambda: {"choices": [x.model_name for x in sd_models.checkpoints_list.values()]}, "refresh_checkpoint_xyz")
                        with gr.Row(visible = False) as row_blocks:
                            gr.HTML(value="<p>BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11<br>,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11,Adjust,VAE,print</p>")

                        with gr.Accordion("Reservation", open=False):
                            with gr.Row():
                                components.s_reserve = gr.Button(value="Reserve XY Plot",variant='primary')
                                s_reloadreserve = gr.Button(value="Reloat List",variant='primary')
                                components.s_startreserve = gr.Button(value="Start XY plot",variant='primary')
                                s_delreserve = gr.Button(value="Delete list(-1 for all)",variant='primary')
                                s_delnum = gr.Number(value=1, label="Delete num : ", interactive=True, visible = True,precision =0)
                            with gr.Row():
                                components.numaframe = gr.Dataframe(
                                    headers=["No.","status","xtype","xmenber","ytype","ymenber","ztype","zmenber","model A","model B","model C","alpha","beta","mode","use MBW","weights alpha","weights beta"],
                                    row_count=5,)

                    components.dtrue =  gr.Checkbox(value = True, visible = False)
                    components.dfalse =  gr.Checkbox(value = False,visible = False)
                    dummy_t =  gr.Textbox(value = "",visible = False)

                    with gr.Accordion("Elemental Merge",open = False):
                        with gr.Row():
                            components.esettings1 = gr.CheckboxGroup(label = "settings",choices=["print change"],type="value",interactive=True)
                        with gr.Row():
                            deep = gr.Textbox(label="Blocks:Element:Ratio,Blocks:Element:Ratio,...",lines=2,value="")

                    with gr.Accordion("Adjust", open=False) as acc_ad:
                        with gr.Row(variant="compact"):
                            finetune = gr.Textbox(label="Adjust", show_label=False, info="Adjust IN,OUT,OUT2,Contrast,Brightness,COL1,COL2,COL3", visible=True, value="", lines=1)
                            finetune_write = gr.Button(value="↑", elem_classes=["tool"])
                            finetune_read = gr.Button(value="↓", elem_classes=["tool"])
                            finetune_reset = gr.Button(value="\U0001f5d1\ufe0f", elem_classes=["tool"])
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                detail1 = gr.Slider(label="IN", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail2 = gr.Slider(label="OUT", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                            with gr.Column(scale=1, min_width=100):
                                detail3 = gr.Slider(label="OUT2", minimum=-6, maximum=6, step=0.01, value=0, info="Detail/Noise")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                contrast = gr.Slider(label="Contrast", minimum=-10, maximum=10, step=0.01, value=0, info="Contrast/Detail")
                            with gr.Column(scale=1, min_width=100):
                                bri = gr.Slider(label="Brightness", minimum=-10, maximum=10, step=0.01, value=0, info="Dark(Minius)-Bright(Plus)")
                        with gr.Row(variant="compact"):
                            with gr.Column(scale=1, min_width=100):
                                col1 = gr.Slider(label="Cyan-Red", minimum=-10, maximum=10, step=0.01, value=0, info="Cyan(Minius)-Red(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col2 = gr.Slider(label="Magenta-Green", minimum=-10, maximum=10, step=0.01, value=0, info="Magenta(Minius)-Green(Plus)")
                            with gr.Column(scale=1, min_width=100):
                                col3 = gr.Slider(label="Yellow-Blue", minimum=-10, maximum=10, step=0.01, value=0, info="Yellow(Minius)-Blue(Plus)")
                        
                            finetune.change(fn=lambda x:gr.update(label = f"Adjust : {x}"if x != "" and x !="0,0,0,0,0,0,0,0" else "Adjust"),inputs=[finetune],outputs = [acc_ad])

                    with gr.Accordion("Let the Dice roll",open = False,visible=True):
                        with gr.Row():
                            gr.HTML(value="<p>R:0~1, U: -0.5~1.5</p>")
                        with gr.Row():
                            luckmode = gr.Radio(label = "Random Mode",choices = ["off", "R", "U", "X", "ER", "EU", "EX","custom"], value = "off") 
                        with gr.Row():
                            lucksets = gr.CheckboxGroup(label = "Settings",choices=["alpha","beta","save E-list"],value=["alpha"],type="value",interactive=True)
                        with gr.Row():
                            luckseed = gr.Number(minimum=-1, maximum=4294967295, step=1, label='Seed for Random Ratio', value=-1, elem_id="luckseed")
                            luckround = gr.Number(minimum=1, maximum=4294967295, step=1, label='Round', value=3, elem_id="luckround")
                            luckserial = gr.Number(minimum=1, maximum=4294967295, step=1, label='Num of challenge', value=1, elem_id="luckchallenge")
                        with gr.Row():  
                            luckcustom = gr.Textbox(label="custom",value = "U,0,0,0,0,0,0,0,0,0,0,0,0,R,R,R,R,R,R,R,R,R,R,R,R,R")
                        with gr.Row():  
                            lucklimits_u = gr.Textbox(label="Upper limit for X",value = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1")
                        with gr.Row(): 
                            lucklimits_l = gr.Textbox(label="Lower limit for X",value = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
                        components.rand_merge = gr.Button(elem_id="runrandmerge", value="Run Rand",variant='primary')

                    with gr.Accordion("Generation Parameters",open = False):
                        gr.HTML(value='If blank or set to 0, parameters in the "txt2img" tab are used.<br>batch size, restore face, hires fix settigns must be set here')
                        prompt = gr.Textbox(label="prompt",lines=1,value="")
                        neg_prompt = gr.Textbox(label="neg_prompt",lines=1,value="")
                        with gr.Row():
                            sampler = gr.Dropdown(label='Sampling method', elem_id=f"sampling", choices=[" ",*[x.name for x in samplers]], value=" ", type="index")
                            steps = gr.Slider(minimum=0.0, maximum=150, step=1, label='Steps',value=0, elem_id="Steps")
                            cfg = gr.Slider(minimum=0.0, maximum=30, step=0.5, label='CFG scale', value=0, elem_id="cfg")
                        with gr.Row():
                            width = gr.Slider(minimum=0, maximum=2048, step=8, label="Width", value=0, elem_id="txt2img_width")
                            height = gr.Slider(minimum=0, maximum=2048, step=8, label="Height", value=0, elem_id="txt2img_height")
                            seed = gr.Number(minimum=-1, maximum=4294967295, step=1, label='Seed', value=0, elem_id="seed")
                        batch_size = denois_str = gr.Slider(minimum=0, maximum=8, step=1, label='Batch size', value=1, elem_id="sm_txt2img_batch_size")
                        genoptions = gr.CheckboxGroup(label = "Gen Options",choices=["Restore faces", "Tiling", "Hires. fix"], visible = True,interactive=True,type="value")    
                        with gr.Row(elem_id="txt2img_hires_fix_row1", variant="compact"):
                            hrupscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                            hr2ndsteps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                            denois_str = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")
                            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                        with gr.Row():
                            setdefault = gr.Button(elem_id="setdefault", value="set to default",variant='primary')
                            resetdefault = gr.Button(elem_id="resetdefault", value="reset default",variant='primary')
                            resetcurrent = gr.Button(elem_id="resetcurrent", value="reset current",variant='primary')

                    with gr.Accordion("Include/Exclude", open=False) as acc_ex:
                        with gr.Row():
                            inex = gr.Radio(label="Mode", choices=["Off","Include","Exclude"], value="Off")
                        with gr.Row():
                            ex_blocks = gr.CheckboxGroup(choices=EXCLUDE_CHOICES + ["print"], visible = True,interactive=True,type="value")
                        with gr.Row():
                            ex_elems = gr.Textbox(label="Elements")
                        inex.change(fn=lambda i, x,y: gr.update(label =f"{i} : " + ",".join(x) +","+ y if x != [] or y != "" else "Include/Exclude"), inputs = [inex,ex_blocks,ex_elems],outputs = [acc_ex])
                        ex_blocks.change(fn=lambda i, x,y: gr.update(label =f"{i} : " + ",".join(x) +","+ y if x != [] or y != "" else "Include/Exclude"), inputs = [inex,ex_blocks,ex_elems],outputs = [acc_ex])
                        ex_elems.change(fn=lambda i, x,y: gr.update(label =f"{i} : " + ",".join(x) +","+ y if x != [] or y != "" else "Include/Exclude"),inputs=[inex,ex_blocks,ex_elems],outputs = [acc_ex])

                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            currentcache = gr.Textbox(label="Current Cache")
                            loadcachelist = gr.Button(elem_id="model_merger_merge", value="Reload Cache List",variant='primary')
                            unloadmodel = gr.Button(value="unload model",variant='primary')

                with gr.Column(variant="compact"):
                    components.currentmodel = gr.Textbox(label="Current Model",lines=1,value="")
                    components.submit_result = gr.Textbox(label="Message")
                    
                    output_panel = create_output_panel("txt2img", opts.outdir_txt2img_samples)
                    
                    mgallery = output_panel[0]  if isinstance(output_panel, tuple) else output_panel.gallery
                    mgeninfo = output_panel[1]   if isinstance(output_panel, tuple) else output_panel.generation_info
                    mhtmlinfo = output_panel[2]   if isinstance(output_panel, tuple) else output_panel.infotext
                    mhtmllog = output_panel[3]   if isinstance(output_panel, tuple) else output_panel.html_log
                    
        # main ui end 
    
        with gr.Tab("LoRA", elem_id="tab_lora"):
            pluslora.on_ui_tabs()
                    
        with gr.Tab("Analysis", elem_id="tab_analysis"):
            with gr.Tab("Models"):
                with gr.Row():
                    an_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint A",interactive=True)
                    create_refresh_button(an_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z") 
                    an_model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint B",interactive=True)
                    create_refresh_button(an_model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z") 
                with gr.Row():
                    an_mode  = gr.Radio(label = "Analysis Mode",choices = ["ASimilarity","Block","Element","Both"], value = "ASimilarity",type  = "value") 
                    an_calc  = gr.Radio(label = "Block method",choices = ["Mean","Min","attn2"], value = "Mean",type  = "value") 
                    an_include  = gr.CheckboxGroup(label = "Include",choices = ["Textencoder(BASE)","U-Net","VAE"], value = ["Textencoder(BASE)","U-Net"],type  = "value") 
                    an_settings = gr.CheckboxGroup(label = "Settings",choices=["save as txt", "save as csv"],type="value",interactive=True)
                with gr.Row():
                    run_analysis = gr.Button(value="Run Analysis",variant='primary')
                with gr.Row():
                    analysis_cosdif = gr.Dataframe(headers=["block","key","similarity[%]"],)
            with gr.Tab("Text Encoder"):
                    with gr.Row():
                        te_smd_loadkeys = gr.Button(value="Calculate Textencoer",variant='primary')
                        te_smd_searchkeys = gr.Button(value="Search Word(red,blue,girl,...)",variant='primary')
                        exclude = gr.Checkbox(label="exclude non numeric,alphabet,symbol word")
                    pickupword = gr.TextArea()
                    encoded = gr.Dataframe()

        run_analysis.click(fn=calccosinedif,inputs=[an_model_a,an_model_b,an_mode,an_settings,an_include,an_calc],outputs=[analysis_cosdif])    

        with gr.Tab("History", elem_id="tab_history"):
            
            with gr.Row():
                with gr.Column(scale = 2):
                    with gr.Row():
                        count = gr.Dropdown(choices=["20", "30", "40", "50", "100"], value="20", label="Load count")
                        load_history = gr.Button(value="Load history",variant='primary', elem_classes=["reset"])
                        reload_history = gr.Button(value="Reload history", elem_classes=["reset"])
                with gr.Column(scale = 2):
                    with gr.Row():
                        searchwrods = gr.Textbox(label="",lines=1,value="")
                        search = gr.Button(value="search", elem_classes=["reset"])
                        searchmode = gr.Radio(label = "Search Mode",choices = ["or","and"], value = "or",type  = "value") 
            with gr.Row():
                history = gr.Dataframe(
                        headers=["ID","Time","Name","Weights alpha","Weights beta","Model A","Model B","Model C","alpha","beta","Mode","use MBW","custum name","save setting","use ID"],
                )
    
        import lora

        with gr.Tab("Elements", elem_id="tab_deep"):
                with gr.Row():
                    smd_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint",interactive=True)
                    create_refresh_button(smd_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")    
                    smd_loadkeys = gr.Button(value="load keys",variant='primary')
                with gr.Row():
                    smd_lora = gr.Dropdown(list(lora.available_loras.keys()),elem_id="model_converter_model_name",label="LoRA",interactive=True)
                    create_refresh_button(smd_lora, lora.list_available_loras, lambda: {"choices": list(lora.available_loras.keys())},"refresh_checkpoint_Z")
                    smd_loadkeys_l = gr.Button(value="load keys",variant='primary')
                with gr.Row():
                    keys = gr.Dataframe(headers=["No.","block","key"],)

        with gr.Tab("Metadata", elem_id="tab_metadata"):
                with gr.Row():
                    meta_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="read metadata",interactive=True)
                    create_refresh_button(meta_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")    
                    smd_loadmetadata = gr.Button(value="load keys",variant='primary')
                with gr.Row():
                    metadata = gr.TextArea()

        smd_loadmetadata.click(
            fn=loadmetadata,
            inputs=[meta_model_a],
            outputs=[metadata]
        )                 

        mclearcache.click(fn=clearcache)
        smd_loadkeys.click(fn=loadkeys,inputs=[smd_model_a,components.dfalse],outputs=[keys])
        smd_loadkeys_l.click(fn=loadkeys,inputs=[smd_lora,components.dtrue],outputs=[keys])

        te_smd_loadkeys.click(fn=encodetexts,inputs=[exclude],outputs=[encoded])
        te_smd_searchkeys.click(fn=pickupencode,inputs=[pickupword],outputs=[encoded])
        

        def unload():
            if shared.sd_model == None: return "already unloaded"
            load_model,unload_model_weights()
            return "model unloaded"

        unloadmodel.click(fn=unload,outputs=[components.submit_result])

        load_history.click(fn=load_historyf,inputs=[history,count],outputs=[history])
        reload_history.click(fn=load_historyf,inputs=[history,count,components.dtrue],outputs=[history])

        components.msettings=[weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,calcmode,useblocks,custom_name,save_sets,components.id_sets,wpresets,deep,finetune,bake_in_vae,opt_value,inex,ex_blocks,ex_elems]
        components.imagegal = [mgallery,mgeninfo,mhtmlinfo,mhtmllog]
        components.xysettings=[x_type,xgrid,y_type,ygrid,z_type,zgrid,esettings]
        components.genparams=[prompt,neg_prompt,steps,sampler,cfg,seed,width,height,batch_size]
        components.hiresfix = [genoptions,hrupscaler,hr2ndsteps,denois_str,hr_scale]
        components.lucks = [luckmode,lucksets,lucklimits_u,lucklimits_l,luckseed,luckserial,luckcustom,luckround]

        setdefault.click(fn = configdealer,
            inputs =[*components.genparams,*components.hiresfix[1:],components.dfalse],
        )

        resetdefault.click(fn = configdealer,
            inputs =[*components.genparams,*components.hiresfix[1:],components.dtrue],
        )

        resetcurrent.click(fn = lambda x : [gr.update(value = x) for x in RESETVALS] ,outputs =[*components.genparams,*components.hiresfix[1:]],)

        s_reverse.click(fn = reversparams,
            inputs =mergeid,
            outputs = [components.submit_result,*components.msettings[0:8],*components.msettings[9:13],deep,calcmode,luckseed,finetune,opt_value,inex,ex_blocks,ex_elems]
        )

        search.click(fn = searchhistory,inputs=[searchwrods,searchmode],outputs=[history])

        s_reloadreserve.click(fn=nulister,inputs=[components.dfalse],outputs=[components.numaframe])
        s_delreserve.click(fn=nulister,inputs=[s_delnum],outputs=[components.numaframe])
        loadcachelist.click(fn=getcachelist,inputs=[],outputs=[currentcache])
        addtox.click(fn=lambda x:gr.Textbox.update(value = x),inputs=[inputer],outputs=[xgrid])
        addtoy.click(fn=lambda x:gr.Textbox.update(value = x),inputs=[inputer],outputs=[ygrid])
        addtoz.click(fn=lambda x:gr.Textbox.update(value = x),inputs=[inputer],outputs=[zgrid])

        stopgrid.click(fn=freezetime)
        stopmerge.click(fn=freezemtime)

        checkpoints.change(fn=lambda x:",".join(x),inputs=[checkpoints],outputs=[inputer])
        blockids.change(fn=lambda x:" ".join(x),inputs=[blockids],outputs=[inputer])
        calcmodes.change(fn=lambda x:",".join(x),inputs=[calcmodes],outputs=[inputer])

        menbers = [base,in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,mi00,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11]
        menbers_plus = menbers + [resetval]

        lower.change(fn = lambda x: [gr.update(minimum = x) for i in range(len(menbers_plus))],inputs = [lower],outputs = menbers_plus)
        upper.change(fn = lambda x: [gr.update(maximum = x) for i in range(len(menbers_plus))],inputs = [upper],outputs = menbers_plus)

        setalpha.click(fn=slider2text,inputs=[*menbers,wpresets, dd_preset_weight,isxl],outputs=[weights_a])
        setbeta.click(fn=slider2text,inputs=[*menbers,wpresets, dd_preset_weight,isxl],outputs=[weights_b])
        setx.click(fn=add_to_seq,inputs=[xgrid,weights_a],outputs=[xgrid])     
        sety.click(fn=add_to_seq,inputs=[ygrid,weights_b],outputs=[ygrid])

        mode_info = {
            "Weight sum": "A*(1-alpha)+B*alpha",
            "Add difference": "A+(B-C)*alpha",
            "Triple sum": "A*(1-alpha-beta)+B*alpha+C*beta",
            "sum Twice": "(A*(1-alpha)+B*alpha)*(1-beta)+C*beta"
        }
        mode.change(fn=lambda mode,calcmode: [gr.update(info=mode_info[mode]), gr.update(interactive=True if mode in ["Triple sum", "sum Twice"] or calcmode in ["tensor", "tensor2"] else False)], inputs=[mode,calcmode], outputs=[mode, base_beta], show_progress=False)
        calcmode.change(fn=lambda calcmode: gr.update(interactive=True) if calcmode in ["tensor", "tensor2","extract"] else gr.update(), inputs=[calcmode], outputs=base_beta, show_progress=False)
        useblocks.change(fn=lambda mbw: gr.update(visible=False if mbw else True), inputs=[useblocks], outputs=[alpha_group])

        def save_current_merge(custom_name, save_settings):
            msg = savemodel(None,None,custom_name,save_settings)
            return gr.update(value=msg)

        def addblockweights(val, blockopt, *blocks):
            if val == "none":
                val = 0

            value = float(val)

            if "BASE" in blockopt:
                vals = [blocks[0] + value]
            else:
                vals = [blocks[0]]

            if "INP*" in blockopt:
                inp = [blocks[i + 1] + value for i in range(12)]
            else:
                inp = [blocks[i + 1] for i in range(12)]
            vals = vals + inp

            if "MID" in blockopt:
                mid = [blocks[13] + value]
            else:
                mid = [blocks[13]]
            vals = vals + mid

            if "OUT*" in blockopt:
                out = [blocks[i + 14] + value for i in range(12)]
            else:
                out = [blocks[i + 14] for i in range(12)]
            vals = vals + out

            return setblockweights(vals, blockopt)

        def mulblockweights(val, blockopt, *blocks):
            if val == "none":
                val = 0

            value = float(val)

            if "BASE" in blockopt:
                vals = [blocks[0] * value]
            else:
                vals = [blocks[0]]

            if "INP*" in blockopt:
                inp = [blocks[i + 1] * value for i in range(12)]
            else:
                inp = [blocks[i + 1] for i in range(12)]
            vals = vals + inp

            if "MID" in blockopt:
                mid = [blocks[13] * value]
            else:
                mid = [blocks[13]]
            vals = vals + mid

            if "OUT*" in blockopt:
                out = [blocks[i + 14] * value for i in range(12)]
            else:
                out = [blocks[i + 14] for i in range(12)]
            vals = vals + out

            return setblockweights(vals, blockopt)

        def resetblockweights(val, blockopt):
            if val == "none":
                val = 0
            vals = [float(val)] * 26
            return setblockweights(vals, blockopt)

        def setblockweights(vals, blockopt):
            if "BASE" in blockopt:
                ret = [gr.update(value = vals[0])]
            else:
                ret = [gr.update()]

            if "INP*" in blockopt:
                inp = [gr.update(value = vals[i + 1]) for i in range(12)]
            else:
                inp = [gr.update() for _ in range(12)]
            ret = ret + inp

            if "MID" in blockopt:
                mid = [gr.update(value = vals[13])]
            else:
                mid = [gr.update()]
            ret = ret + mid

            if "OUT*" in blockopt:
                out = [gr.update(value = vals[i + 14]) for i in range(12)]
            else:
                out = [gr.update() for _ in range(12)]
            ret = ret + out

            return ret

        def resetvalopt(opt):
            if opt == "none":
                value = 0.0
            else:
                value = float(opt)

            return gr.update(value = value)

        def finetune_update(finetune, detail1, detail2, detail3, contrast, bri, col1, col2, col3):
            arr = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
            tmp = ",".join(map(lambda x: str(int(x)) if x == 0.0 else str(x), arr))
            if finetune != tmp:
                return gr.update(value=tmp)
            return gr.update()

        def finetune_reader(finetune):
            tmp = [t.strip() for t in finetune.split(",")]
            ret = [gr.update()]*7
            for i, f in enumerate(tmp[0:7]):
                try:
                    f = float(f)
                    ret[i] = gr.update(value=f)
                except:
                    pass
            return ret

        # update finetune
        finetunes = [detail1, detail2, detail3, contrast, bri, col1, col2, col3]
        finetune_reset.click(fn=lambda: [gr.update(value="")]+[gr.update(value=0.0)]*8, inputs=[], outputs=[finetune, *finetunes])
        finetune_read.click(fn=finetune_reader, inputs=[finetune], outputs=[*finetunes])
        finetune_write.click(fn=finetune_update, inputs=[finetune, *finetunes], outputs=[finetune])
        detail1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
        detail2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
        detail3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
        contrast.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
        bri.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
        col1.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
        col2.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)
        col3.release(fn=finetune_update, inputs=[finetune, *finetunes], outputs=finetune, show_progress=False)

        savecurrent.click(fn=save_current_merge, inputs=[custom_name, save_sets], outputs=[components.submit_result])

        resetopt.change(fn=resetvalopt,inputs=[resetopt],outputs=[resetval])
        resetweight.click(fn=resetblockweights,inputs=[resetval,resetblockopt],outputs=menbers)
        addweight.click(fn=addblockweights,inputs=[resetval,resetblockopt,*menbers],outputs=menbers)
        mulweight.click(fn=mulblockweights,inputs=[resetval,resetblockopt,*menbers],outputs=menbers)

        readalpha.click(fn=text2slider,inputs=[weights_a,isxl],outputs=menbers)
        readbeta.click(fn=text2slider,inputs=[weights_b,isxl],outputs=menbers)

        dd_preset_weight.change(fn=on_change_dd_preset_weight,inputs=[wpresets, dd_preset_weight],outputs=menbers)
        dd_preset_weight_r.change(fn=on_change_dd_preset_weight_r,inputs=[wpresets, dd_preset_weight_r,luckab],outputs=[weights_a,weights_b])

        def refresh_presets(presets,rand,ab = ""):
            choices = preset_name_list(presets,rand)
            return gr.update(choices = choices)

        preset_refresh.click(fn=refresh_presets,inputs=[wpresets,components.dfalse],outputs=[dd_preset_weight])
        preset_refresh_r.click(fn=refresh_presets,inputs=[wpresets,components.dtrue],outputs=[weights_a,weights_b])

        def changexl(isxl):
            out = [True] * 26
            if isxl:
                for i,id in enumerate(BLOCKID[:-1]):
                    if id not in BLOCKIDXLL[:-1]:
                        out[i] = False
            return [gr.update(visible = x) for x in out]

        isxl.change(fn=changexl,inputs=[isxl], outputs=menbers)

        x_type.change(fn=showxy,inputs=[x_type,y_type,z_type], outputs=[row_blockids,row_checkpoints,row_inputers,ygrid,zgrid,row_blocks,row_calcmode])
        y_type.change(fn=showxy,inputs=[x_type,y_type,z_type], outputs=[row_blockids,row_checkpoints,row_inputers,ygrid,zgrid,row_blocks,row_calcmode])
        z_type.change(fn=showxy,inputs=[x_type,y_type,z_type], outputs=[row_blockids,row_checkpoints,row_inputers,ygrid,zgrid,row_blocks,row_calcmode])
        x_randseednum.change(fn=makerand,inputs=[x_randseednum],outputs=[xgrid])

        import subprocess
        def openeditors():
            subprocess.Popen(['start', filepath], shell=True)

        def reloadpresets():
            try:
                with open(filepath) as f:
                    weights_presets = f.read()
                    choices = preset_name_list(weights_presets)
                    return [weights_presets, gr.update(choices = choices)]
            except OSError as e:
                pass

        def savepresets(text):
            with open(filepath,mode = 'w') as f:
                f.write(text)

        s_reloadtext.click(fn=reloadpresets,inputs=[],outputs=[wpresets, dd_preset_weight])
        s_reloadtags.click(fn=tagdicter,inputs=[wpresets],outputs=[weightstags])
        s_savetext.click(fn=savepresets,inputs=[wpresets],outputs=[])
        s_openeditor.click(fn=openeditors,inputs=[],outputs=[])

        def savexyzpreset_f(xtype, xvals, ytype, yvals, ztype, zvals, name, mode_overwrite):
            new_data = {"xtype": TYPESEG[xtype], "xvalues": xvals,
                                "ytype": TYPESEG[ytype], "yvalues": yvals,
                                "ztype": TYPESEG[ztype], "zvalues": zvals
                                }
            data = get_xyzpreset_data()

            if mode_overwrite:
                data[name] = new_data
            else:
                if name in data:
                    gr.Info(f"Supermerger: Preset {name} already exists.")
                else:
                    data[name] = new_data

            with open(xyzpath, 'w') as file:
                json.dump(data, file, indent=4)
            
            data_keys = list(data.keys())
            return gr.update(choices = sorted(data_keys))
        
        def deletexyzpreset_f(name):
            data = get_xyzpreset_data()

            try: del data[name] 
            except KeyError: gr.Info(f"Supermerger: Preset {name} not found.")

            with open(xyzpath, 'w') as file:
                json.dump(data, file, indent=4)
                
            keys_list = list(data.keys())
            return gr.update(choices = sorted(keys_list))

        def loadxyzpreset_f(name):
                data = get_xyzpreset_data()

                preset_data = data.get(name)
                if not preset_data:
                    gr.Info(f"Supermerger: Preset {name} not found.")
                    return [gr.update(value = x) for x in ["alpha","","none","","none",""]]

                sets = [("xtype"),"xvalues","ytype","yvalues","ztype","zvalues"]

                return [gr.update(value = preset_data.get(x)) for x in sets]
        
        def refreshxyzpresets_f(): 
            return gr.update(choices = get_xyzpreset_keylist())
        
        savexyzpreset_b.click(fn=savexyzpreset_f,inputs=[x_type, xgrid, y_type, ygrid, z_type, zgrid,xyzpresets,savexyzpreset_overwrite],outputs=[xyzpresets])
        loadxyzpreset_b.click(fn=loadxyzpreset_f,inputs=[xyzpresets],outputs=[x_type, xgrid, y_type, ygrid, z_type, zgrid])
        deletexyzpreset_b.click(fn=deletexyzpreset_f,inputs=[xyzpresets],outputs=[xyzpresets])
        refreshxyzpresets_b.click(fn=refreshxyzpresets_f,outputs=[xyzpresets])

    return (supermergerui, "SuperMerger", "supermerger"),

msearch = []
mlist=[]

def loadmetadata(model):
    import json
    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    if ".safetensors" not in checkpoint_info.filename: return "no metadata(not safetensors)"
    sdict = sd_models.read_metadata_from_safetensors(checkpoint_info.filename)
    if sdict == {}: return "no metadata"
    return json.dumps(sdict,indent=4)

def load_historyf(data, count=20, reload=False):
    filepath = os.path.join(path_root,"mergehistory.csv")
    global mlist,msearch
    try:
        with  open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader) # skip header
            row_count = sum(1 for row in reader)
            count = int(count)

            nth = None
            if not reload and data is not None and len(data) > 1:
                old = data.loc[len(data)-1, 'ID']
                if old != '':
                    nth = int(old) - count - 1

            if nth is None:
                msearch = []
                mlist = []
                nth = row_count - count

            f.seek(0)
            next(reader)
            nlist = [raw for n,raw in enumerate(reader, start=1) if n > nth and n <= (nth + count)]
            nlist.reverse()
            for m in nlist:
                msearch.append(" ".join(m))
            maxlen = len(nlist[-1][0])
            for i,m in enumerate(nlist):
                nlist[i][0] = nlist[i][0].zfill(maxlen)
            mlist += nlist
            return mlist
    except:
        return [["no data","",""],]

def searchhistory(words,searchmode):
    outs =[]
    ando = "and" in searchmode
    words = words.split(" ") if " " in words else [words]
    for i, m in  enumerate(msearch):
        hit = ando
        for w in words:
            if ando:
                if w not in m:hit = False
            else:
                if w in m:hit = True
        if hit :outs.append(mlist[i])

    if outs == []:return [["no result","",""],]
    return outs

#msettings=[0 weights_a,1 weights_b,2 model_a,3 model_b,4 model_c,5 base_alpha,6 base_beta,7 mode,8 useblocks,9 custom_name,10 save_sets,11 id_sets,12 wpresets]
#13  deep,14 calcmode,15 luckseed 16:opt_value 17 include/exclude 18: exclude_blocks, 19: exclude_elements
MSETSNUM = 20

def reversparams(id):
    def selectfromhash(hash):
        for model in sd_models.checkpoint_tiles():
            if hash in model:
                return model
        return ""
    try:
        idsets = rwmergelog(id = id)
    except:
        return [gr.update(value = "ERROR: history file could not open"),*[gr.update() for x in range(MSETSNUM)]]
    if type(idsets) == str:
        print("ERROR")
        return [gr.update(value = idsets),*[gr.update() for x in range(MSETSNUM)]]
    if idsets[0] == "ID":return  [gr.update(value ="ERROR: no history"),*[gr.update() for x in range(MSETSNUM)]]
    mgs = idsets[3:]
    if mgs[0] == "":mgs[0] = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
    if mgs[1] == "":mgs[1] = "0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2"
    def cutter(text):
        text = text.replace("[","").replace("]","").replace("'", "") 
        return [x.strip() for x in text.split(",") if x != ""]
    mgs[2] = selectfromhash(mgs[2]) if len(mgs[2]) > 5 else ""
    mgs[3] = selectfromhash(mgs[3]) if len(mgs[3]) > 5 else ""
    mgs[4] = selectfromhash(mgs[4]) if len(mgs[4]) > 5 else ""
    mgs[7] = mgs[7].split(":")[0] # get mode name only
    mgs[8] = mgs[8] =="True"
    mgs[10] = cutter(mgs[10])
    mgs[11] = cutter(mgs[11])
    while len(mgs) < MSETSNUM:
        mgs.append("")
    mgs[13] = "normal" if mgs[13] == "" else mgs[13] 
    mgs[14] = -1 if mgs[14] == "" else mgs[14]
    mgs[16] = 0.3 if mgs[16] == "" else float(mgs[16]) 
    mgs[17] = "Off" if mgs[17] == "" else mgs[17]
    mgs[18] = cutter(mgs[18])
    mgs[18] = [x for x in mgs[18] if x in EXCLUDE_CHOICES + ["print"]]
    return [gr.update(value = "setting loaded") ,*[gr.update(value = x) for x in mgs[0:MSETSNUM]]]

def add_to_seq(seq,maker):
    return gr.Textbox.update(value = maker if seq=="" else seq+"\r\n"+maker)

def load_cachelist():
    text = ""
    for x in checkpoints_loaded.keys():
        text = text +"\r\n"+ x.model_name
    return text.replace("\r\n","",1)

def makerand(num):
    text = ""
    for x in range(int(num)):
        text = text +"-1,"
    text = text[:-1]
    return text

#0 row_blockids, 1 row_checkpoints, 2 row_inputers,3 ygrid, 4 zgrid, 5 row_blocks, 6 row_calcmode
def showxy(x,y,z):
    flags =[False]*7
    t = TYPESEG
    txy = t[x] + t[y] + t[z]
    if "model" in txy : flags[1] = flags[2] = True
    if "pinpoint" in txy : flags[0] = flags[2] = True
    if "clude" in txy in txy : flags[5] = True
    if "calcmode" in txy : flags[6] = True
    if not "none" in t[y] : flags[3] = flags[2] = True
    if not "none" in t[z] : flags[4] = flags[2] = True
    return [gr.update(visible = x) for x in flags]

def get_xyzpreset_data():
    try:
        with open(xyzpath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        with open(xyzpath, 'w') as file:
            json.dump({}, file, indent=4)
        return {}
    
def get_xyzpreset_keylist():
    keys_list = list(get_xyzpreset_data())
    return sorted(keys_list)

def text2slider(text, isxl=False):
    vals = [t.strip() for t in text.split(",")]
    vals = [0 if v in "RUX" else v for v in vals]

    if isxl:
        j = 0
        ret = []
        for i, v in enumerate(ISXLBLOCK):
            if v:
                ret.append(gr.update(value = float(vals[j])))
                j += 1
            else:
                ret.append(gr.update())
        return ret

    return [gr.update(value = float(v)) for v in vals]

def slider2text(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,presets, preset, isxl):
    az = find_preset_by_name(presets, preset)
    if az is not None:
        if any(element in az for element in RANCHA):return az
    numbers = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]
    if isxl:
        newnums = []
        for i,id in enumerate(BLOCKID[:-1]):
            if id in BLOCKIDXLL[:-1]:
                newnums.append(numbers[i])
        numbers = newnums
    numbers = [str(x) for x in numbers]
    return gr.update(value = ",".join(numbers) )

def on_change_dd_preset_weight(presets, preset):
    weights = find_preset_by_name(presets, preset)
    if weights is not None:
        return text2slider(weights)

def on_change_dd_preset_weight_r(presets, preset, ab):
    weights = find_preset_by_name(presets, preset)
    if weights is not None:
        if "none" in ab : return gr.update(),gr.update()
        if "alpha" in ab : return gr.update(value = weights),gr.update()
        if "beta" in ab : return gr.update(),gr.update(value = weights)
    return gr.update(),gr.update()

RANCHA = ["R","U","X"]

def tagdicter(presets, rand = False):
    presets=presets.splitlines()
    wdict={}
    for l in presets:
        w=""
        if ":" in l :
            key = l.split(":",1)[0]
            w = l.split(":",1)[1]
        if "\t" in l:
            key = l.split("\t",1)[0]
            w = l.split("\t",1)[1]
        if len([w for w in w.split(",")]) == 26:
            if rand and not any(element in w for element in RANCHA) : continue
            wdict[key.strip()]=w
    return ",".join(list(wdict.keys()))

def preset_name_list(presets, rand = False):
    return tagdicter(presets, rand).split(",")

def find_preset_by_name(presets, preset):
    presets = presets.splitlines()
    for l in presets:
        if ":" in l:
            key = l.split(":",1)[0]
            w = l.split(":",1)[1]
        elif "\t" in l:
            key = l.split("\t",1)[0]
            w = l.split("\t",1)[1]
        else:
            continue
        if key == preset and len([w for w in w.split(",")]) == 26:
            return w

    return None

BLOCKID=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11","Not Merge"]
BLOCKIDXL=['BASE', 'IN0', 'IN1', 'IN2', 'IN3', 'IN4', 'IN5', 'IN6', 'IN7', 'IN8', 'M', 'OUT0', 'OUT1', 'OUT2', 'OUT3', 'OUT4', 'OUT5', 'OUT6', 'OUT7', 'OUT8', 'VAE']
BLOCKIDXLL=['BASE', 'IN00', 'IN01', 'IN02', 'IN03', 'IN04', 'IN05', 'IN06', 'IN07', 'IN08', 'M00', 'OUT00', 'OUT01', 'OUT02', 'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'VAE']
ISXLBLOCK=[True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, True,   True,   True,   True,   True,   True,   True,   True,   True,   True,  False,  False,  False]

def modeltype(sd):
    if "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in sd.keys():
        modeltype = "XL"
    else:
        modeltype = "1.X or 2.X"
    return modeltype

def loadkeys(model_a, lora):
    if lora:
        import lora
        sd = sd_models.read_state_dict(lora.available_loras[model_a].filename,"cpu")
    else:
        sd = loadmodel(model_a)
    keys = []
    mtype = modeltype(sd)
    if lora:
        for i, key in enumerate(sd.keys()):
            keys.append([i,"LoRA",key,sd[key].shape])
    else:    
        for i, key in enumerate(sd.keys()):
            keys.append([i,blockfromkey(key,mtype),key,sd[key].shape])

    return keys

def loadmodel(model):
    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    sd = sd_models.read_state_dict(checkpoint_info.filename,"cpu")
    return sd

ADDRAND = "\n\
ALL_R	R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R\n\
ALL_U	U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U,U\n\
ALL_X	X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X\n\
"

def calccosinedif(model_a,model_b,mode,settings,include,calc):
    inc = " ".join(include)
    settings = " ".join(settings)
    a, b = loadmodel(model_a), loadmodel(model_b)
    name = filenamecutter(model_a) + "-" + filenamecutter(model_b)
    cosine_similarities = []
    blocksim = {}
    blockvals = []
    attn2 = {}
    isxl = "XL" == modeltype(a)
    blockids = BLOCKIDXLL if isxl else BLOCKID
    for bl in blockids:
        blocksim[bl] = []
    blocksim["VAE"] = []

    if "ASim" in mode:
        result = asimilarity(a,b,isxl)
        if len(settings) > 1: savecalc(result,name,settings,True,"Asim")
        del a ,b
        gc.collect()
        return result
    else:
        for key in tqdm(a.keys(), desc="Calculating cosine similarity"):
            block = None
            if blockfromkey(key,isxl) == "Not Merge": continue
            if "model_ema" in key: continue
            if "model" not in key:continue
            if "first_stage_model" in key and not ("VAE" in inc):
                continue
            elif "first_stage_model" in key and "VAE" in inc:
                block = "VAE"
            if "diffusion_model" in key and not ("U-Net" in inc): continue
            if "encoder" in key and not ("encoder" in inc): continue
            if key in b and a[key].size() == b[key].size():
                a_flat = a[key].view(-1).to(torch.float32)
                b_flat = b[key].view(-1).to(torch.float32)
                simab = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))
                if block is None: block,blocks26 = blockfromkey(key,isxl)
                if block =="Not Merge" :continue
                cosine_similarities.append([block, key, round(simab.item()*100,3)])
                blocksim[blocks26].append(round(simab.item()*100,3))
                if "attn2.to_out.0.weight" in key: attn2[block] = round(simab.item()*100,3)

        for bl in blockids:
            val = None
            if bl == "Not Merge": continue
            if bl not in blocksim.keys():continue
            if blocksim[bl] == []: continue
            if "Mean" in calc:
                val = mean(blocksim[bl])
            elif "Min" in calc:
                val = min(blocksim[bl])
            else:
                if bl in attn2.keys():val = attn2[bl]
            if val:blockvals.append([bl,"",round(val,3)])
            if mode != "Element": cosine_similarities.insert(0,[bl,"",round(mean(blocksim[bl]),3)])

        if mode == "Block":
            if len(settings) > 1: savecalc(blockvals,name,settings,True,"Blocks")
            del a ,b
            gc.collect()
            return blockvals
        else:
            if len(settings) > 1: savecalc(cosine_similarities,name,settings,False,"Elements",)
            del a ,b
            gc.collect()
            return cosine_similarities

def savecalc(data,name,settings,blocks,add):
    name = name + "_" + add
    csvpath = os.path.join(path_root,f"{name}.csv")
    txtpath = os.path.join(path_root,f"{name}.txt")

    txt = ""
    for row in data:
        row = [str(r) for r in row]
        txt = txt + ",".join(row)+"\n"
        if blocks: txt = txt.replace(",,",",")

    if "txt" in settings:
        with  open(txtpath, 'w+') as f:
            f.writelines(txt)
            print("file saved to ",txtpath)
    if "csv" in settings:
        with  open(csvpath, 'w+') as f:
            f.writelines(txt)
            print("file saved to ",csvpath)

#code from https://huggingface.co/JosephusCheung/ASimilarityCalculatior

def cal_cross_attn(to_q, to_k, to_v, rand_input):
    hidden_dim, embed_dim = to_q.shape
    attn_to_q = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_k = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_v = nn.Linear(hidden_dim, embed_dim, bias=False)
    attn_to_q.load_state_dict({"weight": to_q})
    attn_to_k.load_state_dict({"weight": to_k})
    attn_to_v.load_state_dict({"weight": to_v})
    
    return torch.einsum(
        "ik, jk -> ik", 
        F.softmax(torch.einsum("ij, kj -> ik", attn_to_q(rand_input), attn_to_k(rand_input)), dim=-1),
        attn_to_v(rand_input)
    )
       
def eval(model, n, input, block):
    qk = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_q.weight"
    uk = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_k.weight"
    vk = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_v.weight"
    atoq, atok, atov = model[qk], model[uk], model[vk]

    attn = cal_cross_attn(atoq, atok, atov, input)
    return attn

ATTN1BLOCKS = [[1,"input"],[2,"input"],[4,"input"],[5,"input"],[7,"input"],[8,"input"],["","middle"],
[3,"output"],[4,"output"],[5,"output"],[6,"output"],[7,"output"],[8,"output"],[9,"output"],[10,"output"],[11,"output"]]

def asimilarity(model_a,model_b,mtype):
    torch.manual_seed(2244096)
    sims = []
  
    for nblock in  tqdm(ATTN1BLOCKS, desc="Calculating cosine similarity"):
        n,block = nblock[0],nblock[1]
        if n != "": n = f"s.{n}"
        key = f"model.diffusion_model.{block}_block{n}.1.transformer_blocks.0.attn1.to_q.weight"

        hidden_dim, embed_dim = model_a[key].shape
        rand_input = torch.randn([embed_dim, hidden_dim])

        attn_a = eval(model_a, n, rand_input, block)
        attn_b = eval(model_b, n, rand_input, block)

        sim = torch.mean(torch.cosine_similarity(attn_a, attn_b))
        sims.append([blockfromkey(key,mtype),"",round(sim.item() * 100,3)])
        
    return sims

CONFIGS = ["prompt","neg_prompt","Steps","Sampling method","CFG scale","Seed","Width","Height","Batch size","Upscaler","Hires steps","Denoising strength","Upscale by"]
RESETVALS = ["","",0," ",0,0,0,0,1,"Latent",0,0.7,2]

def configdealer(prompt,neg_prompt,steps,sampler,cfg,seed,width,height,batch_size,
                        hrupscaler,hr2ndsteps,denois_str,hr_scale,reset):

    data = [prompt,neg_prompt,steps,sampler,cfg,seed,width,height,batch_size,
                        hrupscaler,hr2ndsteps,denois_str,hr_scale]

    current_directory = os.getcwd()
    jsonpath = os.path.join(current_directory,"ui-config.json")
    print(jsonpath)

    with open(jsonpath, 'r') as file:
        json_data = json.load(file)

    for name,men,default in zip(CONFIGS,data,RESETVALS):
        key = f"supermerger/{name}/value"
        json_data[key] = default if reset else men

    with open(jsonpath, 'w') as file:
        json.dump(json_data, file, indent=4)

sorted_output = []

def encodetexts(exclude):
    isxl = hasattr(shared.sd_model,"conditioner")
    model = shared.sd_model.conditioner.embedders[0] if isxl else shared.sd_model.cond_stage_model
    encoder = model.encode_with_transformers
    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    byte_decoder = tokenizer.byte_decoder

    batch = 500

    b_texts = [list(vocab.items())[i:i + batch] for i in range(0, len(vocab), batch)]

    output = []

    for texts in tqdm(b_texts):    
        batch = []
        words = []
        for word, idx in texts:
            tokens = [model.id_start, idx, model.id_end] + [model.id_end] * 74
            batch.append(tokens)
            words.append((idx, word))
        
        embedding = encoder(torch.IntTensor(batch).to("cuda"))[:,1,:] # (bs,768)
        embedding = embedding.to('cuda')
        emb_norms = torch.linalg.vector_norm(embedding, dim=-1) # (bs,)
        
        for i, (word, token) in enumerate(texts):
            try:
                word = bytearray([byte_decoder[x] for x in word]).decode("utf-8")
            except UnicodeDecodeError:
                pass
            if exclude:
                if has_alphanumeric(word) : output.append([word,token,emb_norms[i].item()])
            else:
                output.append([word,token,emb_norms[i].item()])

    output = sorted(output, key=lambda x: x[2], reverse=True)
    for i in range(len(output)):
        output[i].insert(0,i)

    global sorted_output
    sorted_output = output

    return output[:1000]

def pickupencode(texts):
    wordlist = [x[1] for x in sorted_output]
    texts = texts.split(",")
    output = []
    for text in texts:
        if text in wordlist:
            output.append(sorted_output[wordlist.index(text)])
        if text+"</w>" in wordlist:
            output.append(sorted_output[wordlist.index(text+"</w>")])
    return output

def has_alphanumeric(text):
    pattern = re.compile(r'[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;"\'<>,.?/\|\\]')
    return bool(pattern.search(text.replace("</w>","")))

if __package__ == "supermerger":
    script_callbacks.on_ui_tabs(on_ui_tabs)
