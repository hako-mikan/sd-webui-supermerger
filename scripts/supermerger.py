from linecache import clearcache
import safetensors.torch
import gradio as gr
import html
import os
import gc
import os.path
import argparse
import re
import torch
import modules.ui
import tqdm
import random
from scripts.mergers.mergers import genxyplot,runrun,mergen,savemerged

from PIL import Image, PngImagePlugin
from collections import namedtuple
from tqdm import tqdm
from torch import Tensor
from modules import shared, devices, sd_hijack, processing, sd_models, images, sd_samplers,sd_vae,script_callbacks
from modules.ui import create_refresh_button, create_output_panel, plaintext_to_html
from modules.shared import opts,state
from modules.processing import create_infotext,Processed
from modules.sd_models import  load_model,checkpoints_loaded,CheckpointInfo

gensets=argparse.Namespace()

def on_ui_train_tabs(params):
    txt2img_preview_params=params.txt2img_preview_params
    gensets.txt2img_preview_params=txt2img_preview_params
    return None

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="<p>Merge models and load it for generation</p>")

                with gr.Row():
                    model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model A",interactive=True)
                    create_refresh_button(model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                    model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model B",interactive=True)
                    create_refresh_button(model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                    model_c = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model C",interactive=True)
                    create_refresh_button(model_c, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")

                device = gr.Textbox(label="device", elem_id="device",value ="cup",visible = False)
                mode = gr.Radio(label = "Merge Mode",choices = ["Weight sum(A*(1-alpha)+B*alpha", "Add difference(A+(B-C)*alpha)"], value = "Weight sum(A*(1-alpha)+B*alpha") 
                with gr.Row(): 
                    useblocks =  gr.Checkbox(label="by blocks(Setting input is below)")
                    base_alpha = gr.Slider(label="alpha(Ignored in case of block merge)", minimum=0, maximum=1, step=0.001, value=0.5)
                    #weights = gr.Textbox(label="weights,base alpha,IN00,IN02,...IN11,M00,OUT00,...,OUT11",lines=2,value="0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
                with gr.Row():
                    with gr.Column(variant='panel'):
                        merge = gr.Button(elem_id="model_merger_merge", value="Merge!",variant='primary')
                        mergeandgen = gr.Button(elem_id="model_merger_merge", value="Merge And Gen",variant='primary')
                    with gr.Column(variant='panel'):
                        gen = gr.Button(elem_id="model_merger_merge", value="Gen",variant='primary')
                        savemodel = gr.Button(elem_id="model_merger_merge", value="Save Current Model",variant='primary')
                    save = gr.Checkbox(label="save checkpoint after merge")
                    overwrite =  gr.Checkbox(label="allow overwrite")
                    custom_name = gr.Textbox(label="Custom Name (Optional)", elem_id="model_converter_custom_name")
                xgrid = gr.Textbox(label="Sequential Merge Parameters",lines=3,value="0.25,0.5,0.75")
                ygrid = gr.Textbox(label="Y grid (Disabled if blank)",lines=3,value="",visible=False)
                gengrid = gr.Button(elem_id="model_merger_merge", value="Sequential Merge and Generation",variant='primary')
                

                with gr.Row():
                    currentcache = gr.Textbox(label="Current Cache")
                    loadcachelist = gr.Button(elem_id="model_merger_merge", value="Reload Cache List",variant='primary')
                addtoseq = gr.Button(elem_id="copytogen", value="Add to Sequence",variant='primary')
                weights = gr.Textbox(label="weights,base alpha,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")

                dummytrue =  gr.Checkbox(value = True, visible = False)                
                dummyfalse =  gr.Checkbox(value = False,visible = False)     
                dummy_t =  gr.Textbox(value = "",visible = False)    

            with gr.Column(variant='panel'):
                currentmodel = gr.Textbox(label="Current Model",lines=1,value="")  
                submit_result = gr.Textbox(label="Message")
                merge_gallery, merge_generation_info, merge_html_info, merge_html_log = create_output_panel("txt2img", opts.outdir_txt2img_samples)

        with gr.Row():
            aa00= gr.Slider(label="Base", minimum=0, maximum=1, step =0.01, value=0.5)
            in00 = gr.Slider(label="IN00", minimum=0, maximum=1, step=0.01, value=0.5)
            in01 = gr.Slider(label="IN01", minimum=0, maximum=1, step=0.01, value=0.5)
            in02 = gr.Slider(label="IN02", minimum=0, maximum=1, step=0.01, value=0.5)
            in03 = gr.Slider(label="IN03", minimum=0, maximum=1, step=0.01, value=0.5)
            in04 = gr.Slider(label="IN04", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Row():
            in05 = gr.Slider(label="IN05", minimum=0, maximum=1, step=0.01, value=0.5)
            in06 = gr.Slider(label="IN06", minimum=0, maximum=1, step=0.01, value=0.5)
            in07 = gr.Slider(label="IN07", minimum=0, maximum=1, step=0.01, value=0.5)
            in08 = gr.Slider(label="IN08", minimum=0, maximum=1, step=0.01, value=0.5)
            in09 = gr.Slider(label="IN09", minimum=0, maximum=1, step=0.01, value=0.5)
            in10 = gr.Slider(label="IN10", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Row():
            in11 = gr.Slider(label="IN11", minimum=0, maximum=1, step=0.01, value=0.5)
            mi00 = gr.Slider(label="M00", minimum=0, maximum=1, step=0.01, value=0.5)
            ou00 = gr.Slider(label="OUT00", minimum=0, maximum=1, step=0.01, value=0.5)
            ou01 = gr.Slider(label="OUT01", minimum=0, maximum=1, step=0.01, value=0.5)
            ou02 = gr.Slider(label="OUT02", minimum=0, maximum=1, step=0.01, value=0.5)
            ou03 = gr.Slider(label="OUT03", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Row():
            ou04 = gr.Slider(label="OUT04", minimum=0, maximum=1, step=0.01, value=0.5)
            ou05 = gr.Slider(label="OUT05", minimum=0, maximum=1, step=0.01, value=0.5)
            ou06 = gr.Slider(label="OUT06", minimum=0, maximum=1, step=0.01, value=0.5)
            ou07 = gr.Slider(label="OUT07", minimum=0, maximum=1, step=0.01, value=0.5)
            ou08 = gr.Slider(label="OUT08", minimum=0, maximum=1, step=0.01, value=0.5)
            ou09 = gr.Slider(label="OUT09", minimum=0, maximum=1, step=0.01, value=0.5)       
        with gr.Row(): 
            ou10 = gr.Slider(label="OUT10", minimum=0, maximum=1, step=0.01, value=0.5)
            ou11 = gr.Slider(label="OUT11", minimum=0, maximum=1, step=0.01, value=0.5)

        with gr.Row():
            gr.HTML(value="<p> exampls: Change base alpha from 0.1 to 0.9 <br>0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9<br>If you want to display the original model as well for comparison<br>0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1</p>")
            gr.HTML(value="<p> For block-by-block merging <br>0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5<br>1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1<br>0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1</p>")


            merge.click(
                fn=mergen,
                inputs=[weights,model_a,model_b,model_c,device,base_alpha,custom_name,mode,overwrite,save,useblocks],
                outputs=[submit_result,currentmodel]
            )

            mergeandgen.click(
                fn=mergen,
                inputs=[weights,model_a,model_b,model_c,device,base_alpha,custom_name,mode,overwrite,save,useblocks,*gensets.txt2img_preview_params,currentmodel],
                outputs=[submit_result,merge_gallery,merge_generation_info,merge_html_info,merge_html_log,currentmodel]
            )

            gen.click(
                fn=runrun,
                inputs=[*gensets.txt2img_preview_params,currentmodel],
                outputs=[merge_gallery,merge_generation_info,merge_html_info,merge_html_log],
            )

            savemodel.click(
                fn=savemerged,
                inputs=[custom_name,overwrite],
                outputs=[submit_result,dummy_t],
            )

            gengrid.click(
                fn=genxyplot,
                inputs=[xgrid,ygrid,weights,model_a,model_b,model_c,base_alpha,mode,useblocks,*gensets.txt2img_preview_params],
                outputs=[submit_result,merge_gallery,merge_generation_info,merge_html_info,merge_html_log,currentmodel],
            )

            loadcachelist.click(fn=load_cachelist,inputs=[],outputs=[currentcache])
            addtoseq.click(fn=add_to_seq,inputs=[xgrid,weights],outputs=[xgrid])

            menbers = [aa00,in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,mi00,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11]
            aa00.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in00.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in01.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in02.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in03.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in04.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in05.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in06.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in07.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in08.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in09.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in10.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            in11.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            mi00.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou00.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou01.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou02.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou03.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou04.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou05.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou06.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou07.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou08.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou09.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou11.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])
            ou10.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights])

    return [(ui, "SuperMerger", "SuperMerger")]

def add_to_seq(seq,maker):
    if seq=="":
        text = maker
    else:
        text = seq+"\r\n"+maker
    return gr.Textbox.update(value = text)

def reload_mbmaker(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z):
    numbers = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]
    text=",".join(str(s) for s in numbers)
    return gr.Textbox.update(value = text)

def load_cachelist():
    text = ""
    for x in checkpoints_loaded.keys():
        text = text +"\r\n"+ x.model_name
    return text.replace("\r\n","",1)

script_callbacks.on_ui_train_tabs(on_ui_train_tabs)
script_callbacks.on_ui_tabs(on_ui_tabs)