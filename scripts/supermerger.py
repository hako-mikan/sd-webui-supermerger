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
from scripts.mergers.mergers import genxyplot,runrun,mergen,savemerged,freezetime,typesg

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
                mode = gr.Radio(label = "Merge Mode",choices = ["Weight sum:A*(1-alpha)+B*alpha", "Add difference:A+(B-C)*alpha",
                                                    "Triple sum:A*(1-alpha-beta)+B*alpha+C*beta","sum Twice:(A*(1-alpha)+B*alpha)*(1-beta)+C*beta"], value = "Weight sum:A*(1-alpha)+B*alpha") 
                with gr.Row(): 
                    useblocks =  gr.Checkbox(label="use MBW")
                    base_alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=0.5)
                    base_beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=0.25)
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
                with gr.Row():
                    x_type = gr.Dropdown(label="X type", choices=[x for x in typesg], value="alpha", type="index")
                    x_randseednum = gr.Number(value=3, label="number of -1", interactive=True, visible = True)
                xgrid = gr.Textbox(label="Sequential Merge Parameters",lines=3,value="0.25,0.5,0.75")
                y_type = gr.Dropdown(label="Y type", choices=[y for y in typesg], value="none", type="index")    
                ygrid = gr.Textbox(label="Y grid (Disabled if blank)",lines=3,value="",visible =False)
                with gr.Row():
                    gengrid = gr.Button(elem_id="model_merger_merge", value="Sequential XY Merge and Generation",variant='primary')
                    stopgrid = gr.Button(elem_id="model_merger_merge", value="Stop XY",variant='primary')
                dummytrue =  gr.Checkbox(value = True, visible = False)                
                dummyfalse =  gr.Checkbox(value = False,visible = False)     
                dummy_t =  gr.Textbox(value = "",visible = False)    
            blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
    
            with gr.Column(variant='panel'):
                currentmodel = gr.Textbox(label="Current Model",lines=1,value="")  
                submit_result = gr.Textbox(label="Message")
                merge_gallery, merge_generation_info, merge_html_info, merge_html_log = create_output_panel("txt2img", opts.outdir_txt2img_samples)
        with gr.Row():
            modelnames = gr.Textbox(label="",lines=2,value="",visible =False)
            addtox = gr.Button(elem_id="copytogen", value="Add model names to Sequence X",variant='primary',visible =False)
            addtoy = gr.Button(elem_id="copytogen", value="Add model names to Sequence Y",variant='primary',visible =False)
        with gr.Row():
            blockids = gr.CheckboxGroup(label = "checkpoint",choices=[x for x in blockid],type="value",interactive=True,visible = False)
        with gr.Row():
            checkpoints = gr.CheckboxGroup(label = "checkpoint",choices=[x.model_name for x in modules.sd_models.checkpoints_list.values()],type="value",interactive=True,visible = False)
        with gr.Row():
            addtoseq = gr.Button(elem_id="copytogen", value="Add weights to Sequence X",variant='primary')
        with gr.Row():
            weights_a = gr.Textbox(label="weights for alpha, base alpha,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
            weights_b = gr.Textbox(label="weights,for beta, base beta,IN00,IN02,...IN11,M00,OUT00,...,OUT11",value = "0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2")
        with gr.Row():
            editweights = gr.Radio(label = "edit",choices = ["alpha","beta"], value = "alpha") 
            base= gr.Slider(label="Base", minimum=0, maximum=1, step =0.01, value=0.5)
            in00 = gr.Slider(label="IN00", minimum=0, maximum=1, step=0.01, value=0.5)
            in01 = gr.Slider(label="IN01", minimum=0, maximum=1, step=0.01, value=0.5)
            in02 = gr.Slider(label="IN02", minimum=0, maximum=1, step=0.01, value=0.5)
            in03 = gr.Slider(label="IN03", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Row():
            in04 = gr.Slider(label="IN04", minimum=0, maximum=1, step=0.01, value=0.5)
            in05 = gr.Slider(label="IN05", minimum=0, maximum=1, step=0.01, value=0.5)
            in06 = gr.Slider(label="IN06", minimum=0, maximum=1, step=0.01, value=0.5)
            in07 = gr.Slider(label="IN07", minimum=0, maximum=1, step=0.01, value=0.5)
            in08 = gr.Slider(label="IN08", minimum=0, maximum=1, step=0.01, value=0.5)
            in09 = gr.Slider(label="IN09", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Row():
            in10 = gr.Slider(label="IN10", minimum=0, maximum=1, step=0.01, value=0.5)
            in11 = gr.Slider(label="IN11", minimum=0, maximum=1, step=0.01, value=0.5)
            mi00 = gr.Slider(label="M00", minimum=0, maximum=1, step=0.01, value=0.5)
            ou00 = gr.Slider(label="OUT00", minimum=0, maximum=1, step=0.01, value=0.5)
            ou01 = gr.Slider(label="OUT01", minimum=0, maximum=1, step=0.01, value=0.5)
            ou02 = gr.Slider(label="OUT02", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Row():
            ou03 = gr.Slider(label="OUT03", minimum=0, maximum=1, step=0.01, value=0.5)
            ou04 = gr.Slider(label="OUT04", minimum=0, maximum=1, step=0.01, value=0.5)
            ou05 = gr.Slider(label="OUT05", minimum=0, maximum=1, step=0.01, value=0.5)
            ou06 = gr.Slider(label="OUT06", minimum=0, maximum=1, step=0.01, value=0.5)
            ou07 = gr.Slider(label="OUT07", minimum=0, maximum=1, step=0.01, value=0.5)
            ou08 = gr.Slider(label="OUT08", minimum=0, maximum=1, step=0.01, value=0.5)
        with gr.Row(): 
            ou09 = gr.Slider(label="OUT09", minimum=0, maximum=1, step=0.01, value=0.5)       
            ou10 = gr.Slider(label="OUT10", minimum=0, maximum=1, step=0.01, value=0.5)
            ou11 = gr.Slider(label="OUT11", minimum=0, maximum=1, step=0.01, value=0.5)

        with gr.Row():
            gr.HTML(value="<p> exampls: Change base alpha from 0.1 to 0.9 <br>0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9<br>If you want to display the original model as well for comparison<br>0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1</p>")
            gr.HTML(value="<p> For block-by-block merging <br>0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5<br>1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1<br>0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1</p>")
        
        with gr.Row():
            currentcache = gr.Textbox(label="Current Cache")
            loadcachelist = gr.Button(elem_id="model_merger_merge", value="Reload Cache List",variant='primary')

            merge.click(
                fn=mergen,
                inputs=[weights_a,weights_b,model_a,model_b,model_c,device,base_alpha,base_beta,custom_name,mode,overwrite,save,useblocks],
                outputs=[submit_result,currentmodel]
            )

            mergeandgen.click(
                fn=mergen,
                inputs=[weights_a,weights_b,model_a,model_b,model_c,device,base_alpha,base_beta,custom_name,mode,overwrite,save,useblocks,*gensets.txt2img_preview_params,currentmodel],
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
                inputs=[xgrid,ygrid,x_type,y_type,weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,useblocks,*gensets.txt2img_preview_params],
                outputs=[submit_result,merge_gallery,merge_generation_info,merge_html_info,merge_html_log,currentmodel],
            )

            loadcachelist.click(fn=load_cachelist,inputs=[],outputs=[currentcache])
            addtox.click(fn=lambda x:gr.Textbox.update(value = x),inputs=[modelnames],outputs=[xgrid])
            addtoy.click(fn=lambda x:gr.Textbox.update(value = x),inputs=[modelnames],outputs=[ygrid])
            addtoseq.click(fn=add_to_seq,inputs=[xgrid,weights_a],outputs=[xgrid])
            stopgrid.click(fn=freezetime)

            checkpoints.change(fn=lambda x:",".join(x),inputs=[checkpoints],outputs=[modelnames])
            blockids.change(fn=lambda x:" ".join(x),inputs=[checkpoints],outputs=[modelnames])

            menbers = [weights_a,weights_b,editweights,base,in00,in01,in02,in03,in04,in05,in06,in07,in08,in09,in10,in11,mi00,ou00,ou01,ou02,ou03,ou04,ou05,ou06,ou07,ou08,ou09,ou10,ou11]
            base.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in00.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in01.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in02.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in03.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in04.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in05.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in06.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in07.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in08.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in09.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in10.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            in11.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            mi00.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou00.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou01.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou02.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou03.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou04.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou05.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou06.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou07.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou08.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou09.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou11.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            ou10.change(fn=reload_mbmaker,inputs=[x for x in menbers],outputs=[weights_a,weights_b])
            editweights.change(fn=changesliders,inputs=[weights_a,weights_b,editweights],outputs=[x for x in menbers[3:]])

            x_type.change(fn=showxy,inputs=[x_type], outputs=[checkpoints,modelnames,addtox,addtoy])
            y_type.change(fn=showxy,inputs=[y_type], outputs=[checkpoints,modelnames,addtox,addtoy,ygrid])
            checkpoints.change(fn=lambda x:",".join(x),inputs=[checkpoints],outputs=[modelnames])
            x_randseednum.change(fn=makerand,inputs=[x_randseednum],outputs=[xgrid])

    return [(ui, "SuperMerger", "SuperMerger")]

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

def showxy(type):
    if typesg[type] !="none":
        if  "model" in typesg[type] :
            print("koko")
            return [gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)]  
        else:
            return [gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=True)]
    else:
        return [gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)]

def changesliders(texta,textb,target):
    text = texta if target =="alpha" else textb
    vals = [t.strip() for t in text.split(",")]
    return [gr.update(value = float(v)) for v in vals]

def reload_mbmaker(weights_a,weights_b,editweights,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z):
    numbers = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]
    text=",".join(str(s) for s in numbers)
    if editweights == "alpha" :
        if text ==weights_a:return gr.Textbox.update() ,gr.Textbox.update() 
        return gr.Textbox.update(value = text),gr.Textbox.update() 
    else:
        if text ==weights_b:return gr.Textbox.update() ,gr.Textbox.update() 
        return gr.Textbox.update(),gr.Textbox.update(value = text)

script_callbacks.on_ui_train_tabs(on_ui_train_tabs)
script_callbacks.on_ui_tabs(on_ui_tabs)
