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

from PIL import Image, PngImagePlugin
from collections import namedtuple
from tqdm import tqdm
from torch import Tensor
from modules import shared, devices, sd_hijack, processing, sd_models, images, sd_samplers,sd_vae,script_callbacks
from modules.ui import create_refresh_button, create_output_panel, plaintext_to_html
from modules.shared import opts,state,restricted_opts
from modules.processing import create_infotext,Processed
from modules.sd_models import  load_model,checkpoints_loaded,CheckpointInfo

mergedmodel=[]
temp_p=None

def genxyplot(xmen:str,ymen:str,weights:list,model_a,model_b,model_c,alpha,mode,blocks,prompt,nprompt,steps,sampler,cfg,seed,w,h):
    if xmen =="":
        return "parameter is empty",None

    if seed == -1:
        seed = int(random.randrange(4294967294))

    image=None

    if blocks:
        xs = xmen.splitlines()
    else:
        if "\n"in xmen:
            return "error in parameter, clear line included",None
        xs = [x.strip() for x in xmen.split(',')]
        ys = [x.strip() for x in ymen.split(',')]

    for x in xs:
        if "," in x:
            weights=x
            xx = x.split(',',1)
            if len(xx) != 26:
                "error in parameter, number of parameter is not 26"
            x  = xx[0]
            print(x)
            print(weights)
 
        image_temp=mergen(weights, model_a, model_b,model_c, "cpu", float(x), "",mode,False,False,blocks,prompt, nprompt, steps, sampler, cfg, seed, w, h,"")
        if image ==None:
            image = image_temp
        else:
            image[1].append(*image_temp[1])
        print("alpha = "+x +" end")
    
    if blocks:
        if "Add" in mode:
            currentmodel =model_a[:model_a.find('.ckpt')]+" + (" +model_b[:model_b.find('.ckpt')]+" - " +model_c[:model_c.find('.ckpt')]+")"+ " x "+"alpha"
        else:
            currentmodel =model_a[:model_a.find('.ckpt')]+" x (1-alpha) + " +model_b[:model_b.find('.ckpt')] + " x "+"alpha"
    else:
        if "Add" in mode:
            currentmodel =model_a[:model_a.find('.ckpt')]+" + (" +model_b[:model_b.find('.ckpt')]+" - " +model_c[:model_c.find('.ckpt')]+")"+ " x "+"alpha"
        else:
            currentmodel =model_a[:model_a.find('.ckpt')]+" x "+"(1-alpha)"+" + " +model_b[:model_b.find('.ckpt')] + " x "+"alpha"

    ys = [currentmodel]
    image[1].insert(0,makegrid(image[1],xs,ys))
    return image

def runrun(prompt, nprompt, steps, sampler, cfg, seed, w, h,mergeinfo=""):
    shared.state.begin()
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        do_not_save_grid=False,
        do_not_save_samples=True,
        do_not_reload_embeddings=True,
    )
    p.batch_size = 1
    p.prompt = prompt
    p.negative_prompt = nprompt
    p.steps = steps
    p.sampler_name = sd_samplers.samplers[sampler].name
    p.cfg_scale = cfg
    p.seed = seed
    p.width = w
    p.height = h
    p.seed_resize_from_w=0
    p.seed_resize_from_h=0
    p.denoising_strength=None

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    processed:Processed = processing.process_images(p)
    image = processed.images[0]
    infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds)
    if infotext.count("Steps: ")>1:
        infotext = infotext[:infotext.rindex("Steps")]

    infotexts = infotext.split(",")
    for i,x in enumerate(infotexts):
        if "Model:"in x:
            infotexts[i] = " Model: "+mergeinfo
    infotext= ",".join(infotexts)
    images.save_image(image, opts.outdir_txt2img_samples, "",p.seed, p.prompt,shared.opts.samples_format, infotext, p=p)
    shared.state.end()
    global temp_p
    temp_p=p
    return processed.images,infotext,plaintext_to_html(processed.info), plaintext_to_html(processed.comments)

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

def mergen(weights:list, model_a, model_b,model_c, device="cpu", base_alpha=0.5, output_file="",mode="Weight", 
overwrite=False,save=True,useblocks=False,prompt="none", nprompt="", steps=20, sampler=0, cfg=8, seed=-1, w=512, h=512,
currentmodel=""):
 
    global mergedmodel

    mergedmodel = [weights,model_a, model_b,model_c, device, base_alpha, output_file,mode, overwrite,True,useblocks]

    weights_t = weights.split(',',1)
    if useblocks:
        base_alpha  = float(weights_t[0])    
        print("block merge : "+weights)
    else:
        print("merge : "+str(base_alpha))
    weights=weights_t[1]

    if model_a =="" or model_b =="" or ("Add" in mode and model_c==""):
        print("no model selected")
        return "model is no selected",None,None,None,None,currentmodel

    if weights is None:
        weights = None
    else:
        weights = [float(w) for w in weights.split(',')]
    
    if len(weights) != NUM_TOTAL_BLOCKS:
        return f"weights value must be {NUM_TOTAL_BLOCKS+1}.",None,None,None,None,currentmodel

    device = device if device in ["cpu", "cuda"] else "cpu"

 
    theta_1=load_model_weights_m(model_b,False,True,save).copy()

    if "Add" in mode:
        print(f"Loading", model_c)
        theta_2 = load_model_weights_m(model_c,False,False,save).copy()

        for key in tqdm(theta_1.keys()):
            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
                    theta_1[key] = theta_1[key]- t2
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])
        del theta_2

    theta_0=load_model_weights_m(model_a,True,False,save).copy()

    alpha = base_alpha
    print(alpha)
    re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
    re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
    re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

    chckpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

    count_target_of_basealpha = 0
    for key in (tqdm(theta_0.keys(), desc="Stage 1/2") if not False else theta_0.keys()):
        if "model" in key and key in theta_1:
            current_alpha = alpha

            if key in chckpoint_dict_skip_on_merge:
                continue

            # check weighted and U-Net or not
            if weights is not None and 'model.diffusion_model.' in key:
                # check block index
                weight_index = -1

                if 'time_embed' in key:
                    weight_index = 0                # before input blocks
                elif '.out.' in key:
                    weight_index = NUM_TOTAL_BLOCKS - 1     # after output blocks
                else:
                    m = re_inp.search(key)
                    if m:
                        inp_idx = int(m.groups()[0])
                        weight_index = inp_idx
                    else:
                        m = re_mid.search(key)
                        if m:
                            weight_index = NUM_INPUT_BLOCKS
                        else:
                            m = re_out.search(key)
                            if m:
                                out_idx = int(m.groups()[0])
                                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx

                if weight_index >= NUM_TOTAL_BLOCKS:
                    print(f"error. illegal block index: {key}")
                    return "",None,None,None,None,None
                
                if weight_index >= 0 and useblocks:
                    current_alpha = weights[weight_index]
            else:
                count_target_of_basealpha = count_target_of_basealpha + 1

            if "Add" in mode:
            #    print("model A["+key+"] + " + str(current_alpha) + " * (model B - model C)["+key+"]")
                theta_0[key] = theta_0[key] + current_alpha * theta_1[key]
            else:
            #    print(str(1 - current_alpha)+" model A["+key+"] + " + str(current_alpha) + " * (model B)["+key+"]")
                if current_alpha == 1:
                    theta_0[key] = theta_1[key]
                elif current_alpha !=0:
                    theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]
            #print(key,current_alpha)


    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if key in chckpoint_dict_skip_on_merge:
            continue
        if "model" in key and key not in theta_0:
            theta_0.update({key:theta_1[key]})
            
    if useblocks:
        if "Add" in mode:
            currentmodel =model_a[:model_a.find('.ckpt')]+" + (" +model_b[:model_b.find('.ckpt')]+" - " +model_c[:model_c.find('.ckpt')]+")"+ " x "+"alpha"+","+" ("+str(round(base_alpha,3))+','.join(str(s) for s in weights)+")"
        else:
            currentmodel =model_a[:model_a.find('.ckpt')]+" x (1-alpha) + " +model_b[:model_b.find('.ckpt')] + " x "+"alpha"+" ("+str(round(base_alpha,3))+","+','.join(str(s) for s in weights)+")"
    else:
        if "Add" in mode:
            currentmodel =model_a[:model_a.find('.ckpt')]+" + (" +model_b[:model_b.find('.ckpt')]+" - " +model_c[:model_c.find('.ckpt')]+")"+ " x "+str(round(base_alpha,3))
        else:
            currentmodel =model_a[:model_a.find('.ckpt')]+" x "+str(round(1-base_alpha,3))+" + " +model_b[:model_b.find('.ckpt')] + " x "+str(round(base_alpha,3))

    comments=""

    sd_hijack.model_hijack.undo_hijack(shared.sd_model)
    shared.sd_model.load_state_dict(theta_0, strict=False)
    


    if shared.cmd_opts.opt_channelslast:
        shared.sd_model.to(memory_format=torch.channels_last)

    if not shared.cmd_opts.no_half:
        vae = shared.sd_model.first_stage_model

        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.cmd_opts.no_half_vae:
            shared.sd_model.first_stage_model = None

        shared.sd_model.half()
        shared.sd_model.first_stage_model = vae

    devices.dtype = torch.float32 if shared.cmd_opts.no_half else torch.float16
    devices.dtype_vae = torch.float32 if shared.cmd_opts.no_half or shared.cmd_opts.no_half_vae else torch.float16

    shared.sd_model.first_stage_model.to(devices.dtype_vae)


    sd_hijack.model_hijack.hijack(shared.sd_model)

    if save:
        comments = savemodel(theta_0,currentmodel,output_file,overwrite)
    else:
        comments = "Merged model loaded:"+currentmodel

    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()

    vae_file, vae_source = sd_vae.resolve_vae(model_a)
    sd_vae.load_vae(shared.sd_model, vae_file, vae_source)
    torch.cuda.empty_cache()
    if prompt != "none":
        a,b,c,d = runrun(prompt, nprompt, steps, sampler, cfg, seed, w, h,currentmodel)
        return comments,a,b,c,d,currentmodel

    return comments,currentmodel


def savemerged(output_file,overwrite):
    global mergedmodel
    if len(mergedmodel)!=11:
        print(mergedmodel)
        return "ERROR:no info for merged model",None
    #    mergedmode = [weights,model_a, model_b,model_c, device, base_alpha, output_file,mode, overwrite,True,useblocks]
    mergedmodel[6] = output_file
    mergedmodel[8] = overwrite

    return mergen(*mergedmodel)

def savemodel(state_dict,currentmodel,output_file,overwrite):
    if not output_file or output_file == "":
        output_file = currentmodel.replace(" ","").replace(",","_").replace("(","_").replace(")","_")+".ckpt"
    else:
        output_file = output_file if ".ckpt" in output_file else output_file + ".ckpt"
    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path
 
    output_file = os.path.join(ckpt_dir, output_file)

    # check if output file already exists
    if os.path.isfile(output_file) and not overwrite:
        _err_msg = f"Output file ({output_file}) existed and was not saved]"
        print(_err_msg)
        return _err_msg

    print("Saving...")
    torch.save({"state_dict":state_dict}, output_file)
    print("Done!")
    return "Merged model saved in "+output_file


def load_model_weights_m(model,model_a,model_b,save):
    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    sd_model_name = checkpoint_info.model_name

    cachenum = shared.opts.sd_checkpoint_cache
    
    if save:        
        if model_a:
            load_model(checkpoint_info)
        print(f"Loading weights [{sd_model_name}] from file")
        return sd_models.read_state_dict(checkpoint_info.filename,"cuda")

    if checkpoint_info in checkpoints_loaded:
        print(f"Loading weights [{sd_model_name}] from cache")
        return checkpoints_loaded[checkpoint_info]
    elif cachenum>0 and model_a:
        load_model(checkpoint_info)
        print(f"Loading weights [{sd_model_name}] from cache")
        return checkpoints_loaded[checkpoint_info]
    elif cachenum>1 and model_b:
        load_model(checkpoint_info)
        print(f"Loading weights [{sd_model_name}] from cache")
        return checkpoints_loaded[checkpoint_info]
    elif cachenum>2:
        load_model(checkpoint_info)
        print(f"Loading weights [{sd_model_name}] from cache")
        return checkpoints_loaded[checkpoint_info]
    else:
        if model_a:
            load_model(checkpoint_info)
        print(f"Loading weights [{sd_model_name}] from file")
        return sd_models.read_state_dict(checkpoint_info.filename,"cuda")

def makegrid(outputimages,xs,ys):
    def cell(x, y):
        i = xs.index(x) * len(ys) + ys.index(y)
        return outputimages[i]

    ver_texts = [[images.GridAnnotation(y)] for y in ys]
    hor_texts = [[images.GridAnnotation(x)] for x in xs]
    
    global temp_p
    image_cache = []

    for y in ys:
        for x in xs:
            image_cache.append(cell(x, y))

    grid = images.image_grid(image_cache, rows=len(ys))
    grid = images.draw_grid_annotations(grid,int(temp_p.width), int(temp_p.height), hor_texts, ver_texts)

    if opts.grid_save:
        images.save_image(grid, opts.outdir_txt2img_grids, "xy_grid", extension=opts.grid_format, prompt=temp_p.prompt, seed=temp_p.seed, grid=True, p=temp_p)

    return grid
