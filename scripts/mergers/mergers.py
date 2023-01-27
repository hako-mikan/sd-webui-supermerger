from linecache import clearcache
import math
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
import datetime

from PIL import Image, ImageFont, ImageDraw, PngImagePlugin
from fonts.ttf import Roboto
from tqdm import tqdm
from modules import shared, devices, sd_hijack, processing, sd_models, images, sd_samplers,sd_vae,scripts
from modules.ui import  plaintext_to_html
from modules.shared import opts
from modules.processing import create_infotext,Processed
from modules.sd_models import  load_model,checkpoints_loaded
from inspect import currentframe

state_mergen = False
mergedmodel=[]
temp_p=None
typesg = ["none","alpha","beta (if Triple or Twice is not selected,Twice automatically enable)","seed", "mbw weights", "model_A","model_B","model_C","pinpoint blocks (alpha or beta must be selected for another axis)"]
types = ["none","alpha","beta","seed", "mbw weights", "model_A","model_B","model_C","pinpoint blocks"]
#type[0:aplha,1:beta,2:seed,3:mbw,4:model_A,5:model_B,6:model_C]

def freezetime():
    global state_mergen
    state_mergen = True

hear = False
hearm = False
def caster(news,hear):
    if hear: print(news)

def casterr(*args,hear=hear):
    if hear:
        names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
        print('\n'.join([names.get(id(arg), '???') + ' = ' + repr(arg) for arg in args]))

def genxyplot(xmen:str,ymen:str,xtype,ytype,weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,prompt,nprompt,steps,sampler,cfg,seed,w,h):
    global hear
    #type[0:none,1:aplha,2:beta,3:seed,4:mbw,5:model_A,6:model_B,7:model_C,8:pinpoint ]
    xtype,ytype = types[xtype],types[ytype]

    modes=["Weight" ,"Add" ,"Triple","Twice"]
    xs=ys=0
    weights_a_in=weights_b_in="0"

    def castall(hear):
        if hear :print(f"xmen:{xmen}, ymen:{ymen}, xtype:{xtype}, ytype:{ytype}, weights_a:{weights_a_in}, weights_b:{weights_b_in}, model_A:{model_a},model_B :{model_b}, model_C:{model_c}, alpha:{alpha},\
        beta :{beta}, mode:{mode}, blocks:{useblocks}")
    pinpoint = True if "pinpoint" in xtype or "pinpoint" in ytype else False

    usebeta = modes[2] in mode or modes[3] in mode

    #check and adjust format

    print(f"XY plot start, mode:{mode}, X: {xtype}, Y: {ytype}, MBW: {useblocks}")
    castall(hear)
    None5 = [None,None,None,None,None]
    if xmen =="": return "ERROR: parameter X is empty",*None5
    if ymen =="" and not ytype=="none": return "ERROR: parameter Y is empty",*None5
    if model_a =="" and (not "model_A" in xtype or not "model_A" in ytype):return "ERROR: model_A is not selected",*None5
    if model_b ==""and (not "model_B" in xtype or not "model_B" in ytype):return "ERROR: model_B is not selected",*None5
    if model_c =="" and usebeta and (not "model_C" in xtype or not "model_C" in ytype):return "ERROR: model_C is not selected",*None5
    if xtype == ytype: return "ERROR: same type selected for X,Y",*None5

    #for X only plot, use same seed
    if seed == -1: seed = int(random.randrange(4294967294))

    #for XY plot, use same seed
    def dicedealer(zs):
        for i,z in enumerate(zs):
            if z =="-1": zs[i] = str(random.randrange(4294967294))
        print(f"the die was thrown : {zs}")

    #adjust parameters, alpha,beta,models,seed: list of single parameters, mbw(no beta):list of text,mbw(usebeta); list of pair text
    def adjuster(zmen,ztype):
        if "mbw" in ztype:#men separated by newline
            zs = zmen.splitlines()
            caster(zs,hear)
            for z in zs:
                if len([z.strip() for z in z.split(',')]) !=26: return "ERROR, in mbw mode, number of parameters must be 26",*None5
            if usebeta:
                zs = [zs[i:i+2] for i in range(0,len(zs),2)]
                caster(zs,hear)
        else:
            zs = [z.strip() for z in zmen.split(',')]
            caster(zs,hear)
        if "seed" in ztype:dicedealer(zs)
        return zs

    xs = adjuster(xmen,xtype)
    ys = adjuster(ymen,ytype)

    #in case beta selected but mode is Weight sum or Add
    if ("beta" in xtype or "beta" in ytype) and not usebeta:
        mode = modes[3]
        print(f"{modes[3]} mode automatically selected)")

    #in case mbw or pinpoint selected but useblocks not chekced
    if ("mbw" in xtype or "pinpoint" in xtype) and not useblocks:
        useblocks = True
        print(f"MBW mode enabled")

    if ("mbw" in ytype or "pinpoint" in ytype) and not useblocks:
        useblocks = True
        print(f"MBW mode enabled")

    image=None
    xcount =ycount=0
    allcount = len(xs)*len(ys)

    #for STOP XY bottun
    flag = False
    global state_mergen
    state_mergen = False

    #type[0:none,1:aplha,2:beta,3:seed,4:mbw,5:model_A,6:model_B,7:model_C,8:pinpoint ]
    blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
    #format ,IN00 IN03,IN04-IN09,OUT4,OUT05
    def weightsdealer(x,xtype,y,weights):
        caster(f"weights_a, from   : {weights}",hear)
        zz = x if "pinpoint" in xtype else y
        za = y if "pinpoint" in xtype else x
        zz = [z.strip() for z in zz.split(' ')]
        weights_t = [w.strip() for w in weights.split(',')]
        casterr(weights_t,hear=hear)
        if zz[0]!="NOT":
            flagger=[False]*26
            changer = True
        else:
            flagger=[True]*26
            changer = False
        for z in zz:
            if z =="NOT":continue
            if "-" in z:
                zt = [zt.strip() for zt in z.split('-')]
                if  blockid.index(zt[1]) > blockid.index(zt[0]):
                    flagger[blockid.index(zt[0]):blockid.index(zt[1])+1] = [changer]*(blockid.index(zt[1])-blockid.index(zt[0])+1)
                else:
                    flagger[blockid.index(zt[1]):blockid.index(zt[0])+1] = [changer]*(blockid.index(zt[0])-blockid.index(zt[1])+1)
            else:
                print(f"{z},{blockid.index(z)}")
                flagger[blockid.index(z)] =changer
        casterr(weights_t,hear)       
        for i,f in enumerate(flagger):
            if f:weights_t[i]=za
        casterr(flagger,hear=hear)
        outext = ",".join(weights_t)
        caster(f"weights_a, changed: {outext}",hear)
        return ",".join(weights_t)

    def xydealer(z,zt):
        nonlocal alpha,beta,seed,weights_a_in,weights_b_in,model_a,model_b,model_c
        if pinpoint:return
        if "alpha" in zt:alpha = z
        if "beta" in zt: beta = z
        if "seed" in zt:seed = int(z)
        if "mbw" in zt:
            def weightser(z):return z, z.split(',',1)[0]
            if usebeta:
                weights_a_in,alpha = weightser(z[0])
                weights_b_in,beta = weightser(z[1])
            else:
                weights_a_in,alpha = weightser(z)
        if "model_A" in zt:model_a = z
        if "model_B" in zt:model_b = z
        if "model_C" in zt:model_c = z

    # plot start
    for y in ys:
        xydealer(y,ytype)
        xcount = 0
        for x in xs:
            xydealer(x,xtype)
            if "alpha" in xtype and pinpoint:weights_a_in = weightsdealer(x,xtype,y,weights_a)
            if "beta" in xtype and pinpoint:weights_b_in = weightsdealer(x,xtype,y,weights_b)
            castall(True)
            print(f"XY plot: X: {xtype}, {str(x)}, Y: {ytype}, {str(y)} ({xcount+ycount*len(xs)+1}/{allcount})")
            #currentmodel = makemodelname("","",model_a, model_b,model_c, alpha,beta,blocks,mode)
            if xtype=="seed" and xcount > 0:
                image_temp=("",)+runrun(prompt, nprompt, steps, sampler, cfg, seed, w, h,currentmodel)
            else:
                image_temp=mergen(weights_a_in,weights_b_in, model_a, model_b,model_c, "cpu", float(alpha),float(beta), "",mode,False,False,useblocks,prompt, nprompt, steps, sampler, cfg, seed, w, h,"")
                currentmodel = image_temp[5]
            if image ==None:
                image = image_temp
            else:
                image[1].append(*image_temp[1])
            xcount+=1
            if state_mergen:
                flag = True
                break
        ycount+=1
        if flag:break

    if flag and ycount ==1:
        xs = xs[:xcount]
        ys = [ys[0],]
        print(f"stopped at x={xcount},y={ycount}")
    else:
        ys=ys[:ycount]
        print(f"stopped at x={xcount},y={ycount}")

    if "mbw" in xtype and usebeta: xs = [f"alpha:({x[0]}),beta({x[1]})" for x in xs ]
    if "mbw" in ytype and usebeta: ys = [f"alpha:({y[0]}),beta({y[1]})" for y in ys ]

    xs[0]=xtype+" = "+xs[0] #draw X label
    if ytype!=types[0] or "model" in ytype:ys[0]=ytype+" = "+ys[0]  #draw Y label

    currentmodel = makegridmodelname(model_a, model_b,model_c, useblocks,mode,xtype,ytype,alpha,beta,weights_a,weights_b,usebeta)
    grid = makegrid(image[1],xs,ys,currentmodel)

    image[1].insert(0,grid)
  
    state_mergen = False
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
            infotexts[i] = " Model: "+mergeinfo.replace(","," ")
    infotext= ",".join(infotexts)
    images.save_image(image, opts.outdir_txt2img_samples, "",p.seed, p.prompt,shared.opts.samples_format, p=p,info=infotext)
    shared.state.end()
    global temp_p
    temp_p=p
    return processed.images,infotext,plaintext_to_html(processed.info), plaintext_to_html(processed.comments)

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

def mergen(weights_a,weights_b, model_a, model_b,model_c, device="cpu", base_alpha=0.5,base_beta=0.25, output_file="",mode="Weight", 
overwrite=False,save=True,useblocks=False,prompt="none", nprompt="", steps=20, sampler=0, cfg=8, seed=-1, w=512, h=512,
currentmodel=""):
    caster("merge start",hearm)
    modes=["Weight" ,"Add" ,"Triple","Twice"]
    global hear
    global mergedmodel
    gc.collect()

    usebeta = modes[2] in mode or modes[3] in mode
    mergedmodel = [weights_a,weights_b,model_a, model_b,model_c, device, base_alpha,base_beta, output_file,mode, overwrite,True,useblocks].copy()
    caster(mergedmodel,True)
    if model_a =="" or model_b =="" or ((not modes[0] in mode) and model_c=="") : 
        return "ERROR: Necessary model is not selected",None,None,None,None,currentmodel
    if useblocks:
        weights_a_t=weights_a.split(',',1)
        weights_b_t=weights_b.split(',',1)
        base_alpha  = float(weights_a_t[0])    
        weights_a = [float(w) for w in weights_a_t[1].split(',')]
        caster(f"from {weights_a}, alpha = {base_alpha},weights_a ={weights_a}",hearm)
        if len(weights_a) != 25:return f"ERROR: weights alpha value must be {26}.",None,None,None,None,currentmodel
        if usebeta:
            base_beta = float(weights_b_t[0]) 
            weights_b = [float(w) for w in weights_b_t[1].split(',')]
            caster(f"from {weights_b}, beta = {base_beta},weights_a ={weights_b}",hearm)
            if len(weights_b) != 25: return f"ERROR: weights beta value must be {26}.",None,None,None,None,currentmodel

    device = device if device in ["cpu", "cuda"] else "cpu"
    caster("model load start",hearm)
    theta_1=load_model_weights_m(model_b,False,True,save).copy()

    if modes[1] in mode:#Add
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

    if modes[2] in mode or modes[3] in mode:#Tripe or Twice
        theta_2 = load_model_weights_m(model_c,False,False,save).copy()

    alpha = base_alpha
    beta = base_beta

    re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
    re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
    re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

    chckpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

    count_target_of_basealpha = 0
    for key in (tqdm(theta_0.keys(), desc="Stage 1/2") if not False else theta_0.keys()):
        if "model" in key and key in theta_1:
            current_alpha = alpha
            current_beta = beta

            if key in chckpoint_dict_skip_on_merge:
                continue

            # check weighted and U-Net or not
            if weights_a is not None and 'model.diffusion_model.' in key:
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
                    current_alpha = weights_a[weight_index]
                    if usebeta: current_beta = weights_b[weight_index]
            else:
                count_target_of_basealpha = count_target_of_basealpha + 1

            if modes[1] in mode:#Add
                caster(f"model A[{key}] +  {current_alpha} + * (model B - model C)[{key}]",hear)
                theta_0[key] = theta_0[key] + current_alpha * theta_1[key]
            elif modes[2] in mode:#Triple
                caster(f"model A[{key}] +  {1-current_alpha-current_beta} +  model B[{key}]*{current_alpha} + model C[{key}]*{current_beta}",hear)
                theta_0[key] = (1 - current_alpha-current_beta) * theta_0[key] + current_alpha * theta_1[key]+current_beta * theta_2[key]
            elif modes[3] in mode:#Twice
                caster(f"model A[{key}] +  {1-current_alpha} + * model B[{key}]*{alpha}",hear)
                caster(f"model A+B[{key}] +  {1-current_beta} + * model C[{key}]*{beta}",hear)
                theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]
                theta_0[key] = (1 - current_beta) * theta_0[key] + current_beta * theta_2[key]
            else:#Weight
                if current_alpha == 1:
                    caster(f"alpha = 0,model A[{key}=model B[{key}",hear)
                    theta_0[key] = theta_1[key]
                elif current_alpha !=0:
                    caster(f"model A[{key}] +  {1-current_alpha} + * (model B)[{key}]*{alpha}",hear)
                    theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]

    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if key in chckpoint_dict_skip_on_merge:
            continue
        if "model" in key and key not in theta_0:
            theta_0.update({key:theta_1[key]})
            
    currentmodel = makemodelname(weights_a,weights_b,model_a, model_b,model_c, base_alpha,base_beta,useblocks,mode)
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
    
    savelog(currentmodel,mergedmodel)
    caster(mergedmodel,True)
    if save:
        comments = savemodel(theta_0,currentmodel,output_file,overwrite)
    else:
        comments = "Merged model loaded:"+currentmodel

    def _setvae():
        sd_vae.delete_base_vae()
        sd_vae.clear_loaded_vae()
        vae_file, vae_source = sd_vae.resolve_vae(model_a)
        sd_vae.load_vae(shared.sd_model, vae_file, vae_source)

    try:
        _setvae()
    except:
        print("ERROR:setting VAE skipped")

    gc.collect()
    if prompt != "none":
        a,b,c,d = runrun(prompt, nprompt, steps, sampler, cfg, seed, w, h,currentmodel)
        return comments,a,b,c,d,currentmodel

    return comments,currentmodel

def savemerged(output_file,overwrite):
    global mergedmodel
    print(mergedmodel)
    if len(mergedmodel)!=13:
        print(mergedmodel)
        return "ERROR:no info for merged model",None
    #    mergedmode = [0 weights_a,1 weights_b,2 model_a, 3 model_b, 4 model_c, 5 device, 6 base_alpha,7 base_beta, 8 output_file,9 mode,10 overwrite,11 True,12 useblocks]
    mergedmodel[8] = output_file
    mergedmodel[10] = overwrite

    return mergen(*mergedmodel)

def savemodel(state_dict,currentmodel,output_file,overwrite):
    if not output_file or output_file == "":
        output_file = currentmodel.replace(" ","").replace(",","_").replace("(","_").replace(")","_")+".ckpt"
        if output_file[0]=="_":output_file = output_file[1:]
    else:
        output_file = output_file if ".ckpt" in output_file else output_file + ".ckpt"
    ckpt_dir = shared.cmd_opts.ckpt_dir or sd_models.model_path

    if len(output_file) > 255:
       output_file.replace(".ckpet","")
       output_file=output_file[:250]+".ckpt"

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

def makemodelname(weights_a,weights_b,model_a, model_b,model_c, alpha,beta,useblocks,mode):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    model_c=filenamecutter(model_c)

    modes=["Weight" ,"Add" ,"Triple","Twice"]

    if type(alpha) == str:alpha = float(alpha)
    if type(beta)== str:beta  = float(beta)

    if useblocks:
        if modes[1] in mode:#add
            currentmodel =f"{model_a} + ({model_b} - {model_c}) x alpha ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)}"
        elif modes[2] in mode:#triple
            currentmodel =f"{model_a} x (1-alpha-beta) + {model_b} x alpha + {model_c} x beta (alpha = {str(round(alpha,3))},{','.join(str(s) for s in weights_a)},beta = {beta},{','.join(str(s) for s in weights_b)})"
        elif modes[3] in mode:#twice
            currentmodel =f"({model_a} x (1-alpha) + {model_b} x alpha)*(1-beta)+  {model_c} x beta ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)})_({str(round(beta,3))},{','.join(str(s) for s in weights_b)})"
        else:
            currentmodel =f"{model_a} x (1-alpha) + {model_b} x alpha ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)})"
    else:
        if modes[1] in mode:#add
            currentmodel =f"{model_a} + ({model_b} -  {model_c}) x {str(round(alpha,3))}"
        elif modes[2] in mode:#triple
            currentmodel =f"{model_a} x {str(round(1-alpha-beta,3))} + {model_b} x {str(round(alpha,3))} + {model_c} x {str(round(beta,3))}"
        elif modes[3] in mode:#twice
            currentmodel =f"({model_a} x {str(round(1-alpha,3))} +{model_b} x {str(round(alpha,3))}) x {str(round(1-beta,3))} + {model_c} x {str(round(beta,3))}"
        else:
            currentmodel =f"{model_a} x {str(round(1-alpha,3))} + {model_b} x {str(round(alpha,3))}"
    return currentmodel

def makegridmodelname(model_a, model_b,model_c, useblocks,mode,xtype,ytype,alpha,beta,wa,wb,usebeta):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    model_c=filenamecutter(model_c)

    if not usebeta:beta,wb = "not used","not used"
    vals = ""
    modes=["Weight" ,"Add" ,"Triple","Twice"]

    wa = "alpha = " + wa
    wb = "beta = " + wb

    x = 50
    while len(wa) > x:
        wa  = wa[:x] + '\n' + wa[x:]
        x = x + 50

    x = 50
    while len(wb) > x:
        wb  = wb[:x] + '\n' + wb[x:]
        x = x + 50

    if "model" in xtype:
        if "A" in xtype:model_a = "model A"
        elif "B" in xtype:model_b="model B"
        elif "C" in xtype:model_c="model C"

    if "model" in ytype:
        if "A" in ytype:model_a = "model A"
        elif "B" in ytype:model_b="model B"
        elif "C" in ytype:model_c="model C"

    if modes[1] in mode:
        currentmodel =f"{model_a} \n {model_b} - {model_c})\n x alpha"
    elif modes[2] in mode:
        currentmodel =f"{model_a} x \n(1-alpha-beta) {model_b} x alpha \n+ {model_c} x beta"
    elif modes[3] in mode:
        currentmodel =f"({model_a} x(1-alpha) \n + {model_b} x alpha)*(1-beta)\n+  {model_c} x beta"
    else:
        currentmodel =f"{model_a} x (1-alpha) \n {model_b} x alpha"

    vals = f"\nalpha = {alpha},beta = {beta}" if not useblocks else f"\n{wa}\n{wb}"
    currentmodel = currentmodel+vals
    return currentmodel

def filenamecutter(name):
    if "ckpt" in name:name =name[:name.find('.ckpt')]
    if "safetensor" in name:name=name[:name.find('.safetensor')]
    return name

path_root = scripts.basedir()

def savelog(name,settings):
    setting = settings.copy()
    filepath = os.path.join(path_root, "mergelog.csv")
    log = open(filepath, 'a')
    for i,x in enumerate(setting):
        if "," in str(x):setting[i] = f'"{str(setting[i])}"'
    text = ",".join(map(str, setting))
    text=datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S.%f')[:-7]+","+ name +","+ text
    log.write(text+"\n")
    log.close()

def makegrid(imgs,xs,ys,currentmodel):
    ver_texts = [[images.GridAnnotation(y)] for y in ys]
    hor_texts = [[images.GridAnnotation(x)] for x in xs]
    
    global temp_p

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(len(xs) * w, len(ys) * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % len(xs) * w, i // len(xs) * h))

    grid = images.draw_grid_annotations(grid,int(temp_p.width), int(temp_p.height), hor_texts, ver_texts)
    grid = draw_origin(grid, currentmodel,w*len(xs),h*len(ys),w)
    if opts.grid_save:
        images.save_image(grid, opts.outdir_txt2img_grids, "xy_grid", extension=opts.grid_format, prompt=temp_p.prompt, seed=temp_p.seed, grid=True, p=temp_p)

    return grid

def draw_origin(grid, text,width,height,width_one):
    grid_d= Image.new("RGB", (grid.width,grid.height), "white")
    grid_d.paste(grid,(0,0))
    def get_font(fontsize):
        try:
            return ImageFont.truetype(opts.font or Roboto, fontsize)
        except Exception:
            return ImageFont.truetype(Roboto, fontsize)
    d= ImageDraw.Draw(grid_d)
    color_active = (0, 0, 0)
    fontsize = (width+height)//25
    fnt = get_font(fontsize)

    while d.multiline_textsize(text, font=fnt)[0] > width_one*0.8 and fontsize > 0:
        fontsize -=1
        fnt = get_font(fontsize)
    d.multiline_text((1,1), text, font=fnt, fill=color_active,align="center")
    return grid_d
