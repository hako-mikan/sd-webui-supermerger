import random
import os
import gc
import hashlib
import numpy as np
import os.path
import re
import torch
import tqdm
import datetime
from typing import Union, Optional
import csv
import json
import launch
import torch.nn as nn
import scipy.ndimage
from copy import deepcopy
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from functools import partial
from torch import Tensor, lerp
from torch.nn.functional import cosine_similarity, relu, softplus
from modules import shared, processing, sd_models, sd_vae, images, sd_samplers, scripts,devices, extras
from modules.ui import  plaintext_to_html
from modules.shared import opts
from modules.processing import create_infotext,Processed
from modules.sd_models import  load_model,unload_model_weights
from modules.generation_parameters_copypaste import create_override_settings_dict
from scripts.mergers.model_util import filenamecutter,savemodel
from math import ceil
import sys
from multiprocessing import cpu_count
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.mergers.bcolors import bcolors
import collections

PREFIXFIX = ("double_blocks","single_blocks","time_in","vector_in","txt_in")
BNB = ".quant_state.bitsandbytes__"
PREFIX_M = "model.diffusion_model."
QTYPES = ["fp4", "nf4"]

try:
    ui_version = int(launch.git_tag().split("-",1)[0].replace("v","").replace(".",""))
except:
    ui_version = 100

try:
    from backend import memory_management
    from backend.utils import load_torch_file
    forge = True
except:
    forge = False

revert_target = ""
orig_cache = 0
modelcache = collections.OrderedDict()

from inspect import currentframe

SELFKEYS = ["to_out","proj_out","norm"]

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))
scriptpath = os.path.dirname(module_path)

def tryit(func):
    try:
        func() 
    except:
        pass 

stopmerge = False

def freezemtime():
    global stopmerge
    stopmerge = True

mergedmodel=[]
FINETUNEX = ["IN","OUT","OUT2","CONT","BRI","COL1","COL2","COL3"]
TYPESEG = ["none","alpha","beta (if Triple or Twice is not selected,Twice automatically enable)","alpha and beta","seed",
                    "mbw alpha","mbw beta","mbw alpha and beta", "model_A","model_B","model_C","pinpoint blocks (alpha or beta must be selected for another axis)",
                    "include blocks", "exclude blocks","add include", "add exclude","elemental","add elemental","pinpoint element","effective elemental checker","adjust","pinpoint adjust (IN,OUT,OUT2,CONT,BRI,COL1,COL2,COL3)",
                    "calcmode","prompt","random"]
TYPES = ["none","alpha","beta","alpha and beta","seed", "mbw alpha ","mbw beta","mbw alpha and beta",
                "model_A","model_B","model_C","pinpoint blocks","include blocks","exclude blocks","add include", "add exclude","elemental","add elemental","pinpoint element",
                "effective","adjust","pinpoint adjust","calcmode","prompt","random"]
MODES=["Weight" ,"Add" ,"Triple","Twice"]
SAVEMODES=["save model", "overwrite"]
EXCLUDE_CHOICES = ["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11",
                                  "M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11",
                                  "Adjust","VAE"]           
CHCKPOINT_DICT_SKIP_ON_MERGE = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

#type[0:aplha,1:beta,2:seed,3:mbw,4:model_A,5:model_B,6:model_C]
#msettings=[0 weights_a,1 weights_b,2 model_a,3 model_b,4 model_c,5 base_alpha,6 base_beta,7 mode,8 useblocks,9 custom_name,10 save_sets,11 id_sets,12 wpresets]
#id sets "image", "PNG info","XY grid"

hear = False
hearm = False
NON4 = [None]*4

informer = sd_models.get_closet_checkpoint_match

#msettings=[weights_a,weights_b,model_a,model_b,model_c,device,base_alpha,base_beta,mode,loranames,useblocks,custom_name,save_sets,id_sets,wpresets,deep]  

def smergegen(weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,
                       calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor,bake_in_vae,opt_value,inex,ex_blocks,ex_elems,
                       esettings,
                       s_prompt,s_nprompt,s_steps,s_sampler,s_cfg,s_seed,s_w,s_h,s_batch_size,
                       genoptions,s_hrupscaler,s_hr2ndsteps,s_denois_str,s_hr_scale,
                       lmode,lsets,llimits_u,llimits_l,lseed,lserial,lcustom,lround,
                       currentmodel,imggen,
                       *txt2imgparams):

    lucks = {"on":False, "mode":lmode,"set":lsets,"upp":llimits_u,"low":llimits_l,"seed":lseed,"num":lserial,"cust":lcustom,"round":int(lround)}
    deepprint  = "print change" in esettings

    cachedealer(True)

    result,currentmodel,modelid,theta_0,metadata = smerge(
                        weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,calcmode,
                        useblocks,custom_name,save_sets,id_sets,wpresets,deep,tensor,bake_in_vae,opt_value,inex,ex_blocks,ex_elems,deepprint,lucks
                        )

    if "ERROR" in result or "STOPPED" in result: 
        return result,"not loaded",*NON4

    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)

    save = True if SAVEMODES[0] in save_sets else False

    result = savemodel(theta_0,currentmodel,custom_name,save_sets,metadata) if save else "Merged model loaded:"+currentmodel

    model_loader(checkpoint_info, theta_0, metadata, currentmodel)

    cachedealer(False)

    del theta_0
    devices.torch_gc()

    debug = "debug" in save_sets

    if ("copy config" in save_sets) and ("(" not in result):
        try:
            extras.create_config(result.replace("Merged model saved in ",""), 0, informer(model_a), informer(model_b), informer(model_b))
        except:
            pass

    if imggen :
        images = simggen(s_prompt,s_nprompt,s_steps,s_sampler,s_cfg,s_seed,s_w,s_h,s_batch_size,
                        genoptions,s_hrupscaler,s_hr2ndsteps,s_denois_str,s_hr_scale,
                        currentmodel,id_sets,modelid,
                        *txt2imgparams,debug = debug)

        return result,currentmodel,*images[:4]
    else:
        return result,currentmodel

def checkpointer_infomer(name):
    return sd_models.get_closet_checkpoint_match(name)

# XXX hack. fake checkpoint_info
def fake_checkpoint_info(checkpoint_info,metadata={},currentmodel=""):
    from modules import cache
    dump_cache = cache.dump_cache
    c_cache = cache.cache
    
    checkpoint_info = deepcopy(checkpoint_info)
    # change model name etc.
    sha256 = hashlib.sha256(json.dumps(metadata).encode("utf-8")).hexdigest()
    checkpoint_info.sha256 = sha256
    checkpoint_info.name_for_extra = currentmodel
    checkpoint_info.isfake = True

    checkpoint_info.name = checkpoint_info.name_for_extra + ".safetensors"
    checkpoint_info.model_name = checkpoint_info.name_for_extra.replace("/", "_").replace("\\", "_")
    checkpoint_info.title = f"{checkpoint_info.name} [{sha256[0:10]}]"
    checkpoint_info.metadata = metadata

    # for sd-webui  v1.5.x
    sd_models.checkpoints_list[checkpoint_info.title] = checkpoint_info

        # force to set a new sha256 hash
    if c_cache is not None: 
        hashes = c_cache("hashes")
        hashes[f"checkpoint/{checkpoint_info.name}"] = {
        "mtime": os.path.getmtime(checkpoint_info.filename),
        "sha256": sha256,
        }
        # save cache
        dump_cache()

    # set ids for a fake checkpoint info
    checkpoint_info.ids = [checkpoint_info.model_name, checkpoint_info.name, checkpoint_info.name_for_extra]
    return checkpoint_info

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS
BLOCKID=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKIDXLL=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","VAE"]
BLOCKIDXL=['BASE', 'IN0', 'IN1', 'IN2', 'IN3', 'IN4', 'IN5', 'IN6', 'IN7', 'IN8', 'M', 'OUT0', 'OUT1', 'OUT2', 'OUT3', 'OUT4', 'OUT5', 'OUT6', 'OUT7', 'OUT8', 'VAE']
BLOCKIDFLUX = ["CLIP", "T5", "IN"] + ["D{:002}".format(x) for x in range(19)] + ["S{:002}".format(x) for x in range(38)] + ["OUT"] # Len: 61

RANDMAP = [0,50,100] #alpha,beta,elements

statistics = {"sum":{},"mean":{},"max":{},"min":{}}
    
################################################
##### Main Merging Code

def smerge(weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,calcmode,
                useblocks,custom_name,save_sets,id_sets,wpresets,deep,fine,bake_in_vae,opt_value,inex,ex_blocks,ex_elems,deepprint,lucks,main = [False,False,False]):

    caster("merge start",hearm)
    theta_0 = theta_1 = theta_2 = None
    qdtypes = [None, None, None]
    global hear,mergedmodel,stopmerge,statistics, revert_target
    stopmerge = False
    
    debug = "debug" in save_sets
    uselerp = "use old calc method" not in save_sets
    device = "cuda" if "use cuda" in save_sets else "cpu"

    if forge:
        fcinfo = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
        revert_target = revert_target if hasattr(fcinfo, "isfake") else fcinfo
        unload_forge()
    else:
        unload_model_weights(sd_models.model_data.sd_model)

    # for from file
    if type(useblocks) is str:
        useblocks = True if useblocks =="True" else False
    if type(base_alpha) == str:base_alpha = float(base_alpha)
    if type(base_beta) == str:base_beta  = float(base_beta)

    #random
    if lucks != {}:
        if lucks["seed"] == -1: lucks["ceed"] = str(random.randrange(4294967294))
        else: lucks["ceed"] = lucks["seed"] 
    else: lucks["ceed"]  = 0
    np.random.seed(int(lucks["ceed"]))
    randomer = np.random.rand(3000)

    cachetarget =[]
    for model,num in zip([model_a,model_b,model_c],main):
        if model != "" and num:
            cachetarget.append(model)

    weights_a,deep = randdealer(weights_a,randomer,0,lucks,deep)
    weights_b,_ = randdealer(weights_b,randomer,1,lucks,None)

    weights_a_orig = weights_a
    weights_b_orig = weights_b

    # preset to weights
    if wpresets != False and useblocks:
        weights_a = wpreseter(weights_a,wpresets)
        weights_b = wpreseter(weights_b,wpresets)

    # mode select booleans
    usebeta = MODES[2] in mode or MODES[3] in mode or "tensor" in calcmode
    metadata = {"format": "pt"}

    if (calcmode == "trainDifference" or calcmode == "extract") and "Add" not in mode:
        print(f"{bcolors.WARNING}Mode changed to add difference{bcolors.ENDC}")
        mode = "Add"
    if model_c == "" or model_c is None:
        #fallback to avoid crash
        model_c = model_a
        print(f"{bcolors.WARNING}Substituting empty model_c with model_a{bcolors.ENDC}")

    if not useblocks:
        weights_a = weights_b = ""
    #for save log and save current model
    mergedmodel =[weights_a,weights_b,
                            hashfromname(model_a),hashfromname(model_b),hashfromname(model_c),
                            base_alpha,base_beta,mode,useblocks,custom_name,save_sets,id_sets,deep,calcmode,lucks["ceed"],fine,opt_value,inex,ex_blocks,ex_elems].copy()

    model_a = namefromhash(model_a)
    model_b = namefromhash(model_b)
    model_c = namefromhash(model_c)

    caster(mergedmodel,False)

    #elementals
    if len(deep) > 0:
        deep = deep.replace("\n",",")
        deep = deep.replace(calcmode+",","")
        deep = deep.split(",")

    #format check
    if model_a =="" or model_b =="" or ((not MODES[0] in mode) and model_c=="") : 
        return "ERROR: Necessary model is not selected",*NON4
    
    #exclude/include
    ex_elems = ex_elems.split(",")

    #adjust
    if fine.rstrip(",0") != "":
        fine = fineman(fine,isxl)
    else:
        fine = ""

    #for MBW text to list
    if useblocks:
        weights_a_t=weights_a.split(',',1)
        weights_b_t=weights_b.split(',',1)
        base_alpha  = float(weights_a_t[0])    
        weights_a = [float(w) for w in weights_a_t[1].split(',')]
        caster(f"from {weights_a_t}, alpha = {base_alpha},weights_a ={weights_a}",hearm)
        if not (len(weights_a) == 25 or len(weights_a) == 19 or len(weights_a) == 60):return f"ERROR: weights alpha value must be 20 or 26 or 61.",*NON4
        if usebeta:
            base_beta = float(weights_b_t[0]) 
            weights_b = [float(w) for w in weights_b_t[1].split(',')]
            caster(f"from {weights_b_t}, beta = {base_beta},weights_a ={weights_b}",hearm)
            if not(len(weights_b) == 25 or len(weights_b) == 19 or len(weights_a) == 60): return f"ERROR: weights beta value must be 20 or 26 or 61.",*NON4

    #model loading start 
    caster("Model loading start",hearm)
    printstart(model_a,model_b,model_c,base_alpha,base_beta,weights_a,weights_b,mode,useblocks,calcmode,deep,lucks['ceed'],fine,inex,ex_blocks,ex_elems,device)

    theta_1 = load_model_weights_m(model_b,2,cachetarget,device).copy()
    qdtypes[1] = qdtyper(theta_1)
    prefixer(theta_1)
    
    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_1.keys()
    isflux = any("double_block" in k for k in theta_1.keys())

    if isxl and useblocks:
        if len(weights_a) == 25:
            weights_a = weighttoxl(weights_a)
            print(f"alpha weight converted for XL{weights_a}")
        if usebeta:
            if len(weights_b) == 25:
                weights_b = weighttoxl(weights_b)
                print(f"beta weight converted for XL{weights_b}")
        if len(weights_a) == 19: weights_a = weights_a + [0]
        if len(weights_b) == 19: weights_b = weights_b + [0]

    if stopmerge: return "STOPPED", *NON4
    if not (MODES[0] in mode): #Add, Twice, Triple
        theta_2 = load_model_weights_m(model_c,3,cachetarget,device).copy()
        prefixer(theta_2)
        qdtypes[2] = qdtyper(theta_2)
  
    if MODES[1] in mode: #Add
        if not(calcmode == "trainDifference" or calcmode == "extract"):
            if isflux and qdtypes[1] != qdtypes[2]:
                to_qdtype(theta_1, theta_2, qdtypes[1], qdtypes[2], device, "Model B", "Model C")
            for key in tqdm(theta_1, desc="Stage 0/2, Add difference"):
                if 'model' in key:
                    if stopmerge: return "STOPPED", *NON4
                    if not ("weight" in key or "bias" in key): continue
                    if key in theta_2:
                        a = list(theta_1[key].shape)
                        b = list(theta_2[key].shape)
                        assert_inpaint(a, b, key)
                        if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
                            # Merge only the vectors the models have in common.  Otherwise we get an error due to dimension mismatch.
                            theta_1_a = theta_1[key][:, 0:4, :, :]
                        else:
                            theta_1_a = theta_1[key]
                        theta_1[key] = (theta_1_a.to(torch.float32) -theta_2[key].to(torch.float32)).to(theta_1[key].dtype)
                    else:
                        theta_1[key] = torch.zeros_like(theta_1[key].to(torch.float16))
            del theta_2
            theta_2 = None
            devices.torch_gc()
    
    if  "tensor" in calcmode or "self" in calcmode:
        theta_t = load_model_weights_m(model_a,1,cachetarget,device).copy()
        theta_0 ={}
        for key in theta_t:
            theta_0[key] = theta_t[key].clone()
        del theta_t
    else:
        theta_0 = load_model_weights_m(model_a,1,cachetarget,device).copy()

    qdtypes[0] = qdtyper(theta_0)
    need_revert = prefixer(theta_0)

    print(f"Model precisions : {qdtypes}")

    if qdtypes[0] != qdtypes[1]:
        print(f"Precision of model B (or B-C) is changing to {qdtypes[0]}...")
        to_qdtype(theta_0, theta_1, qdtypes[0], qdtypes[1], device, "Model A", "Model B")

    if theta_2 is not None: 
        to_qdtype(theta_0, theta_2, qdtypes[0], qdtypes[2], device, "Model A", "Model C")

    ##### Stage 0/2 in Cosine
    if "cosine" in calcmode:
        sim, sims = precosine("A" in calcmode,theta_0,theta_1)

    ##### Stage 1/2   

    alpha = base_alpha
    beta = base_beta

    keyratio = []
    key_and_alpha = {}

    for num, key in enumerate(tqdm(theta_0.keys(), desc="Stage 1/2") if not False else theta_0.keys()):
        if stopmerge: return "STOPPED", *NON4
        if (isflux and key not in theta_1) or (not isflux and not ("model" in key and key in theta_1)):continue
        if not ("weight" in key or "bias" in key): continue
        if theta_2 is not None and key not in theta_2: continue

        theta_0[key] = theta_0[key].to(device)
        theta_1[key] = theta_1[key].to(device)
        if theta_2 is not None:
            theta_2[key] = theta_2[key].to(device)

        weight_index = -1
        current_alpha = alpha
        current_beta = beta

        a = list(theta_0[key].shape)
        b = list(theta_1[key].shape)

        assert_inpaint(a, b, key)

        block,blocks26 = blockfromkey(key,isxl,isflux)
        #if block == "Not Merge": continue
        if inex != "Off" and (ex_blocks or (ex_elems != [""])) and excluder(blocks26,inex,ex_blocks,ex_elems,key): continue
        if isflux and blocks26 in BLOCKIDFLUX:
            weight_index = BLOCKIDFLUX.index(blocks26)
        elif isxl and blocks26 in BLOCKIDXLL:
            weight_index = BLOCKIDXLL.index(blocks26)
        elif blocks26 in BLOCKID:
            weight_index = BLOCKID.index(blocks26)
        else:
            continue

        if useblocks:
            if weight_index > 0: 
                current_alpha = weights_a[weight_index - 1] 
                if usebeta:
                    current_beta = weights_b[weight_index - 1] 

        if len(deep) > 0:
            current_alpha = elementals(key,weight_index,deep,randomer,num,lucks,deepprint,current_alpha)

        keyratio.append([key,current_alpha, current_beta])
        #keyratio.append([key,current_alpha, current_beta,list(theta_0[key].shape),torch.sum(theta_0[key]).item(), torch.mean(theta_0[key]).item(), torch.max(theta_0[key]).item(),  torch.min(theta_0[key]).item()])

        if calcmode == "normal":
            if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
                # Merge only the vectors the models have in common.  Otherwise we get an error due to dimension mismatch.
                theta_0_a = theta_0[key][:, 0:4, :, :]
            else:
                theta_0_a = theta_0[key]

            if MODES[1] in mode:#Add
                caster(f"{num}, {block}, {model_a}+{current_alpha}+*({model_b}-{model_c}),{key}",hear)
                if uselerp:
                    theta_0_a = torch.lerp(theta_0_a.to(torch.float32),theta_0_a.to(torch.float32) + current_alpha * theta_1[key].to(torch.float32),1.0).to(theta_0_a.dtype)
                else:
                    theta_0_a = (theta_0_a.to(torch.float32) + current_alpha * theta_1[key].to(torch.float32)).to(theta_0_a.dtype)
                
            elif MODES[2] in mode:#Triple
                caster(f"{num}, {block}, {model_a}+{1-current_alpha-current_beta}+{model_b}*{current_alpha}+ {model_c}*{current_beta}",hear)
                #
                if uselerp and current_alpha + current_beta != 0:
                    theta_0_a =lerp(theta_0_a.to(torch.float32),lerp(theta_1[key].to(torch.float32),theta_2[key].to(torch.float32),current_beta/(current_alpha + current_beta)),current_alpha + current_beta).to(theta_0_a.dtype)
                else:
                    theta_0_a = (1 - current_alpha-current_beta) * theta_0_a + current_alpha * theta_1[key]+current_beta * theta_2[key] 

            elif MODES[3] in mode:#Twice
                caster(f"{num}, {block}, {key},{model_a} +  {1-current_alpha} + {model_b}*{current_alpha}",hear)
                caster(f"{num}, {block}, {key}({model_a}+{model_b}) +{1-current_beta}+{model_c}*{current_beta}",hear)
                if uselerp:
                    theta_0_a = torch.lerp(torch.lerp(theta_0_a.to(torch.float32), theta_1[key].to(torch.float32), current_alpha), theta_2[key].to(torch.float32), current_beta).to(theta_0_a.dtype)
                else:
                    theta_0_a = (1 - current_alpha) * theta_0_a + current_alpha * theta_1[key]
                    theta_0_a = (1 - current_beta) * theta_0_a + current_beta * theta_2[key]

            else:#Weight
                if current_alpha == 1:
                    caster(f"{num}, {block}, {key} alpha = 1,{model_a}={model_b}",hear)
                    theta_0_a = theta_1[key]
                elif current_alpha !=0:
                    caster(f"{num}, {block}, {key}, {model_a}*{1-current_alpha}+{model_b}*{current_alpha}",hear)
                    if uselerp:
                        theta_0_a = torch.lerp(theta_0_a.to(torch.float32), theta_1[key].to(torch.float32), current_alpha).to(theta_0_a.dtype)
                    else:
                        theta_0_a = (1 - current_alpha) * theta_0_a + current_alpha * theta_1[key]

            if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
                theta_0[key][:, 0:4, :, :] = theta_0_a
            else:
                theta_0[key] = theta_0_a

            theta_0_a = a = b = None
            del theta_0_a, a, b

        elif "cosine" in calcmode:
            if "first_stage_model" in key: continue
            cosine(calcmode,key,sim,sims,current_alpha,theta_0,theta_1,num,block,uselerp)

        elif calcmode == "trainDifference":
            if torch.allclose(theta_1[key].float(), theta_2[key].float().to(device=theta_1[key].device), rtol=0, atol=0):
                theta_2[key] = theta_0[key]
                continue
            traindiff(key,current_alpha,theta_0,theta_1,theta_2)

        elif calcmode == "smoothAdd":
            caster(f"{num}, {block}, model A[{key}] +  {current_alpha} + * (model B - model C)[{key}]", hear)
            # Apply median filter to the weight differences
            filtered_diff = scipy.ndimage.median_filter(theta_1[key].to(torch.float32).cpu().numpy(), size=3)
            # Apply Gaussian filter to the filtered differences
            filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)
            theta_1[key] = torch.tensor(filtered_diff).to(theta_0[key].device)
            # Add the filtered differences to the original weights
            theta_0[key] = theta_0[key] + current_alpha * theta_1[key]

        elif calcmode == "smoothAdd MT":
            key_and_alpha[key] = current_alpha

        elif "tensor" in calcmode:
            dim = theta_0[key].dim()
            if dim == 0 : continue
            tensormerge("2" not in calcmode,key,dim,theta_0,theta_1,current_alpha,current_beta)

        elif "extract" == calcmode:
            theta_0[key] = extract_super(theta_0[key],theta_1[key],theta_2[key],current_alpha,current_beta,opt_value)

        elif calcmode == "self":
            if any(selfkey in key for selfkey in SELFKEYS):continue
            if current_alpha == 0: continue
            theta_0[key] = (theta_0[key].clone()) * current_alpha

        elif calcmode == "plus random":
            if any(selfkey in key for selfkey in SELFKEYS):continue
            if current_alpha == 0: continue
            theta_0[key] +=  torch.randn_like(theta_0[key].clone()) * current_alpha

        ##### Adjust
        if any(item in key for item in FINETUNES) and fine:
            index = FINETUNES.index(key)
            if 5 > index : 
                theta_0[key] =theta_0[key]* fine[index] 
            else :theta_0[key] =theta_0[key] + torch.tensor(fine[5]).to(theta_0[key].device)

        ##### del quantize info
        if isflux and not calcmode == "smoothAdd MT":
            theta_1[key] = None
            del theta_1[key]

        theta_0[key] = theta_0[key].to("cpu")
        try:
            theta_1[key] = theta_1[key].to("cpu")
        except:
            pass
    
    if calcmode == "smoothAdd MT":
        # setting threads to higher than 8 doesn't significantly affect the time for merging
        threads = cpu_count()
        tasks_per_thread = 8

        theta_0, theta_1, stopped = multithread_smoothadd(key_and_alpha, theta_0, theta_1, threads, tasks_per_thread, hear)
        if stopped:
            return "STOPPED", *NON4

    if need_revert:
        keys = list(theta_0.keys())
        for key in keys:
            theta_0[key.replace(PREFIX_M,"")] = theta_0.pop(key)

    currentmodel = makemodelname(weights_a,weights_b,model_a, model_b,model_c, base_alpha,base_beta,useblocks,mode,calcmode)

    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if key in CHCKPOINT_DICT_SKIP_ON_MERGE or isflux:
            continue
        if "model" in key and key not in theta_0:
            theta_0.update({key:theta_1[key]})

    theta_1 = None
    del theta_1
    if calcmode == "trainDifference" or calcmode == "extract":
        theta_2 = None
        del theta_2
    gc.collect()

    ##### BakeVAE
    bake_in_vae_filename = sd_vae.vae_dict.get(bake_in_vae, None)
    if bake_in_vae_filename is not None:
        print(f"Baking in VAE from {bake_in_vae_filename}")
        vae_dict = load_torch_file(bake_in_vae_filename, device = torch.device(device)) if forge else sd_vae.load_vae_dict(bake_in_vae_filename, map_location=device)

        for key in vae_dict.keys():
            theta_0_key = 'first_stage_model.' + key
            if theta_0_key in theta_0:
                theta_0[theta_0_key] = vae_dict[key]
        vae_dict = None
        del vae_dict

    modelid = rwmergelog(currentmodel,mergedmodel)
    if "save E-list" in lucks["set"]: saveekeys(keyratio,modelid)

    caster(mergedmodel,False)
    if "Reset CLIP ids" in save_sets: resetclip(theta_0)

    if True: # always set metadata. savemodel() will check save_sets later
        merge_recipe = {
            "type": "sd-webui-supermerger",
            "weights_alpha": weights_a if useblocks else None,
            "weights_beta": weights_b if useblocks else None,
            "weights_alpha_orig": weights_a_orig if useblocks else None,
            "weights_beta_orig": weights_b_orig if useblocks else None,
            "model_a": longhashfromname(model_a),
            "model_b": longhashfromname(model_b),
            "model_c": longhashfromname(model_c),
            "base_alpha": base_alpha,
            "base_beta": base_beta,
            "mode": mode,
            "mbw": useblocks,
            "elemental_merge": deep,
            "calcmode" : calcmode,
            f"{inex}":ex_blocks + ex_elems
            }
        metadata["sd_merge_recipe"] = json.dumps(merge_recipe)
        metadata["sd_merge_models"] = {}

        def add_model_metadata(checkpoint_name):
            checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_name)
            checkpoint_info.calculate_shorthash()
            metadata["sd_merge_models"][checkpoint_info.sha256] = {
                "name": checkpoint_name,
                "legacy_hash": checkpoint_info.hash
            }

            #metadata["sd_merge_models"].update(checkpoint_info.metadata.get("sd_merge_models", {}))

        if model_a:
            add_model_metadata(model_a)
        if model_b:
            add_model_metadata(model_b)
        if model_c:
            add_model_metadata(model_c)

        metadata["sd_merge_models"] = json.dumps(metadata["sd_merge_models"])

    return "",currentmodel,modelid,theta_0,metadata

################################################
##### cosineA/B
def precosine(calcmode,theta_0,theta_1):
    if calcmode: #favors modelA's structure with details from B
        if stopmerge: return "STOPPED", *NON4
        sim = torch.nn.CosineSimilarity(dim=0)
        sims = np.array([], dtype=np.float64)
        for key in (tqdm(theta_0.keys(), desc="Stage 0/2")):
            # skip VAE model parameters to get better results
            if "first_stage_model" in key: continue
            if "model" in key and key in theta_1:
                theta_0_norm = nn.functional.normalize(theta_0[key].to(torch.float32), p=2, dim=0)
                theta_1_norm = nn.functional.normalize(theta_1[key].to(torch.float32), p=2, dim=0)
                simab = sim(theta_0_norm, theta_1_norm)
                sims = np.append(sims,simab.cpu().numpy())
        sims = sims[~np.isnan(sims)]
        sims = np.delete(sims, np.where(sims<np.percentile(sims, 1 ,method = 'midpoint')))
        sims = np.delete(sims, np.where(sims>np.percentile(sims, 99 ,method = 'midpoint')))
    else: #favors modelB's structure with details from A
        if stopmerge: return "STOPPED", *NON4
        sim = torch.nn.CosineSimilarity(dim=0)
        sims = np.array([], dtype=np.float64)
        for key in (tqdm(theta_0.keys(), desc="Stage 0/2")):
            # skip VAE model parameters to get better results
            if "first_stage_model" in key: continue
            if "model" in key and key in theta_1:
                simab = sim(theta_0[key].to(torch.float32), theta_1[key].to(torch.float32))
                dot_product = torch.dot(theta_0[key].view(-1).to(torch.float32), theta_1[key].view(-1).to(torch.float32))
                magnitude_similarity = dot_product / (torch.norm(theta_0[key].to(torch.float32)) * torch.norm(theta_1[key].to(torch.float32)))
                combined_similarity = (simab + magnitude_similarity) / 2.0
                sims = np.append(sims, combined_similarity.cpu().numpy())
        sims = sims[~np.isnan(sims)]
        sims = np.delete(sims, np.where(sims < np.percentile(sims, 1, method='midpoint')))
        sims = np.delete(sims, np.where(sims > np.percentile(sims, 99, method='midpoint')))
    return sim, sims

def cosine(mode,key,sim,sims,current_alpha,theta_0,theta_1,num,block,uselerp):
    if "A" in mode: #favors modelA's structure with details from B
        # skip VAE model parameters to get better results
        if "model" in key and key in theta_0:
            # Normalize the vectors before merging
            theta_0_norm = nn.functional.normalize(theta_0[key].to(torch.float32), p=2, dim=0)
            theta_1_norm = nn.functional.normalize(theta_1[key].to(torch.float32), p=2, dim=0)
            simab = sim(theta_0_norm, theta_1_norm)
            dot_product = torch.dot(theta_0_norm.view(-1), theta_1_norm.view(-1))
            magnitude_similarity = dot_product / (torch.norm(theta_0_norm) * torch.norm(theta_1_norm))
            combined_similarity = (simab + magnitude_similarity) / 2.0
            k = (combined_similarity - sims.min()) / (sims.max() - sims.min())
            k = k - abs(current_alpha)
            k = k.clip(min=0,max=1.0)
            caster(f"{num}, {block}, model A[{key}] {1-k} +  (model B)[{key}]*{k}",hear)
            if uselerp:
                theta_0[key] = lerp(theta_1[key].to(torch.float32), theta_0[key].to(torch.float32),k).to(theta_0[key].dtype)
            else:
                theta_0[key] = theta_1[key] * (1 - k) + theta_0[key] * k

    else: #favors modelB's structure with details from A
        # skip VAE model parameters to get better results
        if "model" in key and key in theta_0:
            simab = sim(theta_0[key].to(torch.float32), theta_1[key].to(torch.float32))
            dot_product = torch.dot(theta_0[key].view(-1).to(torch.float32), theta_1[key].view(-1).to(torch.float32))
            magnitude_similarity = dot_product / (torch.norm(theta_0[key].to(torch.float32)) * torch.norm(theta_1[key].to(torch.float32)))
            combined_similarity = (simab + magnitude_similarity) / 2.0
            k = (combined_similarity - sims.min()) / (sims.max() - sims.min())
            k = k - current_alpha
            k = k.clip(min=0,max=1.0)
            caster(f"{num}, {block}, model A[{key}] *{1-k} + (model B)[{key}]*{k}",hear)
            if uselerp:
                theta_0[key] = lerp(theta_1[key].to(torch.float32), theta_0[key].to(torch.float32),k).to(theta_0[key].dtype)
            else:
                theta_0[key] = theta_1[key] * (1 - k) + theta_0[key] * k

################################################
##### Traindiff
def traindiff(key,current_alpha,theta_0,theta_1,theta_2):
            # Check if theta_1[key] is equal to theta_2[key]
    diff_AB = theta_1[key].float() - theta_2[key].float()

    distance_A0 = torch.abs(theta_1[key].float() - theta_2[key].float())
    distance_A1 = torch.abs(theta_1[key].float() - theta_0[key].float())

    sum_distances = distance_A0 + distance_A1

    scale = torch.where(sum_distances != 0, distance_A1 / sum_distances, torch.tensor(0.).float())
    sign_scale = torch.sign(theta_1[key].float() - theta_2[key].float())
    scale = sign_scale * torch.abs(scale)

    new_diff = scale * torch.abs(diff_AB)
    theta_0[key] = theta_0[key] + (new_diff * (current_alpha*1.8))

################################################
##### Extract
def extract_super(base: Optional[Tensor], a: Tensor, b: Tensor, alpha: float, beta: float, gamma: float) -> Tensor:
    assert base is None or base.shape == a.shape
    assert a.shape == b.shape
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert 0 <= gamma
    dtype = base.dtype if base is not None else a.dtype
    base = base.float() if base is not None else 0
    a = a.float() - base
    b = b.float() - base
    c = cosine_similarity(a, b, -1).clamp(-1, 1).unsqueeze(-1)
    d = ((c + 1) / 2) ** gamma
    result = base + lerp(a, b, alpha) * lerp(d, 1 - d, beta)
    return result.to(dtype)

def extract(a: Tensor, b: Tensor, p: float, smoothness: float) -> Tensor:
    assert a.shape == b.shape
    assert 0 <= p <= 1
    assert 0 <= smoothness <= 1
    
    r = relu if smoothness == 0 else partial(softplus, beta=1 / smoothness)
    c = r(cosine_similarity(a, b, dim=-1)).unsqueeze(dim=-1).repeat_interleave(b.shape[-1], -1)
    m = torch.lerp(c, torch.ones_like(c) - c, p)
    return a * m

################################################
##### Tensor Merge
def tensormerge(mode,key,dim, theta_0,theta_1,current_alpha,current_beta):
    if mode:
        if current_alpha+current_beta <= 1 :
            talphas = int(theta_0[key].shape[0]*(current_beta))
            talphae = int(theta_0[key].shape[0]*(current_alpha+current_beta))
            if dim == 1:
                theta_0[key][talphas:talphae] = theta_1[key][talphas:talphae].clone()

            elif dim == 2:
                theta_0[key][talphas:talphae,:] = theta_1[key][talphas:talphae,:].clone()

            elif dim == 3:
                theta_0[key][talphas:talphae,:,:] = theta_1[key][talphas:talphae,:,:].clone()

            elif dim == 4:
                theta_0[key][talphas:talphae,:,:,:] = theta_1[key][talphas:talphae,:,:,:].clone()

        else:
            talphas = int(theta_0[key].shape[0]*(current_alpha+current_beta-1))
            talphae = int(theta_0[key].shape[0]*(current_beta))
            theta_t = theta_1[key].clone()
            if dim == 1:
                theta_t[talphas:talphae] = theta_0[key][talphas:talphae].clone()

            elif dim == 2:
                theta_t[talphas:talphae,:] = theta_0[key][talphas:talphae,:].clone()

            elif dim == 3:
                theta_t[talphas:talphae,:,:] = theta_0[key][talphas:talphae,:,:].clone()

            elif dim == 4:
                theta_t[talphas:talphae,:,:,:] = theta_0[key][talphas:talphae,:,:,:].clone()
            theta_0[key] = theta_t

    else:
        if current_alpha+current_beta <= 1 :
            talphas = int(theta_0[key].shape[0]*(current_beta))
            talphae = int(theta_0[key].shape[0]*(current_alpha+current_beta))
            if dim > 1:
                if theta_0[key].shape[1] > 100:
                    talphas = int(theta_0[key].shape[1]*(current_beta))
                    talphae = int(theta_0[key].shape[1]*(current_alpha+current_beta))
            if dim == 1:
                theta_0[key][talphas:talphae] = theta_1[key][talphas:talphae].clone()

            elif dim == 2:
                theta_0[key][:,talphas:talphae] = theta_1[key][:,talphas:talphae].clone()

            elif dim == 3:
                theta_0[key][:,talphas:talphae,:] = theta_1[key][:,talphas:talphae,:].clone()

            elif dim == 4:
                theta_0[key][:,talphas:talphae,:,:] = theta_1[key][:,talphas:talphae,:,:].clone()

        else:
            talphas = int(theta_0[key].shape[0]*(current_alpha+current_beta-1))
            talphae = int(theta_0[key].shape[0]*(current_beta))
            theta_t = theta_1[key].clone()
            if dim > 1:
                if theta_0[key].shape[1] > 100:
                    talphas = int(theta_0[key].shape[1]*(current_alpha+current_beta-1))
                    talphae = int(theta_0[key].shape[1]*(current_beta))
            if dim == 1:
                theta_t[talphas:talphae] = theta_0[key][talphas:talphae].clone()

            elif dim == 2:
                theta_t[:,talphas:talphae] = theta_0[key][:,talphas:talphae].clone()

            elif dim == 3:
                theta_t[:,talphas:talphae,:] = theta_0[key][:,talphas:talphae,:].clone()

            elif dim == 4:
                theta_t[:,talphas:talphae,:,:] = theta_0[key][:,talphas:talphae,:,:].clone()
            theta_0[key] = theta_t

################################################
##### Multi Thread SmoothAdd

def multithread_smoothadd(key_and_alpha, theta_0, theta_1, threads, tasks_per_thread, hear):  
    lock_theta_0 = Lock()
    lock_theta_1 = Lock()
    lock_progress = Lock()

    def thread_callback(keys):
        nonlocal theta_0, theta_1
        if stopmerge:
            return False

        for key in keys:
            if key not in theta_1:
                continue
            caster(f"model A[{key}] +  {key_and_alpha[key]} + * (model B - model C)[{key}]", hear)
            filtered_diff = scipy.ndimage.median_filter(theta_1[key].to(torch.float32).cpu().numpy(), size=3)
            filtered_diff = scipy.ndimage.gaussian_filter(filtered_diff, sigma=1)
            with lock_theta_1:
                theta_1[key] = torch.tensor(filtered_diff)
            with lock_theta_0:
                theta_0[key] = theta_0[key] + key_and_alpha[key] * theta_1[key]

        with lock_progress:
            progress.update(len(keys))

        return True

    def extract_and_remove(input_list, count):
        extracted = input_list[:count]
        del input_list[:count]

        return extracted

    keys = list(key_and_alpha.keys())

    total_threads = ceil(len(keys) / int(tasks_per_thread))
    print(f"max threads = {threads}, total threads = {total_threads}, tasks per thread = {tasks_per_thread}")

    progress = tqdm(key_and_alpha.keys(), desc="smoothAdd MT")

    futures = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(thread_callback, extract_and_remove(keys, int(tasks_per_thread))) for i in range(total_threads)]
        for future in as_completed(futures):
            if not future.result():
                executor.shutdown()
                return theta_0, theta_1, True
        del progress

    return theta_0, theta_1, False

################################################
##### Elementals
def elementals(key,weight_index,deep,randomer,num,lucks,deepprint,current_alpha):
    skey = key + BLOCKID[weight_index]
    for d in deep:
        if d.count(":") != 2 :continue
        dbs,dws,dr = d.split(":")[0],d.split(":")[1],d.split(":")[2]
        dbs = blocker(dbs,BLOCKID)
        dbs,dws = dbs.split(" "), dws.split(" ")
        dbn,dbs = (True,dbs[1:]) if dbs[0] == "NOT" else (False,dbs)
        dwn,dws = (True,dws[1:]) if dws[0] == "NOT" else (False,dws)
        flag = dbn
        for db in dbs:
            if db in skey:
                flag = not dbn
        if flag:flag = dwn
        else:continue
        for dw in dws:
            if dw in skey:
                flag = not dwn
        if flag:
            dr = eratiodealer(dr,randomer,weight_index,num,lucks)
            if deepprint :print(" ", dbs,dws,key,dr)
            current_alpha = dr
    return current_alpha

def forkforker(filename,device):
    if forge:
        return load_torch_file(filename, device = torch.device(device))
    try:
        return sd_models.read_state_dict(filename, map_location = device)
    except:
        return sd_models.read_state_dict(filename)

################################################
##### Load Model

def load_model_weights_m(model,abc,cachetarget,device):
    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    sd_model_name = checkpoint_info.model_name

    if checkpoint_info in modelcache:
        print(f"Loading weights [{sd_model_name}] from cache")
        return {k: v.to(device) for k, v in modelcache[checkpoint_info].items()}
    else:
        print(f"Loading weights [{sd_model_name}] from file")
        state_dict = forkforker(checkpoint_info.filename,device)
        if orig_cache >= abc:
            modelcache[checkpoint_info] = state_dict
            modelcache[checkpoint_info] = {k: v.to("cpu") for k, v in modelcache[checkpoint_info].items()}
        dontdelete = []
        for model in cachetarget:
            dontdelete.append(sd_models.get_closet_checkpoint_match(model))
        while len(modelcache) > orig_cache:
            for key in modelcache.keys():
                if key in dontdelete:continue
                modelcache.pop(key)
                break
        return state_dict

def makemodelname(weights_a,weights_b,model_a, model_b,model_c, alpha,beta,useblocks,mode,calc):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    model_c=filenamecutter(model_c)

    if type(alpha) == str:alpha = float(alpha)
    if type(beta)== str:beta  = float(beta)

    if useblocks:
        if MODES[1] in mode:#add
            currentmodel =f"{model_a} + ({model_b} - {model_c}) x alpha ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)})"
        elif MODES[2] in mode:#triple
            currentmodel =f"{model_a} x (1-alpha-beta) + {model_b} x alpha + {model_c} x beta (alpha = {str(round(alpha,3))},{','.join(str(s) for s in weights_a)},beta = {beta},{','.join(str(s) for s in weights_b)})"
        elif MODES[3] in mode:#twice
            currentmodel =f"({model_a} x (1-alpha) + {model_b} x alpha)x(1-beta)+  {model_c} x beta ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)})_({str(round(beta,3))},{','.join(str(s) for s in weights_b)})"
        else:
            currentmodel =f"{model_a} x (1-alpha) + {model_b} x alpha ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)})"
    else:
        if MODES[1] in mode:#add
            currentmodel =f"{model_a} + ({model_b} -  {model_c}) x {str(round(alpha,3))}"
        elif MODES[2] in mode:#triple
            currentmodel =f"{model_a} x {str(round(1-alpha-beta,3))} + {model_b} x {str(round(alpha,3))} + {model_c} x {str(round(beta,3))}"
        elif MODES[3] in mode:#twice
            currentmodel =f"({model_a} x {str(round(1-alpha,3))} +{model_b} x {str(round(alpha,3))}) x {str(round(1-beta,3))} + {model_c} x {str(round(beta,3))}"
        else:
            currentmodel =f"{model_a} x {str(round(1-alpha,3))} + {model_b} x {str(round(alpha,3))}"
    if calc != "normal":
        currentmodel = currentmodel + "_" + calc
        if calc == "tensor":
            currentmodel = currentmodel + f"_beta_{beta}"
    return currentmodel

path_root = scripts.basedir()


################################################
##### Logging

def rwmergelog(mergedname = "",settings= [],id = 0):
    # for compatible
    mode_info = {
        "Weight sum": "Weight sum:A*(1-alpha)+B*alpha",
        "Add difference": "Add difference:A+(B-C)*alpha",
        "Triple sum": "Triple sum:A*(1-alpha-beta)+B*alpha+C*beta",
        "sum Twice": "sum Twice:(A*(1-alpha)+B*alpha)*(1-beta)+C*beta",
    }
    setting = settings.copy()
    if len(setting) > 7 and setting[7] in mode_info:
        setting[7] = mode_info[setting[7]] # fix mode entry for compatible
    filepath = os.path.join(path_root, "mergehistory.csv")
    is_file = os.path.isfile(filepath)

    csv.field_size_limit(2244096)

    if not is_file:
        with open(filepath, 'a') as f:
                                       #msettings=[0 weights_a,1 weights_b,2 model_a,3 model_b,4 model_c,5 base_alpha,6 base_beta,7 mode,8 useblocks,9 custom_name,10 save_sets,11 id_sets, 12 deep 13 calcmode]
            f.writelines('"ID","time","name","weights alpha","weights beta","model A","model B","model C","alpha","beta","mode","use MBW","plus lora","custum name","save setting","use ID"\n')
    with  open(filepath, 'r+') as f:
        reader = csv.reader(f)
        mlist = [raw for raw in reader]
        if mergedname != "":
            mergeid = len(mlist)
            setting.insert(0,mergedname)
            for i,x in enumerate(setting):
                if "," in str(x) or "\n" in str(x):setting[i] = f'"{str(setting[i])}"'
            text = ",".join(map(str, setting))
            text=str(mergeid)+","+datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S.%f')[:-7]+"," + text + "\n"
            f.writelines(text)
            return mergeid
        try:
            out = mlist[int(id)]
        except:
            out = "ERROR: OUT of ID index"
        return out

def saveekeys(keyratio,modelid):
    import csv
    path_root = scripts.basedir()
    dir_path = os.path.join(path_root,"extensions","sd-webui-supermerger","scripts", "data")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    filepath = os.path.join(dir_path,f"{modelid}.csv")

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(keyratio)

def savestatics(modelid):
    for key in statistics.keys():
        result = [[tkey] + list(statistics[key][tkey]) for tkey in statistics[key].keys()]
        saveekeys(result,f"{modelid}_{key}")

def get_font(fontsize):
    fontpath = os.path.join(scriptpath, "Roboto-Regular.ttf")
    try:
        return ImageFont.truetype(opts.font or fontpath, fontsize)
    except Exception:
        return ImageFont.truetype(fontpath, fontsize)

def draw_origin(grid, text,width,height,width_one):
    grid_d= Image.new("RGB", (grid.width,grid.height), "white")
    grid_d.paste(grid,(0,0))

    d= ImageDraw.Draw(grid_d)
    color_active = (0, 0, 0)
    fontsize = (width+height)//25
    fnt = get_font(fontsize)

    if grid.width != width_one:
        while d.multiline_textbbox((0,0), text, font=fnt)[2] > width_one*0.75 and fontsize > 0:
            fontsize -=1
            fnt = get_font(fontsize)
    d.multiline_text((0,0), text, font=fnt, fill=color_active,align="center")
    return grid_d

def wpreseter(w,presets):
    if "," not in w and w != "":
        presets=presets.splitlines()
        wdict={}
        for l in presets:
            if ":" in l :
                key = l.split(":",1)[0]
                wdict[key.strip()]=l.split(":",1)[1]
            if "\t" in l:
                key = l.split("\t",1)[0]
                wdict[key.strip()]=l.split("\t",1)[1]
        if w.strip() in wdict:
            name = w
            w = wdict[w.strip()]
            print(f"weights {name} imported from presets : {w}")
    return w

def fullpathfromname(name):
    if hash == "" or hash ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    return checkpoint_info.filename

def namefromhash(hash):
    if hash == "" or hash ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(hash)
    return checkpoint_info.model_name

def hashfromname(name):
    from modules import sd_models
    if name == "" or name ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    if checkpoint_info.shorthash is not None:
        return checkpoint_info.shorthash
    return checkpoint_info.calculate_shorthash()

def longhashfromname(name):
    from modules import sd_models
    if name == "" or name ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    if checkpoint_info.sha256 is not None:
        return checkpoint_info.sha256
    checkpoint_info.calculate_shorthash()
    return checkpoint_info.sha256


################################################
##### Random

RANCHA = ["R","U","X"]

def randdealer(w:str,randomer,ab,lucks,deep):
    up,low = lucks["upp"],lucks["low"]
    up,low = (up.split(","),low.split(","))
    out = []
    outd = {"R":[],"U":[],"X":[]}
    add = RANDMAP[ab]
    for i, r in enumerate (w.split(",")):
        if r.strip() =="R":
            out.append(str(round(randomer[i+add],lucks["round"])))
        elif r.strip() == "U":
            out.append(str(round(-2 * randomer[i+add] + 1.5,lucks["round"])))
        elif r.strip() == "X":
            out.append(str(round((float(low[i])-float(up[i]))* randomer[i+add] + float(up[i]),lucks["round"])))
        elif "E" in r:
            key = r.strip().replace("E","")
            outd[key].append(BLOCKID[i])
            out.append("0")
        else:
            out.append(r)
    for key in outd.keys():
        if outd[key] != []:
            deep = deep + f",{' '.join(outd[key])}::{key}" if deep else f"{' '.join(outd[key])}::{key}"
    return ",".join(out), deep

def eratiodealer(dr,randomer,block,num,lucks):
    if  any(element in dr for element in RANCHA):
        up,low = lucks["upp"],lucks["low"]
        up,low = (up.split(","),low.split(","))
        add = RANDMAP[2]
        if dr.strip() =="R":
            return round(randomer[num+add],lucks["round"])
        elif dr.strip() == "U":
            return round(-2 * randomer[num+add] + 1,lucks["round"])
        elif dr.strip() == "X":
            return round((float(low[block])-float(up[block]))* randomer[num+add] + float(up[block]),lucks["round"])
    else:
        return float(dr)


################################################
##### Generate Image

def simggen(s_prompt,s_nprompt,s_steps,s_sampler,s_cfg,s_seed,s_w,s_h,s_batch_size,
            genoptions,s_hrupscaler,s_hr2ndsteps,s_denois_str,s_hr_scale,
            mergeinfo,id_sets,modelid,
            *txt2imgparams,
            debug = False
            ):
    shared.state.begin()
    from scripts.mergers.components import paramsnames
    if debug: print(paramsnames)

    #[None, 'Prompt', 'Negative prompt', 'Styles', 'Sampling steps', 'Sampling method', 'Batch count', 'Batch size', 'CFG Scale', 
    # 'Height', 'Width', 'Hires. fix', 'Denoising strength', 'Upscale by', 'Upscaler', 'Hires steps', 'Resize width to', 'Resize height to', 
    # 'Hires checkpoint', 'Hires sampling method', 'Hires prompt', 'Hires negative prompt', 'Override settings', 'Script', 'Refiner', 
    # 'Checkpoint', 'Switch at', 'Seed', 'Extra', 'Variation seed', 'Variation strength', 'Resize seed from width', 'Resize seed from height', '', 'Active', 'Active', 'X Types', 'X Values', 'Y Types', 'Y Values']  

    def g(wanted,wantedv=None):
        if wanted in paramsnames:return txt2imgparams[paramsnames.index(wanted)]
        elif wantedv and wantedv in paramsnames:return txt2imgparams[paramsnames.index(wantedv)]
        else:return None

    sampler_index = g("Sampling method")
    if type(sampler_index) is str:
        sampler_name = sampler_index
    else:       
        sampler_name = sd_samplers.samplers[sampler_index].name

    hr_sampler_index = g("Hires sampling method")
    if hr_sampler_index is None: hr_sampler_index = 0
    if type(sampler_index) is str:
        hr_sampler_name = hr_sampler_index
    else:       
        hr_sampler_name = "Use same sampler" if hr_sampler_index == 0 else  sd_samplers.samplers[hr_sampler_index+1].name

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=g("Prompt"),
        styles=g("Styles"),
        negative_prompt=g('Negative prompt'),
        seed=g("Seed","Initial seed"),
        subseed=g("Variation seed"),
        subseed_strength=g("Variation strength"),
        seed_resize_from_h=g("Resize seed from height"),
        seed_resize_from_w=g("Resize seed from width"),
        seed_enable_extras=g("Extra"),
        sampler_name=sampler_name,
        batch_size=g("Batch size"),
        n_iter=g("Batch count"),
        steps=g("Sampling steps"),
        cfg_scale=g("CFG Scale"),
        width=g("Width"),
        height=g("Height"),
        restore_faces=g("Restore faces","Face restore"),
        tiling=g("Tiling"),
        enable_hr=g("Hires. fix","Second pass"),
        hr_scale=g("Upscale by"),
        hr_upscaler=g("Upscaler"),
        hr_second_pass_steps=g("Hires steps","Secondary steps"),
        hr_resize_x=g("Resize width to"),
        hr_resize_y=g("Resize height to"),
        override_settings=create_override_settings_dict(g("Override settings")),
        do_not_save_grid=True,
        do_not_save_samples=True,
        do_not_reload_embeddings=True,
    )
    p.hr_checkpoint_name=None if g("Hires checkpoint") == 'Use same checkpoint' else g("Hires checkpoint")
    p.hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else  hr_sampler_name
    p.hr_additional_modules = []

    if s_sampler is None: s_sampler = 0

    if s_batch_size != 1 :p.batch_size = int(s_batch_size)
    if s_prompt: p.prompt = s_prompt
    if s_nprompt: p.negative_prompt = s_nprompt
    if s_steps: p.steps = s_steps
    if s_sampler: p.sampler_name = sampler_name
    if s_cfg: p.cfg_scale = s_cfg
    if s_seed: p.seed = s_seed
    if s_w: p.width = s_w
    if s_h: p.height = s_h

    if not p.cfg_scale: p.cfg_scale = 7

    p.scripts = scripts.scripts_txt2img
    p.script_args = txt2imgparams[paramsnames.index("Override settings")+1:]
                
    p.denoising_strength=g("Denoising strength") if p.enable_hr else None

    p.hr_prompt=g("Hires prompt","Secondary Prompt")
    p.hr_negative_prompt=g("Hires negative prompt","Secondary negative prompt")

    if "Hires. fix" in genoptions:
        p.enable_hr = True
        if s_hrupscaler: p.hr_upscaler = s_hrupscaler
        if s_hr2ndsteps:p.hr_second_pass_steps = s_hr2ndsteps
        if s_denois_str:p.denoising_strength = s_denois_str
        if s_hr_scale:p.hr_scale = s_hr_scale

    if "Restore faces" in genoptions:
        p.restore_faces = True

    if "Tiling" in genoptions:
        p.tiling = True

    p.cached_c = [None,None]
    p.cached_uc = [None,None]

    p.cached_hr_c = [None, None]
    p.cached_hr_uc = [None, None]

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    if forge:
        global orig_reload_model_weights
        orig_reload_model_weights = sd_models.reload_model_weights
        sd_models.reload_model_weights = reload_model_weights

        processed:Processed = processing.process_images(p)

        sd_models.reload_model_weights = orig_reload_model_weights
    else:
        processed:Processed = processing.process_images(p)

    if "image" in id_sets:
        for i, image in enumerate(processed.images):
            processed.images[i] = draw_origin(image, str(modelid),p.width,p.height,p.width)

    if "PNG info" in id_sets:mergeinfo = mergeinfo + " ID " + str(modelid)

    infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds)
    if infotext.count("Steps: ")>1:
        infotext = infotext[:infotext.rindex("Steps")]

    infotexts = infotext.split(",")
    for i,x in enumerate(infotexts):
        if "Model:"in x:
            infotexts[i] = " Model: "+mergeinfo.replace(","," ")
    infotext= ",".join(infotexts)

    for i, image in enumerate(processed.images):
        images.save_image(image, opts.outdir_txt2img_samples, "",p.seed, p.prompt,shared.opts.samples_format, p=p,info=infotext)

    if s_batch_size > 1:
        grid = images.image_grid(processed.images, p.batch_size)
        processed.images.insert(0, grid)
        images.save_image(grid, opts.outdir_txt2img_grids, "grid", p.seed, p.prompt, opts.grid_format, info=infotext, short_filename=not opts.grid_extended_filename, p=p, grid=True)
    shared.state.end()
    return processed.images,infotext,plaintext_to_html(processed.info), plaintext_to_html(processed.comments),p


################################################
##### Block Ids

def blocker(blocks,blockids):
    blocks = blocks.split(" ")
    output = ""
    for w in blocks:
        flagger=[False]*len(blockids)
        changer = True
        if "-" in w:
            wt = [wt.strip() for wt in w.split('-')]
            if  blockids.index(wt[1]) > blockids.index(wt[0]):
                flagger[blockids.index(wt[0]):blockids.index(wt[1])+1] = [changer]*(blockids.index(wt[1])-blockids.index(wt[0])+1)
            else:
                flagger[blockids.index(wt[1]):blockids.index(wt[0])+1] = [changer]*(blockids.index(wt[0])-blockids.index(wt[1])+1)
        else:
            output = output + " " + w if output else w
        for i in range(len(blockids)):
            if flagger[i]: output = output + " " + blockids[i] if output else blockids[i]
    return output


def blockfromkey(key,isxl,isflux=False):
    if not isxl and not isflux:
        re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
        re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
        re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

        weight_index = -1

        NUM_INPUT_BLOCKS = 12
        NUM_MID_BLOCK = 1
        NUM_OUTPUT_BLOCKS = 12
        NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

        if 'time_embed' in key:
            weight_index = -2                # before input blocks
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
        return BLOCKID[weight_index+1] ,BLOCKID[weight_index+1] 
    
    elif isflux:
        # Extract the two-digit number using regex
        if "vae" in key:
            return "VAE", "Not Merge"
        if "t5xxl" in key:
            return "T5", "T5"
        if "text_encoders.clip" in key:
            return "CLIP", "CLIP"
        
        match = re.search(r'\.(\d+)\.', key)
        if "double_blocks" in key:
            return f"D{match.group(1).zfill(2) }", f"D{match.group(1).zfill(2) }"
        if "single_blocks" in key:
            return f"S{match.group(1).zfill(2) }", f"S{match.group(1).zfill(2) }"
        if "_in" in key:
            return "IN", "IN"
        if "final_layer" in key:
            return "OUT", "OUT"
        return "Not Merge"
    
    else:
        if not ("weight" in key or "bias" in key):return "Not Merge","Not Merge"
        if "label_emb" in key or "time_embed" in key: return "Not Merge","Not Merge"
        if "conditioner.embedders" in key : return "BASE","BASE"
        if "first_stage_model" in key : return "VAE","BASE"
        if "model.diffusion_model" in key:
            if "model.diffusion_model.out." in key: return "OUT8","OUT08"
            block = re.findall(r'input|mid|output', key)
            block = block[0].upper().replace("PUT","") if block else ""
            nums = re.sub(r"\D", "", key)[:1 if "MID" in block else 2] + ("0" if "MID" in block else "")
            if len(nums) == 0:return "M00"
            add = re.findall(r"transformer_blocks\.(\d+)\.",key)[0] if "transformer" in key else ""
            return block + nums + add, block + "0" + nums[0] if "MID" not in block else "M00"

    return "Not Merge", "Not Merge"

################################################
##### Assert Inpaint
def assert_inpaint(a, b, key):
    """ this enables merging an inpainting model (A) with another one (B);
        where normal model would have 4 channels, for latenst space, inpainting model would
        have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9 
    """
    if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
        if a[1] == 4 and b[1] == 9:
            raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")
        if a[1] == 4 and b[1] == 8:
            raise RuntimeError("When merging instruct-pix2pix model with a normal one, A must be the instruct-pix2pix model.")

        if a[1] == 8 and b[1] == 4:#If we have an Instruct-Pix2Pix model...
            result_is_instruct_pix2pix_model = True
        else:
            assert a[1] == 9 and b[1] == 4, f"Bad dimensions for merged layer {key}: A={a}, B={b}"
            result_is_inpainting_model = True

################################################
##### Adjust

def fineman(fine,isxl):
    if fine.find(",") != -1:
        tmp = [t.strip() for t in fine.split(",")]
        fines = [0.0]*8
        for i,f in enumerate(tmp[0:8]):
            try:
                f = float(f)
                fines[i] = f
            except Exception:
                pass

        fine = fines
    else:
        return None

    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3]*0.02] + colorcalc(fine[4:8],isxl)
        ]
    return fine

def colorcalc(cols,isxl):
    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]

COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
COLSXL = [[0,0,1],[1,0,0],[-1,-1,0],[-1,1,0]]

def weighttoxl(weight):
    weight = weight[:9] + weight[12:22] +[0]
    return weight

FINETUNES = [
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.weight",
"model.diffusion_model.out.2.bias",
]

################################################
##### Include/Exclude
def excluder(block:str,inex:bool,ex_blocks:list,ex_elems:list, key:str):
    if ex_blocks == [] and ex_elems == [""]:
        return False
    out = True if inex == "Include" else False
    if block in ex_blocks:out = not out
    if "Adjust" in ex_blocks and key in FINETUNES:out = not out
    for ke in ex_elems:
        if ke != "" and ke in key:out = not out
    if "VAE" in ex_blocks and "first_stage_model"in key:out = not out
    if "print" in ex_blocks and (out ^ (inex == "Include")):
        print("Include" if inex else "Exclude",block,ex_blocks,ex_elems,key)
    return out

################################################
##### Reset Broken CliP IDs

def resetclip(theta):
    idkey = "cond_stage_model.transformer.text_model.embeddings.position_ids"
    broken = []
    if idkey in theta.keys():
        correct = torch.Tensor([list(range(77))]).to(torch.int64)
        current = theta[idkey].to(torch.int64)

        broken = correct.ne(current)
        broken = [i for i in range(77) if broken[0][i]]

        if broken != []: print("Clip IDs broken and fixed: ",broken)
        
        theta[idkey] = correct


################################################
##### cache
def cachedealer(start):
    if start:
        global orig_cache
        orig_cache = shared.opts.sd_checkpoint_cache
        shared.opts.sd_checkpoint_cache = 0
    else:
        shared.opts.sd_checkpoint_cache = orig_cache

def clearcache(model_c):
    global modelcache
    del modelcache
    modelcache = {}
    
    if forge:
        unload_forge()
        unload_forge()
        from modules.sd_models import forge_model_reload, model_data
        from modules_forge.main_entry import forge_unet_storage_dtype_options
        unet_storage_dtype, _ = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))
        forge_model_params = dict(
            checkpoint_info=revert_target,
            additional_modules=shared.opts.forge_additional_modules,
            unet_storage_dtype=unet_storage_dtype
        )
        model_data.forge_hash = None
        model_data.forge_loading_parameters = forge_model_params
        forge_model_reload()

    gc.collect()
    devices.torch_gc()

def getcachelist():
    output = []
    for key in modelcache.keys():
        if hasattr(key, "model_name"):
            output.append(key.model_name)
    return ",".join(output)

################################################
##### print

def printstart(model_a,model_b,model_c,base_alpha,base_beta,weights_a,weights_b,mode,useblocks,calcmode,deep,lucks,fine,inex,ex_blocks,ex_elems,device):
    print(f"  model A  \t: {model_a}")
    print(f"  model B  \t: {model_b}")
    print(f"  model C  \t: {model_c}")
    print(f"  alpha,beta\t: {base_alpha,base_beta}")
    print(f"  weights_alpha\t: {weights_a}")
    print(f"  weights_beta\t: {weights_b}")
    print(f"  mode\t\t: {mode}")
    print(f"  MBW \t\t: {useblocks}")
    print(f"  CalcMode \t: {calcmode}")
    print(f"  Elemental \t: {deep}")
    print(f"  Weights Seed\t: {lucks}")
    print(f"  {inex} \t: {ex_blocks,ex_elems}")
    print(f"  Adjust \t: {fine}")
    print(f"  Device \t: {device}")

def caster(news,hear):
    if hear: print(news)

def casterr(*args,hear=hear):
    if hear:
        names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
        print('\n'.join([names.get(id(arg), '???') + ' = ' + repr(arg) for arg in args]))

################################################
##### model_loader
def model_loader(checkpoint_info, state_dict,metadata, currentmodel): 
    if ui_version >= 150: checkpoint_info = fake_checkpoint_info(checkpoint_info,metadata,currentmodel)

    if not forge:
        sd_models.model_data.__init__()
        load_model(checkpoint_info, already_loaded_state_dict=state_dict)
    else:
        memory_management.free_memory(1e30,torch.device("cpu"))
        
        load_forge_model(state_dict,checkpoint_info)

################################################
##### forge
def unload_forge():
    sd_models.model_data.sd_model = None
    sd_models.model_data.loaded_sd_models = []
    memory_management.unload_all_models()
    memory_management.soft_empty_cache()
    gc.collect()

def reload_model_weights():
    pass

orig_reload_model_weights = None


if forge:
    from modules import sd_models as fsd
    from modules.timer import Timer
    from backend import loader as fld
    import huggingface_guess
    from modules_forge.main_entry import refresh_model_loading_parameters

@torch.inference_mode()
def load_forge_model(state_dict,checkpoint_info = None):
    current_hash = str(fsd.model_data.forge_loading_parameters)
    print('Loading Model: ' + str(fsd.model_data.forge_loading_parameters))

    timer = Timer()

    if fsd.model_data.sd_model:
        fsd.model_data.sd_model = None
        fsd.memory_management.unload_all_models()
        fsd.memory_management.soft_empty_cache()
        gc.collect()

    timer.record("unload existing model")

    checkpoint_info = fsd.model_data.forge_loading_parameters.get('checkpoint_info', checkpoint_info)
 
    if checkpoint_info is None:
        raise ValueError('You do not have any model! Please download at least one model in [models/Stable-diffusion].')

    additional_state_dicts = fsd.model_data.forge_loading_parameters.get('additional_modules', [])
    timer.record("cache state dict")

    fsd.dynamic_args['forge_unet_storage_dtype'] = fsd.model_data.forge_loading_parameters.get('unet_storage_dtype', None)
    fsd.dynamic_args['embedding_dir'] = fsd.cmd_opts.embeddings_dir
    fsd.dynamic_args['emphasis_name'] = opts.emphasis
    sd_model = forge_loader(state_dict, additional_state_dicts)
    timer.record("forge model load")

    sd_model.extra_generation_params = {}
    sd_model.comments = []
    sd_model.sd_checkpoint_info = checkpoint_info
    sd_model.filename = checkpoint_info.filename
    sd_model.sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    fsd.model_data.set_sd_model(sd_model)

    fsd.script_callbacks.model_loaded_callback(sd_model)

    timer.record("scripts callbacks")

    print(f"Model loaded in {timer.summary()}.")

    fsd.model_data.forge_hash = current_hash

    fsd.model_data.forge_loading_parameters = dict(
        checkpoint_info=checkpoint_info,
        additional_modules=shared.opts.forge_additional_modules,
        unet_storage_dtype=fsd.dynamic_args['forge_unet_storage_dtype']
    )

COMP_NAME_AND_PREFIX = {"transformer":PREFIX_M, "text_encoder": "clip_l" , "text_encoder2": "t5xxl", "vae": "vae."}

@torch.inference_mode()
def forge_loader(state_dict, additional_state_dicts):

    state_dicts, estimated_config = split_state_dict(state_dict, additional_state_dicts)
    state_dict = None
    del state_dict
    
    repo_name = estimated_config.huggingface_repo

    local_path = os.path.join(fld.dir_path, 'huggingface', repo_name)
    config: dict = fld.DiffusionPipeline.load_config(local_path)
    huggingface_components = {}
    for component_name, v in config.items():
        if isinstance(v, list) and len(v) == 2:
            lib_name, cls_name = v
            component_sd = state_dicts.get(component_name, None)
            component = fld.load_huggingface_component(estimated_config, component_name, lib_name, cls_name, local_path, component_sd)
            if component_sd is not None:
                del state_dicts[component_name]
            if component is not None:
                huggingface_components[component_name] = component

    for M in fld.possible_models:
        if any(isinstance(estimated_config, x) for x in M.matched_guesses):
            return M(estimated_config=estimated_config, huggingface_components=huggingface_components)

    print('Failed to recognize model type!')
    return None

def split_state_dict(sd, additional_state_dicts: list = None):
    sd = fld.preprocess_state_dict(sd)
    guess = huggingface_guess.guess(sd)

    if isinstance(additional_state_dicts, list):
        for asd in additional_state_dicts:
            asd = load_torch_file(asd)
            sd = fld.replace_state_dict(sd, asd, guess)

    guess.clip_target = guess.clip_target(sd)

    state_dict = {
        guess.unet_target: fld.try_filter_state_dict(sd, guess.unet_key_prefix),
        guess.vae_target: fld.try_filter_state_dict(sd, guess.vae_key_prefix)
    }

    sd = guess.process_clip_state_dict(sd)

    for k, v in guess.clip_target.items():
        state_dict[v] = fld.try_filter_state_dict(sd, [k + '.'])

    state_dict['ignore'] = sd
    print_dict = {k: len(v) for k, v in state_dict.items()}
    print(f'StateDict Keys: {print_dict}')

    del state_dict['ignore']

    return state_dict, guess

def prefixer(t):
    keys = list(t.keys())
    need_revert = False
    for key in keys:
        if key.startswith(PREFIXFIX):
            t["model.diffusion_model." + key] = t.pop(key)
            need_revert = True
    gc.collect()
    return need_revert

def forge_save(filename):
    print("Saveing Model...")
    from modules.paths import models_path
    if ".safetensors" not in filename:
        filename = filename +".safetensors"
    long_filename = os.path.join(models_path, 'Stable-diffusion', filename)
    os.makedirs(os.path.dirname(long_filename), exist_ok=True)
    from modules import shared
    p = shared.sd_model.save_checkpoint(long_filename)
    print(f'Saved checkpoint at: {p}')
    return f'Saved checkpoint at: {p}'

###############################################################
######## QLoRA   
def qdtyper(sd):
    if any("fp4" in k for k in sd):
        return "fp4"
    elif any("nf4" in k for k in sd):
        return "nf4"
    for key in sd:
        if hasattr(sd[key],"dtype"):
            return sd[key].dtype

def to_qdtype(sd_1, sd_2, qd_1, qd_2, device, m1, m2):
    if qd_1 in QTYPES and qd_2 in QTYPES:
        t1 = t2 = torch.float16
    else:
        t1 = t2 = None
    
    if qd_1 in QTYPES: 
        print(f"Changing dtype of {m1} to {qd_2 if t1 is None else t1}")
        q_dequantize(sd_1,qd_1,device,qd_2)

    if qd_2 in QTYPES: 
        print(f"Changing dtype of {m2} to {qd_1 if t2 is None else t2}")
        q_dequantize(sd_2,qd_2,device,qd_1)

    devices.torch_gc()
    
def q_dequantize(sd,qtype,device,dtype):
    dellist = [] 
    from bitsandbytes.functional import dequantize_4bit
    for key in tqdm(sd):
        if ("weight" in key) and ("weight." not in key) and (key + BNB + qtype in sd):
            qs = q_tensor_to_dict(sd[key + BNB + qtype])
            out = torch.empty(qs["shape"],device="cuda:0")
            sd[key] = dequantize_4bit(sd[key].to("cuda:0"),out=out, absmax=sd[key + ".absmax"].to("cuda:0"),blocksize=qs["blocksize"],quant_type=qs["quant_type"]).to(device,dtype)
            dellist.append(key + ".absmax")
            dellist.append(key + BNB + qtype)
            dellist.append(key + ".quant_map")
        elif isinstance(sd[key], torch.Tensor):
            sd[key] = sd[key].to(dtype)

    for key in dellist:
        if key in sd:
            del sd[key]

def q_quantize(weight,qtype):
    from bitsandbytes.functional import quantize_4bit
    return quantize_4bit(weight, quant_type=qtype)

def q_tensor_to_dict(tensor):
    num_list = tensor.tolist()
    char_list = [chr(num) for num in num_list]
    json_string = ''.join(char_list)

    tensor_dict = json.loads(json_string)
    return tensor_dict
