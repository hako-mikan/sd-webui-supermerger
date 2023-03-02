from linecache import clearcache
import os
import gc
import os.path
import re
import torch
import tqdm
import datetime
import csv
from PIL import Image, ImageFont, ImageDraw
from fonts.ttf import Roboto
from tqdm import tqdm
from modules import shared, processing, sd_models, images, sd_samplers,scripts
from modules.ui import  plaintext_to_html
from modules.shared import opts
from modules.processing import create_infotext,Processed
from modules.sd_models import  load_model,checkpoints_loaded
from scripts.mergers.model_util import usemodelgen,filenamecutter,savemodel

from inspect import currentframe

mergedmodel=[]
typesg = ["none","alpha","beta (if Triple or Twice is not selected,Twice automatically enable)","alpha and beta","seed", "mbw alpha","mbw beta","mbw alpha and beta", "model_A","model_B","model_C","pinpoint blocks (alpha or beta must be selected for another axis)","elemental","pinpoint element","effective elemental checker"]
types = ["none","alpha","beta","alpha and beta","seed", "mbw alpha ","mbw beta","mbw alpha and beta", "model_A","model_B","model_C","pinpoint blocks","elemental","pinpoint element","effective"]
modes=["Weight" ,"Add" ,"Triple","Twice"]
sevemodes=["save model", "overwrite"]
#type[0:aplha,1:beta,2:seed,3:mbw,4:model_A,5:model_B,6:model_C]
#msettings=[0 weights_a,1 weights_b,2 model_a,3 model_b,4 model_c,5 base_alpha,6 base_beta,7 mode,8 useblocks,9 custom_name,10 save_sets,11 id_sets,12 wpresets]
#id sets "image", "PNG info","XY grid"

hear = False
hearm = False
non3 = [None]*3

def caster(news,hear):
    if hear: print(news)

def casterr(*args,hear=hear):
    if hear:
        names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
        print('\n'.join([names.get(id(arg), '???') + ' = ' + repr(arg) for arg in args]))
    
  #msettings=[weights_a,weights_b,model_a,model_b,model_c,device,base_alpha,base_beta,mode,loranames,useblocks,custom_name,save_sets,id_sets,wpresets,deep]  
def smergegen(weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,esettings,
                    prompt,nprompt,steps,sampler,cfg,seed,w,h,currentmodel,imggen):

    deepprint  = True if "print change" in esettings else False

    result,currentmodel,modelid,theta_0 = smerge(weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,deepprint=deepprint)

    if "ERROR" in result: return result, *non3

    usemodelgen(theta_0,model_a)

    save = True if sevemodes[0] in save_sets else False

    result = savemodel(theta_0,currentmodel,custom_name,save_sets,model_a) if save else "Merged model loaded:"+currentmodel

    gc.collect()

    if imggen :
        images = simggen(prompt,nprompt,steps,sampler,cfg,seed,w,h,currentmodel,id_sets,modelid)
        return result,currentmodel,*images[:4]
    else:
        return result,currentmodel

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS
blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
     
def smerge(weights_a,weights_b,model_a,model_b,model_c,base_alpha,base_beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,deepprint = False):
    caster("merge start",hearm)
    global hear
    global mergedmodel

    gc.collect()

    # for from file
    if type(useblocks) is str:
        useblocks = True if useblocks =="True" else False
    if type(base_alpha) == str:base_alpha = float(base_alpha)
    if type(base_beta) == str:base_beta  = float(base_beta)

    # preset to weights
    if wpresets != False and useblocks:
        weights_a = wpreseter(weights_a,wpresets)
        weights_b = wpreseter(weights_b,wpresets)

    # mode select booleans
    save = True if sevemodes[0] in save_sets else False
    usebeta = modes[2] in mode or modes[3] in mode
    
    if not useblocks:
        weights_a = weights_b = ""
    #for save log and save current model
    mergedmodel =[weights_a,weights_b,
                            hashfromname(model_a),hashfromname(model_b),hashfromname(model_c),
                            base_alpha,base_beta,mode,useblocks,custom_name,save_sets,id_sets,deep].copy()
    
    model_a = namefromhash(model_a)
    model_b = namefromhash(model_b)
    model_c = namefromhash(model_c)

    caster(mergedmodel,False)

    if len(deep) > 0:
        deep = deep.replace("\n",",")
        deep = deep.split(",")

    #format check
    if model_a =="" or model_b =="" or ((not modes[0] in mode) and model_c=="") : 
        return "ERROR: Necessary model is not selected",*non3
    
    #for MBW text to list
    if useblocks:
        weights_a_t=weights_a.split(',',1)
        weights_b_t=weights_b.split(',',1)
        base_alpha  = float(weights_a_t[0])    
        weights_a = [float(w) for w in weights_a_t[1].split(',')]
        caster(f"from {weights_a_t}, alpha = {base_alpha},weights_a ={weights_a}",hearm)
        if len(weights_a) != 25:return f"ERROR: weights alpha value must be {26}.",*non3
        if usebeta:
            base_beta = float(weights_b_t[0]) 
            weights_b = [float(w) for w in weights_b_t[1].split(',')]
            caster(f"from {weights_b_t}, beta = {base_beta},weights_a ={weights_b}",hearm)
            if len(weights_b) != 25: return f"ERROR: weights beta value must be {26}.",*non3

    caster("model load start",hearm)

    print(f"  model A  \t: {model_a}")
    print(f"  model B  \t: {model_b}")
    print(f"  model C  \t: {model_c}")
    print(f"  alpha,beta\t: {base_alpha,base_beta}")
    print(f"  weights_alpha\t: {weights_a}")
    print(f"  weights_beta\t: {weights_b}")
    print(f"  mode\t\t: {mode}")
    print(f"  MBW \t\t: {useblocks}")

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
            if usebeta and not key in theta_2:
                continue

            weight_index = -1
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
                    print(f"ERROR: illegal block index: {key}")
                    return f"ERROR: illegal block index: {key}",None,None
                
                if weight_index >= 0 and useblocks:
                    current_alpha = weights_a[weight_index]
                    if usebeta: current_beta = weights_b[weight_index]
            else:
                count_target_of_basealpha = count_target_of_basealpha + 1

            if len(deep) > 0:
                skey = key + blockid[weight_index+1]
                for d in deep:
                    if d.count(":") != 2 :continue
                    dbs,dws,dr = d.split(":")[0],d.split(":")[1],d.split(":")[2]
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
                        dr = float(dr)
                        if deepprint :print(dbs,dws,key,dr)
                        current_alpha = dr

            if modes[1] in mode:#Add
                caster(f"model A[{key}] +  {current_alpha} + * (model B - model C)[{key}]",hear)

                if theta_0[key].size != theta_1[key].size:
                    print(theta_0[key].size)
                else:
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

    currentmodel = makemodelname(weights_a,weights_b,model_a, model_b,model_c, base_alpha,base_beta,useblocks,mode)

    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if key in chckpoint_dict_skip_on_merge:
            continue
        if "model" in key and key not in theta_0:
            theta_0.update({key:theta_1[key]})

    modelid = rwmergelog(currentmodel,mergedmodel)

    caster(mergedmodel,False)

    return "",currentmodel,modelid,theta_0

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

    modes=["Weight" ,"Add" ,"Triple","Twice","Diff"]

    if type(alpha) == str:alpha = float(alpha)
    if type(beta)== str:beta  = float(beta)

    if useblocks:
        if modes[1] in mode:#add
            currentmodel =f"{model_a} + ({model_b} - {model_c}) x alpha ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)}"
        elif modes[2] in mode:#triple
            currentmodel =f"{model_a} x (1-alpha-beta) + {model_b} x alpha + {model_c} x beta (alpha = {str(round(alpha,3))},{','.join(str(s) for s in weights_a)},beta = {beta},{','.join(str(s) for s in weights_b)})"
        elif modes[3] in mode:#twice
            currentmodel =f"({model_a} x (1-alpha) + {model_b} x alpha)x(1-beta)+  {model_c} x beta ({str(round(alpha,3))},{','.join(str(s) for s in weights_a)})_({str(round(beta,3))},{','.join(str(s) for s in weights_b)})"
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

path_root = scripts.basedir()

def rwmergelog(mergedname = "",settings= [],id = 0):
    setting = settings.copy()
    filepath = os.path.join(path_root, "mergehistory.csv")
    is_file = os.path.isfile(filepath)
    if not is_file:
        with open(filepath, 'a') as f:
                                       #msettings=[0 weights_a,1 weights_b,2 model_a,3 model_b,4 model_c,5 base_alpha,6 base_beta,7 mode,8 useblocks,9 custom_name,10 save_sets,11 id_sets,12 wpresets]
            f.writelines('"ID","time","name","weights alpha","weights beta","model A","model B","model C","alpha","beta","mode","use MBW","plus lora","custum name","save setting","use ID"\n')
    with  open(filepath, 'r+') as f:
        reader = csv.reader(f)
        mlist = [raw for raw in reader]
        if mergedname != "":
            mergeid = len(mlist)
            setting.insert(0,mergedname)
            for i,x in enumerate(setting):
                if "," in str(x):setting[i] = f'"{str(setting[i])}"'
            text = ",".join(map(str, setting))
            text=str(mergeid)+","+datetime.datetime.now().strftime('%Y.%m.%d %H.%M.%S.%f')[:-7]+"," + text + "\n"
            f.writelines(text)
            return mergeid
        try:
            out = mlist[int(id)]
        except:
            out = "ERROR: OUT of ID index"
        return out

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

    if grid.width != width_one:
        while d.multiline_textsize(text, font=fnt)[0] > width_one*0.75 and fontsize > 0:
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

def simggen(prompt, nprompt, steps, sampler, cfg, seed, w, h,mergeinfo="",id_sets=[],modelid = "no id"):
    shared.state.begin()
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        do_not_save_grid=True,
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
    if "image" in id_sets: processed.images[0] =  draw_origin(processed.images[0], str(modelid),w,h,w)
    image = processed.images[0]
    if "PNG info" in id_sets:mergeinfo = mergeinfo + " ID " + str(modelid)

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
    return processed.images,infotext,plaintext_to_html(processed.info), plaintext_to_html(processed.comments),p
