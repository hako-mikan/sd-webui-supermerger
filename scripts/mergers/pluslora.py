import gc
import hashlib
import json
import math
import os
import sys
import traceback
from io import BytesIO
import gradio as gr
import launch
import modules.shared as shared
import numpy as np
import safetensors.torch
import scripts.mergers.components as components
import torch
from modules import extra_networks, scripts, sd_models, lowvram
from modules.ui import create_refresh_button
from safetensors.torch import load_file, save_file
from scripts.kohyas import extract_lora_from_models as ext
from scripts.kohyas import lora as klora
from scripts.mergers.model_util import (filenamecutter, savemodel)
from scripts.mergers.mergers import extract_super, unload_forge
from tqdm import tqdm

selectable = []
pchanged = False

try:
    from ldm_patched.modules import model_management
    forge = True
except:
    forge = False

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
BLOCKNUMS = [12,17,20,26]
BLOCKIDS=[BLOCKID12,BLOCKID17,BLOCKID20,BLOCKID26]

def to26(ratios):
    if len(ratios) == 26: return ratios
    ids = BLOCKIDS[BLOCKNUMS.index(len(ratios))]
    output = [0]*26
    for i, id in enumerate(ids):
        output[BLOCKID26.index(id)] = ratios[i]
    return output

def f_changediffusers(version):
    launch.run_pip(f"install diffusers=={version}", f"diffusers ver {version}")

def on_ui_tabs():
    import lora
    global selectable
    selectable = [x[0] for x in lora.available_loras.items()]
    sml_path_root = scripts.basedir()
    LWEIGHTSPRESETS="\
    NONE:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
    ALL:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n\
    INS:1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0\n\
    IND:1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
    INALL:1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0\n\
    MIDD:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0\n\
    OUTD:1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0\n\
    OUTS:1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1\n\
    OUTALL:1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1\n\
    ALL0.5:0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5"
    lbwpath = os.path.join(sml_path_root,"scripts", "lbwpresets.txt")
    lbwpathn = os.path.join(sml_path_root,"extensions","sd-webui-lora-block-weight","scripts", "lbwpresets.txt")
    sml_lbwpresets=""

    if os.path.isfile(lbwpath):
        with open(lbwpath,encoding="utf-8") as f:
            sml_lbwpresets = f.read()
    elif os.path.isfile(lbwpathn):
        with open(lbwpathn,encoding="utf-8") as f:
            sml_lbwpresets = f.read()
    else:
        sml_lbwpresets=LWEIGHTSPRESETS

    try:
        import diffusers
        d_ver = diffusers.__version__
    except:
        d_ver = None

    with gr.Blocks(analytics_enabled=False) :
        sml_submit_result = gr.Textbox(label="Message")
        with gr.Row(equal_height=False):
            with gr.Column(equal_height=False):
                sml_cpmerge = gr.Button(elem_id="model_merger_merge", value="Merge to Checkpoint",variant='primary')
                sml_merge = gr.Button(elem_id="model_merger_merge", value="Merge LoRAs",variant='primary')
                with gr.Row(equal_height=False):
                    sml_settings = gr.CheckboxGroup(["same to Strength", "overwrite"], label="settings")
                    sml_filename = gr.Textbox(label="filename(option)",lines=1,visible =True,interactive  = True)  
                sml_metasettings = gr.Radio(value = "create new",choices = ["create new","create new without output_name", "merge","save all", "use first lora"], label="metadata")
                with gr.Row(equal_height=False):
                    save_precision = gr.Radio(label = "save precision",choices=["float","fp16","bf16"],value = "fp16",type="value")
                    calc_precision = gr.Radio(label = "calc precision(fp16:cuda only)" ,choices=["float","fp16","bf16"],value = "float",type="value")
                    device = gr.Radio(label = "device",choices=["cuda","cpu"],value = "cuda",type="value")
            with gr.Column(equal_height=False):
                sml_makelora = gr.Button(elem_id="model_merger_merge", value="Make LoRA (alpha * Tuned - beta * Original)",variant='primary')
                sml_extract = gr.Button(elem_id="model_merger_merge", value="Extract from two LoRAs",variant='primary')
                with gr.Row(equal_height=False):
                    sml_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint Tuned",interactive=True)
                    create_refresh_button(sml_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                with gr.Row(equal_height=False):
                    sml_model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint Original",interactive=True)
                    create_refresh_button(sml_model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                with gr.Row(equal_height=False):
                    alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=1)
                    beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=1)
                    smooth = gr.Slider(label="gamma(smooth)", minimum=-1, maximum=20, step=0.1, value=1)
        
        sml_dim = gr.Radio(label = "remake dimension",choices = ["no","auto",4,8,16,32,64,128,256,512,768,1024],value = "no",type = "value") 
        sml_loranames = gr.Textbox(label='LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,...(":blocks" is option, not necessary)',lines=1,value="",visible =True)
        sml_dims = gr.CheckboxGroup(label = "limit dimension",choices=[],value = [],type="value",interactive=True,visible = False)
        with gr.Row(equal_height=False):
            sml_calcdim = gr.Button(elem_id="calcloras", value="Calculate LoRA dimensions (this may take time for multiple LoRAs)",variant='primary')
            sml_update = gr.Button(elem_id="calcloras", value="update list",variant='primary')
            sml_lratio = gr.Slider(label="default LoRA multiplier", minimum=-1.0, maximum=2, step=0.1, value=1)

        with gr.Row():
            sml_selectall = gr.Button(elem_id="sml_selectall", value="select all",variant='primary')
            sml_deselectall = gr.Button(elem_id="slm_deselectall", value="deselect all",variant='primary')
            components.frompromptb = gr.Button(elem_id="slm_deselectall", value="get from prompt",variant='primary')
            hidenb = gr.Checkbox(value = False,visible = False)
        sml_loras = gr.CheckboxGroup(label = "LoRAs on disk",choices = selectable,type="value",interactive=True,visible = True)
        sml_loraratios = gr.TextArea(label="",value=sml_lbwpresets,visible =True,interactive  = True)  

        sml_selectall.click(fn = lambda x:gr.update(value = selectable),outputs = [sml_loras])
        sml_deselectall.click(fn = lambda x:gr.update(value =[]),outputs = [sml_loras])

        with gr.Row():
            changediffusers = gr.Button(elem_id=f"change_diffusers_version", value=f"change diffusers version(now:{d_ver})",variant='primary')
            dversion = gr.Textbox(label="diffusers version",lines=1,visible =True,interactive  = True)  
        components.sml_loranames = [sml_loras, sml_loranames, hidenb]

        changediffusers.click(
            fn=f_changediffusers,
            inputs=[dversion],
            outputs=[sml_submit_result]
        )

        sml_merge.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_dim,save_precision,calc_precision,sml_metasettings,alpha,beta,smooth,gr.Checkbox(value = True,visible = False),device],
            outputs=[sml_submit_result]
        )

        sml_extract.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_dim,save_precision,calc_precision,sml_metasettings,alpha,beta,smooth,gr.Checkbox(value = False,visible = False),device],
            outputs=[sml_submit_result]
        )

        sml_makelora.click(
            fn=makelora,
            inputs=[sml_model_a,sml_model_b,sml_dim,sml_filename,sml_settings,alpha,beta,save_precision,calc_precision,sml_metasettings,device],
            outputs=[sml_submit_result]
        )

        sml_cpmerge.click(
            fn=pluslora,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_model_a,save_precision,calc_precision,sml_metasettings,device],
            outputs=[sml_submit_result]
        )



        llist ={}
        dlist =[]
        dn = []

        def updateloras():
            lora.list_available_loras()
            names = []
            dels = []
            for n in  lora.available_loras.items():
                if n[0] not in llist:llist[n[0]] = ""
                names.append(n[0])
            for l in list(llist.keys()):
                if l not in names:llist.pop(l)

            global selectable
            selectable = [f"{x[0]}({x[1]})" for x in llist.items()]
            return gr.update(choices = [f"{x[0]}({x[1]})" for x in llist.items()])

        sml_update.click(fn = updateloras,outputs = [sml_loras])

        def calculatedim():
            print("listing dimensions...")
            for n in  tqdm(lora.available_loras.items()):
                if n[0] in llist:
                    if llist[n[0]] !="": continue
                c_lora = lora.available_loras.get(n[0], None) 
                d,t,s = dimgetter(c_lora.filename)
                if t == "LoCon":
                    if len(list(set(d.values()))) > 1:
                        d = "multi dim"
                    else:
                        d = f"{list(set(d.values()))}"
                    d = f"{d}:{t}"
                if s =="XL":
                    if len(list(set(d.values()))) > 1:
                        d = "multi dim"
                    else:
                        d = f"{list(set(d.values()))}"
                    d = f"{d}:XL"
                if d not in dlist:
                    if type(d) == int :dlist.append(d)
                    elif d not in dn: dn.append(d)
                llist[n[0]] = d
            dlist.sort()
            global selectable
            selectable = [f"{x[0]}({x[1]})" for x in llist.items()]
            return gr.update(choices = [f"{x[0]}({x[1]})" for x in llist.items()],value =[]),gr.update(visible =True,choices = [x for x in (dlist+dn)])

        sml_calcdim.click(
            fn=calculatedim,
            inputs=[],
            outputs=[sml_loras,sml_dims]
        )

        def dimselector(dims):
            if dims ==[]:return gr.update(choices = [f"{x[0]}({x[1]})" for x in llist.items()])
            rl=[]
            for d in dims:
                for i in llist.items():
                    if d == i[1]:rl.append(f"{i[0]}({i[1]})")

            global selectable
            selectable = rl.copy()

            return gr.update(choices = [l for l in rl],value =[])

        def llister(names,ratio, hiden):
          if hiden:return gr.update()
          if names ==[] : return ""
          else:
            for i,n in enumerate(names):
              if "(" in n:names[i] = n[:n.rfind("(")]
            return f":{ratio},".join(names)+f":{ratio} "

        hidenb.change(fn=lambda x: False, outputs = [hidenb])
        sml_loras.change(fn=llister,inputs=[sml_loras,sml_lratio, hidenb],outputs=[sml_loranames])     
        sml_dims.change(fn=dimselector,inputs=[sml_dims],outputs=[sml_loras])  

##############################################################
####### make LoRA from checkpoint

def makelora(model_a,model_b,dim,saveto,settings,alpha,beta,save_precision,calc_precision,metasets,device):
    print("make LoRA start")
    if model_a == "" or model_b =="":
      return "ERROR: No model Selected"
    gc.collect()

    currentinfo = shared.sd_model.sd_checkpoint_info

    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
    sd_models.load_model(checkpoint_info)

    model = shared.sd_model

    is_sdxl = hasattr(model, 'conditioner')
    is_sd2 = not model.is_sdxl and hasattr(model.cond_stage_model, 'model')
    is_sd1 = not model.is_sdxl and not model.is_sd2

    print(f"Detected model type: SDXL: {is_sdxl}, SD2.X: {is_sd2}, SD1.X: {is_sd1}")

    if forge:
        unload_forge()
    else:
        sd_models.unload_model_weights()

    if saveto =="" : saveto = makeloraname(model_a,model_b)
    if not ".safetensors" in saveto :saveto  += ".safetensors"
    saveto = os.path.join(shared.cmd_opts.lora_dir,saveto)

    dim = 128 if type(dim) != int else int(dim)
    if os.path.isfile(saveto ) and not "overwrite" in settings:
        _err_msg = f"Output file ({saveto}) existed and was not saved"
        print(_err_msg)
        return _err_msg

    args = Kohya_extract_args(
        v2=is_sd2,
        v_parameterization=True,
        sdxl=is_sdxl,
        save_precision=save_precision,
        model_org=fullpathfromname(model_b),
        model_tuned=fullpathfromname(model_a),
        save_to=saveto,
        dim=dim,
        conv_dim=None,
        device=device,
        no_metadata=False,
        alpha = alpha,
        beta = beta
    )

    result = ext.svd(args)

    sd_models.load_model(currentinfo)
    return result

##############################################################
####### merge LoRAs

def lmerge(loranames,loraratioss,settings,filename,dim,save_precision,calc_precision,metasets,alpha,beta,smooth,merge,device):
    try:
        import lora
        loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]
        if any([x is None for x in loras_on_disk]):
            lora.list_available_loras()

            loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]

        lnames = loranames.split(",")

        #LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,.
        #LoRAname1:ratio1:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,LoRAname2:ratio2:Blocks2,.

        temp = []
        for n in lnames:
            if ":" in n:
                temp.append(n.split(":"))
            else:
                temp[-1].append(n)

        lnames = temp

        loraratios=loraratioss.splitlines()
        ldict ={}

        for i,l in enumerate(loraratios):
            if ":" not in l or not any(l.count(",") == x - 1 for x in BLOCKNUMS) : continue
            ldict[l.split(":")[0]]=l.split(":")[1]

        ln, lr, ld, lt, lm, ls = [], [], [], [], [], [] #lm: 各LoRAのマージ用メタデータ #ls: SD-?
        dmax = 1

        for i,n in enumerate(lnames):
            if len(n) ==2:
                ratio = [float(n[1])]*26
            elif len(n) ==3:
                if n[2].strip() in ldict:
                    ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
                    ratio = to26(ratio)
                else:ratio = [float(n[1])]*26
            elif len(n[2:]) in BLOCKNUMS:
                ratio = [float(x) for x in n[2:]]
                ratio = to26(ratio)
            else:
                print("ERROR:Number of Blocks must be 12,17,20,26")
                ratio = [float(n[1])]*26
            c_lora = lora.available_loras.get(n[0], None) 
            ln.append(c_lora.filename)
            lr.append(ratio)
            d, t, s = dimgetter(c_lora.filename)
            if t == "LoCon":
                d = list(set(d.values()))
                d = d[0]
            lt.append(t)
            ld.append(d)
            ls.append(s)
            if d != "LyCORIS" and type(d) == int:
                if d > dmax : dmax = d
            
            # LoRA毎のメタデータを保存
            meta = prepare_merge_metadata( n[1], ",".join( [str(n) for n in ratio] ), c_lora )
            lm.append( meta )

        if filename =="":filename =loranames.replace(",","+").replace(":","_")
        if not ".safetensors" in filename:filename += ".safetensors"
        loraname = filename.replace(".safetensors", "")
        filename = os.path.join(shared.cmd_opts.lora_dir,filename)

        auto = True if dim == "auto" else False
    
        dim = int(dim) if dim != "no" and dim != "auto" else 0

        if merge:
            if "LyCORIS" in ld:
                if len(ld) !=1:
                    return "multiple merge of LyCORIS is not supported"
                sd = lycomerge(ln[0], lr[0], calc_precision)
            elif dim > 0:
                print("change demension to ", dim)
                sd = merge_lora_models_dim(ln, lr, dim,settings,device,calc_precision)
            elif auto and ld.count(ld[0]) != len(ld):
                print("change demension to ",dmax)
                sd = merge_lora_models_dim(ln, lr, dmax,settings,device,calc_precision)
            else:
                sd = merge_lora_models(ln, lr, settings, False, calc_precision)

            if os.path.isfile(filename) and not "overwrite" in settings:
                _err_msg = f"Output file ({filename}) existed and was not saved"
                print(_err_msg)
                return _err_msg
        else:
            a = merge_lora_models(ln[0:1], lr[0:1], settings, False, calc_precision)
            b = merge_lora_models(ln[1:2], lr[1:2], settings, False, calc_precision)
            sd = extract_two(a,b,alpha,beta,smooth)
        
        # マージ後のメタデータを取得
        metadata = create_merge_metadata( sd, lm, loraname, save_precision,metasets )

        save_to_file(filename,sd,sd, str_to_dtype(save_precision), metadata)
        sd = None
        del sd
        gc.collect()
        torch.cuda.empty_cache()

        return "saved : "+filename
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exc()
        return exc_value

def merge_lora_models(models, ratios, sets, locon, calc_precision):
    base_alphas = {}                          # alpha for merged model
    base_dims = {}
    merge_dtype = str_to_dtype(calc_precision)
    merged_sd = {}
    fugou = 1
    for model, ratios in zip(models, ratios):
        keylist = LBLCOKS26

        print(f"merging {model}: {ratios}")
        lora_sd, metadata, isv2 = load_state_dict(model, merge_dtype)

        # get alpha and dim
        alphas = {}                             # alpha for current model
        dims = {}                               # dims for current model

        base_dims, base_alphas, dims, alphas = dimalpha(lora_sd, base_dims, base_alphas)

        print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        print(f"merging...")
        for key in lora_sd.keys():
            if 'alpha' in key:
                continue

            lora_module_name = key[:key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            ratio = ratios[blockfromkey(key, keylist, isv2)]
            if "same to Strength" in sets:
                ratio, fugou = (ratio ** 0.5, 1) if ratio > 0 else (abs(ratio) ** 0.5, -1)

            if "lora_down" in key:
                ratio = ratio * fugou

            scale = math.sqrt(alpha / base_alpha) * ratio

            if key in merged_sd:
                assert merged_sd[key].size() == lora_sd[key].size(), (
                    f"weights shape mismatch merging v1 and v2, different dims? "
                    f"/ 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                    f" {merged_sd[key].size()} ,{lora_sd[key].size()}, {lora_module_name}"
                )
                merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale
        del lora_sd

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    return merged_sd

def merge_lora_models_dim(models, ratios, new_rank, sets, device, calc_precision):
    merged_sd = {}
    fugou = 1
    isv2 = False
    merge_dtype = str_to_dtype(calc_precision)
    for model, ratios in zip(models, ratios):

        lora_sd, medadata, isv2 = load_state_dict(model, merge_dtype, device)

        # merge
        print(f"merging {model}: {ratios}")
        for key in tqdm(list(lora_sd.keys())):
            if 'lora_down' not in key:
                continue
            lora_module_name = key[:key.rfind(".lora_down")]

            down_weight = lora_sd[key]
            network_dim = down_weight.size()[0]

            up_weight = lora_sd[lora_module_name + '.lora_up.weight']
            alpha = lora_sd.get(lora_module_name + '.alpha', network_dim)

            in_dim = down_weight.size()[1]
            out_dim = up_weight.size()[0]
            conv2d = len(down_weight.size()) == 4
            # print(lora_module_name, network_dim, alpha, in_dim, out_dim)

            # make original weight if not exist
            if lora_module_name not in merged_sd:
                weight = torch.zeros((out_dim, in_dim, 1, 1) if conv2d else (out_dim, in_dim), dtype=merge_dtype, device=device)
            else:
                weight = merged_sd[lora_module_name]

            ratio = ratios[blockfromkey(key, LBLCOKS26,isv2)]
            if "same to Strength" in sets:
                ratio, fugou = (ratio ** 0.5, 1) if ratio > 0 else (abs(ratio) ** 0.5, -1)
            # print(lora_module_name, ratio)
            # W <- W + U * D
            scale = (alpha / network_dim)
            if not conv2d:  # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale * fugou
            else:
                weight = weight + ratio * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale * fugou

            merged_sd[lora_module_name] = weight
            
        lora_sd = None
        del lora_sd
        torch.cuda.empty_cache()
    
    for key in merged_sd.keys():
        merged_sd[key] = merged_sd[key].to(torch.float)

    # extract from merged weights
    print("extract new lora...")
    merged_lora_sd = {}
    with torch.no_grad():
        for lora_module_name, mat in tqdm(list(merged_sd.items())):
            conv2d = (len(mat.size()) == 4)
            if conv2d:
                mat = mat.squeeze()

            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :new_rank]
            S = S[:new_rank]
            U = U @ torch.diag(S)

            Vh = Vh[:new_rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            up_weight = U
            down_weight = Vh

            if conv2d:
                up_weight = up_weight.unsqueeze(2).unsqueeze(3)
                down_weight = down_weight.unsqueeze(2).unsqueeze(3)

            merged_lora_sd[lora_module_name + '.lora_up.weight'] = up_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + '.lora_down.weight'] = down_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + '.alpha'] = torch.tensor(new_rank)

    del merged_sd
    gc.collect()
    torch.cuda.empty_cache()

    return merged_lora_sd

def extract_two(a,b,pa,pb,ps):
    base_alphas = {}                          # alpha for merged model
    base_dims = {}
    merged_sd = {}
    alphas = {}                             # alpha for current model
    dims = {}                               # dims for current model

    base_dims_a, base_alphas_a, dims, alphas_a = dimalpha(a, base_dims, base_alphas)
    base_dims_b, base_alphas_b, dims, alphas_b = dimalpha(b, base_dims, base_alphas)

    print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

    # merge
    print(f"merging...")
    for key in a.keys():
        if 'alpha' in key:
            continue

        lora_module_name = key[:key.rfind(".lora_")]

        base_alpha_a = base_alphas_a[lora_module_name]
        base_alpha_b = base_alphas_b[lora_module_name]
        alpha_a = alphas_a[lora_module_name]
        alpha_b = alphas_b[lora_module_name]

        scale_a = math.sqrt(alpha_a / base_alpha_a) 
        scale_b = math.sqrt(alpha_b / base_alpha_b)

        merged_sd[key] = extract_super(None,a[key] * scale_a,b[key] * scale_b,pa,pb,ps)

        merged_sd[key] = merged_sd[key] + a[key] * scale_a
        merged_sd[key] = a[key] * scale_a

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    return merged_sd

def lycomerge(filename,ratios,calc_precision):
    merge_dtype = str_to_dtype(calc_precision)
    sd, metadata, isv2 = load_state_dict(filename, merge_dtype)

    if len(ratios) == 17:
      r0 = 1
      ratios = [ratios[0]] + [r0] + ratios[1:3]+ [r0] + ratios[3:5]+[r0] + ratios[5:7]+[r0,r0,r0] + [ratios[7]] + [r0,r0,r0] + ratios[8:]

    print("LyCORIS: " , ratios)

    keys_failed_to_match = []

    for lkey, weight in sd.items():
        ratio = 1
        picked = False
        if 'alpha' in lkey:
          continue
        
        try:
            import networks as lora
        except:
            import lora as lora

        fullkey = lora.convert_diffusers_name_to_compvis(lkey,isv2)

        if "." not in fullkey:continue

        key, lora_key = fullkey.split(".", 1)

        for i,block in enumerate(LBLCOKS26):
            if block in key:
                ratio = ratios[i]
                picked = True
        if not picked: keys_failed_to_match.append(key)

        sd[lkey] = weight * math.sqrt(abs(float(ratio)))

        if "down" in lkey and ratio < 0:
          sd[key] = sd[key] * -1
        
    if len(keys_failed_to_match) > 0:
      print(keys_failed_to_match)
  
    return sd 

##############################################################
####### merge to checkpoint
def pluslora(lnames,loraratios,settings,output,model,save_precision,calc_precision,metasets,device):
    if model == []: return "ERROR: No model Selected"
    if lnames == "":return "ERROR: No LoRA Selected"

    add = ""

    print("Plus LoRA start")
    import lora
    lnames = lnames.split(",")

    for i, n in enumerate(lnames):
        lnames[i] = n.split(":")

    loraratios=loraratios.splitlines()
    ldict ={}

    for i,l in enumerate(loraratios):
        if ":" not in l or not any(l.count(",") == x - 1 for x in BLOCKNUMS) : continue
        ldict[l.split(":")[0].strip()]=l.split(":")[1]

    names, filenames, loratypes, lweis = [], [], [], []

    for n in lnames:
        if len(n) ==3:
            if n[2].strip() in ldict:
                ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
                ratio = to26(ratio)
            else:ratio = [float(n[1])]*26
        else:ratio = [float(n[1])]*26
        c_lora = lora.available_loras.get(n[0], None) 
        names.append(n[0])
        filenames.append(c_lora.filename)
        lweis.append(ratio)

    modeln=filenamecutter(model,True)   
    dname = modeln
    for n in names:
      dname = dname + "+"+n

    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    print(f"Loading {model}")
    theta_0 = sd_models.read_state_dict(checkpoint_info.filename,map_location=device)

    isxl = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in theta_0.keys()
    isv2 = "cond_stage_model.model.transformer.resblocks.0.attn.out_proj.weight" in theta_0.keys()

    try:
        import networks
        is15 = True
    except:
        is15 = False

    keychanger = {}
    for key in theta_0.keys():
        if "model" in key:
            skey = key.replace(".","_").replace("_weight","")
            if "conditioner_embedders_" in skey:
                keychanger[skey.split("conditioner_embedders_",1)[1]] = key
            else:
                if "wrapped" in skey:
                    keychanger[skey.split("wrapped_",1)[1]] = key
                else:
                    keychanger[skey.split("model_",1)[1]] = key

    if is15:
        if shared.sd_model is not None:
            orig_checkpoint = shared.sd_model.sd_checkpoint_info
        else:
            orig_checkpoint = None
        checkpoint_info = sd_models.get_closet_checkpoint_match(model)
        if orig_checkpoint != checkpoint_info:
            sd_models.reload_model_weights(info=checkpoint_info)
        theta_0 = newpluslora(theta_0,filenames,lweis,names, isxl,isv2, keychanger)
        
        if orig_checkpoint:
            sd_models.reload_model_weights(info=orig_checkpoint)
    else:
        for name,filename, lwei in zip(names,filenames, lweis):
            print(f"loading: {name}")
            lora_sd, metadata, isv2 = load_state_dict(filename, torch.float)

            print(f"merging..." ,lwei)
            for key in lora_sd.keys():
                ratio = 1

                import lora
                fullkey = lora.convert_diffusers_name_to_compvis(key,isv2)

                msd_key, _ = fullkey.split(".", 1)
                if isxl:
                    if "lora_unet" in msd_key:
                        msd_key = msd_key.replace("lora_unet", "diffusion_model")
                    elif "lora_te1_text_model" in msd_key:
                        msd_key = msd_key.replace("lora_te1_text_model", "0_transformer_text_model")

                for i,block in enumerate(LBLCOKS26):
                    if block in fullkey or block in msd_key:
                        ratio = lwei[i]

                if "lora_down" in key:
                    up_key = key.replace("lora_down", "lora_up")
                    alpha_key = key[:key.index("lora_down")] + 'alpha'

                    # print(f"apply {key} to {module}")

                    down_weight = lora_sd[key].to(device="cpu")
                    up_weight = lora_sd[up_key].to(device="cpu")

                    dim = down_weight.size()[0]
                    alpha = lora_sd.get(alpha_key, dim)
                    scale = alpha / dim
                    # W <- W + U * D
                    weight = theta_0[keychanger[msd_key]].to(device="cpu")

                    if len(weight.size()) == 2:
                        # linear
                        weight = weight + ratio * (up_weight @ down_weight) * scale

                    elif down_weight.size()[2:4] == (1, 1):
                        # conv2d 1x1
                        weight = (
                            weight
                            + ratio
                            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                            * scale
                        )
                    else:
                        # conv2d 3x3
                        conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                        # print(conved.size(), weight.size(), module.stride, module.padding)
                        weight = weight + ratio * conved * scale
                        
                    theta_0[keychanger[msd_key]] = torch.nn.Parameter(weight)
    #usemodelgen(theta_0,model)
    settings.append(save_precision)
    settings.append("safetensors")
    result = savemodel(theta_0,dname,output,settings)
    del theta_0
    gc.collect()
    return result + add

def newpluslora(theta_0,filenames,lweis,names, isxl,isv2, keychanger):
    import networks as nets
    nets.load_networks(names)

    for l, loaded in enumerate(nets.loaded_networks):
        for n, name in enumerate(names):
            changed = False
            if name == loaded.name:
                lbw(nets.loaded_networks[l],to26(lweis[n]),isv2)
                changed = True
            if not changed: "ERROR: {name}weight is not changed"
    
    for net in nets.loaded_networks:
        net.dyn_dim = None
        for name,module in  tqdm(net.modules.items(), desc=f"{net.name}"):
            fullkey = nets.convert_diffusers_name_to_compvis(name,isv2)
            msd_key = fullkey.split(".")[0]
            if isxl:
                if "lora_unet" in msd_key:
                    msd_key = msd_key.replace("lora_unet", "diffusion_model")
                elif "lora_te1_text_model" in msd_key:
                    msd_key = msd_key.replace("lora_te1_text_model", "0_transformer_text_model")

            qvk = ["_q_proj","_k_proj","_v_proj","_out_proj"]

            if msd_key in keychanger.keys():
                wkey = keychanger[msd_key]
                bkey = wkey.replace("weight","bias")
                if bkey in theta_0.keys():
                    theta_0[wkey], theta_0[bkey]= plusweights(theta_0[wkey], module, bias = theta_0[bkey])
                else:
                    theta_0[wkey], _ = plusweights(theta_0[wkey] ,module)

            else:
                if any(x in name for x in qvk):
                    for x in qvk:
                        if x in name:
                            inkey,outkey = name.replace(x,"") + "_in_proj" ,name.replace(x,"") + "_out_proj"
                    bkey = keychanger[outkey].replace("weight","bias")
                    if bkey in theta_0.keys():
                        theta_0[keychanger[inkey]] ,theta_0[keychanger[outkey]], theta_0[bkey]= plusweightsqvk(theta_0[keychanger[inkey]],theta_0[keychanger[outkey]], name ,module, net, bias = theta_0[bkey])
                    else:
                        theta_0[keychanger[inkey]] ,theta_0[keychanger[outkey]], _= plusweightsqvk(theta_0[keychanger[inkey]],theta_0[keychanger[outkey]], name ,module, net)
                else:
                    print(msd_key)
        gc.collect()
    return theta_0

def plusweights(weight, module, bias = None):
    with torch.no_grad():
        updown = module.calc_updown(weight.to(dtype=torch.float))
        if len(weight.shape) == 4 and weight.shape[1] == 9:
            # inpainting model. zero pad updown to make channel[1]  4 to 9
            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))
        if type(updown) == tuple:
            updown, ex_bias = updown
            if ex_bias is not None and bias is not None:
                bias += ex_bias

        weight += updown
    return weight, bias

def plusweightsqvk(inweight, outweight, network_layer_name, module ,net,bias = None):
    with torch.no_grad():
        module_q = net.modules.get(network_layer_name + "_q_proj", None)
        module_k = net.modules.get(network_layer_name + "_k_proj", None)
        module_v = net.modules.get(network_layer_name + "_v_proj", None)
        module_out = net.modules.get(network_layer_name + "_out_proj", None)

        if module_q and module_k and module_v and module_out:
            with torch.no_grad():
                updown_q = module_q.calc_updown(inweight)
                updown_k = module_k.calc_updown(inweight)
                updown_v = module_v.calc_updown(inweight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                updown_out = module_out.calc_updown(outweight)
                if type(updown_out) is tuple:
                    updown_out,ex_bias = updown_out

                inweight += updown_qkv
                outweight += updown_out
                if bias is not None and ex_bias is not None:
                    bias += ex_bias

    return inweight,outweight,bias

def lbw(lora,lwei,isv2):
    errormodules = []

    blocks = LBLCOKS26
    if isv2:
        blocks[0] = V2ENCODER

    for key in lora.modules.keys():
        ratio = 1
        picked = False

        for i,block in enumerate(blocks):
            if block in key:
                if i == 26: i=0
                ratio = lwei[i]
                picked = True

        if not picked:
            errormodules.append(key)

        ltype = type(lora.modules[key]).__name__
        set = False
        if ltype in LORAANDSOON.keys():
            setattr(lora.modules[key],LORAANDSOON[ltype],torch.nn.Parameter(getattr(lora.modules[key],LORAANDSOON[ltype]) * ratio))
            #print(ltype)
            set = True
        else:
            if hasattr(lora.modules[key],"up_model"):
                lora.modules[key].up_model.weight= torch.nn.Parameter(lora.modules[key].up_model.weight *ratio)
                #print("LoRA using LoCON")
                set = True
            else:
                lora.modules[key].up.weight= torch.nn.Parameter(lora.modules[key].up.weight *ratio)
                #print("LoRA")
                set = True
        if not set : 
            print("unkwon LoRA")

    if errormodules:
        print("unchanged modules:", errormodules)
    else:
        print(f"{lora.name}: Successfully set the ratio {lwei} ")

    return lora

LORAANDSOON = {
    "LoraHadaModule" : "w1a",
    "LycoHadaModule" : "w1a",
    "NetworkModuleHada": "w1a",
    "FullModule" : "weight",
    "NetworkModuleFull": "weight",
    "IA3Module" : "w",
    "NetworkModuleIa3" : "w",
    "LoraKronModule" : "w1",
    "LycoKronModule" : "w1",
    "NetworkModuleLokr": "w1",
}

def save_to_file(file_name, model, state_dict, dtype, metadata):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(model, file_name, metadata=metadata)
    else:
        torch.save(model, file_name)

CLAMP_QUANTILE = 0.99


def str_to_dtype(p):
  if p == 'float':
    return torch.float
  if p == 'fp16':
    return torch.float16
  if p == 'bf16':
    return torch.bfloat16
  return None


def get_safetensors_header(filename):
    import json
    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        if metadata_len > 2 and json_start in (b'{"', b"{'"):
            json_data = json_start + file.read(metadata_len-2)
            return json.loads(json_data)

        # invalid safetensors
        return {}

def load_state_header(file_name, dtype):
  """load safetensors header if available"""
  if os.path.splitext(file_name)[1] == '.safetensors':
    sd = get_safetensors_header(file_name)
  else:
    sd = torch.load(file_name, map_location='cpu')
  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)
  return sd

def load_state_dict(file_name, dtype, device = "cpu"):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name,device=device)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location=device)
        metadata = {}

    isv2 = False

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype = dtype, device = device)
            if "resblocks" in key:
                isv2 = True
    
    if isv2: print("SD2.X")

    return sd, metadata, isv2

def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata

def dimgetter(filename):
    lora_sd = load_state_header(filename, torch.float)
    alpha = None
    dim = None
    ltype = None

    if "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight" in lora_sd.keys():
      ltype = "LoCon"
      if type(lora_sd["lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight"]) is dict:
          lora_sd, _, _ = load_state_dict(filename, torch.float)
      _, _, dim, _ = dimalpha(lora_sd)

    if "lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight" in lora_sd.keys():
        sdx = "XL"
        if type(lora_sd["lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight"]) is dict:
            lora_sd, _, _ = load_state_dict(filename, torch.float)
        _, _, dim, _ = dimalpha(lora_sd)
    else:
        sdx = ""

    for key, value in lora_sd.items():
  
        if alpha is None and 'alpha' in key:
            alpha = value
        if dim is None and 'lora_down' in key:
            if type(value) == torch.Tensor and len(value.size()) == 2:
                dim = value.size()[0]
            elif type(value) == dict:
                dim = value.get("shape",[0,0])[0]
        if "hada_" in key:
            dim,ltype, sdx = "LyCORIS","LyCORIS", "LyCORIS"
        if alpha is not None and dim is not None:
            break
    if alpha is None:
        alpha = dim
    if ltype == None:ltype = "LoRA"
    if dim :
      return dim, ltype, sdx
    else:
      return "unknown","unknown","unknown"

def blockfromkey(key,keylist,isv2 = False):
    try:
        import networks as lora
    except:
        import lora as lora
    fullkey = lora.convert_diffusers_name_to_compvis(key,isv2)

    if "lora_unet" in fullkey:
        fullkey = fullkey.replace("lora_unet", "diffusion_model")
    elif "lora_te1_text_model" in fullkey:
        fullkey = fullkey.replace("lora_te1_text_model", "0_transformer_text_model")
    
    for i,n in enumerate(keylist):
        if n in  fullkey: return i
    if "1_model_transformer_resblocks_" in fullkey:return 0
    print(f"ERROR:Block is not deteced:{fullkey}")
    return 0

def dimalpha(lora_sd, base_dims={}, base_alphas={}):
    alphas = {}                             # alpha for current model
    dims = {}                               # dims for current model
    for key in lora_sd.keys():
        if 'alpha' in key:
            lora_module_name = key[:key.rfind(".alpha")]
            alpha = float(lora_sd[key].detach().numpy())
            alphas[lora_module_name] = alpha
            if lora_module_name not in base_alphas:
                base_alphas[lora_module_name] = alpha
        elif "lora_down" in key:
            lora_module_name = key[:key.rfind(".lora_down")]
            dim = lora_sd[key].size()[0]
            dims[lora_module_name] = dim
            if lora_module_name not in base_dims:
                base_dims[lora_module_name] = dim

    for lora_module_name in dims.keys():
        if lora_module_name not in alphas:
            alpha = dims[lora_module_name]
            alphas[lora_module_name] = alpha
            if lora_module_name not in base_alphas:
                base_alphas[lora_module_name] = alpha
    return base_dims, base_alphas, dims, alphas


def fullpathfromname(name):
    if hash == "" or hash ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    return checkpoint_info.filename

def makeloraname(model_a,model_b):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    return "lora_"+model_a+"-"+model_b

V2ENCODER = "resblocks"

LBLCOKS26=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders"]

###########################################################
##### metadata

def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash

def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]

def prepare_merge_metadata( ratio, blocks, fromLora ):
    """
    メタデータに ratio, blocks などの情報を付加しておく

    Parameters
    ----
    ratio : string
        name:ratio:blocks の ratio 部分
    blocks : string
        name:ratio:bloks の blocks 部分(ラベルではなくて実パラメータ)
    fromLora : NetworkOnDisk
        マージ対象のLoRA
    
    Returns
    ----
    dict[str, str]
        メタデータ
    """
    meta = fromLora.metadata
    meta["sshs_ratio"] = str.strip( ratio )
    meta["sshs_blocks"] = str.strip( blocks )
    meta["ss_output_name"] = str.strip( fromLora.name )

    return meta

BASE_METADATA = [
    "sshs_ratio", "sshs_blocks", "ss_output_name",
    "sshs_model_hash", "sshs_legacy_hash",
    "ss_network_module",
    "ss_network_alpha", "ss_network_dim",
    "ss_mixed_precision", "ss_v2",
    "ss_training_comment",
    "ss_sd_model_name", "ss_new_sd_model_hash",
    "ss_clip_skip",
    "ss_base_model_version"
]

MINIMUM_METADATA = [
    "ss_network_module","ss_network_alpha", "ss_network_dim","ss_v2","ss_sd_model_name", "ss_base_model_version"
]

def create_merge_metadata( sd, lmetas, lname, lprecision, metasets ):
    """
    LoRAマージ後のメタデータを作成する

    Parameters
    ----
    sd : NetworkOnDisk
        マージ後のLoRA
    lmetas : dict[str, str]
        マージされるLoRAのメタデータ
    lname : str
        マージ後のLoRA名
    lprecision : str
        save precision の値
    mergeAll : bool
        メタデータの残し方。ただしタグ情報はディレクトリ名が後勝ちでマージします
        True 全メタデータを残す。単マージの場合はTrue固定
        False 一部のメタデータのみ残す
    
    Returns
    ----
    dict[str, str]
        メタデータ
    """

    metadata = {}
    networkModule = None

    if "first" in metasets:
        # 単なるweightマージならそのままコピー
        metadata = lmetas[0]
    elif "new" in metasets:
        for key in MINIMUM_METADATA:
            if key in lmetas[0].keys():
                metadata[key] = lmetas[0][key]
        
    else:
        # 複数マージの場合はマージしたタグと主要メタデータを保存
        metadata = lmetas[0]
        tags = {}
        for i, lmeta in enumerate( lmetas ):
            meta = {}
            metadata[ f"sshs_cp{i}" ] = json.dumps( lmeta )

            # 最初の network_module を保持
            if networkModule is None and "ss_network_module" in lmeta:
                networkModule = lmeta["ss_network_module"]

            # タグをマージ
            if "merge" in metasets:
                if "ss_tag_frequency" in lmeta:
                    ldict = lmeta["ss_tag_frequency"]
                    if "ss_tag_frequency" in metadata:
                        mdict = metadata["ss_tag_frequency"]
                        if type(ldict) is dict and type(mdict) is dict:
                            for key in ldict:
                                if key not in mdict:
                                    mdict[key] = ldict[key]

    # network_moduleからLoRA種別判定する場合が多いため、最初に見つけたものにする
    if networkModule is not None:
        metadata["ss_network_module"] = networkModule

    # output名とprecision、dimは変更された可能性がある
    if "without" not in metasets:
        metadata["ss_output_name"] = lname
    else:
        if "ss_output_name" in metadata:
            del metadata["ss_output_name"]
    metadata["ss_mixed_precision"] = lprecision

    # metadataで保存できる形式に変換
    for key in metadata:
        if type(metadata[key] ) is not str:
            metadata[key] = json.dumps( metadata[key] )
    # データ変更によりhashが変わるので計算
    model_hash, legacy_hash = precalculate_safetensors_hashes( sd, metadata )
    metadata[ "sshs_model_hash" ] = model_hash
    metadata[ "sshs_legacy_hash" ] = legacy_hash

    return metadata


##############################################################
####### Get loranames from prompt
def frompromptf(*args):
    outst = []
    outss = []
    prompt = args[1]
    names, multis, lbws = loradealer(prompt, "", "")
    for name, multi, lbw in zip(names, multis, lbws):
        nml = [name,str(multi),lbw] if lbw is not None else [name,str(multi)]
        outst.append(":".join(nml))
        if name in selectable:
            outss.append(name)
    global pchanged
    pchanged = True
    return outss,",".join(outst), True

def loradealer(prompts,lratios,elementals):
    _, extra_network_data = extra_networks.parse_prompts([prompts])
    moduletypes = extra_network_data.keys()

    outnames = []
    outmultis = []
    outlbws = []

    for ltype in moduletypes:
        lorans = []
        lorars = []
        loraps = []
        multipliers = []
        elements = []
        if not (ltype == "lora" or ltype == "lyco") : continue
        for called in extra_network_data[ltype]:
            multiple = float(syntaxdealer(called.items,"unet=","te=",1))
            multipliers.append(multiple)
            lorans.append(called.items[0])
            loraps.append(syntaxdealer(called.items,"lbw=",None,2))

        if len(lorans) > 0:
            outnames.extend(lorans)
            outmultis.extend(multipliers)
            outlbws.extend(loraps)

    return outnames, outmultis, outlbws

def syntaxdealer(items,type1,type2,index): #type "unet=", "x=", "lwbe=" 
    target = [type1,type2] if type2 is not None else [type1]
    for t in target:
        for item in items:
            if t in item:
                return item.replace(t,"")
    if index > len(items) - 1 :return None
    return items[index] if "@" not in items[index] else 1

##############################################################
####### Extract lora from checkpoints args
class Kohya_extract_args:
    def __init__(
        self,
        v2=False,
        v_parameterization=None,
        sdxl=False,
        save_precision=None,
        model_org=None,
        model_tuned=None,
        save_to=None,
        dim=4,
        conv_dim=None,
        device=None,
        no_metadata=False,
        alpha = 1,
        beta = 1
    ):
        self.v2 = v2
        self.v_parameterization = v_parameterization
        self.sdxl = sdxl
        self.save_precision = save_precision
        self.model_org = model_org
        self.model_tuned = model_tuned
        self.save_to = save_to
        self.dim = dim
        self.conv_dim = conv_dim
        self.device = device
        self.no_metadata = no_metadata
        self.alpha = alpha
        self.beta = beta
