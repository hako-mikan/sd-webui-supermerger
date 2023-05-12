import re
from sklearn.linear_model import PassiveAggressiveClassifier
import torch
import math
import os
import gc
import gradio as gr
from torchmetrics import Precision
import modules.shared as shared
import gc
from safetensors.torch import load_file, save_file
from typing import List
from tqdm import tqdm
from modules import  sd_models,scripts
from scripts.mergers.model_util import load_models_from_stable_diffusion_checkpoint,filenamecutter,savemodel
from modules.ui import create_refresh_button

def on_ui_tabs():
    import lora
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
    sml_filepath = os.path.join(sml_path_root,"scripts", "lbwpresets.txt")
    sml_lbwpresets=""
    try:
        with open(sml_filepath,encoding="utf-8") as f:
            sml_lbwpresets = f.read()
    except OSError as e:
        sml_lbwpresets=LWEIGHTSPRESETS 

    with gr.Blocks(analytics_enabled=False) :
        sml_submit_result = gr.Textbox(label="Message")
        with gr.Row().style(equal_height=False):
            sml_cpmerge = gr.Button(elem_id="model_merger_merge", value="Merge to Checkpoint",variant='primary')
            sml_makelora = gr.Button(elem_id="model_merger_merge", value="Make LoRA (alpha * A - beta * B)",variant='primary')
            sml_model_a = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint A",interactive=True)
            create_refresh_button(sml_model_a, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
            sml_model_b = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Checkpoint B",interactive=True)
            create_refresh_button(sml_model_b, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
        with gr.Row().style(equal_height=False):
            sml_merge = gr.Button(elem_id="model_merger_merge", value="Merge LoRAs",variant='primary')
            alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=1)
            beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=1)
        with gr.Row().style(equal_height=False):
            sml_settings = gr.CheckboxGroup(["same to Strength", "overwrite"], label="settings")
            precision = gr.Radio(label = "save precision",choices=["float","fp16","bf16"],value = "fp16",type="value")
        with gr.Row().style(equal_height=False):
          sml_dim = gr.Radio(label = "remake dimension",choices = ["no","auto",*[2**(x+2) for x in range(9)]],value = "no",type = "value") 
          sml_filename = gr.Textbox(label="filename(option)",lines=1,visible =True,interactive  = True)  
        sml_loranames = gr.Textbox(label='LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,...(":blocks" is option, not necessary)',lines=1,value="",visible =True)
        sml_dims = gr.CheckboxGroup(label = "limit dimension",choices=[],value = [],type="value",interactive=True,visible = False)
        with gr.Row().style(equal_height=False):
          sml_calcdim = gr.Button(elem_id="calcloras", value="calculate dimension of LoRAs(It may take a few minutes if there are many LoRAs)",variant='primary')
          sml_update = gr.Button(elem_id="calcloras", value="update list",variant='primary')
        sml_loras = gr.CheckboxGroup(label = "Lora",choices=[x[0] for x in lora.available_loras.items()],type="value",interactive=True,visible = True)
        sml_loraratios = gr.TextArea(label="",value=sml_lbwpresets,visible =True,interactive  = True)  

        sml_merge.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_dim,precision],
            outputs=[sml_submit_result]
        )

        sml_makelora.click(
            fn=makelora,
            inputs=[sml_model_a,sml_model_b,sml_dim,sml_filename,sml_settings,alpha,beta,precision],
            outputs=[sml_submit_result]
        )

        sml_cpmerge.click(
            fn=pluslora,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_model_a,precision],
            outputs=[sml_submit_result]
        )
        llist ={}
        dlist =[]
        dn = []

        def updateloras():
          lora.list_available_loras()
          for n in  lora.available_loras.items():
            if n[0] not in llist:llist[n[0]] = ""
          return gr.update(choices = [f"{x[0]}({x[1]})" for x in llist.items()])

        sml_update.click(fn = updateloras,outputs = [sml_loras])

        def calculatedim():
          print("listing dimensions...")
          for n in  tqdm(lora.available_loras.items()):
              if n[0] in llist:
                if llist[n[0]] !="": continue
              c_lora = lora.available_loras.get(n[0], None) 
              d,t = dimgetter(c_lora.filename)
              if t == "LoCon" : d = f"{d}:{t}"
              if d not in dlist:
                if type(d) == int :dlist.append(d)
                elif d not in dn: dn.append(d)
              llist[n[0]] = d
          dlist.sort()
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
          return gr.update(choices = [l for l in rl],value =[])

        def llister(names):
          if names ==[] : return ""
          else:
            for i,n in enumerate(names):
              if "(" in n:names[i] = n[:n.rfind("(")]
            return ":1.0,".join(names)+":1.0"
        sml_loras.change(fn=llister,inputs=[sml_loras],outputs=[sml_loranames])     
        sml_dims.change(fn=dimselector,inputs=[sml_dims],outputs=[sml_loras])  

def makelora(model_a,model_b,dim,saveto,settings,alpha,beta,precision):
    print("make LoRA start")
    if model_a == "" or model_b =="":
      return "ERROR: No model Selected"
    gc.collect()

    if saveto =="" : saveto = makeloraname(model_a,model_b)
    if not ".safetensors" in saveto :saveto  += ".safetensors"
    saveto = os.path.join(shared.cmd_opts.lora_dir,saveto)

    dim = 128 if type(dim) != int else int(dim)
    if os.path.isfile(saveto ) and not "overwrite" in settings:
        _err_msg = f"Output file ({saveto}) existed and was not saved"
        print(_err_msg)
        return _err_msg

    svd(fullpathfromname(model_a),fullpathfromname(model_b),False,dim,precision,saveto,alpha,beta)
    return f"saved to {saveto}"

def lmerge(loranames,loraratioss,settings,filename,dim,precision):
    import lora
    loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]
    if any([x is None for x in loras_on_disk]):
        lora.list_available_loras()

        loras_on_disk = [lora.available_loras.get(name, None) for name in loranames]

    lnames = [loranames] if "," not in loranames else loranames.split(",")

    for i, n in enumerate(lnames):
        lnames[i] = n.split(":")

    loraratios=loraratioss.splitlines()
    ldict ={}

    for i,l in enumerate(loraratios):
        if ":" not in l or not (l.count(",") == 16 or l.count(",") == 25) : continue
        ldict[l.split(":")[0]]=l.split(":")[1]

    ln = []
    lr = []
    ld = []
    lt = []
    dmax = 1

    for i,n in enumerate(lnames):
        if len(n) ==3:
            if n[2].strip() in ldict:
                ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
            else:ratio = [float(n[1])]*17
        else:ratio = [float(n[1])]*17
        c_lora = lora.available_loras.get(n[0], None) 
        ln.append(c_lora.filename)
        lr.append(ratio)
        d,t = dimgetter(c_lora.filename)
        lt.append(t)
        ld.append(d)
        if d != "LyCORIS":
          if d > dmax : dmax = d

    if filename =="":filename =loranames.replace(",","+").replace(":","_")
    if not ".safetensors" in filename:filename += ".safetensors"
    filename = os.path.join(shared.cmd_opts.lora_dir,filename)
  
    dim = int(dim) if dim != "no" and dim != "auto" else 0

    if "LyCORIS" in ld or "LoCon" in lt:
      if len(ld) !=1:
        return "multiple merge of LyCORIS is not supported"
      sd = lycomerge(ln[0],lr[0])
    elif dim > 0:
      print("change demension to ", dim)
      sd = merge_lora_models_dim(ln, lr, dim,settings)
    elif "auto" in settings and ld.count(ld[0]) != len(ld):
      print("change demension to ",dmax)
      sd = merge_lora_models_dim(ln, lr, dmax,settings)
    else:
      sd = merge_lora_models(ln, lr,settings)

    if os.path.isfile(filename) and not "overwrite" in settings:
        _err_msg = f"Output file ({filename}) existed and was not saved"
        print(_err_msg)
        return _err_msg

    save_to_file(filename,sd,sd, str_to_dtype(precision))
    return "saved : "+filename

def pluslora(lnames,loraratios,settings,output,model,precision):
    if model == []:
      return "ERROR: No model Selected"
    if lnames == "":
      return "ERROR: No LoRA Selected"

    print("plus LoRA start")
    import lora
    lnames = [lnames] if "," not in lnames else lnames.split(",")

    for i, n in enumerate(lnames):
        lnames[i] = n.split(":")

    loraratios=loraratios.splitlines()
    ldict ={}

    for i,l in enumerate(loraratios):
        if ":" not in l or not (l.count(",") == 16 or l.count(",") == 25) : continue
        ldict[l.split(":")[0].strip()]=l.split(":")[1]

    names=[]
    filenames=[]
    loratypes=[]
    lweis=[]

    for n in lnames:
        if len(n) ==3:
            if n[2].strip() in ldict:
                ratio = [float(r)*float(n[1]) for r in ldict[n[2]].split(",")]
            else:ratio = [float(n[1])]*17
        else:ratio = [float(n[1])]*17
        c_lora = lora.available_loras.get(n[0], None) 
        names.append(n[0])
        filenames.append(c_lora.filename)
        _,t = dimgetter(c_lora.filename)
        if "LyCORIS" in t: return "LyCORIS merge is not supported"
        lweis.append(ratio)

    modeln=filenamecutter(model,True)   
    dname = modeln
    for n in names:
      dname = dname + "+"+n

    checkpoint_info = sd_models.get_closet_checkpoint_match(model)
    print(f"Loading {model}")
    theta_0 = sd_models.read_state_dict(checkpoint_info.filename,"cpu")

    keychanger = {}
    for key in theta_0.keys():
        if "model" in key:
            skey = key.replace(".","_").replace("_weight","")
            keychanger[skey.split("model_",1)[1]] = key

    for name,filename, lwei in zip(names,filenames, lweis):
      print(f"loading: {name}")
      lora_sd = load_state_dict(filename, torch.float)

      print(f"merging..." ,lwei)
      for key in lora_sd.keys():
        ratio = 1

        fullkey = convert_diffusers_name_to_compvis(key)

        for i,block in enumerate(LORABLOCKS):
            if block in fullkey:
                ratio = lwei[i]

        msd_key, lora_key = fullkey.split(".", 1)

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

          if not len(down_weight.size()) == 4:
            # linear
            weight = weight  + ratio * (up_weight @ down_weight) * scale
          else:
            # conv2d
            weight = weight  + ratio * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)
                                        ).unsqueeze(2).unsqueeze(3) * scale
          theta_0[keychanger[msd_key]] = torch.nn.Parameter(weight)
    #usemodelgen(theta_0,model)
    settings.append(precision)
    result = savemodel(theta_0,dname,output,settings,model)
    del theta_0
    gc.collect()
    return result

def save_to_file(file_name, model, state_dict, dtype):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == '.safetensors':
        save_file(model, file_name)
    else:
        torch.save(model, file_name)

re_digits = re.compile(r"\d+")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")


def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key

CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6

def str_to_dtype(p):
  if p == 'float':
    return torch.float
  if p == 'fp16':
    return torch.float16
  if p == 'bf16':
    return torch.bfloat16
  return None

def svd(model_a,model_b,v2,dim,save_precision,save_to,alpha,beta):
  save_dtype = str_to_dtype(save_precision)

  if model_a == model_b:
    text_encoder_t, _, unet_t = load_models_from_stable_diffusion_checkpoint(v2, model_a)
    text_encoder_o, _, unet_o = text_encoder_t, _, unet_t
  else:
    print(f"loading SD model : {model_b}")
    text_encoder_o, _, unet_o = load_models_from_stable_diffusion_checkpoint(v2, model_b)
    
    print(f"loading SD model : {model_a}")
    text_encoder_t, _, unet_t = load_models_from_stable_diffusion_checkpoint(v2, model_a)

  # create LoRA network to extract weights: Use dim (rank) as alpha
  lora_network_o = create_network(1.0, dim, dim, None, text_encoder_o, unet_o)
  lora_network_t = create_network(1.0, dim, dim, None, text_encoder_t, unet_t)
  assert len(lora_network_o.text_encoder_loras) == len(
      lora_network_t.text_encoder_loras), f"model version is different (SD1.x vs SD2.x) / それぞれのモデルのバージョンが違います（SD1.xベースとSD2.xベース） "
  # get diffs
  diffs = {}
  text_encoder_different = False
  for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.text_encoder_loras, lora_network_t.text_encoder_loras)):
    lora_name = lora_o.lora_name
    module_o = lora_o.org_module
    module_t = lora_t.org_module
    diff = alpha*module_t.weight - beta*module_o.weight

    # Text Encoder might be same
    if torch.max(torch.abs(diff)) > MIN_DIFF:
      text_encoder_different = True

    diff = diff.float()
    diffs[lora_name] = diff

  if not text_encoder_different:
    print("Text encoder is same. Extract U-Net only.")
    lora_network_o.text_encoder_loras = []
    diffs = {}

  for i, (lora_o, lora_t) in enumerate(zip(lora_network_o.unet_loras, lora_network_t.unet_loras)):
    lora_name = lora_o.lora_name
    module_o = lora_o.org_module
    module_t = lora_t.org_module
    diff = alpha*module_t.weight - beta*module_o.weight
    diff = diff.float()

    diffs[lora_name] = diff

  # make LoRA with svd
  print("calculating by svd")
  rank = dim
  lora_weights = {}
  with torch.no_grad():
    for lora_name, mat in tqdm(list(diffs.items())):
      conv2d = (len(mat.size()) == 4)
      if conv2d:
        mat = mat.squeeze()

      U, S, Vh = torch.linalg.svd(mat)

      U = U[:, :rank]
      S = S[:rank]
      U = U @ torch.diag(S)

      Vh = Vh[:rank, :]

      dist = torch.cat([U.flatten(), Vh.flatten()])
      hi_val = torch.quantile(dist, CLAMP_QUANTILE)
      low_val = -hi_val

      U = U.clamp(low_val, hi_val)
      Vh = Vh.clamp(low_val, hi_val)

      lora_weights[lora_name] = (U, Vh)

  # make state dict for LoRA
  lora_network_o.apply_to(text_encoder_o, unet_o, text_encoder_different, True)   # to make state dict
  lora_sd = lora_network_o.state_dict()
  print(f"LoRA has {len(lora_sd)} weights.")

  for key in list(lora_sd.keys()):
    if "alpha" in key:
      continue

    lora_name = key.split('.')[0]
    i = 0 if "lora_up" in key else 1

    weights = lora_weights[lora_name][i]
    # print(key, i, weights.size(), lora_sd[key].size())
    if len(lora_sd[key].size()) == 4:
      weights = weights.unsqueeze(2).unsqueeze(3)

    assert weights.size() == lora_sd[key].size(), f"size unmatch: {key}"
    lora_sd[key] = weights

  # load state dict to LoRA and save it
  info = lora_network_o.load_state_dict(lora_sd)
  print(f"Loading extracted LoRA weights: {info}")

  dir_name = os.path.dirname(save_to)
  if dir_name and not os.path.exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)

  # minimum metadata
  metadata = {"ss_network_dim": str(dim), "ss_network_alpha": str(dim)}

  lora_network_o.save_weights(save_to, save_dtype, metadata)
  print(f"LoRA weights are saved to: {save_to}")
  return save_to  

def load_state_dict(file_name, dtype):
  if os.path.splitext(file_name)[1] == '.safetensors':
    sd = load_file(file_name)
  else:
    sd = torch.load(file_name, map_location='cpu')
  for key in list(sd.keys()):
    if type(sd[key]) == torch.Tensor:
      sd[key] = sd[key].to(dtype)
  return sd

def dimgetter(filename):
    lora_sd = load_state_dict(filename, torch.float)
    alpha = None
    dim = None
    type = None

    if "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight" in lora_sd.keys():
      type = "LoCon"

    for key, value in lora_sd.items():
  
        if alpha is None and 'alpha' in key:
            alpha = value
        if dim is None and 'lora_down' in key and len(value.size()) == 2:
            dim = value.size()[0]
        if "hada_" in key:
            dim,type = "LyCORIS","LyCORIS"
        if alpha is not None and dim is not None:
            break
    if alpha is None:
        alpha = dim
    if type == None:type = "LoRA"
    if dim :
      return dim,type
    else:
      return "unknown","unknown"

def blockfromkey(key):
    fullkey = convert_diffusers_name_to_compvis(key)
    for i,n in enumerate(LORABLOCKS):
      if n in  fullkey: return i
    return 0

def merge_lora_models_dim(models, ratios, new_rank,sets):
  merged_sd = {}
  fugou = 1
  for model, ratios in zip(models, ratios):
    merge_dtype = torch.float
    lora_sd = load_state_dict(model, merge_dtype)

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
        weight = torch.zeros((out_dim, in_dim, 1, 1) if conv2d else (out_dim, in_dim), dtype=merge_dtype)
      else:
        weight = merged_sd[lora_module_name]

      ratio = ratios[blockfromkey(key)]
      if "same to Strength" in sets:
        ratio, fugou = (ratio**0.5,1) if ratio > 0 else (abs(ratio)**0.5,-1)
      #print(lora_module_name, ratio)
      # W <- W + U * D
      scale = (alpha / network_dim)
      if not conv2d:        # linear
        weight = weight + ratio * (up_weight @ down_weight) * scale * fugou
      else:
        weight = weight + ratio * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)
                                   ).unsqueeze(2).unsqueeze(3) * scale * fugou

      merged_sd[lora_module_name] = weight

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

  return merged_lora_sd

def merge_lora_models(models, ratios,sets):
  base_alphas = {}                          # alpha for merged model
  base_dims = {}
  merge_dtype = torch.float
  merged_sd = {}
  fugou = 1
  for model, ratios in zip(models, ratios):
    print(f"merging {model}: {ratios}")
    lora_sd = load_state_dict(model, merge_dtype)

    # get alpha and dim
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
    
    print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

    # merge
    print(f"merging...")
    for key in lora_sd.keys():
      if 'alpha' in key:
        continue
      if "lora_down" in key: dwon = True
      lora_module_name = key[:key.rfind(".lora_")]

      base_alpha = base_alphas[lora_module_name]
      alpha = alphas[lora_module_name]

      ratio = ratios[blockfromkey(key)]
      if "same to Strength" in sets:
        ratio, fugou = (ratio**0.5,1) if ratio > 0 else (abs(ratio)**0.5,-1)

      if "lora_down" in key:
        ratio = ratio * fugou

      scale = math.sqrt(alpha / base_alpha) * ratio

      if key in merged_sd:
        assert merged_sd[key].size() == lora_sd[key].size(
        ), f"weights shape mismatch merging v1 and v2, different dims? / 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
        merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
      else:
        merged_sd[key] = lora_sd[key] * scale
  
  # set alpha to sd
  for lora_module_name, alpha in base_alphas.items():
    key = lora_module_name + ".alpha"
    merged_sd[key] = torch.tensor(alpha)

  print("merged model")
  print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

  return merged_sd

def fullpathfromname(name):
    if hash == "" or hash ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    return checkpoint_info.filename

def makeloraname(model_a,model_b):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    return "lora_"+model_a+"-"+model_b

def lycomerge(filename,ratios):
    sd = load_state_dict(filename, torch.float)

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
        
        fullkey = convert_diffusers_name_to_compvis(lkey)
        key, lora_key = fullkey.split(".", 1)

        for i,block in enumerate(LYCOBLOCKS):
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

LORABLOCKS=["encoder",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_"]

LYCOBLOCKS=["encoder",
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
"diffusion_model_output_blocks_11_"]

class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.region = None
        self.region_mask = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def merge_to(self, sd, dtype, device):
        # get up/down weight
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)

        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)

        # merge weight
        if len(weight.size()) == 2:
            # linear
            weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # print(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + self.multiplier * conved * self.scale

        # set weight to org_module
        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def forward(self, x):
        if self.region is None:
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

        # regional LoRA   FIXME same as additional-network extension
        if x.size()[1] % 77 == 0:
            # print(f"LoRA for context: {self.lora_name}")
            self.region = None
            return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

        # calculate region mask first time
        if self.region_mask is None:
            if len(x.size()) == 4:
                h, w = x.size()[2:4]
            else:
                seq_len = x.size()[1]
                ratio = math.sqrt((self.region.size()[0] * self.region.size()[1]) / seq_len)
                h = int(self.region.size()[0] / ratio + 0.5)
                w = seq_len // h

            r = self.region.to(x.device)
            if r.dtype == torch.bfloat16:
                r = r.to(torch.float)
            r = r.unsqueeze(0).unsqueeze(1)
            # print(self.lora_name, self.region.size(), x.size(), r.size(), h, w)
            r = torch.nn.functional.interpolate(r, (h, w), mode="bilinear")
            r = r.to(x.dtype)

            if len(x.size()) == 3:
                r = torch.reshape(r, (1, x.size()[1], -1))

            self.region_mask = r

        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale * self.region_mask

def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    if network_dim is None:
        network_dim = 4  # default

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    """
    block_dims = kwargs.get("block_dims")
    block_alphas = None
    if block_dims is not None:
    block_dims = [int(d) for d in block_dims.split(',')]
    assert len(block_dims) == NUM_BLOCKS, f"Number of block dimensions is not same to {NUM_BLOCKS}"
    block_alphas = kwargs.get("block_alphas")
    if block_alphas is None:
        block_alphas = [1] * len(block_dims)
    else:
        block_alphas = [int(a) for a in block_alphas(',')]
    assert len(block_alphas) == NUM_BLOCKS, f"Number of block alphas is not same to {NUM_BLOCKS}"
    conv_block_dims = kwargs.get("conv_block_dims")
    conv_block_alphas = None
    if conv_block_dims is not None:
    conv_block_dims = [int(d) for d in conv_block_dims.split(',')]
    assert len(conv_block_dims) == NUM_BLOCKS, f"Number of block dimensions is not same to {NUM_BLOCKS}"
    conv_block_alphas = kwargs.get("conv_block_alphas")
    if conv_block_alphas is None:
        conv_block_alphas = [1] * len(conv_block_dims)
    else:
        conv_block_alphas = [int(a) for a in conv_block_alphas(',')]
    assert len(conv_block_alphas) == NUM_BLOCKS, f"Number of block alphas is not same to {NUM_BLOCKS}"
  """

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
    )
    return network



class LoRANetwork(torch.nn.Module):
    # is it possible to apply conv_in and conv_out?
    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        conv_lora_dim=None,
        conv_alpha=None,
        modules_dim=None,
        modules_alpha=None,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha

        if modules_dim is not None:
            print(f"create LoRA network from weights")
        else:
            print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")

        self.apply_to_conv2d_3x3 = self.conv_lora_dim is not None
        if self.apply_to_conv2d_3x3:
            if self.conv_alpha is None:
                self.conv_alpha = self.alpha
            print(f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}")

        # create module instances
        def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    # TODO get block index here
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            if modules_dim is not None:
                                if lora_name not in modules_dim:
                                    continue  # no LoRA module in this weights file
                                dim = modules_dim[lora_name]
                                alpha = modules_alpha[lora_name]
                            else:
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.apply_to_conv2d_3x3:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha
                                else:
                                    continue

                            lora = LoRAModule(lora_name, child_module, self.multiplier, dim, alpha)
                            loras.append(lora)
            return loras

        self.text_encoder_loras = create_modules(
            LoRANetwork.LORA_PREFIX_TEXT_ENCODER, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
        )
        print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras = create_modules(LoRANetwork.LORA_PREFIX_UNET, unet, target_modules)
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location="cpu")

    def apply_to(self, text_encoder, unet, apply_text_encoder=None, apply_unet=None):
        if self.weights_sd:
            weights_has_text_encoder = weights_has_unet = False
            for key in self.weights_sd.keys():
                if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                    weights_has_unet = True

            if apply_text_encoder is None:
                apply_text_encoder = weights_has_text_encoder
            else:
                assert (
                    apply_text_encoder == weights_has_text_encoder
                ), f"text encoder weights: {weights_has_text_encoder} but text encoder flag: {apply_text_encoder} / 重みとText Encoderのフラグが矛盾しています"

            if apply_unet is None:
                apply_unet = weights_has_unet
            else:
                assert (
                    apply_unet == weights_has_unet
                ), f"u-net weights: {weights_has_unet} but u-net flag: {apply_unet} / 重みとU-Netのフラグが矛盾しています"
        else:
            assert apply_text_encoder is not None and apply_unet is not None, f"internal error: flag not set"

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            print(f"weights are loaded: {info}")

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, dtype, device):
        assert self.weights_sd is not None, "weights are not loaded"

        apply_text_encoder = apply_unet = False
        for key in self.weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in self.weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = self.weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)
        print(f"weights are merged")

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    @staticmethod
    def set_regions(networks, image):
        image = image.astype(np.float32) / 255.0
        for i, network in enumerate(networks[:3]):
            # NOTE: consider averaging overwrapping area
            region = image[:, :, i]
            if region.max() == 0:
                continue
            region = torch.tensor(region)
            network.set_region(region)

    def set_region(self, region):
        for lora in self.unet_loras:
            lora.set_region(region)

from io import BytesIO
import safetensors.torch
import hashlib

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
