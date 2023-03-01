import re
import torch
import math
import os
import gc
import gradio as gr
import modules.shared as shared
import gc
from safetensors.torch import load_file, save_file
from typing import List
from tqdm import tqdm
from modules import  sd_models,scripts
from scripts.mergers.model_util import load_models_from_stable_diffusion_checkpoint,filenamecutter,savemodel
from modules.ui import create_refresh_button

LORABLOCKS=["encoder",
"down_blocks_0_attentions_0",
"down_blocks_0_attentions_1",
"down_blocks_1_attentions_0",
"down_blocks_1_attentions_1",
"down_blocks_2_attentions_0",
"down_blocks_2_attentions_1",
"mid_block_attentions_0",
"up_blocks_1_attentions_0",
"up_blocks_1_attentions_1",
"up_blocks_1_attentions_2",
"up_blocks_2_attentions_0",
"up_blocks_2_attentions_1",
"up_blocks_2_attentions_2",
"up_blocks_3_attentions_0",
"up_blocks_3_attentions_1",
"up_blocks_3_attentions_2"]

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
        with open(sml_filepath) as f:
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
            sml_settings = gr.CheckboxGroup(["same to Strength", "overwrite"], label="settings")
            alpha = gr.Slider(label="alpha", minimum=-1.0, maximum=2, step=0.001, value=1)
            beta = gr.Slider(label="beta", minimum=-1.0, maximum=2, step=0.001, value=1)
        with gr.Row().style(equal_height=False):
          sml_dim = gr.Radio(label = "remake dimension",choices = ["no","auto",*[2**(x+2) for x in range(9)]],value = "no",type = "value") 
          sml_filename = gr.Textbox(label="filename(option)",lines=1,visible =True,interactive  = True)  
        sml_loranames = gr.Textbox(label='LoRAname1:ratio1:Blocks1,LoRAname2:ratio2:Blocks2,...(":blocks" is option, not necessary)',lines=1,value="",visible =True)
        sml_dims = gr.CheckboxGroup(label = "limit dimension",choices=[],value = [],type="value",interactive=True,visible = False)
        with gr.Row().style(equal_height=False):
          sml_calcdim = gr.Button(elem_id="calcloras", value="calculate dimension of LoRAs(It may take a few minutes if there are many LoRAs)",variant='primary')
          sml_update = gr.Button(elem_id="calcloras", value="update list",variant='primary')
        sml_loras = gr.CheckboxGroup(label = "Lora",choices=[x[0] for x in lora.available_loras.items()],type="value",interactive=True,visible = True)
        sml_loraratios = gr.TextArea(label="",lines=10,value=sml_lbwpresets,visible =True,interactive  = True)  

        sml_merge.click(
            fn=lmerge,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_dim],
            outputs=[sml_submit_result]
        )

        sml_makelora.click(
            fn=makelora,
            inputs=[sml_model_a,sml_model_b,sml_dim,sml_filename,sml_settings,alpha,beta],
            outputs=[sml_submit_result]
        )

        sml_cpmerge.click(
            fn=pluslora,
            inputs=[sml_loranames,sml_loraratios,sml_settings,sml_filename,sml_model_a],
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
              d = dimgetter(c_lora.filename)
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

def fullpathfromname(name):
    if hash == "" or hash ==[]: return ""
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    return checkpoint_info.filename

def makeloraname(model_a,model_b):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    return "lora_"+model_a+"-"+model_b

def makelora(model_a,model_b,dim,saveto,settings,alpha,beta):
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

    svd(fullpathfromname(model_a),fullpathfromname(model_b),False,dim,"float",saveto,alpha,beta)
    return f"saved to {saveto}"

def lmerge(loranames,loraratioss,settings,filename,dim):
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
        ldict[l.split(":")[0]]=l.split(":")[1]

    ln = []
    lr = []
    ld = []
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
        d = dimgetter(c_lora.filename)
        ld.append(d)
        if d > dmax : dmax = d

    if filename =="":filename =loranames.replace(",","+").replace(":","_")
    if not ".safetensors" in filename:filename += ".safetensors"
    filename = os.path.join(shared.cmd_opts.lora_dir,filename)
  
    dim = int(dim) if dim != "no" and dim != "auto" else 0

    if dim > 0:
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

    save_to_file(filename,sd,sd, torch.float)
    return "saved : "+filename

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

    if match(m, re_text_block):
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key

def pluslora(lnames,loraratios,settings,output,model):
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
        ldict[l.split(":")[0].strip()]=l.split(":")[1]

    names=[]
    filenames=[]
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
        for i,block in enumerate(LORABLOCKS):
            if block in key:
                ratio = lwei[i]

        fullkey = convert_diffusers_name_to_compvis(key)
        msd_key, lora_key = fullkey.split(".", 1)

        if "lora_down" in key:
          up_key = key.replace("lora_down", "lora_up")
          alpha_key = key[:key.index("lora_down")] + 'alpha'

          # print(f"apply {key} to {module}")

          down_weight = lora_sd[key]
          up_weight = lora_sd[up_key]

          dim = down_weight.size()[0]
          alpha = lora_sd.get(alpha_key, dim)
          scale = alpha / dim
          # W <- W + U * D
          weight = theta_0[keychanger[msd_key]]

          if not len(down_weight.size()) == 4:
            # linear
            weight = weight  + ratio * (up_weight @ down_weight) * scale
          else:
            # conv2d
            weight = weight  + ratio * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)
                                        ).unsqueeze(2).unsqueeze(3) * scale
          theta_0[keychanger[msd_key]] = torch.nn.Parameter(weight)
    #usemodelgen(theta_0,model)
    result = savemodel(theta_0,dname,output,settings,model)
    del theta_0
    gc.collect()
    return result

CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6
def svd(model_a,model_b,v2,dim,save_precision,save_to,alpha,beta):
  def str_to_dtype(p):
    if p == 'float':
      return torch.float
    if p == 'fp16':
      return torch.float16
    if p == 'bf16':
      return torch.bfloat16
    return None

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

class LoRAModule(torch.nn.Module):
  """
  replaces forward method of the original Linear, instead of replacing the original Linear module.
  """

  def __init__(self, lora_name, org_module: torch.nn.Module, multiplier=1.0, lora_dim=4, alpha=1):
    """ if alpha == 0 or None, alpha is rank (no scaling). """
    super().__init__()
    self.lora_name = lora_name
    self.lora_dim = lora_dim

    if org_module.__class__.__name__ == 'Conv2d':
      in_dim = org_module.in_channels
      out_dim = org_module.out_channels
      self.lora_down = torch.nn.Conv2d(in_dim, lora_dim, (1, 1), bias=False)
      self.lora_up = torch.nn.Conv2d(lora_dim, out_dim, (1, 1), bias=False)
    else:
      in_dim = org_module.in_features
      out_dim = org_module.out_features
      self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
      self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

    if type(alpha) == torch.Tensor:
      alpha = alpha.detach().float().numpy()                              # without casting, bf16 causes error
    alpha = lora_dim if alpha is None or alpha == 0 else alpha
    self.scale = alpha / self.lora_dim
    self.register_buffer('alpha', torch.tensor(alpha))                    # 定数として扱える

    # same as microsoft's
    torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(self.lora_up.weight)

    self.multiplier = multiplier
    self.org_module = org_module                  # remove in applying

  def apply_to(self):
    self.org_forward = self.org_module.forward
    self.org_module.forward = self.forward
    del self.org_module

  def forward(self, x):
    return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
  if network_dim is None:
    network_dim = 4                     # default
  return LoRANetwork(text_encoder, unet, multiplier=multiplier, lora_dim=network_dim, alpha=network_alpha)

class LoRANetwork(torch.nn.Module):
  UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
  TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
  LORA_PREFIX_UNET = 'lora_unet'
  LORA_PREFIX_TEXT_ENCODER = 'lora_te'

  def __init__(self, text_encoder, unet, multiplier=1.0, lora_dim=4, alpha=1) -> None:
    super().__init__()
    self.multiplier = multiplier
    self.lora_dim = lora_dim
    self.alpha = alpha

    # create module instances
    def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> List[LoRAModule]:
      loras = []
      for name, module in root_module.named_modules():
        if module.__class__.__name__ in target_replace_modules:
          for child_name, child_module in module.named_modules():
            if child_module.__class__.__name__ == "Linear" or (child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)):
              lora_name = prefix + '.' + name + '.' + child_name
              lora_name = lora_name.replace('.', '_')
              lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim, self.alpha)
              loras.append(lora)
      return loras

    self.text_encoder_loras = create_modules(LoRANetwork.LORA_PREFIX_TEXT_ENCODER,
                                             text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
    print(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

    self.unet_loras = create_modules(LoRANetwork.LORA_PREFIX_UNET, unet, LoRANetwork.UNET_TARGET_REPLACE_MODULE)
    print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

    self.weights_sd = None

    # assertion
    names = set()
    for lora in self.text_encoder_loras + self.unet_loras:
      assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
      names.add(lora.lora_name)

  def load_weights(self, file):
    if os.path.splitext(file)[1] == '.safetensors':
      from safetensors.torch import load_file, safe_open
      self.weights_sd = load_file(file)
    else:
      self.weights_sd = torch.load(file, map_location='cpu')

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
        assert apply_text_encoder == weights_has_text_encoder, f"text encoder weights: {weights_has_text_encoder} but text encoder flag: {apply_text_encoder} / 重みとText Encoderのフラグが矛盾しています"

      if apply_unet is None:
        apply_unet = weights_has_unet
      else:
        assert apply_unet == weights_has_unet, f"u-net weights: {weights_has_unet} but u-net flag: {apply_unet} / 重みとU-Netのフラグが矛盾しています"
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
      param_data = {'params': enumerate_params(self.text_encoder_loras)}
      if text_encoder_lr is not None:
        param_data['lr'] = text_encoder_lr
      all_params.append(param_data)

    if self.unet_loras:
      param_data = {'params': enumerate_params(self.unet_loras)}
      if unet_lr is not None:
        param_data['lr'] = unet_lr
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

    if os.path.splitext(file)[1] == '.safetensors':
      from safetensors.torch import save_file

      save_file(state_dict, file)
    else:
      torch.save(state_dict, file)

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
    for key in lora_sd.keys():
          if "lora_down" in key:
            dim = lora_sd[key].size()[0]
            return dim
    return "unknown"

def blockfromkey(key):
    for i,n in enumerate(LORABLOCKS):
      if n in  key: return i
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
