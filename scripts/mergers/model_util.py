import os
import gc
import torch
from transformers import CLIPTextModel,  CLIPTextConfig
from safetensors.torch import load_file
import safetensors.torch
import threading
from modules import shared
from modules.sd_models import read_state_dict
try:
  from modules import sd_models_xl
  xl = True
except:
  xl = False

# DiffUsers版StableDiffusionのモデルパラメータ
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 64  # fixed from old invalid value `32`
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# V2
V2_UNET_PARAMS_ATTENTION_HEAD_DIM = [5, 10, 20, 20]
V2_UNET_PARAMS_CONTEXT_DIM = 1024

# Diffusersの設定を読み込むための参照モデル
DIFFUSERS_REF_MODEL_ID_V1 = "runwayml/stable-diffusion-v1-5"
DIFFUSERS_REF_MODEL_ID_V2 = "stabilityai/stable-diffusion-2-1"


# region StableDiffusion->Diffusersの変換コード
# convert_original_stable_diffusion_to_diffusers をコピーして修正している（ASL 2.0）


def shave_segments(path, n_shave_prefix_segments=1):
  """
  Removes segments. Positive values shave the first segments, negative shave the last segments.
  """
  if n_shave_prefix_segments >= 0:
    return ".".join(path.split(".")[n_shave_prefix_segments:])
  else:
    return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item.replace("in_layers.0", "norm1")
    new_item = new_item.replace("in_layers.2", "conv1")

    new_item = new_item.replace("out_layers.0", "norm2")
    new_item = new_item.replace("out_layers.3", "conv2")

    new_item = new_item.replace("emb_layers.1", "time_emb_proj")
    new_item = new_item.replace("skip_connection", "conv_shortcut")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside resnets to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("nin_shortcut", "conv_shortcut")
    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
    #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

    #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
    #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

    #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
  """
  Updates paths inside attentions to the new naming scheme (local renaming)
  """
  mapping = []
  for old_item in old_list:
    new_item = old_item

    new_item = new_item.replace("norm.weight", "group_norm.weight")
    new_item = new_item.replace("norm.bias", "group_norm.bias")

    new_item = new_item.replace("q.weight", "query.weight")
    new_item = new_item.replace("q.bias", "query.bias")

    new_item = new_item.replace("k.weight", "key.weight")
    new_item = new_item.replace("k.bias", "key.bias")

    new_item = new_item.replace("v.weight", "value.weight")
    new_item = new_item.replace("v.bias", "value.bias")

    new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
    new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

    new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

    mapping.append({"old": old_item, "new": new_item})

  return mapping


def assign_to_checkpoint(
    paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
):
  """
  This does the final conversion step: take locally converted weights and apply a global renaming
  to them. It splits attention layers, and takes into account additional replacements
  that may arise.

  Assigns the weights to the new checkpoint.
  """
  assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

  # Splits the attention layers into three variables.
  if attention_paths_to_split is not None:
    for path, path_map in attention_paths_to_split.items():
      old_tensor = old_checkpoint[path]
      channels = old_tensor.shape[0] // 3

      target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

      num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

      old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
      query, key, value = old_tensor.split(channels // num_heads, dim=1)

      checkpoint[path_map["query"]] = query.reshape(target_shape)
      checkpoint[path_map["key"]] = key.reshape(target_shape)
      checkpoint[path_map["value"]] = value.reshape(target_shape)

  for path in paths:
    new_path = path["new"]

    # These have already been assigned
    if attention_paths_to_split is not None and new_path in attention_paths_to_split:
      continue

    # Global renaming happens here
    new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
    new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
    new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

    if additional_replacements is not None:
      for replacement in additional_replacements:
        new_path = new_path.replace(replacement["old"], replacement["new"])

    # proj_attn.weight has to be converted from conv 1D to linear
    if "proj_attn.weight" in new_path:
      checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
    else:
      checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
  keys = list(checkpoint.keys())
  attn_keys = ["query.weight", "key.weight", "value.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in attn_keys:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0, 0]
    elif "proj_attn.weight" in key:
      if checkpoint[key].ndim > 2:
        checkpoint[key] = checkpoint[key][:, :, 0]


def linear_transformer_to_conv(checkpoint):
  keys = list(checkpoint.keys())
  tf_keys = ["proj_in.weight", "proj_out.weight"]
  for key in keys:
    if ".".join(key.split(".")[-2:]) in tf_keys:
      if checkpoint[key].ndim == 2:
        checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def convert_ldm_unet_checkpoint(v2, checkpoint, config):
  """
  Takes a state dict and a config, and returns a converted checkpoint.
  """

  # extract state_dict for UNet
  unet_state_dict = {}
  unet_key = "model.diffusion_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(unet_key):
      unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

  new_checkpoint = {}

  new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
  new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
  new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
  new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

  new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
  new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

  new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
  new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
  new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
  new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

  # Retrieves the keys for the input blocks only
  num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
  input_blocks = {
      layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}." in key]
      for layer_id in range(num_input_blocks)
  }

  # Retrieves the keys for the middle blocks only
  num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
  middle_blocks = {
      layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}." in key]
      for layer_id in range(num_middle_blocks)
  }

  # Retrieves the keys for the output blocks only
  num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
  output_blocks = {
      layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}." in key]
      for layer_id in range(num_output_blocks)
  }

  for i in range(1, num_input_blocks):
    block_id = (i - 1) // (config["layers_per_block"] + 1)
    layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

    resnets = [
        key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
    ]
    attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

    if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.weight"
      )
      new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
          f"input_blocks.{i}.0.op.bias"
      )

    paths = renew_resnet_paths(resnets)
    meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
    assign_to_checkpoint(
        paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    if len(attentions):
      paths = renew_attention_paths(attentions)
      meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

  resnet_0 = middle_blocks[0]
  attentions = middle_blocks[1]
  resnet_1 = middle_blocks[2]

  resnet_0_paths = renew_resnet_paths(resnet_0)
  assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

  resnet_1_paths = renew_resnet_paths(resnet_1)
  assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

  attentions_paths = renew_attention_paths(attentions)
  meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(
      attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
  )

  for i in range(num_output_blocks):
    block_id = i // (config["layers_per_block"] + 1)
    layer_in_block_id = i % (config["layers_per_block"] + 1)
    output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
    output_block_list = {}

    for layer in output_block_layers:
      layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
      if layer_id in output_block_list:
        output_block_list[layer_id].append(layer_name)
      else:
        output_block_list[layer_id] = [layer_name]

    if len(output_block_list) > 1:
      resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
      attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

      resnet_0_paths = renew_resnet_paths(resnets)
      paths = renew_resnet_paths(resnets)

      meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
      assign_to_checkpoint(
          paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
      )

      # オリジナル：
      # if ["conv.weight", "conv.bias"] in output_block_list.values():
      #   index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])

      # biasとweightの順番に依存しないようにする：もっといいやり方がありそうだが
      for l in output_block_list.values():
        l.sort()

      if ["conv.bias", "conv.weight"] in output_block_list.values():
        index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.bias"
        ]
        new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
            f"output_blocks.{i}.{index}.conv.weight"
        ]

        # Clear attentions as they have been attributed above.
        if len(attentions) == 2:
          attentions = []

      if len(attentions):
        paths = renew_attention_paths(attentions)
        meta_path = {
            "old": f"output_blocks.{i}.1",
            "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
        }
        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )
    else:
      resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
      for path in resnet_0_paths:
        old_path = ".".join(["output_blocks", str(i), path["old"]])
        new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

        new_checkpoint[new_path] = unet_state_dict[old_path]

  # SDのv2では1*1のconv2dがlinearに変わっているので、linear->convに変換する
  if v2:
    linear_transformer_to_conv(new_checkpoint)

  return new_checkpoint


def convert_ldm_vae_checkpoint(checkpoint, config):
  # extract state dict for VAE
  vae_state_dict = {}
  vae_key = "first_stage_model."
  keys = list(checkpoint.keys())
  for key in keys:
    if key.startswith(vae_key):
      vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)
  # if len(vae_state_dict) == 0:
  #   # 渡されたcheckpointは.ckptから読み込んだcheckpointではなくvaeのstate_dict
  #   vae_state_dict = checkpoint

  new_checkpoint = {}

  new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
  new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
  new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
  new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
  new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
  new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

  new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
  new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
  new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
  new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
  new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
  new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

  new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
  new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
  new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
  new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

  # Retrieves the keys for the encoder down blocks only
  num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
  down_blocks = {
      layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
  }

  # Retrieves the keys for the decoder up blocks only
  num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
  up_blocks = {
      layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
  }

  for i in range(num_down_blocks):
    resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

    if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.weight"
      )
      new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
          f"encoder.down.{i}.downsample.conv.bias"
      )

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)

  for i in range(num_up_blocks):
    block_id = num_up_blocks - 1 - i
    resnets = [
        key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
    ]

    if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.weight"
      ]
      new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
          f"decoder.up.{block_id}.upsample.conv.bias"
      ]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
  num_mid_res_blocks = 2
  for i in range(1, num_mid_res_blocks + 1):
    resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

    paths = renew_vae_resnet_paths(resnets)
    meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

  mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
  paths = renew_vae_attention_paths(mid_attentions)
  meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
  assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
  conv_attn_to_linear(new_checkpoint)
  return new_checkpoint


def create_unet_diffusers_config(v2):
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # unet_params = original_config.model.params.unet_config.params

  block_out_channels = [UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT]

  down_block_types = []
  resolution = 1
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnDownBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "DownBlock2D"
    down_block_types.append(block_type)
    if i != len(block_out_channels) - 1:
      resolution *= 2

  up_block_types = []
  for i in range(len(block_out_channels)):
    block_type = "CrossAttnUpBlock2D" if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else "UpBlock2D"
    up_block_types.append(block_type)
    resolution //= 2

  config = dict(
      sample_size=UNET_PARAMS_IMAGE_SIZE,
      in_channels=UNET_PARAMS_IN_CHANNELS,
      out_channels=UNET_PARAMS_OUT_CHANNELS,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
      cross_attention_dim=UNET_PARAMS_CONTEXT_DIM if not v2 else V2_UNET_PARAMS_CONTEXT_DIM,
      attention_head_dim=UNET_PARAMS_NUM_HEADS if not v2 else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
  )

  return config


def create_vae_diffusers_config():
  """
  Creates a config for the diffusers based on the config of the LDM model.
  """
  # vae_params = original_config.model.params.first_stage_config.params.ddconfig
  # _ = original_config.model.params.first_stage_config.params.embed_dim
  block_out_channels = [VAE_PARAMS_CH * mult for mult in VAE_PARAMS_CH_MULT]
  down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
  up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

  config = dict(
      sample_size=VAE_PARAMS_RESOLUTION,
      in_channels=VAE_PARAMS_IN_CHANNELS,
      out_channels=VAE_PARAMS_OUT_CH,
      down_block_types=tuple(down_block_types),
      up_block_types=tuple(up_block_types),
      block_out_channels=tuple(block_out_channels),
      latent_channels=VAE_PARAMS_Z_CHANNELS,
      layers_per_block=VAE_PARAMS_NUM_RES_BLOCKS,
  )
  return config


def convert_ldm_clip_checkpoint_v1(checkpoint):
  keys = list(checkpoint.keys())
  text_model_dict = {}
  for key in keys:
    if key.startswith("cond_stage_model.transformer"):
      text_model_dict[key[len("cond_stage_model.transformer."):]] = checkpoint[key]
  return text_model_dict


def convert_ldm_clip_checkpoint_v2(checkpoint, max_length):
  # 嫌になるくらい違うぞ！
  def convert_key(key):
    if not key.startswith("cond_stage_model"):
      return None

    # common conversion
    key = key.replace("cond_stage_model.model.transformer.", "text_model.encoder.")
    key = key.replace("cond_stage_model.model.", "text_model.")

    if "resblocks" in key:
      # resblocks conversion
      key = key.replace(".resblocks.", ".layers.")
      if ".ln_" in key:
        key = key.replace(".ln_", ".layer_norm")
      elif ".mlp." in key:
        key = key.replace(".c_fc.", ".fc1.")
        key = key.replace(".c_proj.", ".fc2.")
      elif '.attn.out_proj' in key:
        key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
      elif '.attn.in_proj' in key:
        key = None                  # 特殊なので後で処理する
      else:
        raise ValueError(f"unexpected key in SD: {key}")
    elif '.positional_embedding' in key:
      key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
    elif '.text_projection' in key:
      key = None    # 使われない???
    elif '.logit_scale' in key:
      key = None    # 使われない???
    elif '.token_embedding' in key:
      key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
    elif '.ln_final' in key:
      key = key.replace(".ln_final", ".final_layer_norm")
    return key

  keys = list(checkpoint.keys())
  new_sd = {}
  for key in keys:
    # remove resblocks 23
    if '.resblocks.23.' in key:
      continue
    new_key = convert_key(key)
    if new_key is None:
      continue
    new_sd[new_key] = checkpoint[key]

  # attnの変換
  for key in keys:
    if '.resblocks.23.' in key:
      continue
    if '.resblocks' in key and '.attn.in_proj_' in key:
      # 三つに分割
      values = torch.chunk(checkpoint[key], 3)

      key_suffix = ".weight" if "weight" in key else ".bias"
      key_pfx = key.replace("cond_stage_model.model.transformer.resblocks.", "text_model.encoder.layers.")
      key_pfx = key_pfx.replace("_weight", "")
      key_pfx = key_pfx.replace("_bias", "")
      key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
      new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
      new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
      new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

  # rename or add position_ids
  ANOTHER_POSITION_IDS_KEY = "text_model.encoder.text_model.embeddings.position_ids"
  if ANOTHER_POSITION_IDS_KEY in new_sd:
    # waifu diffusion v1.4
    position_ids = new_sd[ANOTHER_POSITION_IDS_KEY]
    del new_sd[ANOTHER_POSITION_IDS_KEY]
  else:
    position_ids = torch.Tensor([list(range(max_length))]).to(torch.int64)

  new_sd["text_model.embeddings.position_ids"] = position_ids
  return new_sd

def is_safetensors(path):
  return os.path.splitext(path)[1].lower() == '.safetensors'

def load_checkpoint_with_text_encoder_conversion(ckpt_path):
  # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
  TEXT_ENCODER_KEY_REPLACEMENTS = [
      ('cond_stage_model.transformer.embeddings.', 'cond_stage_model.transformer.text_model.embeddings.'),
      ('cond_stage_model.transformer.encoder.', 'cond_stage_model.transformer.text_model.encoder.'),
      ('cond_stage_model.transformer.final_layer_norm.', 'cond_stage_model.transformer.text_model.final_layer_norm.')
  ]

  state_dict = read_state_dict(ckpt_path)

  key_reps = []
  for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
    for key in state_dict.keys():
      if key.startswith(rep_from):
        new_key = rep_to + key[len(rep_from):]
        key_reps.append((key, new_key))

  for key, new_key in key_reps:
    state_dict[new_key] = state_dict[key]
    del state_dict[key]

  return state_dict

def to_half(sd):
    for key in sd.keys():
        if 'model' in key and sd[key].dtype == torch.float:
            sd[key] = sd[key].half()
    return sd

def savemodel(state_dict,currentmodel,fname,savesets,model_a,metadata={}):
    from modules import sd_models,shared
    if "fp16" in savesets: 
        state_dict = to_half(state_dict)
        pre = ".fp16"
    else:pre = ""
    ext = ".safetensors" if "safetensors" in savesets else ".ckpt"

    # is it a inpainting or instruct-pix2pix2 model?
    if "model.diffusion_model.input_blocks.0.0.weight" in state_dict.keys():
        shape = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape
        if shape[1] == 9:
            pre += "-inpainting"
        if shape[1] == 8:
            pre += "-instruct-pix2pix"

    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
    model_a_path= checkpoint_info.filename
    modeldir = os.path.split(model_a_path)[0]

    if not fname or fname == "":
        fname = currentmodel.replace(" ","").replace(",","_").replace("(","_").replace(")","_")+pre+ext
        if fname[0]=="_":fname = fname[1:]
    else:
        fname = fname if ext in fname else fname +pre+ext

    fname = os.path.join(modeldir, fname)
    fname = fname.replace("ProgramFiles_x86_","Program Files (x86)")

    if len(fname) > 255:
       fname.replace(ext,"")
       fname=fname[:240]+ext

    # check if output file already exists
    if os.path.isfile(fname) and not "overwrite" in savesets:
        _err_msg = f"Output file ({fname}) existed and was not saved]"
        print(_err_msg)
        return _err_msg

    print("Saving...")
    try:
      if ext == ".safetensors":
          safetensors.torch.save_file(state_dict, fname, metadata=metadata)
      else:
          torch.save(state_dict, fname)
    except Exception as e:
      print(f"ERROR: Couldn't saved:{fname},ERROR is {e}")
      return f"ERROR: Couldn't saved:{fname},ERROR is {e}"
    print("Done!")
    return "Merged model saved in "+fname

def filenamecutter(name,model_a = False):
    from modules import sd_models
    if name =="" or name ==[]: return
    checkpoint_info = sd_models.get_closet_checkpoint_match(name)
    name= os.path.splitext(checkpoint_info.filename)[0]

    if not model_a:
        name = os.path.basename(name)
    return name

# TODO dtype指定の動作が怪しいので確認する text_encoderを指定形式で作れるか未確認
def load_models_from_stable_diffusion_checkpoint(v2, ckpt_path, dtype=None):
  import diffusers
  print("diffusers version : ",diffusers.__version__)

  state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)
  if dtype is not None:
    for k, v in state_dict.items():
      if type(v) is torch.Tensor:
        state_dict[k] = v.to(dtype)

  # Convert the UNet2DConditionModel model.
  unet_config = create_unet_diffusers_config(v2)
  converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)

  unet = diffusers.UNet2DConditionModel(**unet_config)
  info = unet.load_state_dict(converted_unet_checkpoint)
  print("loading u-net:", info)

  # Convert the VAE model.
  vae_config = create_vae_diffusers_config()
  converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

  vae = diffusers.AutoencoderKL(**vae_config)
  info = vae.load_state_dict(converted_vae_checkpoint)
  print("loading vae:", info)

  # convert text_model
  if v2:
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(state_dict, 77)
    cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=23,
        num_attention_heads=16,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=512,
        torch_dtype="float32",
        transformers_version="4.25.0.dev0",
    )
    text_model = CLIPTextModel._from_config(cfg)
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
  else:
    converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    info = text_model.load_state_dict(converted_text_encoder_checkpoint)
  print("loading text encoder:", info)

  return text_model, vae, unet, info

import torch
from modules import devices

module_in_gpu = None
cpu = torch.device("cpu")

import modules.sd_models as msd

sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight = 'conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight = 'conditioner.embedders.0.model.ln_final.weight'

def usemodel(checkpoint_info=None, already_loaded_state_dict=None):
  with threading.Lock():
    msd.load_model(checkpoint_info,already_loaded_state_dict)
  torch.cuda.empty_cache()

def load_model_weights(model, checkpoint_info: msd.CheckpointInfo, state_dict, timer):
    sd_model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title

    if state_dict is None:
        state_dict = msd.get_checkpoint_state_dict(checkpoint_info, timer)

    model.is_sdxl = hasattr(model, 'conditioner')
    model.is_sd2 = not model.is_sdxl and hasattr(model.cond_stage_model, 'model')
    model.is_sd1 = not model.is_sdxl and not model.is_sd2

    if model.is_sdxl:
        sd_models_xl.extend_sdxl(model)

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    timer.record("apply weights to model")

    if shared.cmd_opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("apply channels_last")

    if not shared.cmd_opts.no_half:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.cmd_opts.no_half_vae:
            model.first_stage_model = None
        # with --upcast-sampling, don't convert the depth model weights to float16
        if hasattr(shared.cmd_opts,"upcast_sampling"):
          if shared.cmd_opts.upcast_sampling and depth_model:
              model.depth_model = None

        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

        timer.record("apply half()")

    devices.dtype_unet = torch.float16 if model.is_sdxl and not shared.cmd_opts.no_half else model.model.diffusion_model.dtype
    if hasattr(shared.cmd_opts,"upcast_sampling"):
      devices.unet_needs_upcast = shared.cmd_opts.upcast_sampling and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    model.first_stage_model.to(devices.dtype_vae)
    model.has_accelerate = False
    timer.record("apply dtype to VAE")

    model.sd_model_hash = sd_model_hash
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device)  # fix for training

    def setvae():
        msd.sd_vae.delete_base_vae()
        msd.sd_vae.clear_loaded_vae()
        vae_file, vae_source = msd.sd_vae.resolve_vae(checkpoint_info.filename)
        msd.sd_vae.load_vae(model, vae_file, vae_source)
        timer.record("load VAE")
    
    try:
      setvae()
    except:
      pass

def get_empty_cond(sd_model):
    if hasattr(sd_model, 'conditioner'):
        d = sd_model.get_learned_conditioning([""])
        return d['crossattn']
    else:
        return sd_model.cond_stage_model([""])
