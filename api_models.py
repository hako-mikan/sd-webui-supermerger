"""Purpose: Pydantic models for the API."""
from typing import List, Dict

from modules.api import models as sd_models  # pylint: disable=E0401
from pydantic import BaseModel, Field

from fastapi import File, UploadFile, Form
from typing import Annotated


class TaggerInterrogateRequest(sd_models.InterrogateRequest):
    """Interrogate request model"""
    model: str = Field(
        title='Model',
        description='The interrogate model used.',
    )
    threshold: float = Field(
        title='Threshold',
        description='The threshold used for the interrogate model.',
        default=0.0,
    )
    queue: str = Field(
        title='Queue',
        description='name of queue; leave empty for single response',
        default='',
    )
    name_in_queue: str = Field(
        title='Name',
        description='name to queue image as or use <sha256>. leave empty to '
                    'retrieve the final response',
        default='',
    )


class TaggerInterrogateResponse(BaseModel):
    """Interrogate response model"""
    caption: Dict[str, Dict[str, float]] = Field(
        title='Caption',
        description='The generated captions for the image.'
    )


class TaggerInterrogatorsResponse(BaseModel):
    """Interrogators response model"""
    models: List[str] = Field(
        title='Models',
        description=''
    )


class MergeLoraRequest(BaseModel):
    """Interrogate request model"""
    lnames: str = Field(
        title='Lnames',
        description='Lora name.',
        default='NgocYen001:1',
    )
    output: str = Field(
        title='Output name',
        description='Output name.',
        default='lora_output',
    )
    model: str = Field(
        title='model',
        description='Checkpoint model used for the generation.',
        default='dreamshaper_8.safetensors',
    )
    save_precision: str = Field(
        title='Save precision',
        description='Save precision.',
        default='fp16',
    )
    calc_precision: str = Field(
        title='Calc precision',
        description='Calc precision.',
        default='float'
    )
    metasets: str = Field(
        title='Metasets',
        description='Metasets.',
        default='create new',
    )
    device: str = Field(
        title='Device',
        description='Device.',
        default='cuda',
    )
    loraratios: str = Field(
        title='Loraratios',
        description='Loraratios.',
        default='',
    )


class MergeLoraResponse(BaseModel):
    """Interrogate request model"""
    checkpoint_merged_path: str = Field(
        title='Merged Checkpoint Path',
        description='Merged Checkpoin Path.',
        default='',
    )


class UploadLoraRequest(BaseModel):
    """Interrogate request model"""
    lora_file: UploadFile


class UploadLoraResponse(BaseModel):
    """Interrogate request model"""
    message: str = Field(
        title='Message',
        description='Message.',
        default='',
    )


class ErrorResponse(BaseModel):
    """Error response model"""
    message: str = Field(
        title='Message',
        description='Error message.',
        default='',
    )


class UploadLoraMergeLoraRequest(BaseModel):
    """Interrogate request model"""
    lnames: str = Field(
        title='Lnames',
        description='Lora name.',
        default='NgocYen001:1',
    )
    output: str = Field(
        title='Output name',
        description='Output name.',
        default='lora_output',
    )
    model: str = Field(
        title='model',
        description='Checkpoint model used for the generation.',
        default='dreamshaper_8.safetensors',
    )
    save_precision: str = Field(
        title='Save precision',
        description='Save precision.',
        default='fp16',
    )
    calc_precision: str = Field(
        title='Calc precision',
        description='Calc precision.',
        default='float'
    )
    metasets: str = Field(
        title='Metasets',
        description='Metasets.',
        default='create new',
    )
    device: str = Field(
        title='Device',
        description='Device.',
        default='cuda',
    )
    loraratios: str = Field(
        title='Loraratios',
        description='Loraratios.',
        default='',
    )
    rate: float = Field(
        title='Rate',
        description='Rate.',
        default=0.7,
    )


class UploadLoraMergeLoraResponse(BaseModel):

    message = Field(
        title='Message',
        description='Message.',
        default='',
    )

    checkpoint_merged_name: str = Field(
        title='Checkpoint Merged Name',
        description='Checkpoint Merged Name.',
        default='',
    )
