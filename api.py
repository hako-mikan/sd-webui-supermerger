"""API module for FastAPI"""
import requests
from typing import Callable, Dict, Optional
from threading import Lock
from secrets import compare_digest
import asyncio
from collections import defaultdict
from hashlib import sha256
import string
from random import choices

from modules import shared  # pylint: disable=import-error
from modules.api.api import decode_base64_to_image  # pylint: disable=E0401
from modules.call_queue import queue_lock  # pylint: disable=import-error
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import api_models as models
from scripts.mergers import pluslora

from fastapi import File, UploadFile, Form
from typing import Annotated
import shutil
from modules.progress import create_task_id, add_task_to_queue, start_task, finish_task, current_task
from time import sleep

class Api:
    """Api class for FastAPI"""

    def __init__(
        self, app: FastAPI, qlock: Lock, prefix: Optional[str] = None
    ) -> None:
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.app = app
        self.queue: Dict[str, asyncio.Queue] = {}
        self.res: Dict[str, Dict[str, Dict[str, float]]] = \
            defaultdict(dict)
        self.queue_lock = qlock
        self.tasks: Dict[str, asyncio.Task] = {}

        self.runner: Optional[asyncio.Task] = None
        self.prefix = prefix
        self.running_batches: Dict[str, Dict[str, float]] = \
            defaultdict(lambda: defaultdict(int))

        # self.add_api_route(
        #     'interrogate',
        #     self.endpoint_interrogate,
        #     methods=['POST'],
        #     response_model=models.TaggerInterrogateResponse
        # )

        # self.add_api_route(
        #     'interrogators',
        #     self.endpoint_interrogators,
        #     methods=['GET'],
        #     response_model=models.TaggerInterrogatorsResponse
        # )

        self.add_api_route(
            'unload-interrogators',
            self.endpoint_unload_interrogators,
            methods=['POST'],
            response_model=str,
        )

        self.add_api_route(
            'merge-lora',
            self.merge_lora_api,
            methods=['POST'],
            response_model=models.MergeLoraResponse,
        )

        self.add_api_route(
            'upload-lora',
            self.upload_lora_api,
            methods=['POST'],
            response_model=models.UploadLoraResponse,
        )

        self.add_api_route(
            'upload-lora-merge-checkpoint',
            self.upload_lora_and_merge_lora_to_checkpoint,
            methods=['POST'],
            response_model=models.UploadLoraMergeLoraResponse,
        )

    async def add_to_queue(self, m, q, n='', i=None, t=0.0) -> Dict[
        str, Dict[str, float]
    ]:
        if m not in self.queue:
            self.queue[m] = asyncio.Queue()
        #  loop = asyncio.get_running_loop()
        #  asyncio.run_coroutine_threadsafe(
        task = asyncio.create_task(self.queue[m].put((q, n, i, t)))
        #  , loop)

        if self.runner is None:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(self.batch_process(), loop=loop)
        await task
        return await self.tasks[q+"\t"+n]

    async def do_queued_interrogation(self, m, q, n, i, t) -> Dict[
        str, Dict[str, float]
    ]:
        self.running_batches[m][q] += 1.0
        # queue and name empty to process, not queue
        res = self.endpoint_interrogate(
            models.TaggerInterrogateRequest(
                image=i,
                model=m,
                threshold=t,
                name_in_queue='',
                queue=''
            )
        )
        self.res[q][n] = res.caption["tag"]
        for k, v in res.caption["rating"].items():
            self.res[q][n]["rating:"+k] = v
        return self.running_batches

    async def finish_queue(self, m, q) -> Dict[str, Dict[str, float]]:
        if q in self.running_batches[m]:
            del self.running_batches[m][q]
        if q in self.res:
            return self.res.pop(q)
        return self.running_batches

    async def batch_process(self) -> None:
        #  loop = asyncio.get_running_loop()
        while len(self.queue) > 0:
            for m in self.queue:
                # if zero the queue might just be pending
                while True:
                    try:
                        #  q, n, i, t = asyncio.run_coroutine_threadsafe(
                        #  self.queue[m].get_nowait(), loop).result()
                        q, n, i, t = self.queue[m].get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    self.tasks[q+"\t"+n] = asyncio.create_task(
                        self.do_queued_interrogation(m, q, n, i, t) if n != ""
                        else self.finish_queue(m, q)
                    )

            for model in self.running_batches:
                if len(self.running_batches[model]) == 0:
                    del self.queue[model]
            else:
                await asyncio.sleep(0.1)

        self.running_batches.clear()
        self.runner = None

    def auth(self, creds: Optional[HTTPBasicCredentials] = None):
        if creds is None:
            creds = Depends(HTTPBasic())
        if creds.username in self.credentials:
            if compare_digest(creds.password,
                              self.credentials[creds.username]):
                return True

        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Basic"
            })

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        if self.prefix:
            path = f'{self.prefix}/{path}'

        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[
                Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    async def queue_interrogation(self, m, q, n='', i=None, t=0.0) -> Dict[
        str, Dict[str, float]
    ]:
        """ queue an interrogation, or add to batch """
        if n == '':
            task = asyncio.create_task(self.add_to_queue(m, q))
        else:
            if n == '<sha256>':
                n = sha256(i).hexdigest()
                if n in self.res[q]:
                    return self.running_batches
            elif n in self.res[q]:
                # clobber name if it's already in the queue
                j = 0
                while f'{n}#{j}' in self.res[q]:
                    j += 1
                n = f'{n}#{j}'
            self.res[q][n] = {}
            # add image to queue
            task = asyncio.create_task(self.add_to_queue(m, q, n, i, t))
        return await task

    def endpoint_unload_interrogators(self):
        unloaded_models = 0

        return f"Successfully unload {unloaded_models} model(s)"

    def merge_lora(self, request: models.MergeLoraRequest) -> str:
        """Merge Lora"""
        try:
            # comment:

            request.loraratios = "\
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

            request.lnames = f"{request.lnames}"
            data = request

            res = pluslora.pluslora(
                loraratios=data.loraratios,
                calc_precision=data.calc_precision,
                device=data.device,
                lnames=data.lnames,
                metasets=data.metasets,
                model=data.model,
                output=data.output,
                save_precision=data.save_precision,
                settings=[]
            )

            return res

        except Exception as e:
            raise e
        # end try

    def referesh_loras_request(self):
        """Refresh Loras"""
        try:
            # comment:

            res = requests.post("http://localhost:7860/sdapi/v1/refresh-loras")

            return res

        except Exception as e:
            raise e
        # end try

    def merge_lora_api(self, request: models.MergeLoraRequest):
        """Merge Lora"""
        try:
            # comment:

            res = self.merge_lora(request)

            return models.MergeLoraResponse(checkpoint_merged_path=res)

        except Exception as e:
            raise e
        # end try

    def upload_file(self, file: UploadFile):
        try:
            # save lora file to disk
            file_location = f"models/Lora/{file.filename}"
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
            message = f'{file.filename} saved at {file_location}'

            return message
        except Exception as e:
            raise e
        # end try

    def upload_lora_api(self, lora_file: UploadFile):
        """Upload Lora"""
        try:
            # comment:
            message = self.upload_file(lora_file)
            self.referesh_loras_request()

            return models.UploadLoraResponse(message=message)
        except Exception as e:
            raise e
        # end try

    def upload_lora_and_merge_lora_to_checkpoint(self, lora_file: UploadFile, merge_request: models.UploadLoraMergeLoraRequest = Depends()):
        """Upload Lora and merge Lora to checkpoint"""
        try:

            task_id = create_task_id("txt2img")
            print("Task merge ID:   ", task_id)
            add_task_to_queue(task_id)
            # comment:

            print("Merge Request:   ", merge_request)
            lora_file_name = lora_file.filename.split(".")[0]

            with self.queue_lock:

                try:
                    shared.state.begin(job="scripts_txt2img")
                    start_task(task_id)

                    upload_res = self.upload_file(lora_file)
                    print("Uploaded file successfully:   ", upload_res)
                    self.referesh_loras_request()

                    # merge lora
                    merge_request.lnames = f"{lora_file_name}:0.8"

                    print("Started to merge lora")
                    merged_res = self.merge_lora(merge_request)

                    message = f'Upload and merge lora <{lora_file.filename}> to checkpoint <{merge_request.model}> successfully.'

                    checkpoint_merged_name = merged_res.split("/")[-1]
                finally:
                    shared.state.end()
                    shared.total_tqdm.clear()
                    finish_task(task_id)

     
            

            return models.UploadLoraMergeLoraResponse(message=message, checkpoint_merged_name=checkpoint_merged_name)
        except Exception as e:
            raise e
        finally:
            print("Finish task")
            finish_task(task_id)
        # end try


def on_app_started(_, app: FastAPI):
    Api(app, queue_lock, '/supermerger/v1')
