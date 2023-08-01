import gradio as gr
from scripts.mergers.mergers import simggen, smergegen
from scripts.mergers.numanager import numanager
from modules import scripts as modules_scripts
from modules import script_callbacks

class Components:
    merge = None
    mergeandgen = None
    gen = None
    s_reserve = None
    s_reserve1 = None
    gengrid = None
    s_startreserve = None
    rand_merge = None

    msettings = None
    esettings1 = None
    genparams = None
    lucks = None
    currentmodel = None
    dfalse = None
    dtrue = None
    id_sets = None
    xysettings = None

    submit_result = None
    imagegal = None
    numaframe = None

class GenParamsGetter(modules_scripts.Script):
    txt2img_gen_button = None
    img2img_gen_button = None

    txt2img_params = []
    img2img_params = []

    event_assigned = False

    def title(self):
        return "Super Marger Generation Parameter Getter"
    
    def show(self, is_img2img):
        return modules_scripts.AlwaysVisible

    def after_component(self, component: gr.components.Component, **_kwargs):
        """Find generate button"""
        if component.elem_id == "txt2img_generate":
            GenParamsGetter.txt2img_gen_button = component
        elif  component.elem_id == "img2img_generate":
            GenParamsGetter.img2img_gen_button = component

    def get_components_by_ids(root: gr.Blocks, ids: list[int]):
        components: list[gr.Blocks] = []

        if root._id in ids:
            components.append(root)
            ids = [_id for _id in ids if _id != root._id]

        if isinstance(root, gr.components.BlockContext):
            for block in root.children:
                components.extend(GenParamsGetter.get_components_by_ids(block, ids))

        return components
    
    def compare_components_with_ids(components: list[gr.Blocks], ids: list[int]):
        return len(components) == len(ids) and all(component._id == _id for component, _id in zip(components, ids))

    def get_params_components(demo: gr.Blocks, app):
        for _id, _is_txt2img in zip([GenParamsGetter.txt2img_gen_button._id, GenParamsGetter.img2img_gen_button._id], [True, False]):
            dependencies: list[dict] = [x for x in demo.dependencies if x["trigger"] == "click" and _id in x["targets"]]
            dependency: dict = None
            cnet_dependency: dict = None
            UiControlNetUnit = None
            for d in dependencies:
                if len(d["outputs"]) == 1:
                    outputs = outputs = GenParamsGetter.get_components_by_ids(demo, d["outputs"])
                    output = outputs[0]
                    if (
                        isinstance(output, gr.State)
                        and type(output.value).__name__ == "UiControlNetUnit"
                    ):
                        cnet_dependency = d
                        UiControlNetUnit = type(output.value)

                elif len(d["outputs"]) == 4:
                    dependency = d

            params = [params for params in demo.fns if GenParamsGetter.compare_components_with_ids(params.inputs, dependency["inputs"])]

            if _is_txt2img:
                GenParamsGetter.txt2img_params = params[0].inputs
                GenParams.txt2img_params = params[0].inputs
            elif not _is_txt2img:
                GenParamsGetter.img2img_params = params[0].inputs
                GenParams.img2img_params = params[0].inputs
        
        if not GenParamsGetter.event_assigned:
            with demo:
                Components.merge.click(
                    fn=smergegen,
                    inputs=[*Components.msettings,Components.esettings1,*Components.genparams,*Components.lucks,Components.currentmodel,Components.dfalse,*GenParamsGetter.txt2img_params],
                    outputs=[Components.submit_result,Components.currentmodel]
                )

                Components.mergeandgen.click(
                    fn=smergegen,
                    inputs=[*Components.msettings,Components.esettings1,*Components.genparams,*Components.lucks,Components.currentmodel,Components.dtrue,*GenParamsGetter.txt2img_params],
                    outputs=[Components.submit_result,Components.currentmodel,*Components.imagegal]
                )

                Components.gen.click(
                    fn=simggen,
                    inputs=[*Components.genparams,*GenParamsGetter.txt2img_params,Components.currentmodel,Components.id_sets],
                    outputs=[*Components.imagegal],
                )

                Components.s_reserve.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="reserve",visible=False),*Components.xysettings,*Components.msettings,*Components.genparams,*Components.lucks,*GenParamsGetter.txt2img_params],
                    outputs=[Components.numaframe]
                )

                Components.s_reserve1.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="reserve",visible=False),*Components.xysettings,*Components.msettings,*Components.genparams,*Components.lucks,*GenParamsGetter.txt2img_params],
                    outputs=[Components.numaframe]
                )

                Components.gengrid.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="normal",visible=False),*Components.xysettings,*Components.msettings,*Components.genparams,*Components.lucks,*GenParamsGetter.txt2img_params],
                    outputs=[Components.submit_result,Components.currentmodel,*Components.imagegal],
                )

                Components.s_startreserve.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value=" ",visible=False),*Components.xysettings,*Components.msettings,*Components.genparams,*Components.lucks,*GenParamsGetter.txt2img_params],
                    outputs=[Components.submit_result,Components.currentmodel,*Components.imagegal],
                )

                Components.rand_merge.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="random",visible=False),*Components.xysettings,*Components.msettings,*Components.genparams,*Components.lucks,*GenParamsGetter.txt2img_params],
                    outputs=[Components.submit_result,Components.currentmodel,*Components.imagegal],
                )
            GenParamsGetter.event_assigned = True

class GenParams:
    txt2img_params = []
    img2img_params = []

script_callbacks.on_app_started(GenParamsGetter.get_params_components)