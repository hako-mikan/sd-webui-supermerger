import gradio as gr
import scripts.mergers.components as components
from scripts.mergers.mergers import smergegen, simggen
from scripts.mergers.xyplot import numanager
from scripts.mergers.pluslora import frompromptf
from modules import scripts, script_callbacks

class GenParamGetter(scripts.Script):
    txt2img_gen_button = None
    img2img_gen_button = None

    events_assigned = False

    def title(self):
        return "Super Marger Generation Parameter Getter"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_wanted_params(params,wanted):
        output = []
        for target in wanted:
            if target is None:
                output.append(params[0])
                continue
            for param in params:
                if hasattr(param,"label"):
                    if param.label == target:
                        output.append(param)
        return output

    def after_component(self, component: gr.components.Component, **_kwargs):
        """Find generate button"""
        if component.elem_id == "txt2img_generate":
            GenParamGetter.txt2img_gen_button = component
        elif  component.elem_id == "img2img_generate":
            GenParamGetter.img2img_gen_button = component

    def get_components_by_ids(root: gr.Blocks, ids: list[int]):
        components: list[gr.Blocks] = []

        if root._id in ids:
            components.append(root)
            ids = [_id for _id in ids if _id != root._id]
        
        if hasattr(root,"children"):
            for block in root.children:
                components.extend(GenParamGetter.get_components_by_ids(block, ids))
        return components
    
    def compare_components_with_ids(components: list[gr.Blocks], ids: list[int]):
        return len(components) == len(ids) and all(component._id == _id for component, _id in zip(components, ids))

    def get_params_components(demo: gr.Blocks, app):
        for _id, _is_txt2img in zip([GenParamGetter.txt2img_gen_button._id, GenParamGetter.img2img_gen_button._id], [True, False]):
            dependencies: list[dict] = [x for x in demo.dependencies if x["trigger"] == "click" and _id in x["targets"]]
            dependency: dict = None
            cnet_dependency: dict = None
            UiControlNetUnit = None
            for d in dependencies:
                if len(d["outputs"]) == 1:
                    outputs = GenParamGetter.get_components_by_ids(demo, d["outputs"])
                    output = outputs[0]
                    if (
                        isinstance(output, gr.State)
                        and type(output.value).__name__ == "UiControlNetUnit"
                    ):
                        cnet_dependency = d
                        UiControlNetUnit = type(output.value)

                elif len(d["outputs"]) == 4:
                    dependency = d

            params = [params for params in demo.fns if GenParamGetter.compare_components_with_ids(params.inputs, dependency["inputs"])]

            from pprint import pprint

            if _is_txt2img:
                components.paramsnames = [x.label if hasattr(x,"label") else "None" for x in params[0].inputs]

            if _is_txt2img:
                components.txt2img_params = params[0].inputs 
            else:
                components.img2img_params = params[0].inputs
        
        if not GenParamGetter.events_assigned:
            with demo:
                components.merge.click(
                    fn=smergegen,
                    inputs=[*components.msettings,components.esettings1,*components.genparams,*components.hiresfix,*components.lucks,components.currentmodel,components.dfalse,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel]
                )

                components.mergeandgen.click(
                    fn=smergegen,
                    inputs=[*components.msettings,components.esettings1,*components.genparams,*components.hiresfix,*components.lucks,components.currentmodel,components.dtrue,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal]
                )

                components.gen.click(
                    fn=simggen,
                    inputs=[*components.genparams,*components.hiresfix,components.currentmodel,components.id_sets,gr.Textbox(value="No id",visible=False),*components.txt2img_params],
                    outputs=[*components.imagegal],
                )

                components.s_reserve.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="reserve",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.numaframe]
                )

                components.s_reserve1.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="reserve",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.numaframe]
                )

                components.gengrid.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="normal",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal],
                )

                components.s_startreserve.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value=" ",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal],
                )

                components.rand_merge.click(
                    fn=numanager,
                    inputs=[gr.Textbox(value="random",visible=False),*components.xysettings,*components.msettings,*components.genparams,*components.hiresfix,*components.lucks,*components.txt2img_params],
                    outputs=[components.submit_result,components.currentmodel,*components.imagegal],
                )

                components.frompromptb.click(
                    fn=frompromptf,
                    inputs=[*components.txt2img_params],
                    outputs=components.sml_loranames,
                )
            GenParamGetter.events_assigned = True

if __package__ == "GenParamGetter":
    script_callbacks.on_app_started(GenParamGetter.get_params_components)
