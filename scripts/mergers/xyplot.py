import random
import gc
import re
from tracemalloc import Statistic
import cv2
import numpy as np
import os
import copy
import csv
import textwrap
from PIL import Image, ImageFont, ImageDraw, ImageColor, PngImagePlugin
from modules import images, sd_models, devices
from modules.sd_models import load_model
from modules.shared import opts
from scripts.mergers.mergers import TYPES,FINETUNEX,EXCLUDE_CHOICES,smerge,simggen,filenamecutter,draw_origin,wpreseter,savestatics,cachedealer,get_font
from scripts.mergers.model_util import savemodel
from scripts.mergers.bcolors import bcolors

hear = True
hearm = False

state_mergen = False

RANDCOL = 10
NUM = "num"
RAND = "random"

numadepth = []

def freezetime():
    global state_mergen
    state_mergen = True

def numanager(startmode,xtype,xmen,ytype,ymen,ztype,zmen,esettings,
                    weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,calcmode,
                    useblocks,custom_name,save_sets,id_sets,wpresets,deep,fine,bake_in_vae,optv,inex,ex_blocks,ex_elems,
                    s_prompt,s_nprompt,s_steps,s_sampler,s_cfg,s_seed,s_w,s_h,s_batch_size,
                    genoptions,s_hrupscaler,s_hr2ndsteps,s_denois_str,s_hr_scale,
                    lmode,lsets,llimits_u,llimits_l,lseed,lserial,lcustom,lround,
                    *txt2imgparams):

    cachedealer(True)

    global numadepth
    grids = []
    sep = "|"

    if RAND in startmode:
        if "off" in lmode:
            cachedealer(False)
            return "Random mode is off",*[None]*5
        if lserial > 0 : lseed = -1
        useblocks = True
    if RAND in startmode or TYPES.index(RAND) in [xtype,ytype,ztype]:
        xtype,xmen,ytype,ymen,weights_a,weights_b = crazyslot(lmode,lsets,llimits_u,llimits_l,lseed,lserial,lcustom,xtype,xmen,ytype,ymen,weights_a,weights_b,startmode)

    lucks = {"on":startmode == RAND, "mode":lmode,"set":lsets,"upp":llimits_u,"low":llimits_l,"seed":lseed,"num":lserial,"cust":lcustom,"round":int(lround)}
    gensets_s = [s_prompt,s_nprompt,s_steps,s_sampler,s_cfg,s_seed,s_w,s_h,s_batch_size,genoptions,s_hrupscaler,s_hr2ndsteps,s_denois_str,s_hr_scale]

    allsets = [xtype,xmen,ytype,ymen,ztype,zmen,esettings,
                  weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,calcmode,
                  useblocks,custom_name,save_sets,id_sets,wpresets,deep,fine,bake_in_vae,optv,inex,ex_blocks,ex_elems,
                  txt2imgparams,gensets_s,lucks]

    from scripts.mergers.components import paramsnames
    seed = txt2imgparams[paramsnames.index("Seed" if "Seed" in paramsnames else "Initial seed")]

    if RAND not in startmode:
        if sep in xmen: allsets = separator(allsets,1,sep,xmen,seed,startmode)
        if sep in ymen: allsets = separator(allsets,3,sep,ymen,seed,startmode)
        if sep in zmen: allsets = separator(allsets,5,sep,zmen,seed,startmode)

    if "reserve" in startmode :
        cachedealer(False)
        return numaker(allsets)

    if "normal" or RAND in startmode:
        result,currentmodel,xyimage,a,b,c= sgenxyplot(*allsets)
        if xyimage is not None:grids = xyimage
        else:print(result)
    else:
        if numadepth ==[]:
            cachedealer(False)
            return "no reservation",*[None]*5
        result=currentmodel=xyimage=a=b=c = None

    while True:
        for i,row in enumerate(numadepth):
            if row[1] =="waiting":  
                numadepth[i][1] = "Operating"
                try:
                    result,currentmodel,xyimage,a,b,c = sgenxyplot(*row[2:])
                except Exception as e:
                    print(e)
                    numadepth[i][1] = "Error"
                else:
                    if xyimage is not None:
                        grids.append(xyimage[0])
                        numadepth[i][1] = "Finished"
                    else:
                        print(result)
                        numadepth[i][1] = "Error"
        wcounter = 0
        for row in numadepth:
            if row[1] != "waiting":
                wcounter += 1
        if wcounter == len(numadepth):
            break

    gc.collect()
    cachedealer(False)

    return result,currentmodel,grids,a,b,c

def separator(allsets,index,sep,men,seed,startmode):
    if seed =="-1": allsets[30] = str(random.randrange(4294967294))
    mens = men.split(sep)
    if "reserve" not in startmode:
        allsets[index] = mens[0]
        for men in mens[1:]:
            numaker([*allsets[0:index],men,*allsets[index+1:]])
    else:
        allsets[index] = mens[-1]
        for men in mens[:-1]:
            numaker([*allsets[0:index],men,*allsets[index+1:]])
    return allsets

def numaker(allsets):
    global numadepth
    numadepth.append([len(numadepth)+1,"waiting",*allsets])
    return numalistmaker(copy.deepcopy(numadepth))

def nulister(redel):
    global numadepth
    if redel == False:
        return numalistmaker(copy.deepcopy(numadepth))
    if redel ==-1:
        numadepth = []
    else:
        try:del numadepth[int(redel-1)]
        except Exception as e:print(e)
    return numalistmaker(copy.deepcopy(numadepth))

def numalistmaker(numa):
    if numa ==[]: return [["no data","",""],]
    for i,r in enumerate(numa):
        r[2] =  TYPES[int(r[2])]
        r[4] =  TYPES[int(r[4])]
        r[6] =  TYPES[int(r[6])]
        numa[i] = r[0:8]+r[11:14]+r[14:18]+r[9:11]
    return numa

def caster(news,hear):
    if hear: print(news)

def sgenxyplot(xtype,xmen,ytype,ymen,ztype,zmen,esettings,
                  weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,
                  calcmode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,fine,bake_in_vae,optv,inex,ex_blocks,ex_elems,
                  gensets,gensets_s,lucks):
    global hear
    esettings = " ".join(esettings)
    savestat = "savestat" in deep

    gensets = list(gensets)

    from scripts.mergers.components import paramsnames
    def g(wanted,wantedv=None,value = True):
        if wanted in paramsnames:return gensets[paramsnames.index(wanted)] if value else paramsnames.index(wanted)
        elif wantedv and wantedv in paramsnames:return gensets[paramsnames.index(wantedv)] if value else paramsnames.index(wantedv)
        else:return None if value else 0

    fine = fine.split(",") if fine else [0]*8

    deep_ori = deep
    ex_blocks_ori = ex_blocks.copy()
    #type[0:none,1:aplha,2:beta,3:seed,4:mbw,5:model_A,6:model_B,7:model_C,8:pinpoint 9:deep]
    xtype = TYPES[xtype]
    ytype = TYPES[ytype]
    ztype = TYPES[ztype]
    XYZ =xtype + ytype + ztype 
    mainmodel = mainmodeldealer(XYZ)
    #ALL
    if "ALL" in [xmen,ymen,zmen]:
        xmen,ymen,zmen = alldealer([xmen,ymen,zmen],[xtype,ytype,ztype])

    if ytype == "none": ymen = ""
    if ztype == "none": zmen = ""

    modes=["Weight" ,"Add" ,"Triple","Twice"]
    xs=ys=ys=0
    weights_a_in=weights_b_in="0"

    deepprint  = True if "print change" in esettings else False

    def castall(hear):
        if hear :print(f"xmen:{xmen}, ymen:{ymen},zmen:{zmen}, xtype:{xtype}, ytype:{ytype}, ztype:{ztype}\
        weights_a:{weights_a_in}, weights_b:{weights_b_in}, model_A:{model_a},model_B :{model_b}, model_C:{model_c}, alpha:{alpha},\
        beta :{beta}, mode:{mode}, blocks:{useblocks}")

    pinpoint = "pinpoint blocks" in XYZ and "alpha" in XYZ

    usebeta = modes[2] in mode or modes[3] in mode

    if "prompt" in XYZ: gensets_s[0] = ""
    if "seed" in XYZ: gensets_s[5] = 0

    #check and adjust format
    print(f"\n{bcolors.OKGREEN}XY plot start, mode:{mode}, X: {xtype}, Y: {ytype}, Z: {ztype} MBW: {useblocks}{bcolors.ENDC}")
    castall(hear)
    None5 = [None,None,None,None,None]
    if xmen =="": return "ERROR: parameter X is empty",*None5
    if ymen =="" and not ytype=="none":
        print("Parameter Y is empty, disable Y")
        ytype = "none"
    if zmen =="" and not ztype=="none":
        print("Parameter Z is empty, disable Z")
        ztype = "none"
    if model_a ==[] and "model_A" not in XYZ:return f"ERROR: model_A is not selected",*None5
    if model_b ==[] and "model_B" not in XYZ:return f"ERROR: model_B is not selected",*None5
    if model_c ==[] and usebeta and "model_C" not in XYZ:return "ERROR: model_C is not selected",*None5
    if xtype == ytype and not (xtype == "add elemental" or xtype == RAND): return "ERROR: same type selected for X,Y",*None5

    if useblocks:
        weights_a_in=wpreseter(weights_a,wpresets)
        weights_b_in=wpreseter(weights_b,wpresets)

    #for X only plot, use same seed
    if g("Seed") == -1: gensets[g("Seed",value=False)] = int(random.randrange(4294967294))

    #gensets :prompt:1,seed:11
    #gensets_s :prompt:0, seed:5

    #for XY plot, use same seed
    def dicedealer(zs):
        for i,z in enumerate(zs):
            if z =="-1": zs[i] = str(random.randrange(4294967294))
        print(f"the die was thrown : {zs}")

    #adjust parameters, alpha,beta,models,seed: list of single parameters, mbw(no beta):list of text,mbw(usebeta); list of pair text
    def adjuster(wmen,wtype,awtype,bwtype):
        if "mbw" in wtype or "prompt" in wtype or "adjust" == wtype:#men separated by newline
            ws = wmen.splitlines()
            caster(ws,hear)
            if "mbw alpha and beta" in wtype:
                ws = [ws[i:i+2] for i in range(0,len(ws),2)]
                caster(ws,hear)
        elif "elemental" in wtype:
            ws = wmen.split("\n\n")
        elif RAND in wtype:
            ws = [""] * int(wmen)
        else:
            if "pinpoint element" in wtype:
                wmen = wmen.replace("\n",",")
            if "effective" in wtype:
                wmen = ","+wmen
                wmen = wmen.replace("\n",",")
            ws = [w.strip() for w in wmen.split(',')]
            caster(ws,hear)
        if "alpha" in wtype and ("effective" in awtype or "effective" in bwtype):
            ws = [ws[0]]
        if "seed" in wtype:dicedealer(ws)
        if "alpha" == wtype or "beta" == wtype:
            ow = []
            for w in ws:
                try:
                    float(w)
                    ow.append(w)
                except:
                    pass
            ws = ow
        return ws

    xs = adjuster(xmen,xtype,ytype,ztype)
    ys = adjuster(ymen,ytype,xtype,ztype)
    zs = adjuster(zmen,ztype,xtype,ytype)

    #in case beta selected but mode is Weight sum or Add or Diff
    if ("beta" in XYZ) and (not usebeta and "tensor" not in calcmode):
        mode = modes[3]
        print(f"{bcolors.WARNING}{modes[3]} mode automatically selected){bcolors.ENDC}")

    #in case mbw or pinpoint selected but useblocks not chekced
    if "mbw" in XYZ and not useblocks:
        useblocks = True
        print(f"{bcolors.WARNING}MBW mode enabled{bcolors.ENDC}")

    xcount = ycount = zcount = 0
    allcount = len(xs)*len(ys)*len(zs) if not lucks["on"] else lucks["num"]

    #for STOP XY bottun
    flag = False
    global state_mergen
    state_mergen = False

    #type[0:none,1:aplha,2:beta,3:seed,4:mbw,5:model_A,6:model_B,7:model_C,8:pinpoint ]
    blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
    #format ,IN00 IN03,IN04-IN09,OUT4,OUT05
    def weightsdealer(xyzval: list,xyztype: list,weights: str, mode: str):
        caster(f"weights from : {weights}",hear)
        ww = xyzval[xyztype.index("pinpoint blocks")]
        wa = xyzval[xyztype.index(mode)]
        ww = [w.strip() for w in ww.split(' ')]
        weights_t = [w.strip() for w in weights.split(',')]
        if ww[0]!="NOT":
            flagger=[False]*26
            changer = True
        else:
            flagger=[True]*26
            changer = False
        for w in ww:
            if w =="NOT":continue
            if "-" in w:
                wt = [wt.strip() for wt in w.split('-')]
                if  blockid.index(wt[1]) > blockid.index(wt[0]):
                    flagger[blockid.index(wt[0]):blockid.index(wt[1])+1] = [changer]*(blockid.index(wt[1])-blockid.index(wt[0])+1)
                else:
                    flagger[blockid.index(wt[1]):blockid.index(wt[0])+1] = [changer]*(blockid.index(wt[0])-blockid.index(wt[1])+1)
            else:
                flagger[blockid.index(w)] =changer    
        for i,f in enumerate(flagger):
            if f:weights_t[i]=wa
        outext = ",".join(weights_t)
        caster(f"weights changed: {outext}",hear)
        return outext

    def abdealer(z):
        if " " in z:return z.split(" ")[0],z.split(" ")[1]
        return z,z

    def excluder(w):
        if "NOT" in w:
            outs = EXCLUDE_CHOICES.copy()
            w = w.split(" ")
            for x in w:
                if x in outs:
                    outs.remove(x)
            return outs
        else:
            return w.split(" ")

    def xydealer(w,wt,awt,bwt):
        wta = awt + bwt
        nonlocal alpha,beta,gensets,weights_a_in,weights_b_in,model_a,model_b,model_c,deep,calcmode,fine,inex,ex_blocks
        if "prompt" in wt:
            gensets[1] = w
            return
        if "pinpoint blocks" in wt or "pinpoint element" in wt or "effective" in wt:return
        if "mbw" in wt:
            def weightser(w):return w, w.split(',',1)[0]
            if "mbw alpha and beta" in wt:
                weights_a_in,alpha = weightser(wpreseter(w[0],wpresets))
                weights_b_in,beta = weightser(wpreseter(w[1],wpresets))
            elif "alpha" in wt:
                weights_a_in,alpha = weightser(wpreseter(w,wpresets))
            else:
                weights_b_in,beta = weightser(wpreseter(w,wpresets))
            return
        if "alpha and" in wt:
            alpha,beta = abdealer(w)
            return
        if "alpha" in wt and not ("pinpoint element" in wta or "effective" in wta or "pinpoint adjust" in wta):alpha = w
        if "beta" in wt: beta = w
        if "seed" in wt:
            gensets[g("Seed",value = False)] = int(w)
        if "model_A" in wt:model_a = w
        if "model_B" in wt:model_b = w
        if "model_C" in wt:model_c = w
        if "elemental" in wt:
            deep = deep  +","+ w if "add" in wt else w
        if "calcmode" in wt:calcmode = w
        if "adjust" == wt:fine = w
        if "clude" in wt:
            if "add" in wt:
                inex = wt.split(" ")[1].capitalize()
                ex_blocks = ex_blocks_ori + excluder(w)
            else:
                inex = wt.split(" ")[0].capitalize()
                ex_blocks = excluder(w)
    
    def elementdealer(xyzval,xyztype):
        t = "pinpoint element" if "pinpoint element" in xyztype else "effective"
        return str(xyzval[xyztype.index(t)]) + ":" + str(xyzval[xyztype.index("alpha")])

    def finedealer(fine,xyzval: list,xyztype: list):
        f = xyzval[xyztype.index("pinpoint adjust")]
        a = xyzval[xyztype.index("alpha")]
        fine_in = fine.copy()
        fine_in[FINETUNEX.index(f)] = a
        return fine_in

    # plot start

    xyzimage=[]
    stocker = Stocker()
    for z in zs:
        ycount = 0
        xyimage = []
        xydealer(z,ztype,xtype,ytype)
        for y in ys:
            deep = deep_ori
            xydealer(y,ytype,xtype,ztype)
            xcount = 0
            for x in xs:
                deepy = deep
                xyzv= [x,y,z]
                xyzt = [xtype,ytype,ztype]
                xydealer(x,xtype,ytype,ztype)
                if pinpoint and "alpha" in XYZ:
                    weights_a_in = weightsdealer(xyzv,xyzt,weights_a,"alpha")
                    weights_b_in = weights_b
                if pinpoint and "beta" in XYZ:
                    weights_b_in = weightsdealer(xyzv,xyzt,weights_b,"beta")
                    weights_a_in = weights_a
                if "pinpoint element" in XYZ or "effective" in XYZ:
                    deep_in = deep + "," + elementdealer(xyzv,xyzt) if deep else elementdealer(xyzv,xyzt)
                else:
                    deep_in = deep
                if "pinpoint adjust" in XYZ and "alpha" in XYZ:
                    fine_in = finedealer(fine,xyzv,xyzt)
                else:
                    fine_in = fine
                if type(fine_in) == list:fine_in = ",".join([str(x) for x in fine_in])

                print(f"{bcolors.OKGREEN}XY plot: X: {xtype}, {str(x)}, Y: {ytype}, {str(y)}, Z: {ztype}, {str(z)} ({len(xs)*len(ys)*zcount + ycount*len(xs) +xcount +1}/{allcount}){bcolors.ENDC}")
                
                if "stock" in esettings: stocker.check_alpha(useblocks,alpha,weights_a_in,calcmode)

                if (((xtype=="seed") or (xtype=="prompt")) and xcount > 0) or (stocker.now and stocker.stock is not None):
                    print(f"{bcolors.WARNING}Merge is skipped{bcolors.ENDC}")
                else:
                    _, currentmodel,modelid,theta_0, metadata =smerge(weights_a_in,weights_b_in, model_a,model_b,model_c, float(alpha),float(beta),mode,calcmode,
                                                                                        useblocks,"",save_sets,id_sets,False,deep_in,fine_in,bake_in_vae,optv,inex,ex_blocks,ex_elems,deepprint,lucks,main=mainmodel) 
                    checkpoint_info = sd_models.get_closet_checkpoint_match(model_a)
                    if "save model" in esettings:
                        savemodel(theta_0,currentmodel,custom_name,save_sets,metadata) 
                    
                    sd_models.model_data.__init__()
                    load_model(checkpoint_info, already_loaded_state_dict=theta_0)

                theta_0 = None
                del theta_0

                if xcount == 0: statid = modelid

                if stocker.now and stocker.stock is not None:
                    image_temp = stocker.stock
                    print("Stocked image used")
                else:
                    image_temp = simggen(*gensets_s,currentmodel,id_sets,modelid,*gensets)
                    if stocker.now and stocker.stock is None:
                        stocker.stock = image_temp

                gc.collect()
                devices.torch_gc()

                xyimage.append(image_temp[0][0])
                xcount+=1
                deep = deepy
                if state_mergen:
                    flag = True
                    break
            if lucks["on"] and (len(xs)*len(ys)*zcount + ycount*len(xs) +xcount +1) >= lucks["num"]: flag = True
            ycount+=1
            if flag:break

        if flag and ycount ==1:
            xs = xs[:xcount]
            ys = [ys[0],]
            print(f"stopped at x={xcount},y={ycount}")
        elif flag:
            ys = ys[:ycount]
            print(f"stopped at x={xcount},y={ycount}")
        
        (xs_t, ys_t) = (xs.copy(), ys.copy())

        xs,ys,zs = ajustlegend(xs,XYZ),ajustlegend(ys,XYZ),ajustlegend(zs,XYZ)

        if "mbw alpha and beta" in xtype: xs_t = [f"alpha:({x[0]}),beta({x[1]})" for x in xs ]
        if "mbw alpha and beta" in ytype: ys_t = [f"alpha:({y[0]}),beta({y[1]})" for y in ys ]

        xs_t[0]=xtype+" = "+xs_t[0] #draw X label
        if ytype!=TYPES[0] or "model" in ytype:ys_t[0]=ytype+" = "+str(ys[0])  #draw Y label

        if ys_t==[""]:ys_t = [" "]

        if "effective" in xtype or "effective" in ytype:
            xyimage,xs_t,ys_t = effectivechecker(xyimage,xs_t,ys_t,model_a,model_b,esettings)

        if not "grid" in esettings:
            if "swap XY" in esettings:
                xyimage, xs_t, ys_t = swapxy(xyimage, xs_t, ys_t)
            gridmodel= makegridmodelname(model_a, model_b,model_c, useblocks,mode,xtype,ytype,alpha,beta,weights_a,weights_b,usebeta)
            grid = smakegrid(xyimage,xs_t,ys_t,gridmodel,image_temp[4])
            xyimage.insert(0,grid)
        
        if savestat: savestatics(statid)

        zcount+=1
        xyzimage.append(xyimage[0])
        if flag: break

    state_mergen = False
    return "Finished",currentmodel,xyzimage,*image_temp[1:4]

def ajustlegend(ws,xyz):
    if "pinpoint adjust" in xyz:
        for i, s in enumerate(ws):
            if "COL1" == s:ws[i] = "COL1(Cyan-Red)"
            if "COL2" == s:ws[i] = "COL2(Magenta-Green)"
            if "COL3" == s:ws[i] = "COL3(Yellow-Blue)"
    return ws

def smakegrid(imgs,xs,ys,currentmodel,p):
    xs = [makemultilineweight(x) for x in xs]
    ys = [makemultilineweight(y) for y in ys]

    ver_texts = [[images.GridAnnotation(y)] for y in ys]
    hor_texts = [[images.GridAnnotation(x)] for x in xs]

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(len(xs) * w, len(ys) * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % len(xs) * w, i // len(xs) * h))

    grid = draw_grid_annotations(grid,w,h, hor_texts, ver_texts)
    grid = draw_origin(grid, currentmodel,w*len(xs),h*len(ys),w)
    if opts.grid_save:
        images.save_image(grid, opts.outdir_txt2img_grids, "xy_grid", extension=opts.grid_format, prompt=p.prompt, seed=p.seed, grid=True, p=p)

    return grid

def makemultilineweight(weight):
    i = 0
    o = ""
    for c in weight:
        o += c
        i += 1
        if i > 25:
            if c == ",":
                o += "\n"
                i = 0
    return o

def swapxy(imgs,xs,ys):
    nimgs = []
    for x in range(len(xs)):
        for y in range(len(ys)):
            nimgs.append(imgs[y * len(xs) + x])
    return nimgs, ys, xs

def makegridmodelname(model_a, model_b,model_c, useblocks,mode,xtype,ytype,alpha,beta,wa,wb,usebeta):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
    
    if model_c == "" or model_c is None:
        #fallback to avoid crash
        model_c = "model A"
        print(f"{bcolors.WARNING}Substituting empty model_c with model_a{bcolors.ENDC}")
    else:
        model_c=filenamecutter(model_c)

    if not usebeta:beta,wb = "not used","not used"
    vals = ""
    modes=["Weight" ,"Add" ,"Triple","Twice"]

    if "mbw" in xtype:
        if "alpha" in xtype:wa = "X"
        if usebeta or " beta" in xtype:wb = "X"

    if "mbw" in ytype:
        if "alpha" in ytype:wa = "Y"
        if usebeta or " beta" in ytype:wb = "Y"

    wa = "alpha = " + wa
    wb = "beta = " + wb

    x = 50
    while len(wa) > x:
        wa  = wa[:x] + '\n' + wa[x:]
        x = x + 50

    x = 50
    while len(wb) > x:
        wb  = wb[:x] + '\n' + wb[x:]
        x = x + 50

    if "model" in xtype:
        if "A" in xtype:model_a = "model A"
        elif "B" in xtype:model_b="model B"
        elif "C" in xtype:model_c="model C"

    if "model" in ytype:
        if "A" in ytype:model_a = "model A"
        elif "B" in ytype:model_b="model B"
        elif "C" in ytype:model_c="model C"

    if modes[1] in mode:
        currentmodel =f"{model_a} \n {model_b} - {model_c})\n x alpha"
    elif modes[2] in mode:
        currentmodel =f"{model_a} x \n(1-alpha-beta) {model_b} x alpha \n+ {model_c} x beta"
    elif modes[3] in mode:
        currentmodel =f"({model_a} x(1-alpha) \n + {model_b} x alpha)*(1-beta)\n+  {model_c} x beta"
    else:
        currentmodel =f"{model_a} x (1-alpha) \n {model_b} x alpha"

    if "alpha" in xtype:alpha = "X"
    if "beta" in xtype:beta = "X" 
    if "alpha" in ytype:alpha = "Y"
    if "beta" in ytype:beta = "Y"

    if "mbw" in xtype:
        if "alpha" in xtype: alpha = "X"
        if "beta" in xtype or usebeta: beta = "X"

    if "mbw" in ytype:
        if "alpha" in ytype: alpha = "Y"
        if "beta" in ytype or usebeta: beta = "Y"

    vals = f"\nalpha = {alpha},beta = {beta}" if not useblocks else f"\n{wa}\n{wb}"

    currentmodel = currentmodel+vals
    return currentmodel

def effectivechecker(imgs,xs,ys,model_a,model_b,esettings):
    diffs = []
    outnum =[]
    im1 = np.array(imgs[0])
    
    model_a = filenamecutter(model_a)
    model_b = filenamecutter(model_b)
    dir = os.path.join(opts.outdir_txt2img_samples,f"{model_a+model_b}","difgif")

    if "gif" in esettings:
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass

    ls,ss = (xs.copy(),ys.copy()) if len(xs) > len(ys) else (ys.copy(),xs.copy())

    for i in range(len(imgs)-1):
        im2 = np.array(imgs[i+1])

        abs_diff = cv2.absdiff(im2 ,  im1)

        abs_diff_t = cv2.threshold(abs_diff, 20, 255, cv2.THRESH_BINARY)[1]        
        res = abs_diff_t.astype(np.uint8)
        percentage = (np.count_nonzero(res) * 100)/ res.size
        abs_diff = cv2.bitwise_not(abs_diff)
        outnum.append(percentage)

        abs_diff = Image.fromarray(abs_diff)     

        diffs.append(abs_diff)

        if "gif" in esettings:
            gifpath = gifpath_t = os.path.join(dir,ls[i+1].replace(":","_")+".gif")
            
            is_file = os.path.isfile(gifpath)
            j = 0
            while is_file:
                gifpath = gifpath_t.replace(".gif",f"_{j}.gif")
                print(gifpath)
                is_file = os.path.isfile(gifpath)
                j = j + 1

            imgs[0].save(gifpath, save_all=True, append_images=[imgs[i+1]], optimize=False, duration=1000, loop=0)

    nums = []
    outs = []

    ls = ls[1:]
    for i in range(len(ls)):
        nums.append([ls[i],outnum[i]])
        ls[i] = ls[i] + "\n Diff : " + str(round(outnum[i],3)) + "%"    

    if "csv" in esettings:
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass
        filepath = os.path.join(dir, f"{model_a+model_b}.csv")
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(nums)

    if len(ys) > len (xs):
        for diff,img in zip(diffs,imgs[1:]):
            outs.append(diff)
            outs.append(img)
            outs.append(imgs[0])
        ss = ["diff",ss[0],"source"]
        return outs,ss,ls
    else:
        outs = [imgs[0]]*len(diffs)  + imgs[1:]+ diffs
        ss = ["source",ss[0],"diff"]
        return outs,ls,ss

def crazyslot(lmode,lsets,llimits_u,llimits_l,lseed,lserial,lcustom,xtype,xmen,ytype,ymen,weights_a,weights_b,start):
    if start == RAND:
        if lserial > RANDCOL:
            xtype = ytype = TYPES.index(RAND)
            xmen = RANDCOL
            ymen = lserial // RANDCOL + 1 
        else:
            xtype = TYPES.index(RAND)
            xmen = lserial

    if "alpha" in lsets:
        if "custom" in lmode:
            weights_a = lcustom
        else:
            weights_a = ",".join([lmode]*26)

    if "beta" in lsets:
        if "custom" in lmode:
            weights_b = lcustom
        else:
            weights_b = ",".join([lmode]*26)

    return xtype,xmen,ytype,ymen,weights_a,weights_b

def alldealer(mens,types):
    for i, men in enumerate(mens):
        if men == "ALL":
            if types[i] == "pinpoint blocks":mens[i] = "BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00|OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11"
            if types[i] == "pinpoint adjust":mens[i] ="IN,OUT,OUT2,CONT,COL1,COL2,COL3" 
    return mens

def draw_grid_annotations(im, width, height, hor_texts, ver_texts, margin=0):

    color_active = ImageColor.getcolor(getattr(opts,"grid_text_inactive_color","#000000"), 'RGB')
    color_inactive = ImageColor.getcolor(getattr(opts,"grid_text_inactive_color","#999999"), 'RGB')
    color_background = ImageColor.getcolor(getattr(opts,"grid_background_color","#ffffff"), 'RGB')

    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines, initial_fnt, initial_fontsize):
        minfont = 2244096
        for line in lines:
            fnt = initial_fnt
            fontsize = initial_fontsize
            line_spacing = fontsize // 2
            while drawing.multiline_textbbox((0,0), line.text, font=fnt,spacing = 5)[2] > line.allowed_width and fontsize > 0:
                fontsize -= 1
                fnt = get_font(fontsize)
                line_spacing = fontsize // 2
            if minfont > fontsize:
                minfont = fontsize
        
        fnt = get_font(minfont)
        
        for line in lines:
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="center",spacing = 5)

            if not line.is_active:
                drawing.line((draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2, draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] * 0.3  + line_spacing

    fontsize = (width + height) // 25
    line_spacing = fontsize // 20

    fnt = get_font(fontsize)

    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else width * 3 // 4

    cols = im.width // width
    rows = im.height // height

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), color_background)
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [width] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [images.GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            line.allowed_width = allowed_width

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for lines in ver_texts]

    pad_top = 0 if sum(hor_text_heights) == 0 else max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left + margin * (cols-1), im.height + pad_top + margin * (rows-1)), color_background)

    for row in range(rows):
        for col in range(cols):
            cell = im.crop((width * col, height * row, width * (col+1), height * (row+1)))
            result.paste(cell, (pad_left + (width + margin) * col, pad_top + (height + margin) * row))

    d = ImageDraw.Draw(result)

    for col in range(cols):
        x = pad_left + (width + margin) * col + width / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col], fnt, fontsize)

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + (height + margin) * row + height / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row], fnt, fontsize)

    return result

def mainmodeldealer(xyz):
    abc = [True,True,True]
    if "model" in xyz:
        if "A" in xyz:abc[0] = False
        if "B" in xyz:abc[1] = False       
        if "C" in xyz:abc[2] = False
    return abc


class Stocker:
    def __init__(self):
        self.stocked = False
        self.stock = None
        self.now = False

    def check_alpha(self,useblocks, alpha,weights_a_in,calcmode):
        now = False
        if "cosine" not in calcmode:
            if useblocks:
                if re.fullmatch(r"^(0,)+$", weights_a_in): now = True
            else:
                if float(alpha) == 0: now = True
        self.now = now
