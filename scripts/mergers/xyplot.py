import random
import cv2
import numpy as np
import os
import copy
import csv
from PIL import Image
from modules import images,scripts
from modules.shared import opts
from scripts.mergers.mergers import types,smerge,simggen,filenamecutter,draw_origin,wpreseter
from scripts.mergers.model_util import usemodelgen

hear = True
hearm = False

state_mergen = False

numadepth = []

def freezetime():
    global state_mergen
    state_mergen = True

def numanager(normalstart,xtype,xmen,ytype,ymen,esettings,
                    weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,
                    prompt,nprompt,steps,sampler,cfg,seed,w,h):
    global numadepth
    grids = []
    sep = "|"

    if sep  in xmen:
        xmens = xmen.split(sep)
        xmen = xmens[0]
        if seed =="-1": seed = str(random.randrange(4294967294))
        for men in xmens[1:]:
            numaker(xtype,men,ytype,ymen,esettings,
                        weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,
                        prompt,nprompt,steps,sampler,cfg,seed,w,h)
    elif sep  in ymen:
        ymens = ymen.split(sep)
        ymen = ymens[0]
        if seed =="-1": seed = str(random.randrange(4294967294))
        for men in ymens[1:]:
            numaker(xtype,xmen,ytype,men,esettings,
                        weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,
                        prompt,nprompt,steps,sampler,cfg,seed,w,h)

    if normalstart:
        result,currentmodel,xyimage,a,b,c= sgenxyplot(xtype,xmen,ytype,ymen,esettings,
                                                                             weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,
                                                                             prompt,nprompt,steps,sampler,cfg,seed,w,h)
        if xyimage is not None:grids =[xyimage[0]]
        else:print(result)
    else:
        if numadepth ==[]:
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

    return result,currentmodel,grids,a,b,c

def numaker(xtype,xmen,ytype,ymen,esettings,
#msettings=[weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets]       
                    weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,
                    prompt,nprompt,steps,sampler,cfg,seed,w,h):
    global numadepth
    numadepth.append([len(numadepth)+1,"waiting",xtype,xmen,ytype,ymen,esettings,
                    weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,
                    prompt,nprompt,steps,sampler,cfg,seed,w,h])
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
        r[2] =  types[int(r[2])]
        r[4] =  types[int(r[4])]
        numa[i] = r[0:6]+r[8:11]+r[12:16]+r[6:8]
    return numa

def caster(news,hear):
    if hear: print(news)

def sgenxyplot(xtype,xmen,ytype,ymen,esettings,
                    weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,useblocks,custom_name,save_sets,id_sets,wpresets,deep,
                    prompt,nprompt,steps,sampler,cfg,seed,w,h):
    global hear
    esettings = " ".join(esettings)
    #type[0:none,1:aplha,2:beta,3:seed,4:mbw,5:model_A,6:model_B,7:model_C,8:pinpoint 9:deep]
    xtype = types[xtype]
    ytype = types[ytype]
    if ytype == "none": ymen = ""

    modes=["Weight" ,"Add" ,"Triple","Twice"]
    xs=ys=0
    weights_a_in=weights_b_in="0"

    deepprint  = True if "print change" in esettings else False

    def castall(hear):
        if hear :print(f"xmen:{xmen}, ymen:{ymen}, xtype:{xtype}, ytype:{ytype}, weights_a:{weights_a_in}, weights_b:{weights_b_in}, model_A:{model_a},model_B :{model_b}, model_C:{model_c}, alpha:{alpha},\
        beta :{beta}, mode:{mode}, blocks:{useblocks}")

    pinpoint = "pinpoint blocks" in xtype or "pinpoint blocks" in ytype
    usebeta = modes[2] in mode or modes[3] in mode

    #check and adjust format
    print(f"XY plot start, mode:{mode}, X: {xtype}, Y: {ytype}, MBW: {useblocks}")
    castall(hear)
    None5 = [None,None,None,None,None]
    if xmen =="": return "ERROR: parameter X is empty",*None5
    if ymen =="" and not ytype=="none": return "ERROR: parameter Y is empty",*None5
    if model_a ==[] and not ("model_A" in xtype or "model_A" in ytype):return f"ERROR: model_A is not selected",*None5
    if model_b ==[] and not ("model_B" in xtype or "model_B" in ytype):return f"ERROR: model_B is not selected",*None5
    if model_c ==[] and usebeta and not ("model_C" in xtype or "model_C" in ytype):return "ERROR: model_C is not selected",*None5
    if xtype == ytype: return "ERROR: same type selected for X,Y",*None5

    if useblocks:
        weights_a_in=wpreseter(weights_a,wpresets)
        weights_b_in=wpreseter(weights_b,wpresets)

    #for X only plot, use same seed
    if seed == -1: seed = int(random.randrange(4294967294))

    #for XY plot, use same seed
    def dicedealer(zs):
        for i,z in enumerate(zs):
            if z =="-1": zs[i] = str(random.randrange(4294967294))
        print(f"the die was thrown : {zs}")

    #adjust parameters, alpha,beta,models,seed: list of single parameters, mbw(no beta):list of text,mbw(usebeta); list of pair text
    def adjuster(zmen,ztype,aztype):
        if "mbw" in ztype:#men separated by newline
            zs = zmen.splitlines()
            caster(zs,hear)
            if "mbw alpha and beta" in ztype:
                zs = [zs[i:i+2] for i in range(0,len(zs),2)]
                caster(zs,hear)
        elif "elemental" in ztype:
            zs = zmen.split("\n\n")
        else:
            if "pinpoint element" in ztype:
                zmen = zmen.replace("\n",",")
            if "effective" in ztype:
                zmen = ","+zmen
                zmen = zmen.replace("\n",",")
            zs = [z.strip() for z in zmen.split(',')]
            caster(zs,hear)
        if "alpha" in ztype and "effective" in aztype:
            zs = [zs[0]]
        if "seed" in ztype:dicedealer(zs)
        return zs

    xs = adjuster(xmen,xtype,ytype)
    ys = adjuster(ymen,ytype,xtype)

    #in case beta selected but mode is Weight sum or Add or Diff
    if ("beta" in xtype or "beta" in ytype) and not usebeta:
        mode = modes[3]
        print(f"{modes[3]} mode automatically selected)")

    #in case mbw or pinpoint selected but useblocks not chekced
    if ("mbw" in xtype or "pinpoint blocks" in xtype) and not useblocks:
        useblocks = True
        print(f"MBW mode enabled")

    if ("mbw" in ytype or "pinpoint blocks" in ytype) and not useblocks:
        useblocks = True
        print(f"MBW mode enabled")

    xyimage=[]
    xcount =ycount=0
    allcount = len(xs)*len(ys)

    #for STOP XY bottun
    flag = False
    global state_mergen
    state_mergen = False

    #type[0:none,1:aplha,2:beta,3:seed,4:mbw,5:model_A,6:model_B,7:model_C,8:pinpoint ]
    blockid=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
    #format ,IN00 IN03,IN04-IN09,OUT4,OUT05
    def weightsdealer(x,xtype,y,weights):
        caster(f"weights from : {weights}",hear)
        zz = x if "pinpoint blocks" in xtype else y
        za = y if "pinpoint blocks" in xtype else x
        zz = [z.strip() for z in zz.split(' ')]
        weights_t = [w.strip() for w in weights.split(',')]
        if zz[0]!="NOT":
            flagger=[False]*26
            changer = True
        else:
            flagger=[True]*26
            changer = False
        for z in zz:
            if z =="NOT":continue
            if "-" in z:
                zt = [zt.strip() for zt in z.split('-')]
                if  blockid.index(zt[1]) > blockid.index(zt[0]):
                    flagger[blockid.index(zt[0]):blockid.index(zt[1])+1] = [changer]*(blockid.index(zt[1])-blockid.index(zt[0])+1)
                else:
                    flagger[blockid.index(zt[1]):blockid.index(zt[0])+1] = [changer]*(blockid.index(zt[0])-blockid.index(zt[1])+1)
            else:
                flagger[blockid.index(z)] =changer    
        for i,f in enumerate(flagger):
            if f:weights_t[i]=za
        outext = ",".join(weights_t)
        caster(f"weights changed: {outext}",hear)
        return outext

    def abdealer(z):
        if " " in z:return z.split(" ")[0],z.split(" ")[1]
        return z,z

    def xydealer(z,zt,azt):
        nonlocal alpha,beta,seed,weights_a_in,weights_b_in,model_a,model_b,model_c,deep
        if pinpoint or "pinpoint element" in zt or "effective" in zt:return
        if "mbw" in zt:
            def weightser(z):return z, z.split(',',1)[0]
            if "mbw alpha and beta" in zt:
                weights_a_in,alpha = weightser(wpreseter(z[0],wpresets))
                weights_b_in,beta = weightser(wpreseter(z[1],wpresets))
                return
            elif "alpha" in zt:
                weights_a_in,alpha = weightser(wpreseter(z,wpresets))
                return
            else:
                weights_b_in,beta = weightser(wpreseter(z,wpresets))
                return
        if "and" in zt:
            alpha,beta = abdealer(z)
            return
        if "alpha" in zt and not "pinpoint element" in azt:alpha = z
        if "beta" in zt: beta = z
        if "seed" in zt:seed = int(z)
        if "model_A" in zt:model_a = z
        if "model_B" in zt:model_b = z
        if "model_C" in zt:model_c = z
        if "elemental" in zt:deep = z
    
    # plot start
    for y in ys:
        xydealer(y,ytype,xtype)
        xcount = 0
        for x in xs:
            xydealer(x,xtype,ytype)
            if ("alpha" in xtype or "alpha" in ytype) and pinpoint:
                weights_a_in = weightsdealer(x,xtype,y,weights_a)
                weights_b_in = weights_b
            if ("beta" in xtype or "beta" in ytype) and pinpoint:
                weights_b_in = weightsdealer(x,xtype,y,weights_b)
                weights_a_in =weights_a
            if "pinpoint element" in xtype or "effective" in xtype:
                deep_in = deep +","+ str(x)+":"+ str(y) 
            elif "pinpoint element" in ytype or "effective" in ytype:
                deep_in = deep +","+ str(y)+":"+ str(x) 
            else:
                deep_in = deep

            print(f"XY plot: X: {xtype}, {str(x)}, Y: {ytype}, {str(y)} ({xcount+ycount*len(xs)+1}/{allcount})")
            if not (xtype=="seed" and xcount > 0):
               _ , currentmodel,modelid,theta_0=smerge(weights_a_in,weights_b_in, model_a,model_b,model_c, float(alpha),float(beta),mode,useblocks,"","",id_sets,False,deep_in,deepprint = deepprint) 
               usemodelgen(theta_0,model_a)
                             # simggen(prompt, nprompt, steps, sampler, cfg, seed, w, h,mergeinfo="",id_sets=[],modelid = "no id"):
            image_temp=simggen(prompt, nprompt, steps, sampler, cfg, seed, w, h,currentmodel,id_sets,modelid)
            xyimage.append(*image_temp[0])
            xcount+=1
            if state_mergen:
                flag = True
                break
        ycount+=1
        if flag:break

    if flag and ycount ==1:
        xs = xs[:xcount]
        ys = [ys[0],]
        print(f"stopped at x={xcount},y={ycount}")
    elif flag:
        ys=ys[:ycount]
        print(f"stopped at x={xcount},y={ycount}")

    if "mbw alpha and beta" in xtype: xs = [f"alpha:({x[0]}),beta({x[1]})" for x in xs ]
    if "mbw alpha and beta" in ytype: ys = [f"alpha:({y[0]}),beta({y[1]})" for y in ys ]

    xs[0]=xtype+" = "+xs[0] #draw X label
    if ytype!=types[0] or "model" in ytype:ys[0]=ytype+" = "+ys[0]  #draw Y label

    if ys==[""]:ys = [" "]

    if "effective" in xtype or "effective" in ytype:
        xyimage,xs,ys = effectivechecker(xyimage,xs,ys,model_a,model_b,esettings)

    if not "grid" in esettings:
        gridmodel= makegridmodelname(model_a, model_b,model_c, useblocks,mode,xtype,ytype,alpha,beta,weights_a,weights_b,usebeta)
        grid = smakegrid(xyimage,xs,ys,gridmodel,image_temp[4])
        xyimage.insert(0,grid)

    state_mergen = False
    return "Finished",currentmodel,xyimage,*image_temp[1:4]

def smakegrid(imgs,xs,ys,currentmodel,p):
    ver_texts = [[images.GridAnnotation(y)] for y in ys]
    hor_texts = [[images.GridAnnotation(x)] for x in xs]

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(len(xs) * w, len(ys) * h), color='black')

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % len(xs) * w, i // len(xs) * h))

    grid = images.draw_grid_annotations(grid,int(p.width), int(p.height), hor_texts, ver_texts)
    grid = draw_origin(grid, currentmodel,w*len(xs),h*len(ys),w)
    if opts.grid_save:
        images.save_image(grid, opts.outdir_txt2img_grids, "xy_grid", extension=opts.grid_format, prompt=p.prompt, seed=p.seed, grid=True, p=p)

    return grid

def makegridmodelname(model_a, model_b,model_c, useblocks,mode,xtype,ytype,alpha,beta,wa,wb,usebeta):
    model_a=filenamecutter(model_a)
    model_b=filenamecutter(model_b)
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

        abs_diff_t = cv2.threshold(abs_diff, 5, 255, cv2.THRESH_BINARY)[1]        
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
