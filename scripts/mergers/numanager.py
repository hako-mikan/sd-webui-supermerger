import random
import gc
import copy
from scripts.mergers.mergers import TYPES
from scripts.mergers.xyplot import RAND, RANDCOL, sgenxyplot

def numanager(startmode,xtype,xmen,ytype,ymen,ztype,zmen,esettings,
                    weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,calcmode,
                    useblocks,custom_name,save_sets,id_sets,wpresets,deep,fine,bake_in_vae,
                    prompt_s,nprompt_s,steps_s,sampler_s,cfg_s,seed_s,w_s,h_s,batch_size_s,
                    lmode,lsets,llimits_u,llimits_l,lseed,lserial,lcustom,lround,
                    id_task, prompt, negative_prompt, prompt_styles, steps, sampler_index, restore_faces, tiling, n_iter, batch_size, cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, enable_hr, denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y, hr_sampler_index, hr_prompt, hr_negative_prompt, override_settings_texts, *args):
    global numadepth
    grids = []
    sep = "|"

    if RAND in startmode:
        if "off" in lmode:return "Random mode is off",*[None]*5
        if lserial > 0 : lseed = -1
        useblocks = True
    if RAND in startmode or TYPES.index(RAND) in [xtype,ytype,ztype]:
        xtype,xmen,ytype,ymen,weights_a,weights_b = crazyslot(lmode,lsets,llimits_u,llimits_l,lseed,lserial,lcustom,xtype,xmen,ytype,ymen,weights_a,weights_b,startmode)

    lucks = {"on":startmode == RAND, "mode":lmode,"set":lsets,"upp":llimits_u,"low":llimits_l,"seed":lseed,"num":lserial,"cust":lcustom,"round":int(lround)}
    gensets = [id_task, prompt, negative_prompt, prompt_styles, steps, sampler_index, restore_faces, tiling, n_iter, batch_size, cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, enable_hr, denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y, hr_sampler_index, hr_prompt, hr_negative_prompt, override_settings_texts, *args,]
    gensets_s = [prompt_s,nprompt_s,steps_s,sampler_s,cfg_s,seed_s,w_s,h_s]

    allsets = [xtype,xmen,ytype,ymen,ztype,zmen,esettings,
                  weights_a,weights_b,model_a,model_b,model_c,alpha,beta,mode,calcmode,
                  useblocks,custom_name,save_sets,id_sets,wpresets,deep,fine,bake_in_vae,
                  gensets,gensets_s,batch_size_s,lucks]

    print(xtype,xmen,ytype,ymen,weights_a,weights_b)

    if RAND not in startmode:
        if sep in xmen: allsets = separator(allsets,1,sep,xmen,seed,startmode)
        if sep in ymen: allsets = separator(allsets,3,sep,ymen,seed,startmode)
        if sep in zmen: allsets = separator(allsets,5,sep,zmen,seed,startmode)

    if "reserve" in startmode : return numaker(allsets)

    if "normal" or RAND in startmode:
        result,currentmodel,xyimage,a,b,c= sgenxyplot(*allsets)
        if xyimage is not None:grids = xyimage
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

    gc.collect()

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