# SuperMerger
- Model merge extention for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Merge models can be loaded directly for generation without saving

日本語は[こちら](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/README_ja.md)

# Recent Update
All updates can be found [here](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/changelog.md)  

### update 2023.03.03.0145(JST)
- Add new XY type "mbw alpha","mbw beta","mbw alpha and beta"

### update 2023.03.02.1900(JST)
- Elemental Merge feature added. Details [here](https://github.com/hako-mikan/sd-webui-supermerger/blob/main/elemental_en.md).


### update 2023.02.20.2000(JST)
The timing of importing "diffusers" has been changed. With this update, some environments can be started without installing "diffusers".

diffusers must now be installed. on windows, this can be done by typing "pip install diffusers" at the command prompt in the web-ui folder, but it depends on your environment.

# overview
This extension allows merged models to be loaded as models for image generation without saving them.
This extension can prevent the use of HDD and SSD.

## Usage

### Merge mode
#### Weight sum
Normal merge. alpha is used. if MBW is enabled, MBW base is used as alpha.
#### Add difference
Add difference, if MBW is enabled, MBW base is used as alpha
#### Triple sum
Merge 3 models at the same time. alpha and beta are used. I added this function because there were three model selection windows, but I am not sure if it works effectively. 
#### sum Twice
Weight sum twice, alpha and beta are used. 

### use MBW
If checked, block-by-block merging is enabled. Use the slider at the bottom of the screen to set the ratio of each block.

## Each button
### Merge
After merging, load as a model for generation. **Note that a different model is loaded than the model information in the upper left corner.** It will be reset when you re-select the model in the model selection screen on the top left.

### Gen.
Image generation is performed using the settings in the text2image tab.

### Merge and Gen
Merge and Generate image after merging.

### Set from ID
Read settings from merge log. The log is updated each time a merge is performed, and a sequential ID starting from 1 is assigned. Set on -1 will read the last merged configuration. The merge log is saved in extension/sd-webui-supermerger/mergehistory.csv. You can browse and search in the History tab. You can search and/or by separating with a half-width space.

### Sequential XY Merge and Generation
Performs sequential merge image generation. Effective in all merge modes.
#### alpha,beta
Change alpha and beta.
#### alpha and beta
Change alpha and beta at the same time. Separate alpha and beta with a single space, and separate each element with a comma. If only one number is entered, the same value is entered for both alpha and beta.  
Example: 0,0.5 0.1,0.3 0.4,0.5
#### MBW
Performs a block-byblock merge. Enter ratios separated by newlines. Presets can be used, but be careful to **separate on a new line**.For Triple and Twice, enter two lines as a set. An odd number of lines will result in an error. 
#### seed
Changes the seed. Entering -1 will result in a fixed seed in the opposite axis direction.
#### model_A,B,C
Changes the model. The model selected in the model selection window is ignored.
#### pinpoint blocks
Changes only specific blocks in MBW. Choose alpha or beta for the opposite axis. If you enter a block ID, the alpha (beta) will change only for that block. As with the other types, use commas to separate them. Multiple blocks can be changed at the same time by separating them with a space or hyphen. NOT must be entered first to have any effect.
##### Input example
IN01,OUT10 OUT11, OUT03-OUT06,OUT07-OUT11,NOT M00 OUT03-OUT06
In this case
- 1:Only IN01 changes
- 2:OUT10 and OUT11 change
- 3:OUT03 to OUT06 change
- 4:OUT07 to OUT11 change
- 5:All except for M00 and OUT03 to OUT06 are changed.  

Please be careful not to forget to input "0".
![xy_grid-0006-2934360860 0](https://user-images.githubusercontent.com/122196982/214343111-e82bb20a-799b-4026-8e3c-dd36e26841e3.jpg)

Block ID (only upper case letters are valid)
BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09, OUT10,OUT11

### Reserve XY plot
The Reserve XY plot button reserves the execution of an XY plot for the setting at the time the button is pressed, instead of immediately executing the plot. The reserved XY plot will be executed after the normal XY plot is completed or by pressing the Start XY plot button on the Reservation tab. Reservations can be made at any time during the execution or non-execution of an XY plot. The reservation list is not automatically updated, so use the Reload button. If an error occurs, the plot is discarded and the next reservation is executed. Images will not be displayed until all reservations are finished, but those that have been marked "Finished" have finished generating the grid and can be viewed in the Image Browser or other applications.

It is also possible to move to an appointment at any location by using "|".  
Inputing "0.1,0.2,0.3,0.4,0.5|0.6,0.7,0.8,0.9,1.0"

0.1,0.2,0.3,0.4,0.5  
0.6,0.7,0.8,0.9,1.0  
The grid is divided into two reservations, "0.1,0.2,0.3,0.4,0.5" and "0.6,0.7,0.8,0.9,1.0" executed. This may be useful when there are too many elements and the grid becomes too large.

### About Cache
By storing models in memory, continuous merging and other operations can be sped up.
Cache settings can be configured from web-ui's setting menu.

### unload button
Deletes the currently loaded model. This is used to free up GPU memory when using kohya-ss GUI. Once the model is deleted, you will not be able to generate images. If you want to generate images, please re-select models.

## LoRA
LoRA related functions. It is basically the same as kohya-ss scripts, but it supports block-by-block merging. Currently, it does not support V2.X series merging.
### merge to checkpoint
Merge LoRAs into a model. Multiple LoRAs can be merged at the same time.  
Enter LoRA name1:ratio1:block1,LoRA name2:ratio2:block2,... 
LoRA can also be used alone. The ":block" part can be omitted. The ratio can be any value, including negative values. There is no restriction that the total must sum to 1 (of course, if it greatly exceeds 1, it will break down).

### Make LoRA
Generates a LoRA from the difference of two models.
If you specify a demension, it will be created with the specified dimension. If no demension is specified, LoRAs are created with dim 128.
The blend ratio can be adjusted by alpha and beta. (alpha x Model_A - beta x Model B) alpha, beta = 1 is the normal LoRA creation.

### merge LoRAs
Merges one or more LoRAs. kohya-ss's latest script is used, so LoRAs with different dimensions can be merged, but note that the generated images may differ significantly because LoRAs are recalculated when dimensions are converted. 

The calculate dimention button calculates the dimensions of each LoRA and activates the display and sorting functions. The calculation is rather time-consuming and takes several tens of seconds for a LoRA of about 50. Newly merged LoRAs will not appear in the list, so please press the reload button. Dimension recalculation only calculates the added LoRAs.

### Difference between Normal Merge and SAME TO STRENGTH
If the same to Strength option is not used, the result is the same as the merge in the script created by kohya-ss. In this case, the result is different from the case where LoRA is applied on Web-ui as shown in the figure below. The reason for this is related to the mathematical formula used to adopt LoRA into U-net. kohya-ss's script multiplies the ratio as it is, but the formula used to apply LoRA squares the ratio, so if the ratio is set to a number other than 1, or to a negative value, the result will differ from Strength (strength when applied).  Using the SAME TO STRENGTH option, the square root of the ratio is driven at merge time, so that Strength and the ratio are calculated to have the same meaning at apply time. It is also calculated so that a negative value will have the same effect. If you are not doing additional learning, for example, you may be fine using the SAME TO STRENGTH option, but if you are doing additional learning on the merged LoRA, you may not want to use anyone else's option.  
The following figures show the generated images for each case of normal image generation/same to Strength option/normal merge, using  merged LoRAs of figmization and ukiyoE. You can see that in the case of normal merge, even in the negative direction, the image is squared and positive.
![xyz_grid-0014-1534704891](https://user-images.githubusercontent.com/122196982/218322034-b7171298-5159-4619-be1d-ac684da92ed9.jpg)

For hierarchical merges see

https://github.com/bbc-mc/sdweb-merge-block-weighted-gui

This script uses some scripts from web-ui, mbw-merge and kohya-ss
