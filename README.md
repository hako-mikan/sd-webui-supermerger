# SuperMerger
- Model merge extention for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- Merge models can be loaded directly for generation without saving

# Recent Update
2023.01.20.2350(JST)  
Problem solved where png info could not be saved properly.  
png infoがうまく保存できない問題を解決しました。

2023.01.25.0115(JST)  
added several features 
- added new merge mode "Triple sum","sum Twice"  
- added XY plot
 
- 新しいマージモードを追加しました "Triple sum","sum Twice"  
- XY plot機能を追加しました

After updating, restart webui-user.bat; Reload UI will not reflect the update.  更新後、webui-user.batを再起動してください。Reload UIでは更新が反映されません。
# 

日本語説明は[後半](#概要)後半にあります。

# Overview
This extension allows you to load a merged model for image generation without saving.
Until now, it was necessary to save the merged model and delete it if you did not like it, but by using this extension, you can prevent the consumption of HDD and SSD.

## Usage

### Merge mode
#### Weight sum
Normal merge. alpha is used. if MBW is enabled, MBW base is used as alpha.
#### Add difference
Difference merge, if MBW is enabled, MBW base is used as alpha
#### Triple sum
Merges 3 models at the same time. This function was added because there were three model selection windows, but I am not sure if it works effectively or not. if MBW is enabled, enter MBW parameters for alpha and beta.
#### sum Twice
Weight sum twice.

### merge
After merging, load the model as a generated model. __Note that a different model is loaded from the model information in the upper left corner.__ It will be reset when you re-select the model in the model selection screen in the upper left corner.


### gen
Generate images using the settings in the text2image tab.

### Merge and Gen
Merge and Generate image after merging

### Save merged model
Save the currently loaded merged model (merges a new one, because saving the loaded model saves it as a fp16 model). This is because saving the loaded model assumes it is a single-precision model.)

### Sequential XY Merge and Generation
Sequential merge image generation.
#### alpha,beta
Change alpha, beta.
#### seed
Change the seed. Assuming -1 will result in a fixed seed in the opposite axis direction.
#### model_A,B,C
Change model. The model selected in the model selection window is ignored.
#### pinpoint blocks
Change only specific blocks in MBW. Choose alpha or beta for the opposite axis. If you enter a block ID, only that block's alpha (beta) will change. Separate with commas like any other type. Multiple blocks can be changed at the same time by separating them with spaces or hyphens. Adding NOT at the beginning reverses the change target. If NOT IN09-OUT02 is set, all except IN09-OUT02 will change. NOT has no effect unless entered first.
##### Input example
IN01, OUT10 OUT11, OUT03-OUT06, OUT07-OUT11, NOT M00 OUT03-OUT06
in this case
1: Only IN01 changes
2: OUT10 and OUT11 change
3: OUT06 to OUT03 change
4: OUT11 to OUT07 change
5: All but M00 and OUT03 to OUT06 change 
Be careful not to forget the "0"
![xy_grid-0006-2934360860 0](https://user-images.githubusercontent.com/122196982/214343111-e82bb20a-799b-4026-8e3c-dd36e26841e3.jpg)
### About Cache
The cache is used to store models in memory to speed up sequential merges and other operations.
Cache settings can be configured from web-ui's setting menu.

This script uses some of the web-ui and mbw-merge scripts.

# 概要
このextentionではモデルをマージした際、保存せずに画像生成用のモデルとして読み込むことができます。
これまでマージしたモデルはいったん保存して気に入らなければ削除するということが必要でしたが、このextentionを使うことでHDDやSSDの消耗を防ぐことができます。

## 使い方

### マージモード
#### Weight sum
通常のマージです。alphaが使用されます。MBWが有効になっている場合はMBWのbaseがアルファとして使われます
#### Add difference
差分マージです。MBWが有効になっている場合はMBWのbaseがアルファとして使われます
#### Triple sum
マージを3モデル同時に行います。alpha,betaが使用されます。モデル選択窓が3つあったので追加した機能ですが、有効に働くかはわかりません。MBWでも使えます。それぞれMBWのalpha,betaを入力してください。
#### sum Twice
Weight sumを2回行います。alpha,betaが使用されます。MBWモードでも使えます。それぞれMBWのalpha,betaを入力してください。

### MBW モード
チェックするとブロックごとのマージが有効になります。各ブロックごとの比率は下部のスライダーなどで設定してください。

## 各ボタン
### merge
マージした後、生成用モデルとして読み込みます。 __左上のモデル情報とは違うモデルがロードされていることに注意してください。__ 左上のモデル選択画面でモデルを選択しなおすとリセットされます

### gen
text2imageタブの設定で画像生成を行います

### Merge and Gen
マージしたのち画像を生成します

### Save merged model
現在読み込まれているマージモデルを保存します(新規にマージを行います。これはロードされたモデルを保存すると単精度のモデルとして保存されるためです)

### Sequential XY Merge and Generation
連続マージ画像生成を行います。すべてのマージモードで有効です。
#### alpha,beta
アルファ、ベータを変更します。各要素は半角カンマで区切ります。
#### seed
シードを変更します。-1と入力すると、反対の軸方向には固定されたseedになります。各要素は半角カンマで区切ります。
#### model_A,B,C
モデルを変更します。モデル選択窓で選択されたモデルは無視されます。各要素は半角カンマで区切ります。
#### pinpoint blocks
MBWにおいて特定のブロックのみを変化させます。反対の軸はalphaまたはbetaを選んでください。ブロックIDを入力すると、そのブロックのみalpha(beta)が変わります。他のタイプと同様にカンマで区切ります。スペースまたはハイフンで区切ることで複数のブロックを同時に変化させることもできます。最初にNOTをつけることで変化対象が反転します。NOT IN09-OUT02とすると、IN09-OUT02以外が変化します。NOTは最初に入力しないと効果がありません。
##### 入力例
IN01,OUT10 OUT11, OUT03-OUT06,OUT07-OUT11,NOT M00 OUT03-OUT06
この場合
1:IN01のみ変化
2:OUT10およびOUT11が変化
3:OUT03からOUT06が変化
4:OUT07からOUT11が変化
5:M00およびOUT03からOUT06以外が変化
となります。0の打ち忘れに注意してください。
![xy_grid-0006-2934360860 0](https://user-images.githubusercontent.com/122196982/214343111-e82bb20a-799b-4026-8e3c-dd36e26841e3.jpg)

ブロックID(大文字のみ有効)
BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11

### キャッシュについて
モデルをメモリ上に保存することにより連続マージなどを高速化することができます。
キャッシュの設定はweb-uiのsettingから行ってください。

階層別マージについては下記を参照してください

https://github.com/bbc-mc/sdweb-merge-block-weighted-gui

このスクリプトではweb-uiとmbw-mergeのスクリプトを一部使用しています
