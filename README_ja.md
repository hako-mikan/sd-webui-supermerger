# SuperMerger
- [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 用のモデルマージ拡張
- マージしたモデルを保存せず直接生成に使用できます

[<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](README.md)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](#overview)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)


# Overview
このextentionではモデルをマージした際、保存せずに画像生成用のモデルとして読み込むことができます。
これまでマージしたモデルはいったん保存して気に入らなければ削除するということが必要でしたが、このextentionを使うことでHDDやSSDの消耗を防ぐことができます。
モデルの保存とロードの時間を節約できるため、比率を変更しながら連続生成することによりモデルマージの効率を大幅に向上させます。

# もくじ
- [Merge Models](#merge-models)
    - [Merge Block Weight](#merge-block-weight)
    - [XYZ Plot](#xyz-plot)
    - [Adjust](#adjust)
    - [Let the Dice Roll](#let-the-dice-roll)
    - [Elemental Merge](#elemental-merge)
    - [Generation Parameters](#generation-parameters)
- [LoRA](#lora)
    - [Merge LoRAs](#merge-loras)
    - [Merge to Checipoint](#merge-to-checkpoint)
    - [Extract from Checkpoints](#extract-from-checkpoints)
- [Other Tabs](#other-tabs)

- [Calcomode](calcmode_ja.md)
- [Elemental Merge](elemental_ja.md)


# Recent Update
2023.11.10
- LoRA抽出スクリプトの変更: SDXL,2.Xをサポート
- SD2.XでのモデルへのLoRAマージをサポート
- バグフィックス
- 新しい計算方式extract merge(checkpoint/LoRA)を追加。作成した[subaqua](https://github.com/sbq0)氏に感謝します。

## 計算関数の変更
いくつかのモードでは、計算に使用する機能が変更され、マージ計算の速度が向上しています。計算結果は同じですが、同じ結果が得られない場合は、オプションの「use old calc method」をチェックしてください。影響を受ける方式は以下の通りです:
Weight Sum: normal, cosineA, cosineB
Sum Twice:normal  
提案した[wkpark](https://github.com/wkpark)氏に感謝します。

**注意！** 
XLモデルのマージには最低64GBのCPUメモリが必要です。64Gのメモリであっても併用しているソフトによってはシステムが不安定になる恐れがあるのでシステムが落ちてもいい状態で作業して下さい。私は久しぶりにブルースクリーンに遭遇しました。

## 既知の問題
他の拡張機能（sd-webui-prompt-all-in-oneなど）を同時にインストールしている場合、起動時にブラウザを自動的に開くオプションを有効にすると、動作が不安定になることがあります。Gradioの問題である可能性が高いので、修正は難しいです。そのオプションを無効にしてお使いください。

すべての更新は[ここ](changelog.md)  で確認できます。 

# つかいかた
## Merge Models
ここでマージされたモデルは、Web-UIの生成モデルとしてロードされます。左上のモデル表示は変わりませんが、マージされたモデルは実際にロードされています。別のモデルが左上のモデル選択から選択されるまで、マージされたモデルはロードされたままになります。
### Basic Usage
Select models A/B/(C), the merge mode, and alpha (beta), then press Merge/Merge and Gen to start the merging process. In the case of Merge and Gen, generation is carried out using the prompt and other settings specified in txt2img.The Gen button only generates images, and the Stop button interrupts the merging process.
モデルA/B/(C)、merge mode、alpha (beta)を選択し、Merge/Merge and Genを押すとマージ処理が始まります。Merge and Genの場合は、txt2imgで指定されたプロンプトやその他の設定を使用して生成が行われます。Genボタンは画像のみを生成し、Stopボタンはマージを中断します。

### Load Settings From:
マージログから設定を読み込みます。マージが行われるたびにログが更新され、1から始まる連続IDが割り当てられます。"-1"は最後のマージからの設定に対応し、"-2"は最後から二番目のものに対応します。マージログはextension/sd-webui-supermerger/mergehistory.csvに保存されます。Historyタブで閲覧や検索ができます。半角スペースで区切ってand/orで検索できます。
### Clear Cache
Web-UIのモデルキャッシュ機能が有効になっている場合、SuperMergerは連続マージを高速化するためにモデルキャッシュを作成します。モデルはWeb-UIのキャッシュ機能とは別にキャッシュされます。使用後にキャッシュを削除するにはこのボタンを使用してください。キャッシュはVRAMではなくRAMに作成されます。

# 各種設定
## マージモード
### Weight sum
通常のマージです。alphaが使用されます。α=0の場合Model A, α=1 の時model Bになります。
### Add difference
差分マージです。
### Triple sum
マージを3モデル同時に行います。alpha,betaが使用されます。モデル選択窓が3つあったので追加した機能ですが、ちゃんと動くようです。MBWでも使えます。それぞれMBWのalpha,betaを入力してください。
### sum Twice
Weight sumを2回行います。alpha,betaが使用されます。MBWモードでも使えます。それぞれMBWのalpha,betaを入力してください。

### use MBW
チェックするとブロックごとのマージ(階層マージ)が有効になります。各ブロックごとの比率は下部のスライダーかプリセットで設定してください。


### Merge mode
#### Weight sum $(1-\alpha) A + \alpha B$
通常のマージです。alphaが使用されます。$\alpha$=0の場合Model A, $\alpha$=1 の時model Bになります。
#### Add difference $A + \alpha (B-C)$
差分を加算します。MBWが有効な場合、$\alpha$としてMBWベースが使用されます。
#### Triple sum $(1-\alpha - \beta) A + \alpha B + \beta C$
同時に3つのモデルをマージします。$\alpha$と$\beta$が使用されます。3つのモデル選択ウィンドウがあったためこの機能を追加しましたが、実際に効果的に動作するかはわかりません。
#### sum Twice $(1-\beta)((1-\alpha)A+\alpha B)+\beta C$
Weight sumを2回行います。$\alpha$と$\beta$が使用されます。

### calcmode
各計算方法の詳細については[リンク先](calcmode_ja.md)を参照してください。計算方法とマージモードの対応表は以下の通りです。
| Calcmode  | Description  | Merge Mode  |
|----|----|----|
|normal | 通常の計算方法   |  ALL  |
|cosineA | モデルAを基準にマージ中の損失を最小限にする計算を行います。  | Weight sum    |
|cosineB | モデルBを基準にマージ中の損失を最小限にする計算を行います。  | Weight sum    |
|trainDifference   |モデルAに対してファインチューニングするかのように差分を'トレーニング'します。   | Add difference   |
|smoothAdd  | 中央値フィルタとガウスフィルタの利点を混合した差分の追加  | Add difference   |
|smoothAdd MT| マルチスレッドを使用して計算を高速化します。   | Add difference    |
|extract   | モデルB/Cの共通点・非共通点を抽出して追加します.  | Add difference  |
|tensor| 加算の代わりにテンソルを比率で入れ替えます   | Weight sum |
|tensor2  |テンソルの次元が大きい場合、２次元目を基準にして入れ替えます | Weight sum |
|self  | モデル自身に$\alpha$を掛け合わせます |  Weight sum  |

### use MBW
階層マーを有効にします。Merge Block Weightで重みを設定してください。これを有効にすると、アルファとベータが無効になります。

### Options
| 設定         | 説明                                       |
|-----------------|---------------------------------------------------|
| save model      | マージ後のモデルを保存します。                    |
| overwrite       | モデルの上書きを有効にします。                 |
| safetensors     | safetensors形式で保存します。                      |
| fp16            | 半精度で保存します。                          |
| save metadata   | 保存時にメタデータにマージ情報を保存します。(safetensorsのみ)  |
| prune           | 保存時にモデルから不要なデータを削除します。 |
| Reset CLIP ids  | CLIP idsをリセットします。                              |
| use old calc method           | 古い計算関数を使用します|
| debug           | デバッグ情報をCMDに出力します。                 |

### save merged model ID to
生成された画像またはPNG情報にマージIDを保存するかどうかを選択できます。
### Additional Value
現在、calcmodeで'extract'が選択されている場合にのみ有効です。
### Custom Name (Optional)
モデル名を設定できます。設定されていない場合は、自動的に決定されます。
### Bake in VAE
モデルを保存するとき、選択されたVAEがモデルに組み込まれます。

### Save current Merge
現在ロードされているモデルを保存します。PCのスペックやその他の問題により、マージに時間がかかる場合に効果的です。

## Merge Block Weight
これは階層ごとに割合を設定できるマージ技法です。各階層ごは背景の描写、キャラクター、絵柄などに対応している可能性があるため、階層ごごとの割合を変更することで、様々なマージモデルを作成できます。

階層はSDのバージョンによって異なり、以下のような階層があります。

BASEはテキストエンコーダを指し、プロンプトへの反応などに影響します。IN-OUTはU-Netを指し、画像生成を担当します。

Stable diffusion 1.X, 2.X
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|  
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN00|IN01|IN02|IN03|IN04|IN05|IN06|IN07|IN08|IN09|IN10|IN11|MID|OUT00|OUT01|OUT02|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|OUT09|OUT10|OUT11|

Stable diffusion XL
|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|BASE|IN00|IN01|IN02|IN03|IN04|IN05|IN06|IN07|IN08|MID|OUT00|OUT01|OUT02|OUT03|OUT04|OUT05|OUT06|OUT07|OUT08|

## XYZ Plot
## XYZプロット
連続的にマージ画像を生成します。すべてのマージモードで効果的です。
### alpha, beta
alphaとbetaの値を変更します。
### alpha と beta
alphaとbetaを同時に変更します。alphaとbetaはスペース1つで区切り、各要素はカンマで区切ってください。1つの数字のみ入力された場合、alphaとbetaに同じ値が入力されます。  
例: 0, 0.5 0.1, 0.3 0.4, 0.5
### MBW
階層ごとにマージを実行します。改行で区切った割合を入力してください。プリセットは使用できますが、必ず**改行で区切る**ようにしてください。TripleやTwiceの場合は、2行を1セットで入力してください。奇数行の入力ではエラーになります。
### seed
シード値を変更します。-1を入力すると反対軸方向の固定シードになります。
### model_A, B, C 
モデルを変更します。モデル選択ウィンドウで選択したモデルは無視されます。
### pinpoint block
MBWで特定の階層のみを変更します。反対軸にはalphaかbetaを選択してください。階層Dを入力すると、その階層のalpha (beta)のみが変更されます。他のタイプと同様にカンマで区切って入力してください。スペースやハイフンで区切ることで、同時に複数の階層を変更できます。効果を得るには、必ず先頭にNOTを入力してください。
#### 入力例
IN01, OUT10 OUT11, OUT03-OUT06, OUT07-OUT11, NOT M00 OUT03-OUT06
この場合
- 1:IN01のみ変化
- 2:OUT10およびOUT11が変化
- 3:OUT03からOUT06が変化
- 4:OUT07からOUT11が変化
- 5:M00およびOUT03からOUT06以外が変化  

となります。0の打ち忘れに注意してください。
![xy_grid-0006-2934360860 0](https://user-images.githubusercontent.com/122196982/214343111-e82bb20a-799b-4026-8e3c-dd36e26841e3.jpg)

階層ID (大文字のみ有効)
BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,IN09,IN10,IN11,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08,OUT09,OUT10,OUT11

XL model
BASE,IN00,IN01,IN02,IN03,IN04,IN05,IN06,IN07,IN08,M00,OUT00,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT08

### calcmode
計算方式を変更します。適用できるマージモードとの対応に注意して下さい。カンマで区切ります

### prompt
プロンプトを変更できます。txt2imgのプロンプトは無視されます。ネガティブプロンプトは有効です。
**改行で区切る**ことに注意をして下さい。

### XYプロットの予約
Reserve XY plotボタンはすぐさまプロットを実行せず、ボタンを押したときの設定のXYプロットの実行を予約します。予約したXYプロットは通常のXYプロットが終了した後か、ReservationタブのStart XY plotボタンを押すと実行が開始されます。予約はXYプロット実行時・未実行時いつでも可能です。予約一覧は自動更新されないのでリロードボタンを使用してください。エラー発生時はそのプロットを破棄して次の予約を実行します。すべての予約が終了するまで画像は表示されませんが、Finishedになったものについてはグリッドの生成は終わっているので、Image Browser等で見ることが可能です。  
「|」を使用することで任意の場所で予約へ移動することも可能です。  
0.1,0.2,0.3,0.4,0.5|0.6,0.7,0.8,0.9,1.0とすると  

0.1,0.2,0.3,0.4,0.5  
0.6,0.7,0.8,0.9,1.0  
というふたつの予約に分割され実行されます。これは要素が多すぎてグリッドが大きくなってしまう場合などに有効でしょう。

## Adjust
これは、モデルの細部と色調を補正します。LoRAとは異なるメカニズムを採用しています。U-Netの入力と出力のポイントを調整することで、画像の細部と色調を調整できます。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-supermerger/images/fsample0.jpg)
## 使い方

テキストボックスに直接入力するか、スライダーで値を決めてから上ボタンを押してテキストボックスに反映できます。空白のままにしておくと無視されます。 
カンマで区切った7つの数字を入力してください。

```
0,0,0,0,0,0,0,0
```
これがデフォルトで、これらの値をずらすことで効果が現れます。

### Each setting value
### 各設定値の意味
8つの数字は以下に対応しています。
1. 描き込み/ノイズ  
2. 描き込み/ノイズ
3. 描き込み/ノイズ
4. コントラスト/描き込み  
5. 明るさ
6. 色調1 (シアン-赤)
7. 色調2 (マゼンタ-緑) 
8. 色調3 (黄-青)

描き込みが増えるほどノイズも必然的に増えることに注意してください。また、Hires.fixを使用する場合、出力が異なる可能性があるので、使用される設定でテストすることをおすすめします。

値は+/-5程度までは問題ないと思われますが、モデルによって異なります。正の値を入力すると描き込みが増えます。色調には3種類あり、おおむねカラーバランスに対応しているようです。
### 1,2,3 Detail/Noise
1.はU-Netの入り口に相当する部分です。ここを調節すると画像の描き込み量が調節できます。ここはOUTに比べて構図が変わりやすいです。マイナスにするとフラットに、そして少しぼけた感じに。プラスにすると描き込みが増えノイジーになります。通常の生成でノイジーでもhires.fixできれいになることがあるので注意してください。2,3はOUTに相当する部分です。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-supermerger/images/fsample1.jpg)

### 4. Contrast/Detail
ここを調節するとコントラストや明るさがかわり、同時に描き込み量も変わります。サンプルを見てもらった方が早いですね。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-supermerger/images/fsample3.jpg)

### 5,6,7,8 Brightness, Color Tone
明るさと色調を補正できます。概ねカラーバランスに対応するようです。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-supermerger/images/asample1.jpg)

## Let the Dice roll
ランダムにマージ比率を決定します。一度の複数のランダムマージが行えます。比率は各ブロック、各エレメントごとにランダムにすることが可能です。
## 使い方
　Let the Dice rollで使用できます。`Random Mode`を選択し`Run Rand`を押すと`Num of challenge`の回数分ランダムにウェイトが設定されて画像が生成されます。生成はXYZモードで動作するので`STOP`ボタンが有効です。`Seed for Random Ratio`は`-1`に設定して下さい。Num of challengeの回数が2回以上の場合、自動的に-1に設定されます。同じseedを使うと再現性があります。生成数が10を超える場合グリッドは自動的に2次元になります。`Settings`の`alpha`、`beta`はチェックするとランダム化されます。Elementalの場合`beta`は無効化されます。

## 各モード
### R,U,X
26ブロックすべてに対してランダムなウェイトが設定されます。`R`、`U`、`X`の違いは乱数の値の範囲です。Xは各層に対して`lower limit` ~ `upper limit`で指定します。
R : 0 ~ 1  
U : -0.5 ~ 1.5  
X : lower limit ~ upper limit
### ER,EU,EX
Elementすべてに対してランダムなウェイトが設定されます。`ER`、`EU`、`EX`の違いは乱数の値の範囲です。Xは各層に対して`lower limit` ~ `upper limit`で指定します。

### custom
ランダム化される階層を指定します。`costom`で指定します。  
`R`、`U`、`X`、`ER`、`EU`、`EX`が使用できます。
例：
```
U,0,0,0,0,0,0,0,0,0,0,0,0,R,R,R,R,R,R,R,R,R,R,R,R,R
U,0,0,0,0,0,0,0,0,0,0,0,0,ER,0,0,0,0,0,0,X,0,0,0,0,0
```
### XYZモード
typeに`random`を設定することで使用できます。ランダム化する回数を入力すると、回数分軸の要素が設定されます。
```
X type : seed, -1,-1,-1
Y type : random, 5
```
とすると、3×5のgridができ、5回分ランダムにウェイトが設定されたモデルで生成されます。ランダムかの設定はランダムのパネルで設定して下さい。ここが`off`では正常に動作しません。

### Settings
- `round` は丸める小数点以下の桁数を設定します。初期値は3で、0.123のようになります。
- `save E-list` はElementalのキーと割合をcsv形式で`script/data/`に保存します。

## Elemental Merge
[こちら](elemental_ja.md)を参照して下さい。



## Generation Parameters

ここでは画像生成の条件も設定できます。ここで値を設定すると優先されます。
## Include/Exclude
マージする(しない)階層を設定できます。  Includeの場合ここで設定した階層のみマージされます。Excludeは逆でここで設定した階層のみマージされません。「print」にチェックを入れると、コマンドプロンプト画面で階層が除外されたか確認できます。「Adjust」にチェックを入れると、Adjustで使用する要素がのみマージ/除外されます。`attn`などの文字列も指定でき、この場合`attn`を含む要素のみマージ/除外されます。文字列はカンマで区切ってください。
## unload button
現在ロードされているモデルを削除します。kohya-ss GUI使用時にGPUメモリを解放するために使用します。モデルが削除されると画像生成ができなくなります。画像生成を行いたい場合は、再度モデルを選択してください。


## LoRA
LoRA関連の機能です。基本的にはkohya-ssのスクリプトと同じですが、階層マージに対応します。現時点ではV2.X系のマージには対応していません。

注意：LyCORISは構造が特殊なため単独マージのみに対応しています。単独マージの比率は1,0のみ使用可能です。他の値を用いるとsame to Strengthでも階層LoRAの結果と一致しません。
LoConは整数以外でもそれなりに一致します。

LoCon/LyCoris のモデルへのマージにはweb-ui1.5以上が必要です。
|  1.X,2.X     | LoRA  | LoCon | LyCORIS |
|----------|-------|-------|---------|
| Merge to Model |   Yes   | Yes   | Yes     |
| Merge LoRAs   |    Yes   | Yes    | No     |
| Apply Block Weight(single)|Yes|Yes|Yes|
| Extract From Models   | Yes    | No    | No      |

|  XL     | LoRA  | LoCon | LyCORIS |
|----------|-------|-------|---------|
| Merge to Model |   Yes   | Yes   | Yes     |
| Merge LoRAs   |    Yes   | Yes    | No     |
| Extract From Models   | Yes    | No    | No      |


### Merge LoRAs
ひとつまたは複数のLoRA同士をマージします。kohya-ss氏の最新のスクリプトを使用しているので、dimensionの異なるLoRA同氏もマージ可能ですが、dimensionの変換の際はLoRAの再計算を行うため、生成される画像が大きく異なる可能性があることに注意してください。  

calculate dimentionボタンで各LoRAの次元を計算して表示・ソート機能が有効化します。計算にはわりと時間がかかって、50程度のLoRAでも数十秒かかります。新しくマージされたLoRAはリストに表示されないのでリロードボタンを押してください。次元の再計算は追加されたLoRAだけを計算します。

### Merge to Checkpoint
Merge LoRAs into a model. Multiple LoRAs can be merged at the same time.  
Enter LoRA name1:ratio1:block1,LoRA name2:ratio2:block2,... 
LoRA can also be used alone. The ":block" part can be omitted. The ratio can be any value, including negative values. There is no restriction that the total must sum to 1 (of course, if it greatly exceeds 1, it will break down).
To use ":block", use a block preset name from the bottom list of presets, or create your own. Ex:   
```
LoRAname1:ratio1
LoRAname1:ratio1:ALL
LoRAname1:ratio1:1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0
```

### Extract from checkpoints
ふたつのモデルの差分からLoRAを生成します。
demensionを指定すると指定されたdimensionで作製されます。無指定の場合は128で作製します。
alphaとbetaによって配合比率を調整することができます。$(\alpha A - \beta B)$　alpha, beta = 1が通常のLoRA作成となります。

### Extract from tow LoRAs
[こちら](calcmode_ja.md#extractlora)を参照して下さい。

### Metadata
#### create new
新しく最小限のMetadataを作製します。dim,alpha,basemodelのversion,filename,networktypeのみが作製されます。
#### merge
各LoRAの情報が保存され、タグがマージされます。
(各LoRAの情報はWeb-uiの簡易Metadata読み込み機能では見えません)
#### save all
各LoRAの情報が保存されます。
(各LoRAの情報はWeb-uiの簡易Metadata読み込み機能では見えません)
#### use first LoRA
最初のLoRAの情報をそのままコピーします

### Get Ratios from Prompt
プロンプト欄からLoRAの比率設定を読み込みます。これはLoRA Block Weightの設定も含まれ、そのままマージが行えます。

### Difference between Normal Merge and SAME TO STRENGTH
same to Strengthオプションを使用しない場合は、kohya-ss氏の作製したスクリプトのマージと同じ結果になります。この場合、下図のようにWeb-ui上でLoRAを適用した場合と異なる結果になります。これはLoRAをU-netに組み込む際の数式が関係しています。kohya-ss氏のスクリプトでは比率をそのまま掛けていますが、適用時の数式では比率が２乗されてしまうため、比率を1以外の数値に設定すると、あるいはマイナスに設定するとStrength（適用時の強度）と異なる結果となります。same to Strengthオプションを使用すると、マージ時には比率の平方根を駆けることで、適用時にはStrengthと比率が同じ意味を持つように計算しています。また、マイナスが効果が出るようにも計算しています。追加学習をしない場合などはsame to Strengthオプションを使用しても問題ないと思いますが、マージしたLoRAに対して追加学習をする場合はだれも使用しない方がいいかもしれません。  

下図は通常適用/same to Strengthオプション/通常マージの各場合の生成画像です。figma化とukiyoE LoRAのマージを使用しています。通常マージの場合はマイナス方向でも２乗されてプラスになっていることが分かります。
![xyz_grid-0014-1534704891](https://user-images.githubusercontent.com/122196982/218322034-b7171298-5159-4619-be1d-ac684da92ed9.jpg)

## Other tabs
## Analysis
2つのモデルの違いを分析してください。比較したいモデルを選んでください、モデルAとモデルBを。
### Mode

ASimilalityモードは、qkvから計算されたテンソルを比較します。他のモードは各要素のコサイン類似度から計算します。ASimilalityモード以外では計算された差が小さくなるようです。ASimilalityモードは出力画像の違いに近い結果を与えるため、一般的にはこのモードを使用すべきです。
このAsimilality分析は、[Asimilality script](https://huggingface.co/JosephusCheung/ASimilarityCalculatior)を拡張して作成されました。

### Block Method
ASimilalityモード以外のモードで各階層の比率を計算する方法です。Meanは平均を表し、minは最小値を表し、attn2は階層の計算結果としてattn2の値を出力します。

## History
マージ履歴を検索することができます。検索機能は「and」と「or」の両方の検索に対応しています。

## Elements
モデルに含まれるElementのリスト、階層の割り当て、およびテンソルのサイズを取得できます。

## 謝辞
このスクリプトは[kohya](https://github.com/kohya-ss)氏,[bbc-mc](https://github.com/bbc-mc)氏のスクリプト一部使用しています。また、拡張の開発に貢献した全ての方々にも感謝します。
