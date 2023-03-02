# Changelog
### bug fix 2023.02.19.2330(JST)
いくつかのバグが修正されました
- LOWRAMオプション有効時にエラーになる問題
- Linuxでエラーになる問題
- XY plotが正常に終了しない問題
- 未ロードのモデルを設定時にエラーになる問題

### update to version 3 2023.02.17.2020(JST)
- LoRA関係の機能を追加しました
- Logを保存し、設定を呼び出せるようになりました
- safetensors,fp16形式での保存に対応しました
- weightのプリセットに対応しました
- XYプロットの予約が可能になりました

### bug fix 2023.02.19.2330(JST)
Several bugs have been fixed
- Error when LOWRAM option is enabled
- Error on Linux
- XY plot did not finish properly
- Error when setting unused models

### update to version 3 2023.02.17.2020(JST)
- Added LoRA related functions
- Logs can now be saved and settings can be recalled.
- Save in safetensors and fp16 format is now supported.
- Weight presets are now supported.
- Reservation of XY plots is now possible.

### bug fix 2023.01.29.0000(JST)
pinpoint blocksがX方向で使用できない問題を修正しました。  
pinpoint blocks選択時Triple,Twiceを使用できない問題を解決しました  
XY plot 使用時に一部軸タイプでMBWを使用できない問題を解決しました  
Fixed a problem where pinpoint blocks could not be used in the X axis.  
Fixed a problem in which Triple,Twice could not be used when selecting pinpoint blocks.  
Problem solved where MBW could not be used with some axis types when using XY plot.

### bug fixed 2023.01.28.0100(JST)
MBWモードでSave current modelボタンが正常に動作しない問題を解決しました
ファイル名が長すぎて保存時にエラーが出る問題を解決しました
Problem solved where the "Save current model" button would not work properly in MBW mode
Problem solved where an error would occur when saving a file with too long a file name

### bug fixed 2023.01.26.2100(JST)
XY plotにおいてタイプMBWが使用できない不具合を修正しました
Fixed a bug that type of MBW could work in XY plot

### updated 2023.01.25.0000(JST)
added several features  
- added new merge mode "Triple sum","sum Twice"  
- added XY plot  
- 新しいマージモードを追加しました "Triple sum","sum Twice"  
- XY plot機能を追加しました  

### bug fixed 2023.01.20.2350(JST)
png infoがうまく保存できない問題を解決しました。  
Problem solved where png info could not be saved properly.
