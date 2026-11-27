# FactoryVision Stage 1

題目：智慧工廠鋼材表面瑕疵辨識與定位系統之傳統分類與自製 CNN。

## 環境與執行

Python 3.11。以下命令在本 Stage 資料夾執行；`DATA_ROOT` 請替換為課程
`classification` 資料夾路徑，該資料夾須包含 `train/` 和 `validation/`。
程式與 checkpoint 不依賴作者的絕對路徑。

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python src/train.py --data-root "DATA_ROOT" --device cuda
python src/evaluate.py --data-root "DATA_ROOT" --device cuda
```

無 CUDA 時把 `--device cuda` 改為 `--device cpu`。程式亦可從其他工作目錄
以 `python <stage路徑>/src/train.py ...` 執行；輸出仍寫入本 Stage。
`--data-root` 相對路徑相對於執行時的工作目錄。

單獨重跑與驗證管線：

```powershell
python src/train.py --data-root "DATA_ROOT" --experiment traditional
python src/train.py --data-root "DATA_ROOT" --experiment E1 --device cuda
python src/train.py --data-root "DATA_ROOT" --experiment E2 --device cuda
python src/train.py --data-root "DATA_ROOT" --experiment E3 --device cuda
python src/train.py --data-root "DATA_ROOT" --experiment E4 --device cuda
python src/train.py --data-root "DATA_ROOT" --experiment finalize --device cuda
python src/train.py --data-root "DATA_ROOT" --experiment E1 --smoke --device cpu
```

Smoke 僅跑一個 epoch、兩個訓練 batch，結果獨立放 `experiments/smoke/`，
不產生正式最終模型、不可當正式實驗。

## 設計與公平比較

- 固定課程 split，train 1,260、validation 270；主要實驗 seed=42。
- 類別順序固定為 crazing、inclusion、patches、pitted_surface、rolled_in_scale、scratches。
- 傳統方法：灰階64×64像素/255，PCA100（whiten）+ RBF SVM（C=10、gamma=scale）。PCA及SVM只fit train。
- 自製 CNN：灰階128×128，Conv32/64/128 各接 ReLU、MaxPool，AdaptiveAvgPool4×4，Linear128/ReLU/Dropout/Linear6。從零訓練、不載入預訓練模型。
- E1 baseline 只 resize/tensor/normalize；E2 加 A 資料擴增（水平、垂直翻轉各 p=0.5，旋轉±10度）；E3 加 B Dropout0.5；E4 同時 A+B。
- 四組固定 Adam lr=0.001、weight_decay=0、batch64、40 epochs；無 early stopping/scheduler。每組選驗證 Macro F1 最佳 epoch，最終 improved CNN 從 E2/E3/E4 選擇。
- `config.json` 由 finalize 更新為最終 checkpoint 的設定；所有實驗設定及結果保留。

## 輸出

- `stage1_model.pth`：最終 improved CNN checkpoint（含 state_dict、架構設定、類別順序與最佳 epoch）。
- `experiments/summary.json`、`comparison.csv`：完整傳統方法及四組 CNN 比較。
- 各實驗目錄：metrics、predictions、confusion matrix；CNN 另有 history、train/validation loss/accuracy curves、最佳 checkpoint、實際錯誤案例。
- `evaluation/`：從最終 checkpoint 重新驗證結果。
- `report_notes.md`：實驗設計、結果與分析紀錄；`stage1_report.pdf` 為正式報告。

## 資料與測量說明

僅使用課程提供的 classification 資料。課程固定切分存在完全相同的 patches
影像對 `train/patches/TRN_0504.png` 與 `validation/patches/VAL_0101.png`；
遵守固定 split 保留，報告揭露可能使驗證成績略為樂觀。此處沒有獨立測試集。

CNN部署計時使用全部270張Validation（六類各45張），batch=1、FP32、20次暖機，CUDA前後同步。
從記憶體BGR影像開始，包含灰階轉換、縮放／標準化、GPU傳輸、forward與softmax/argmax及結果回CPU；排除磁碟I/O與模型載入。
完整流程計時存於 `experiments/deployment_benchmark.json` 及 `evaluation/metrics.json`；實驗目錄中的forward-only計時不含前後處理，與完整流程採不同範圍。SVM計時為CPU PCA transform+predict。
Stage2/3採相同計時邊界、裝置及影像ID；各任務使用其官方原圖版本與輸入尺寸，結果代表應用成本，非純架構速度比。
訓練曲線的 train 指標在訓練模式測量，可能含 augmentation/dropout；validation 為 eval 模式。
