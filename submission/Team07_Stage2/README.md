# FactoryVision Stage 2：Transfer Learning

使用固定分類資料切分，比較 **ResNet18／MobileNetV3-Small × Feature Extraction／Full Fine-tuning** 四組實驗。以 Validation Macro F1 選擇最終模型；平手依序選 checkpoint 較小、可訓練參數較少者，全部相同才依實驗名稱排序。

## 本次完成結果

四組均完成15 epochs。最終為 **MobileNetV3-Small / Feature Extraction，第2 epoch**，
Validation 270/270正確，Accuracy、Macro Precision、Macro Recall、Macro F1均為1.0。
ResNet18 FT、MobileNet FE、MobileNet FT三組的最佳checkpoint均全對；
ResNet18 FE為268/270、Macro F1=0.992592。
MobileNet兩個策略同分、同checkpoint大小，依較少可訓練參數選FE：596,998對1,524,006。
本次FE／FT的15 epochs訓練迴圈時間分別44.80／48.44秒，包含每epoch驗證與儲存最佳權重，排除模型建立與最終評估；時間僅作本機成本佐證，不作浮動的選模排序條件。
FE訓練596,998個參數，完整模型1,524,006個參數，checkpoint約5.93 MiB。

RTX3060單張FP32完整分類流程實測平均5.288ms、P95 6.511ms，量測全部270張Validation影像；包含前處理、CPU/GPU傳輸、forward與結果回CPU，完整範圍與重現步驟見下文。
最終模型沒有錯分，因此沒有最終failure image。報告另明確標示未選中ResNet18 FE的2張實際錯誤。
以上為用於選模型與epoch的Validation結果，**不是獨立test成績**；原固定split的重複影像已揭露。
完整四組表格、曲線與分析見 `stage2_report.pdf`、`experiments/summary.json`、`report_notes.md`。

## 重現環境

- Python 3.11.9。
- PyTorch 2.6.0（CUDA 12.4 build）、torchvision 0.21.0。
- GPU 訓練環境：NVIDIA GeForce RTX 3060，12 GB VRAM。
- 其他直接相依套件與完整版本見 `requirements.txt`，已依本次實際安裝版本固定。

在本資料夾執行命令。先安裝與執行環境相符的 PyTorch／torchvision，再安裝其餘相依套件：

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

訓練前已完成四組 CPU smoke check：官方預訓練權重成功載入、真實影像 forward／backward 與梯度數值有限；Feature Extraction 僅更新 classifier，backbone 與 BatchNorm buffers 均保持不變；Full Fine-tuning 可更新 backbone。資料載入確認 Train 1,260／Validation 270 與指定類別順序一致。此檢查不產生正式實驗成績或 checkpoint。

## 資料與前處理

`--data` 指向附件解壓後的 `classification/`，其中應包含 `train/<class>/*.png` 與 `validation/<class>/*.png`。本專案原位置的相對路徑為 `../../data/STUDENT_PUBLIC/classification`；移動專案後只需調整命令列路徑，不需修改程式。

- Train：1,260 張；Validation：270 張。維持助教指定 split。
- 類別順序固定為 `crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches`。
- `ImageFolder` 將輸入讀成 RGB；resize 至 224 × 224、轉成 tensor，使用 ImageNet mean `[0.485, 0.456, 0.406]` 與 std `[0.229, 0.224, 0.225]` 正規化。
- 四組均無資料增強；Validation 不參與梯度更新。
- 原附件的 `train/patches/TRN_0504.png` 與 `validation/patches/VAL_0101.png` 像素完全相同。保留固定 split，並在報告中揭露此資料限制。

## 架構、預訓練來源與策略

| 架構 | torchvision weights | 公開權重來源 |
|---|---|---|
| ResNet18 | `ResNet18_Weights.IMAGENET1K_V1` | [官方權重](https://download.pytorch.org/models/resnet18-f37072fd.pth) |
| MobileNetV3-Small | `MobileNet_V3_Small_Weights.IMAGENET1K_V1` | [官方權重](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) |

首次訓練會下載 torchvision 的 ImageNet 預訓練權重。未使用已對 FactoryVision／鋼材瑕疵資料微調的網路 checkpoint。Validation 評估僅讀取本地 checkpoint，不下載預訓練權重。

- `feature_extraction`：凍結 backbone 參數與 BatchNorm running statistics。ResNet18 訓練新的 `fc`；**MobileNetV3-Small 訓練整個 `classifier`，包含保留的第一個 Linear 與新建的六類輸出 Linear**，不是只訓練最後一層。MobileNet classifier 的 Dropout 在訓練時啟用。
- `full_fine_tuning`：訓練整個模型；分類 head learning rate 為 `1e-3`，backbone 為 `1e-4`。
- 四組共用：15 epochs、batch size 32、input size 224、seed 42、Cross Entropy、AdamW、weight decay `1e-4`、CosineAnnealingLR。Feature Extraction 的 head learning rate 亦為 `1e-3`。
- 每個實驗重設 Python／NumPy／PyTorch seed，關閉 cuDNN benchmark、開啟 deterministic；不同硬體或套件版本的數值仍可能略有差異。

## 完整訓練命令

```powershell
python src/train.py --data ../../data/STUDENT_PUBLIC/classification --epochs 15 --input-size 224 --batch-size 32 --seed 42 --device cuda
```

預設 `--architecture all --strategy all`，依序完成四組實驗。可使用下列選項單獨重跑其中一組：

```powershell
python src/train.py --data ../../data/STUDENT_PUBLIC/classification --epochs 15 --input-size 224 --batch-size 32 --seed 42 --device cuda --architecture mobilenet_v3_small --strategy feature_extraction
```

`--architecture` 可為 `all`、`resnet18`、`mobilenet_v3_small`；`--strategy` 可為 `all`、`feature_extraction`、`full_fine_tuning`。單組重跑後，程式會重新彙整 `experiments/` 中的既有結果；正式四組比較需使用相同的 seed、epochs、input size 與 batch size，不混用不同設定的舊實驗。CPU 執行可指定 `--device cpu`。

每組依當次 Validation Macro F1 保存最佳 epoch 的 `best.pth`，再載入該 checkpoint 重算最終 metrics。訓練完成後，四組最佳 checkpoint 中的選中模型會複製至 `stage2_model.pth`；其 architecture、strategy 與 epoch 記錄在 `config.json` 及 `experiments/summary.json`，報告應與該選擇一致。重訓會保留既有 `config.json` 的 `team_id`，不會覆寫組號。

## Validation 評估命令

```powershell
python src/evaluate.py --data ../../data/STUDENT_PUBLIC/classification --checkpoint stage2_model.pth --output evaluation --device cuda
```

指標包含 Accuracy、Macro Precision、Macro Recall、Macro F1、Cross Entropy、per-class metrics 與 Confusion Matrix。`evaluation/` 另保存逐圖預測、至多六張高信心誤分類案例及推論時間。

推論時間以全部270張Validation（六類各45張）、batch=1、FP32、20次warm-up計算mean／P50／P95；CUDA前後同步。從記憶體BGR影像計時，包含RGB轉換、縮放／標準化、CPU/GPU傳輸、forward與softmax/argmax及結果回CPU，排除磁碟I/O與模型載入。四個已訓練模型的同條件比較存於 `experiments/deployment_benchmark.json`，最終模型計時同步於 `evaluation/metrics.json`。Stage1/3採相同邊界，但輸入尺寸與任務不同，不當成純架構速度比或產線吞吐保證。

## 輸出檔案

- `stage2_model.pth`：最終選中模型 checkpoint；包含 architecture、strategy、class order、前處理、seed 與 best epoch。
- `config.json`：符合規定的最終模型設定；checkpoint 路徑使用相對檔名。
- `experiments/<architecture>__<strategy>/`：各組 `best.pth`、`metrics.json`、`history.json`、`history.csv`、`predictions.json`、`curves.png` 與 `confusion_matrix.png`。
- `experiments/summary.json`：四組結果及選模規則。
- `evaluation/`：選中模型的最終 Validation 證據與效能量測。
- `stage2_report.pdf`：四組比較、曲線、參數／可訓練參數／checkpoint 大小、錯誤分析與部署決策。
