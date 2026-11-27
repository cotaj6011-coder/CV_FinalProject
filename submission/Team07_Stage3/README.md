# FactoryVision Stage 3：YOLO11n 瑕疵偵測

本 Stage 使用官方 COCO 預訓練 YOLO11n，在課程 FactoryVision Detection Public Dataset 的固定 Train 1,260 張影像上微調。Validation 固定 270 張，用於模型選擇及分析；不使用瑕疵資料集的現成 fine-tuned checkpoint。

## 本次完成結果

完整訓練60 epochs，依 Validation mAP@0.5 選中 epoch49（`epoch48.pt`）。最終 checkpoint 重新載入評估：mAP@0.5 **73.35%**、mAP@0.5:0.95 **42.87%**。confidence=0.25 的一對一配對統計為 TP446／FP185／FN195，Micro Precision **70.68%**、Recall **69.58%**。

單張FP32推論流程量測全部270張影像，經20次暖機，包含preprocessing、傳輸、prediction API、NMS與結果回CPU，排除磁碟讀圖及模型載入；平均與P95見 `experiments/deployment_benchmark.json`。報告包含12張GT／Prediction對照、5個失敗案例與信心門檻分析，詳見 `experiments/summary.json`、`report_notes.md`。

## 環境與模型

- Python 3.11；PyTorch 2.6.0、torchvision 0.21.0、Ultralytics 8.3.238；其餘版本見 `requirements.txt`。
- 模型：Ultralytics YOLO11n，卷積 backbone／neck、anchor-free detection head；訓練程式調用公開框架實作。
- 預訓練來源：[官方 COCO checkpoint](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)。
- 最終 checkpoint：`stage3_model.pt`，以本 Stage 根目錄為基準的相對路徑載入。
- 設定：`config.json`。

## 安裝

在此 Stage 根目錄執行：

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

上述安裝指令使用本次 RTX 3060 實驗的 CUDA 12.4 wheel；CPU 可在下列指令使用 `--device cpu`。所有程式均不需要修改原始碼。

## 訓練

`--data-root` 指向課程資料集解壓後的 `STUDENT_PUBLIC`，並非其 `detection` 子目錄。以下示例適用於本 workspace 的目錄配置：

```powershell
python src/train.py --data-root ../../data/STUDENT_PUBLIC --device 0
```

訓練使用 seed 42、60 epochs、batch 16、320×320、AdamW。完整 optimizer、augmentation、AMP 與 deterministic 設定見 `config.json`。輸入 200×200 RGB 圖像以 letterbox 調整為模型輸入尺寸，像素除以 255；不使用 ImageNet mean/std normalization。訓練 augmentation 僅作用於 Train。

`src/dataset.py` 將來源的 `images`／`labels_yolo` 複製成 `experiments/prepared_data/images/{train,val}` 與 `labels/{train,val}`，建立相容 YAML。原始 dataset 完全不變，split 與標註也不重分配。多類別影像逐框讀取原始 YOLO 類別，不以分類目錄標籤代替。

每個 epoch 儲存 checkpoint；以固定 Validation 的 mAP@0.5 最高者選模，分數相同時保留較早 epoch。選模紀錄寫入 `experiments/selection.json`，選中快照保留於原位置，再以官方 `strip_optimizer` 匯出 `stage3_model.pt`，移除 optimizer 等續訓狀態，保存相同 EMA 權重。這與 Ultralytics 預設 `best.pt` 的選擇準則有別，須依本程式產物復現。

## 推論權重與訓練快照

`stage3_model.pt` 為推論用 checkpoint，5,458,948 bytes（5.21 MiB）；選中 `experiments/train/weights/epoch48.pt` 為續訓快照，16,032,468 bytes（15.29 MiB），保留於交付包。公開版本將兩檔的本機訓練路徑改為 Stage3 相對路徑；所有序列化張量與固定輸入的CPU推論結果均保持相同，完整對照見 `experiments/checkpoint_export.json` 的 `privacy_sanitization`。兩者的499個模型 state tensors 經逐項核對，數值與 dtype 完全相同。浮點權重採 FP16 儲存；評估及計時仍以 FP32 載入（`half=False`）。Stage1／2 權重採 FP32 儲存，因此磁碟大小比較包含儲存精度差異，不能視為相同格式的參數效率比較。

匯出不會變更原始快照。訓練完成會自動執行；單獨重做匯出：

```powershell
python src/export.py
```

本次已重新載入推論權重，以 FP32 對270張 Validation 完整重算；官方指標、逐圖預測、固定門檻統計、threshold sweep與失敗案例均與匯出前逐值相同。`experiments/checkpoint_export.json` 保存來源、大小、SHA256、張量與驗證結果等價檢查。重新匯出後 `validation_equal` 會先重設為 `null`，必須重新驗證後才能宣稱驗證結果相同。

## Validation 評估

```powershell
python src/evaluate.py --data-root ../../data/STUDENT_PUBLIC --device 0
```

主要產物：

- `experiments/summary.json`：官方 mAP、各類 AP、固定 threshold P/R/F1、參數與環境資訊。
- `experiments/validation_predictions.json`：每張影像的 GT 與 prediction，所有框均為原始影像座標 `xyxy_pixel`。
- `experiments/threshold_sweep.json`：confidence 0.05～0.90 的比較。
- `experiments/failure_cases.json`：五個失敗案例的預測、GT與匹配結果。
- `experiments/figures/`：至少十張 GT／Prediction 對照、失敗案例、confidence curve、各類 AP。
- `experiments/official_validation/`：框架輸出的 PR／F1／confusion matrix 等圖表。

評估區分兩種統計：

1. 官方 AP：最低 confidence=0.001，NMS IoU=0.7，取得 mAP@0.5／mAP@0.5:0.95 與各類 AP。官方摘要 Precision／Recall 是框架在最大平均 F1 信心位置的結果，不應誤稱為 confidence=0.25 的結果。
2. 固定操作點：confidence=0.25、匹配 IoU≥0.5，依信心由高到低，在同類別未匹配 GT 中選最高 IoU，一對一匹配。額外輸出 micro／macro P、R、F1 與各類 TP／FP／FN。NMS IoU=0.7 用於抑制重複預測，與評估匹配 IoU=0.5 是不同用途。

## 推論時間

先停止其他模型訓練，再執行：

```powershell
python src/evaluate.py --data-root ../../data/STUDENT_PUBLIC --device 0 --benchmark-only
```

batch=1、FP32，20次暖身後量測全部270張Validation影像（六類各45張）；BGR影像已讀入記憶體，排除磁碟I/O及模型載入，CUDA計時前後同步。完整流程包含preprocessing、CPU/GPU傳輸、模型、NMS、prediction API overhead與結果回CPU，另列框架forward time。同一次 `--benchmark-only` 會同步更新 `experiments/summary.json` 的 latency、`experiments/latency.json` 與 `experiments/deployment_benchmark.json`。與Stage1/2使用相同計時邊界、裝置與影像ID；各任務保留官方原圖版本及輸入尺寸。記錄mean／median／p95、硬體及套件版本，不把此量測等同純架構速度比或產線吞吐量。

## 檔案與提交

正式 Stage 根目錄需要 `stage3_report.pdf`、`stage3_model.pt`、`config.json`、`requirements.txt`、`README.md` 及 `src/`。實驗證據存於 `experiments/`；`prepared_data` 不重複打包；epoch checkpoint 僅保留選中的 `epoch48.pt` 續訓快照，另保留最終推論權重及報告所引用的證據。報告包含實驗數據、GT／Prediction對照與失敗分析。

方法參考：[訓練 API](https://docs.ultralytics.com/modes/train/)、[Validation API](https://docs.ultralytics.com/modes/val/)、[Prediction API](https://docs.ultralytics.com/modes/predict/)、[YOLO11 模型](https://docs.ultralytics.com/models/yolo11/)。
