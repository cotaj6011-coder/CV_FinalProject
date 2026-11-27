# FactoryVision 期末專題

題目：**智慧工廠鋼材表面瑕疵辨識與定位系統**。

組別：**Team07**。

## 資料與執行環境

- [資料盤點](dataset_audit.json)：影像、標註、固定 split 與完整性檢查。
- [執行環境](environment.json)：Python、套件及硬體版本。
- [Stage 2 驗證集複核](audits/stage2_validation_audit.json)：獨立推論、重訓與跨 split 裁切重疊證據。
- 原始課程附件與資料集不納入此儲存庫。取得資料後，將 `STUDENT_PUBLIC/` 放在專案的 `data/` 底下；程式不更動原始影像、標註或指定 split。

實驗資料包含分類及偵測各 Train 1,260／Validation 270 張，Hidden data 未提供。保留原指定切分。後續複核確認：分類 Validation 至少148／270張（54.8%）與 Train 共用至少25%面積的完全相同像素，共217對（216對裁切重疊及1對整張重複）。這是原始資料 ZIP 內已存在的重疊；數量是已確認下限。

## 專題內容

| 階段 | 完整實驗設計 | 程式與證據 |
|---|---|---|
| Stage 1 | PCA+SVM；自製 CNN 的 E1、A、B、A+B，四組各40 epochs | [Stage1](submission/Team07_Stage1/README.md) |
| Stage 2 | ResNet18、MobileNetV3-Small × Feature Extraction、Full Fine-tuning，四組各15 epochs | [Stage2](submission/Team07_Stage2/README.md) |
| Stage 3 | 官方 COCO 預訓練 YOLO11n，60 epochs；threshold、IoU、失敗及效率分析 | [Stage3](submission/Team07_Stage3/README.md) |

所有正式實驗 seed=42。預訓練只用一般 ImageNet／COCO 權重，未使用已針對瑕疵資料微調的網路模型。每個 Stage 的 `src/`、config、README、requirements 與最終 checkpoint 可獨立使用。

## 實際完成結果

| 階段 | 最終模型 | 固定 Validation 結果 |
|---|---|---|
| Stage 1 | E2：SurfaceCNN + 資料擴增，epoch 37 | Accuracy 97.78%、Macro F1 97.76% |
| Stage 2 | MobileNetV3-Small / Feature Extraction，epoch 2 | Accuracy 100.00%、Macro F1 100.00%（270/270） |
| Stage 3 | YOLO11n，epoch 49 | mAP@0.5 73.35%；confidence=0.25 時 Micro Precision 70.68%、Recall 69.58% |

完整完成四組 CNN 各40 epochs、四組遷移學習各15 epochs、偵測60 epochs。Stage3 附12張GT／Prediction對照、5個真實失敗案例及confidence門檻分析。所有最終 checkpoint 皆已重新載入評估；上述結果屬模型選擇用的Validation，不是獨立Hidden test成績。

Stage2 的270／270經獨立讀取原始ZIP推論及相同seed重訓確認；MobileNet FE僅第2、9輪達100%，第15輪為269／270。驗證集同時用於選模型與epoch，加上跨 split 裁切重疊，因此100%不能代表獨立來源的泛化準確率，也不能僅憑重疊數量推算高估幅度。原PDF報告記錄了整張重複；新增的裁切重疊結果以本頁與複核紀錄補充，見[重疊對照圖](audits/stage2_overlap_examples.png)。

修訂版部署計時統一為RTX3060、FP32、batch=1、20次暖機、全部270張Validation（六類各45張）。包含記憶體BGR影像前處理、GPU傳輸、模型、後處理及結果回CPU，排除磁碟I/O與模型載入；各Stage的 `experiments/deployment_benchmark.json` 保存逐張時間、影像ID及硬體。輸入尺寸與任務不同，不把應用流程成本當成純架構速度比。

Stage3推論權重已由15.30MiB訓練快照匯出為5.21MiB推論檔，保留原快照。499個模型張量完全相等；重新評估270張的指標及逐圖預測完全一致。正式報告補入Stage1配對錯分、Stage2更新成本、Stage3具體失敗分析及暫緩直接產線部署的量化理由。

## 交付檔位置

總報告為 [Team07_Final_Report.pdf](output/Team07_Final_Report.pdf)。Stage 報告位於 `submission/Team07_StageN/stageN_report.pdf`。三個 Stage ZIP 可由 `tools/package_submission.py` 在本機產生，不重複納入 Git。

`experiments/` 保留實驗 metrics、history、預測、圖表與選模證據；分類模型的獨立重載驗證位於 `evaluation/`。最終模型與選中偵測訓練快照一併提供。Stage3權重中的本機路徑已改為相對路徑，模型張量保持相同；目前檔案大小、SHA256與去識別驗證見 `submission/Team07_Stage3/experiments/checkpoint_export.json`。PDF中的權重檔案大小為報告產生時的紀錄。

`.gitignore` 排除本機環境、原始附件與資料、準備資料副本、暫存、舊版本、未審查的根目錄文件與認證檔案。提交作者使用 GitHub noreply 信箱。

## 重現

本機 Python 環境：`.venv/Scripts/python.exe`，CUDA 12.4、PyTorch 2.6.0、RTX 3060 12GB。所有安裝版本見 `environment.json` 與各 Stage 的 `requirements.txt`。

`tools/validate_submission.py` 可在本機打包後核對必要檔案、設定、ZIP內外一致性、根目錄、大小、CRC、部署計時與權重匯出證據。PDF已經渲染視檢。

各 Stage README 已列出不需修改程式的完整 train／validation 命令。報告生成工具在 `tools/`，與提交模型的執行相依獨立：

```text
tools/build_reports.py --stage 1|2|3|final|all
tools/package_submission.py
tools/validate_submission.py
tools/benchmark_deployment.py --stage 1|2|3 --data-root data/STUDENT_PUBLIC
```

報告工具使用 reportlab 與微軟正黑體；驗證工具使用 pypdf／Pillow。`tmp/` 是報告渲染及檢查暫存，不納入 Git。
