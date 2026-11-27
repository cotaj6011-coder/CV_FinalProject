# Stage 2 實驗完成紀錄

## 實驗與選模

固定 Train 1,260 / Validation 270、六類各210/45張、seed=42。
ResNet18、MobileNetV3-Small 為兩個不同架構家族；各比較 Feature Extraction（FE）與 Full Fine-tuning（FT）。
兩者均使用官方 torchvision ImageNet-1K V1 權重，未使用 FactoryVision fine-tuned 權重。
四組完整訓練15 epochs、batch32、224×224、RGB、ImageNet normalization、無 augmentation。
AdamW，head lr=0.001，FT backbone lr=0.0001，weight_decay=0.0001，CosineAnnealingLR T_max=15。

| 實驗 | Accuracy | Macro Precision | Macro Recall | Macro F1 | 最佳 epoch | 參數 | 可訓練參數 | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small FE | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2 | 1,524,006 | 596,998 | 6,217,034 |
| MobileNetV3-Small FT | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 7 | 1,524,006 | 1,524,006 | 6,217,034 |
| ResNet18 FE | 0.992593 | 0.992673 | 0.992593 | 0.992592 | 8 | 11,179,590 | 3,078 | 44,793,022 |
| ResNet18 FT | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2 | 11,179,590 | 11,179,590 | 44,793,022 |

選擇 MobileNetV3-Small FE，第2 epoch 的 checkpoint。三組最佳checkpoint均在Validation取得270/270正確。
選模規則依 validation Macro F1 最高，同分取較小 checkpoint，再同分取較少可訓練參數，前三項仍相同才依實驗名稱排序。
其 trainable parameters 比同架構 FT 少927,008，但此數字是更新成本分析，不是額外精度優勢。
MobileNet FE/FT 最後第15 epoch各有1張錯分，報告與提交的是早先保存的最佳checkpoint，而非最後epoch。
ResNet18 FT 相對 FE 的 Macro F1 提升約0.7408個百分點；MobileNet 的 FE 與 FT 最佳Macro F1相同。

## 失敗與泛化分析

最終模型270/270正確，每類45/45；沒有最終失敗影像，`evaluation/failure_cases` 為空。
僅 ResNet18 FE 的最佳checkpoint有2個錯誤：

- `validation/inclusion/VAL_0083.png`：inclusion → pitted_surface，預測分數0.761701；可見低對比、細長暗紋。
- `validation/scratches/VAL_0228.png`：scratches → inclusion，預測分數0.635071；可見面積小的細長亮痕。

上述影像觀察不能證明錯分成因。兩張在最終MobileNet FE均分類正確。
失敗圖展示 R18-FE 的錯誤，最終 MobileNet FE 在這兩張均分類正確。

FE 的 backbone 及 BatchNorm running statistics 凍結；ResNet18只訓練新fc，MobileNet訓練整個classifier（保留hidden Linear與新六類輸出）。
FT 更新全模型；MobileNet classifier Dropout 在 train 模式開啟。train/validation模式不同，因此不能將曲線差距全部視為過擬合。

MobileNet第1 epoch：FE validation F1=0.988888、loss=0.046104；FT F1=0.569581、loss=1.493422。FT早期波動可能與backbone及BatchNorm同時適應資料有關；未做單獨凍結BN的控制實驗，不能視為已證實因果。
最終FE模型信心最低的正確例為VAL_0084：GT/prediction均為inclusion，分數0.516610，第二候選scratches為0.469926。這是分數接近的正確例；100% accuracy不表示各樣本有同等決策餘裕或信心已校準。

Validation 被用來選epoch與模型，100%不是獨立test成績；單一seed、較小驗證集和固定split原有重複影像限制泛化解讀。
重複影像為 train/patches/TRN_0504.png 與 validation/patches/VAL_0101.png，遵守規定保留原split。

## 效率與重現

最終checkpoint為6,217,034 bytes（約5.93 MiB）。

| 實驗 | 可訓練參數 | 完整15 epochs時間 |
|---|---:|---:|
| MobileNet FE | 596,998 | 44.804075秒 |
| MobileNet FT | 1,524,006 | 48.436621秒 |
| ResNet FE | 3,078 | 48.419227秒 |
| ResNet FT | 11,179,590 | 71.463770秒 |

計時包含訓練epoch迴圈、每輪驗證與最佳checkpoint寫入；排除初始模型／資料準備及最後重載評估。MobileNet FE少更新927,008個參數（60.827%），本次少3.633秒；只有單次紀錄，不能作跨裝置固定效益。相同架構FE/FT推論圖相同，不能從較少可訓練參數推出更快推論。

部署計時採用 `experiments/deployment_benchmark.json` 的四模型完整流程比較：相同270張Validation、六類各45張、224×224、RTX3060、batch1 FP32、20次暖機、CUDA同步。包含CPU前處理、CPU/GPU傳輸、forward與分類結果回CPU；排除磁碟讀圖與模型載入。影像ID、各樣本延遲、平均/P50/P95均保留。`evaluation/metrics.json` 保存最終模型的完整流程計時；相機、通訊與PLC仍未量測。

正式結果來源：`experiments/summary.json`、各組 `metrics.json/history.json/predictions.json`、`evaluation/metrics.json`。
最終evaluate重新載入stage2_model.pth的分類指標與所選最佳checkpoint一致。
曲線與confusion matrix由plot_history/plot_confusion依保存的JSON紀錄繪製。
