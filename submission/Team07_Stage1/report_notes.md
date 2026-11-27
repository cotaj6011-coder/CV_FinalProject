# Stage 1 實驗紀錄

## 實驗設計

固定課程 train/validation split，seed=42。CNN 僅用灰階影像 resize 128×128、tensor、固定 mean/std=0.5。
自製三層 Conv(32/64/128) + ReLU + MaxPool，AdaptiveAvgPool(4×4)、Linear128、ReLU、Dropout、Linear6；所有權重從零訓練。
A 為資料擴增（水平/垂直翻轉各 p=0.5、旋轉 ±10°）；B 為 classifier Dropout p=0.5。所有 CNN 使用 Adam lr=0.001、weight_decay=0、batch_size=64、40 epochs；不使用 early stopping 或 learning-rate scheduler。
每組依 validation Macro F1 選 epoch，完整訓練40 epochs。最終 improved CNN 僅從 E2/E3/E4 選 Macro F1 最高者；tie 保留最早 epoch/較前實驗。
傳統方法以64×64灰階像素/255作特徵，PCA100(whiten=True) + RBF SVM(C=10,gamma=scale)；PCA、SVM均只fit training。

## 結果

| 實驗 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| traditional | 0.7444 | 0.7455 | 0.7444 | 0.7392 |
| E1 | 0.9741 | 0.9743 | 0.9741 | 0.9740 |
| E2 | 0.9778 | 0.9777 | 0.9778 | 0.9776 |
| E3 | 0.9704 | 0.9725 | 0.9704 | 0.9701 |
| E4 | 0.9519 | 0.9560 | 0.9519 | 0.9506 |

最終選擇 E2，best epoch=37，Macro F1=0.9776。
A only、B only、A+B 相對baseline的Macro F1差為 +0.0035、-0.0039、-0.0234。

## E1 → E2 的逐張改善

使用各自最佳 checkpoint 的相同270張驗證預測配對：E1錯7張，E2錯6張；修正2張、新增1張錯誤、共同錯5張，淨多1張正確。Macro F1增加約0.35個百分點，不代表全面解決混淆或統計顯著。

| 影像 | GT | E1 預測（信心） | E2 預測（信心） | 變化 |
|---|---|---|---|---|
| VAL_0097 | patches | pitted_surface（0.8053） | patches（0.6695） | 修正 |
| VAL_0236 | scratches | pitted_surface（0.8860） | scratches（0.6650） | 修正 |
| VAL_0082 | inclusion | inclusion（0.5865） | pitted_surface（0.7789） | 新增錯分 |

共同錯誤：VAL_0083、VAL_0144、VAL_0159、VAL_0160、VAL_0177。
圖與配對紀錄為 `experiments/E1_E2_paired_changes.png/json`，來源為 E1/E2 的 `predictions.json`。
VAL_0097可見暗色塊狀區，VAL_0236可見細長亮直線，VAL_0082則是較淡的細長暗紋。這是可見影像與預測對照，不能證明特定旋轉或翻轉造成修正／退步。

## 訓練曲線與泛化

| 實驗 | 最佳 epoch | 最佳 epoch train/val accuracy | 最終 epoch train/val accuracy | 最終 epoch train/val loss |
|---|---:|---|---|---|
| E1 | 28 | 0.9905/0.9741 | 0.9532/0.9556 | 0.1443/0.1911 |
| E2 | 37 | 0.9730/0.9778 | 0.9690/0.9444 | 0.1072/0.1814 |
| E3 | 40 | 0.9603/0.9704 | 0.9603/0.9704 | 0.1117/0.1175 |
| E4 | 34 | 0.9230/0.9519 | 0.9056/0.9074 | 0.2656/0.3961 |

E4最佳epoch34的validation loss=0.134527，最後epoch40升至0.396132；Macro F1從0.950630降至0.906812。A+B提高訓練難度、最佳epoch的train accuracy仍偏低，與正則化較強或固定預算不足的假說相容，但不構成欠擬合的因果證明。後段train與validation同時退步也支持更新波動的觀察，不能只以「尚未收斂」解釋。未做低Dropout、延長訓練或調learning rate的控制實驗，因此三種解釋均保留為待驗證假說。

最終模型的非對角誤判（由多至少）：
- pitted_surface → crazing：2 張。
- inclusion → pitted_surface：2 張。
- pitted_surface → scratches：1 張。
- pitted_surface → inclusion：1 張。

## 解讀界線

此為固定驗證集上的模型選擇結果，未宣稱獨立測試成績或統計顯著差異。所有設定共同依規定 seed=42，未進行多seed實驗。
課程固定split中有一對完全相同影像：train/patches/TRN_0504.png 與 validation/patches/VAL_0101.png。遵守固定split保留，驗證成績可能稍樂觀。
CNN訓練accuracy/loss為訓練模式、可能含augmentation/dropout；validation為eval模式，因此兩者差距不能完全等同過擬合。
CNN部署延遲記錄於 `experiments/deployment_benchmark.json`：同270張Validation、六類各45張，batch1、FP32、20次暖機、CUDA同步；包含CPU前處理、傳輸、forward與分類結果回CPU，排除磁碟讀圖及模型載入。Stage1輸入128、Stage2輸入224、Stage3輸入320，故跨階段數字是各任務流程成本，不能當純架構速度倍率。原始訓練記錄的forward-only延遲與CPU SVM PCA+predict時間仍屬不同計時範圍，不混入此對照。
各組 history.json/training_curves.png 顯示 train/validation loss與accuracy；confusion_matrix.png、failure_cases.json/png 提供真實驗證失敗例，信心分數不等同錯誤成因證明。
