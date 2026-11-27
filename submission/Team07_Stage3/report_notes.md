# Stage 3 報告整合筆記

實驗完成60 epochs；報告包含12張GT vs Prediction與5個失敗案例，逐例分析影像與匹配結果。效能分析使用全部270張影像的完整流程計時，分列訓練快照與推論權重大小。

## 方法與重現規格

- YOLO11n 官方 COCO 預訓練，固定課程 Train 1,260／Validation 270。未用外部瑕疵資料或已 fine-tuned checkpoint。
- seed=42，60 epochs，input=320，batch=16，AdamW；完整設定見 config.json。
- Backbone／neck 為 YOLO11n，anchor-free detection head；confidence 控制保留預測，NMS 抑制同類重疊框，評估 IoU 衡量定位重疊。
- 主要選模指標為 Validation mAP@0.5，每個 epoch 有 checkpoint，選最高分；不是直接取官方預設 best.pt。
- 固定 confidence=0.25、NMS IoU=0.7、匹配 IoU=0.5。輸出全為原圖 xyxy_pixel。

## 實際結果

- 選中 epoch 49，訓練 CSV Validation mAP50=0.73335；最終獨立重新評估 mAP50=0.7334757128898547、mAP50:95=0.42871075891277044。訓練階段與重新評估的 batch/執行模式差異使末位小幅不同，報告主結果採後者。
- 固定 confidence=0.25：TP=446、FP=185、FN=195；Micro Precision=0.7068145800、Recall=0.6957878315、F1=0.7012578616。
- 單張 FP32 完整流程延遲以 experiments/deployment_benchmark.json 為準：20 次暖身、全部270張固定 Validation、六個主類別各45張；各階段使用相同影像ID、裝置及計時邊界。包含CPU前處理、裝置傳輸、模型、NMS／後處理及結果返回CPU，CUDA同步，排除磁碟I/O。不同任務與輸入尺寸仍不能視為相同工作量。
- stage3_model.pt 是移除optimizer等續訓狀態的推論權重；原始訓練快照保留於experiments/train/weights/epoch48.pt。大小與匯出驗證記錄見summary.json的checkpoint_export。Stage3以FP16儲存、FP32推論；Stage1／2以FP32 state_dict儲存，大小差異不全來自架構。
- 相同 Validation 上，threshold=0.05 時 Recall=88.77%、Precision=29.45%、FP=1363；threshold=0.50 時 Recall=51.01%、Precision=89.59%。0.25 在本次掃描中得到最高 Micro F1，但仍屬同一驗證集上的分析，不能視為獨立測試。
- 最弱類別為 crazing（AP50 約39.95%、固定門檻 Recall23.53%）；patches AP50 約96.56%。不能用整體mAP掩蓋個別類別的漏檢。

## 五個案例的實際影像觀察

1. VAL_0001：兩個 crazing GT 位於上方及中下方，預測只在上半部框出偏移區域，TP0／FP1／FN2。可見背景與瑕疵皆有碎裂紋理；低對比或紋理相似僅是原因假說。另有同類低信心框 IoU0.835、confidence0.125，證明最終門檻會排除部分足夠重疊的候選。
2. VAL_0046：左側縱向 inclusion 被模型分成較短的框，TP2／FP2／FN1；其中同類預測 IoU0.446 未達匹配門檻，另一個重疊框無法取得一對一配對。這支持分割／重複預測的幾何描述；不能直接由圖像證明網路內部成因。
3. VAL_0077：右側兩個 inclusion 正確定位；左側亮直線被預測為 scratches，但該處沒有提供的GT，依現有標註計2個FP。可描述其外觀接近細長亮線，但不能僅靠圖片斷定它是背景或漏標。
4. VAL_0106：兩個 patches GT均有匹配，另在上緣與左下相鄰區產生2個FP，其中一框同類最大IoU0.454。局部外觀相似及邊界外延可作為假說，報告以提供GT為評分依據。
5. VAL_0145：右側 pitted_surface 被檢出，左側較弱紋理區漏檢，TP1／FP0／FN1。原始低信心候選對左側GT的IoU0.851、confidence0.180，固定0.25門檻濾掉它；降低threshold能救回候選，也會提高整體FP。

## 實驗證據來源

1. experiments/selection.json：選中 epoch 與 mAP50。
2. experiments/summary.json：官方 mAP50、mAP50:95、各類 AP、P/R 與參數／大小。
3. experiments/figures/comparison_*.png：至少十張 GT vs Prediction。
4. experiments/failure_cases.json 與 figures/failure_*.png：五個案例的影像、預測框與配對結果。
5. experiments/figures/confidence_threshold_sweep.png：Precision／Recall 的操作取捨。
6. experiments/deployment_benchmark.json：全部270張、六個主類別均衡、與其他階段統一計時範圍的推論時間。
7. experiments/summary.json 的 checkpoint_export：原始訓練快照與推論權重大小，以及匯出後驗證紀錄。

官方摘要 P/R 在 max-F1 信心位置；固定0.25的P/R是另算的一對一匹配統計，不可混用。mAP 所用 min confidence=0.001，不能把固定confidence輸出拿來重算同一AP而不說明。

部署結論：本專題選用 YOLO11n 作定位模組；實際工廠暫緩直接部署，原因是固定0.25仍漏檢195／641個GT（30.42%）並有185個FP，crazing Recall僅23.53%；降低門檻到0.05會增加至1363個FP，且未提供現場容錯、節拍與資料分布。未訓練更大detector，因此沒有證據聲稱較大的模型一定值得或一定不值得。
