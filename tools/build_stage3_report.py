"""Build Stage 3's Chinese PDF exclusively from measured experiment artifacts."""
import csv
import json
from pathlib import Path

from report_layout import (
    ROOT, PageBreak, Spacer, build, cover, dataset_section, figure, heading,
    mb, para, pct, subheading, table,
)

STAGE = ROOT / "submission" / "Team07_Stage3"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iou(first, second):
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_a + area_b - intersection)


def failure_analysis(case, record, classes, confidence):
    """Separate visible case observations from measured errors and untested causes."""
    false_positives = case["false_positives"]
    false_negatives = case["false_negatives"]
    poor = [prediction for prediction in false_positives if prediction["kind"] == "poor_localization"]
    kinds = []
    if false_negatives:
        kinds.append("FN 漏檢")
    if poor:
        kinds.append("Poor Localization 定位不足")
    if len(false_positives) > len(poor):
        kinds.append("FP 誤報")
    observations = {
        "VAL_0001.png": "上方及中下方各有一個 crazing GT；保留下來的預測位於上半部，框位置向下偏離上方 GT，也未覆蓋中下方 GT。影像中瑕疵與背景都可見碎裂紋理。",
        "VAL_0046.png": "左側縱向 inclusion 有三個 GT。上方與中段已配對；下方長條 GT 被較短框覆蓋，另有一個與中段重疊的預測，形成定位不足與重複／競爭配對。",
        "VAL_0077.png": "右側兩個 inclusion 都正確定位；左側亮直線卻被框成兩個 scratches。提供的 GT 在左側沒有對應標註，因此這兩框依本次評估規則都是 FP。",
        "VAL_0106.png": "右上與左下兩個 patches GT 都已配對；額外預測位於影像上緣及左下相鄰區。左下多餘框向上延伸，覆蓋超過已配對 GT 的範圍。",
        "VAL_0145.png": "右側較明顯的 pitted_surface 已檢出，左側紋理較弱的 GT 區域漏檢。下列低信心候選證據顯示，此區並非完全沒有產生定位框。",
    }
    interpretations = {
        "VAL_0001.png": "錯誤同時涉及保留框的位置偏移與低信心候選被排除。低對比或背景紋理相似可能使判斷更難，但本次未做對照實驗，不能將影像外觀當成已證實原因。",
        "VAL_0046.png": "同類重疊框未取得配對，反映一對一評分下的重複預測；下方長條只被局部覆蓋則是另一個定位問題。調整 NMS 可能影響重複框，也可能抑制相鄰真實缺陷，不能據此宣稱能同時修好兩種錯誤。",
        "VAL_0077.png": "細長亮線的外觀支持描述預測位置，但無法只憑圖片判定該區是背景或漏標；本報告維持提供 GT 的評分結果。此例提醒高於0.5的 confidence 也不保證符合標註。",
        "VAL_0106.png": "已檢出兩個 GT 不代表沒有誤報。額外框顯示模型也回應影像邊緣及相鄰紋理；外觀相似或邊界範圍判斷僅為假說，不能由單張案例證明內部成因。",
        "VAL_0145.png": "此例可確認 confidence 門檻排除了足夠重疊的候選；較弱紋理是否導致低信心仍屬假說。降低門檻可以保留該候選，但整體 threshold sweep 同時顯示誤報增加，不能只看這張改善。",
    }
    evidence = [f"固定 confidence={confidence:.2f}：TP={len(case['matches'])}、"
                f"FP={len(false_positives)}、FN={len(false_negatives)}。定位不足的框計入 FP；"
                "只有未取得任何成功配對的 GT 才計入 FN。"]
    if case["image"] in observations:
        evidence.append(observations[case["image"]])
    if poor:
        prediction = poor[0]
        evidence.append(f"定位不足框預測 {classes[prediction['class_id']]}，confidence={prediction['confidence']:.3f}，"
                        f"與同類 GT 的最大 IoU={prediction['best_same_class_iou']:.3f}，未達配對門檻0.5。")
    if false_negatives:
        truth = false_negatives[0]
        same_class = [prediction for prediction in record["predictions"]
                      if prediction["class_id"] == truth["class_id"]]
        best = max(same_class, key=lambda prediction: iou(prediction["bbox"], truth["bbox"]), default=None)
        if best is not None:
            best_iou = iou(best["bbox"], truth["bbox"])
            evidence.append(f"對首個未配對 GT，保留至 confidence=0.001 的同類候選中，最大 IoU={best_iou:.3f}，"
                            f"該框 confidence={best['confidence']:.3f}。"
                            + ("它符合定位條件，但被0.25操作門檻濾掉。"
                               if best_iou >= 0.5 and best["confidence"] < confidence else ""))
    other_false = [prediction for prediction in false_positives if prediction["kind"] == "false_positive"]
    if other_false:
        prediction = other_false[0]
        overlap = max(iou(prediction["bbox"], truth["bbox"]) for truth in record["ground_truth"])
        evidence.append(f"另一誤報框預測 {classes[prediction['class_id']]}，confidence={prediction['confidence']:.3f}，"
                        f"與任意 GT 的最大 IoU={overlap:.3f}。"
                        + ("它與同類 GT 重疊足夠，但該 GT 已被較高信心框配對。"
                           if prediction["best_same_class_iou"] >= 0.5 else ""))
    interpretation = interpretations.get(case["image"],
        "此例依實際框、類別與一對一配對結果分析；尚無額外對照實驗可確定失敗原因。")
    return "、".join(kinds), evidence, [interpretation]


def main():
    config = read(STAGE / "config.json")
    summary = read(STAGE / "experiments/summary.json")
    records = {record["image"]: record for record in read(STAGE / "experiments/validation_predictions.json")}
    failures = summary["failure_cases"]
    latency = read(STAGE / "experiments/deployment_benchmark.json")["models"][0]
    checkpoint_export = summary["checkpoint_export"]
    if latency is None:
        raise ValueError("Stage 3 latency benchmark must finish before building the completed report.")
    if len(summary["visualizations"]) < 12 or len(failures) < 5:
        raise ValueError("Report requires 12 comparison images and five measured failure cases.")
    training = config["training"]
    classes = config["class_order"]
    official = summary["official_metrics"]
    fixed = summary["fixed_threshold_metrics"]
    selection = summary["selection"]
    crazing = next(row for row in fixed["per_class"] if row["class_name"] == "crazing")
    low_threshold = next(row for row in summary["threshold_sweep"] if row["confidence_threshold"] == 0.05)
    with (STAGE / selection["training_results_csv"]).open(encoding="utf-8") as handle:
        history = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    selected_row = next(row for row in history if int(row["epoch"]) == selection["selected_epoch"])
    final_row = history[-1]
    story = cover("Stage 3 | Object Detection", "YOLO11n 瑕疵定位、驗證誤差與部署取捨")
    story += dataset_section() + [
        Spacer(1, 8), para("偵測依原始逐框類別標註訓練。Train 88 張、Validation 16 張影像含多種類別，故不可直接把分類資料夾的單一類別套到全部框。"),
        PageBreak(), heading("Detector、Backbone 與前處理"),
        para("使用 Ultralytics YOLO11n，COCO 一般物件預訓練後，將偵測輸出調整為課程六類瑕疵。未載入網路上已對此瑕疵資料集 fine-tune 的 checkpoint，也未加入其他瑕疵訓練資料。"),
        para("YOLO11n backbone 由 Conv、C3k2、SPPF、C2PSA 組成；neck 透過上採樣、串接及下採樣融合多尺度特徵。Detect head 使用 P3/8、P4/16、P5/32 三個尺度，採 anchor-free 框預測；沒有預先指定一組 anchor box 長寬模板，也未使用 DETR 的 object query。"),
        table(["項目", "本次設定"], [
            ["模型與輸入", f"{config['detector']}；{config['input_size']}×{config['input_size']} RGB"],
            ["前處理", "原圖 200×200，letterbox／resize 至輸入尺寸，像素除以 255；不套 ImageNet mean/std"],
            ["標註", "來源 normalized YOLO xywh；模型預測與分析輸出回原圖 xyxy_pixel"],
            ["Split", f"Train 1,260；Validation {summary['validation_images']}；Validation instances {summary['validation_instances']}"],
            ["模型選擇", f"固定 Validation mAP@0.5 最高的 epoch；本次選中 epoch {selection['selected_epoch']}"],
            ["最終檔案", summary["checkpoint"]],
        ], [130, 363]),
        subheading("相對路徑與原始資料保留"),
        para("程式另建立 Ultralytics 所需 images／labels 目錄，保留原始 labels_yolo 與固定 split，不修改來源資料。模型從 submission 根目錄相對路徑載入；evaluate 指令明確指定資料根目錄，不依賴訓練機的原始個人路徑。"),
        para("模型來源：https://docs.ultralytics.com/models/yolo11/", "small"),
        para(config["pretrained_source"], "small"),
        PageBreak(), heading("實際訓練設定與 checkpoint 選擇"),
        table(["設定", "值"], [
            ["Epochs / Batch / Seed", f"設定 {training['epochs']} epochs；完成 {selection['epochs_completed']}；batch={training['batch']}；seed={config['seed']}"],
            ["Optimizer", f"{training['optimizer']}；lr0={training['lr0']}；lrf={training['lrf']}；weight_decay={training['weight_decay']}"],
            ["Schedule", f"線性 learning-rate decay；warmup={training['warmup_epochs']} epochs；patience={training['patience']}"],
            ["Flip / Geometry", f"fliplr={training['fliplr']}；flipud={training['flipud']}；degrees={training['degrees']}；translate={training['translate']}；scale={training['scale']}"],
            ["Mosaic / Mixup", f"mosaic={training['mosaic']}；最後 {training['close_mosaic']} epochs 關閉 mosaic；mixup={training['mixup']}"],
            ["HSV", f"h={training['hsv_h']}、s={training['hsv_s']}、v={training['hsv_v']}"],
            ["Execution", f"AMP={training['amp']}；deterministic={training['deterministic']}；workers={training['workers']}"],
            ["Environment", f"Python {summary['environment']['python']}；torch {summary['environment']['torch']}；Ultralytics {summary['environment']['ultralytics']}"],
            ["Hardware", summary["environment"]["device"]],
        ], [130, 363]),
        subheading("選模依據"),
        para(f"選中 epoch {selection['selected_epoch']}，訓練紀錄中的 Validation mAP@0.5={pct(selection['selected_validation_map50'])}。每個 epoch 的權重均先保存，再按此主指標選取；平手取較早 epoch。未直接使用框架預設以其他 fitness 選出的 best.pt。"),
        para("框架在 train 結束時對預設 best.pt 的自動驗證，不是本報告的最終證據。下頁分數來自重新載入實際提交的 stage3_model.pt，再對固定 Validation 評估。"),
        para("訓練 API：https://docs.ultralytics.com/modes/train/", "small"),
        PageBreak(), heading("Validation 指標與各類 AP"),
        table(["指標", "實際結果", "定義"], [
            ["mAP@0.5", pct(official["map50"]), "六類 AP 在 IoU=0.5 的平均"],
            ["mAP@0.5:0.95", pct(official["map50_95"]), "IoU 0.50 到 0.95、間隔0.05 的平均"],
            ["Official Macro Precision", pct(official["macro_precision_at_max_f1"]), "框架最大平均 F1 信心操作點"],
            ["Official Macro Recall", pct(official["macro_recall_at_max_f1"]), "與上列相同操作點，非固定0.25"],
            ["固定門檻 Micro Precision", pct(fixed["micro"]["precision"]), f"confidence={config['confidence_threshold']}，TP / (TP+FP)"],
            ["固定門檻 Micro Recall", pct(fixed["micro"]["recall"]), "TP / (TP+FN)"],
            ["固定門檻 Micro F1", pct(fixed["micro"]["f1"]), "Precision 與 Recall 的調和平均"],
        ], [160, 95, 238]),
        Spacer(1, 12), table(["類別", "AP@0.5", "AP@0.5:0.95", "P*", "R*"], [
            [row["class_name"], pct(row["ap50"]), pct(row["ap50_95"]),
             pct(row["precision_at_official_max_f1"]), pct(row["recall_at_official_max_f1"])]
            for row in official["per_class"]
        ], [151, 83, 103, 78, 78]),
        para("P*／R* 為官方 max-F1 操作點；AP 計算保留 confidence>=0.001 的候選框。固定 threshold 的 P/R 另以預測信心排序，對同類未匹配 GT 做 IoU>=0.5 一對一配對，兩種摘要不可混稱。", "small"),
        para("Validation API：https://docs.ultralytics.com/modes/val/", "small"),
        PageBreak(), heading("固定操作點：類別錯誤與定位指標"),
        table(["類別", "TP", "FP", "FN", "Precision", "Recall", "F1"], [
            [row["class_name"], row["tp"], row["fp"], row["fn"],
             pct(row["precision"]), pct(row["recall"]), pct(row["f1"])]
            for row in fixed["per_class"]
        ], [145, 38, 38, 38, 78, 78, 78]),
        Spacer(1, 10), para(f"整體計數：TP={fixed['micro']['tp']}，FP={fixed['micro']['fp']}，FN={fixed['micro']['fn']}。"
                               f"Macro P={pct(fixed['macro']['precision'])}、Macro R={pct(fixed['macro']['recall'])}、Macro F1={pct(fixed['macro']['f1'])}。"),
        para(f"固定門檻的漏檢比例為 {fixed['micro']['fn']} / {summary['validation_instances']} = {pct(fixed['micro']['fn'] / summary['validation_instances'])}；crazing 的 TP={crazing['tp']}、FN={crazing['fn']}、Recall={pct(crazing['recall'])}。整體 mAP 不能代替這類瑕疵的漏檢風險。"),
        figure(STAGE / "experiments/figures/per_class_ap50.png", "各類 AP@0.5 取自同一個最終 checkpoint。", max_height=240),
        subheading("IoU、NMS 與 confidence 的分工"),
        para(f"IoU=交集面積／聯集面積，評估時需同類別且 IoU>={config['evaluation']['matching_iou_threshold']} 才能成為 TP。confidence>={config['confidence_threshold']} 決定哪些框進入操作點統計；NMS IoU={config['nms_iou_threshold']} 決定同類候選框互相重疊時是否抑制較低信心者。NMS 門檻不是評估的 IoU 門檻。"),
        para("對細長或小型瑕疵，同樣像素的邊界偏移可能造成較大 IoU 損失。過大的框也會因聯集面積增加而降低 IoU；這些是量測的幾何性質，不是已證明的模型失敗原因。"),
        PageBreak(), heading("Confidence Threshold 的取捨"),
        figure(STAGE / "experiments/figures/confidence_threshold_sweep.png", "同一批 NMS 後候選框，逐一套用 confidence 門檻；匹配 IoU 固定0.5。", max_height=245),
        table(["Confidence", "TP", "FP", "FN", "Micro P", "Micro R", "Micro F1"], [
            [f"{row['confidence_threshold']:.2f}", row["micro"]["tp"], row["micro"]["fp"], row["micro"]["fn"],
             pct(row["micro"]["precision"]), pct(row["micro"]["recall"]), pct(row["micro"]["f1"])]
            for row in summary["threshold_sweep"]
        ], [72, 45, 45, 45, 95, 95, 96]),
        Spacer(1, 8), para(f"由0.25降至0.05，Recall由{pct(fixed['micro']['recall'])}升至{pct(low_threshold['micro']['recall'])}，但FP由{fixed['micro']['fp']}增加至{low_threshold['micro']['fp']}；因此不能將降低門檻當作沒有代價的漏檢修正。最終 config 保留0.25，掃描用於解釋操作取捨；同一 Validation 的掃描不構成獨立測試。"),
    ]
    run_dir = (STAGE / selection["training_results_csv"]).parent
    story += [PageBreak(), heading("訓練曲線與驗證波動"),
              figure(run_dir / "results.png", "框架保存的完整訓練／驗證 loss 與偵測 metrics。", max_height=340),
              para(f"選中 epoch {selection['selected_epoch']} 的 Validation mAP50={pct(float(selected_row['metrics/mAP50(B)']))}；"
                   f"最後 epoch {selection['epochs_completed']} 為 {pct(float(final_row['metrics/mAP50(B)']))}。"
                   f"最後 epoch train box loss={float(final_row['train/box_loss']):.4f}、val box loss={float(final_row['val/box_loss']):.4f}。"),
              para("訓練含 augmentation、後期關閉 mosaic，且驗證使用不同模式，故 train／val loss 不能只以數值差直接證明過擬合。選模依明定的 Validation mAP50，不預設最後 epoch 或更低 train loss 必然最好。")]
    visualizations = summary["visualizations"][:12]
    for first in range(0, len(visualizations), 2):
        story += [PageBreak(), heading(f"GT vs Prediction | {first + 1}-{min(first + 2, 12)} / 12")]
        for relative_path in visualizations[first:first + 2]:
            story += [figure(STAGE / relative_path,
                             f"{Path(relative_path).stem}：左為 GT，右為 confidence>={config['confidence_threshold']} 的預測；框與類別來自實際 Validation。",
                             max_height=270)]
        story += [para("展示影像按檔名順序，優先涵蓋每類前兩張，再補齊12張，未依正確率挑選漂亮案例。所有270張仍納入量化指標。", "small")]
    for index, case in enumerate(failures[:5], 1):
        record = records[case["image"]]
        kinds, evidence, hypotheses = failure_analysis(case, record, classes, config["confidence_threshold"])
        story += [PageBreak(), heading(f"失敗案例 {index} / 5 | {case['image']}"),
                  para(kinds, "subtitle"),
                  figure(STAGE / case["figure"], "GT 與 Prediction 對照；預測標籤後為 confidence。", max_height=265),
                  subheading("可由標註與預測證實的證據")]
        story += [para(text) for text in evidence]
        story += [subheading("可能原因與解讀界線")]
        story += [para(text) for text in hypotheses]
    story += [PageBreak(), heading("FP32 推論效率與模型成本"),
              table(["項目", "量測結果"], [
                  ["Parameter count", f"{summary['parameter_count']:,}"],
                  ["推論專用權重", mb(checkpoint_export["inference_checkpoint_bytes"])],
                  ["原始訓練快照", mb(checkpoint_export["training_snapshot_bytes"])],
                  ["裝置", latency["device"]],
                  ["FP32 / Batch", "FP32，batch=1"],
                  ["暖身 / Samples", f"{latency['warmup_iterations']} 次 / {latency['samples']} 張"],
                  ["End-to-end mean", f"{latency['mean_ms']:.3f} ms"],
                  ["End-to-end median / P95", f"{latency['p50_ms']:.3f} / {latency['p95_ms']:.3f} ms"],
                  ["估計順序處理 FPS", f"{1000 / latency['mean_ms']:.2f}（1000 / mean ms）"],
              ], [190, 303]),
              subheading("計時方式"),
              para("所有模型在相同裝置，以 batch=1、FP32、20次暖身及全部270張 Validation 單獨量測；六個主類別各45張，包含多類別框的影像仍保留原標註。與Stage1／2使用相同影像ID及計時邊界：影像預載至記憶體，包含CPU前處理、裝置傳輸、模型、輸出後處理與結果返回CPU，排除磁碟I/O，CUDA前後同步。Stage3的後處理包含NMS與API overhead，輸入尺寸為320；不同階段任務與尺寸仍不同。未包含相機、通訊、PLC或人工反應時間。"),
              para(f"推論檔保留選中epoch的EMA模型並移除optimizer等續訓狀態；原始快照另存於{checkpoint_export['source_checkpoint']}。權重以FP16儲存、載入後以FP32推論；Stage1／2以FP32 state_dict儲存，故檔案大小差異同時包含保存格式與精度，不能全歸因於架構。匯出後重新載入並核對Validation，指標保持一致。"),
              PageBreak(), heading("部署決策、結論與重現"),
              para(f"本專題採用YOLO11n作定位實驗模組；真實工廠暫緩直接部署。固定0.25漏檢{fixed['micro']['fn']} / {summary['validation_instances']}個GT（{pct(fixed['micro']['fn'] / summary['validation_instances'])}），crazing Recall只有{pct(crazing['recall'])}。降至0.05雖提升召回，卻產生{low_threshold['micro']['fp']}個FP；目前缺少現場可接受的漏檢、誤報成本及節拍，不能以FPS或整體mAP宣稱可上線。"),
              para("本次沒有訓練更大的 detector，因此不主張更大模型一定值得或一定不值得；已完成的證據是此模型的定位效能、實際延遲、模型大小與失敗案例。提高輸入尺寸或改模型需另外實驗，不能當作已取得的改善。"),
              subheading("完成內容與重現"),
              para(f"Stage 3 完成 YOLO11n 在固定資料集上的微調與定位評估。最終 Validation mAP@0.5={pct(official['map50'])}；confidence={config['confidence_threshold']} 時 Micro Precision={pct(fixed['micro']['precision'])}、Recall={pct(fixed['micro']['recall'])}。"
                   f"單張 FP32 end-to-end 平均 {latency['mean_ms']:.3f} ms。上述數值均對應同一提交 checkpoint。"),
              para("執行方式：在 Stage 根目錄依 requirements.txt 安裝套件，使用 README 所列 train／evaluate 指令與 --data-root。src/model.py、train.py、evaluate.py 及 helpers 含完整流程，checkpoint 以相對根目錄載入。"),
              para("實驗明細：experiments/selection.json、summary.json、validation_predictions.json、threshold_sweep.json、failure_cases.json、deployment_benchmark.json。"),
              subheading("原始來源")]
    sources = [
        "作業說明：FactoryVision_Final_Project.pdf，第19-35頁；https://lms2020.nchu.edu.tw/course/homework/76885",
        "YOLO11：https://docs.ultralytics.com/models/yolo11/",
        "本次版本架構：https://github.com/ultralytics/ultralytics/blob/v8.3.238/ultralytics/cfg/models/11/yolo11.yaml",
        "預訓練 checkpoint：" + config["pretrained_source"],
        "Training API：https://docs.ultralytics.com/modes/train/",
        "Validation API：https://docs.ultralytics.com/modes/val/",
        "Prediction API：https://docs.ultralytics.com/modes/predict/",
    ]
    story += [para(source, "small") for source in sources]
    build(STAGE / "stage3_report.pdf", story)
    print(STAGE / "stage3_report.pdf")


if __name__ == "__main__":
    main()
