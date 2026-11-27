"""Generate the four reports exclusively from saved, measured experiment evidence."""
import argparse
import json
from pathlib import Path

from report_layout import (ROOT, PageBreak, Spacer, build, cover, dataset_section,
                           figure, heading, mb, para, pct, subheading, table)

SUB = ROOT / 'submission'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metric_row(name, metrics):
    return [name] + [pct(metrics[k]) for k in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')]


def per_class_rows(metrics):
    return [[name, pct(values['precision']), pct(values['recall']), pct(values['f1-score']), int(values['support'])]
            for name, values in metrics.items() if isinstance(values, dict) and name not in ('macro avg', 'weighted avg')]


def top_confusions(matrix, classes, limit=3):
    pairs = sorted([(matrix[i][j], classes[i], classes[j]) for i in range(6) for j in range(6) if i != j], reverse=True)
    return '；'.join(f'{true} → {pred}：{count} 張' for count, true, pred in pairs[:limit] if count) or '此驗證集沒有錯分。'


def sources(stage, items):
    story = [heading('重現方式與來源'),
        para(f'本階段完整訓練與 Validation 指令見同層 README.md；必要設定保存在 config.json，環境版本列於 requirements.txt。src/ 提供完整 model、train、evaluate 與必要 helper。最終模型使用同層 stage{stage}_model.' + ('pt' if stage == 3 else 'pth') + '，無須修改程式路徑。'),
        para('作業規格：FactoryVision_Final_Project.pdf；iLearning 作業頁：https://lms2020.nchu.edu.tw/course/homework/76885。')]
    story.extend(para(item, 'small') for item in items)
    return story


def stage1():
    root = SUB / 'Team07_Stage1'
    data = read(root / 'experiments/summary.json')
    config = read(root / 'config.json')
    results = data['experiments']
    chosen = data['selected_experiment']
    final = results[chosen]
    classes = config['class_order']
    paired = read(root / 'experiments/E1_E2_paired_changes.json')
    e4_history = read(root / 'experiments/E4/history.json')
    e4_best = next(row for row in e4_history if row['epoch'] == results['E4']['best_epoch'])
    e4_last = e4_history[-1]
    story = cover('Stage 1 | Baseline 與 Improved CNN', '傳統特徵基準、自製卷積網路與控制變因實驗')
    story += dataset_section() + [PageBreak(), heading('方法與公平比較設計'),
        subheading('傳統方法：PCA + RBF SVM'),
        para('將灰階圖縮至 64×64，像素除以 255 後展平。只以 Training fit PCA（100 components、whiten=True、randomized SVD、seed=42）與 RBF SVM（C=10、gamma=scale）。Validation 僅 transform / predict；未用其資訊估計 PCA。'),
        para(f"PCA 解釋變異比例 {pct(results['traditional']['pca_explained_variance_ratio'])}；support vectors={results['traditional']['support_vectors']}。此基準用於衡量直接像素與線性降維的辨識能力。傳統方法輸入64×64、CNN為128×128，因此兩者差異同時包含輸入解析度，不能全歸因於模型架構。"),
        subheading('自行設計的 SurfaceCNN'),
        para('輸入為 1×128×128 灰階影像；Resize → ToTensor → Normalize(mean=0.5, std=0.5)。三組 Conv3×3(32、64、128 channels) + ReLU + MaxPool2d，再以 AdaptiveAvgPool2d(4×4)、Flatten、Linear(2048→128)、ReLU、Dropout、Linear(128→6) 分類。所有權重從零訓練；未使用既有架構或預訓練權重。'),
        table(['設定', '所有 CNN 共用'], [['Optimizer', 'Adam，learning rate=0.001，weight_decay=0'],
            ['Budget', '40 epochs，batch=64，seed=42；無 scheduler / early stopping'],
            ['選模', '各組依 Validation Macro F1 選 epoch；平手保留較早 epoch'],
            ['A：Data augmentation', '水平／垂直翻轉各 p=0.5；RandomRotation ±10°，只用於 Training'],
            ['B：Dropout', 'Classifier dropout p=0.5；無此方法時 p=0'],
            ['最終 improved CNN', '只從 E2 / E3 / E4 依 Macro F1 選最高者']], [140, 353]),
        PageBreak(), heading('完整比較與 Ablation'),
        table(['實驗', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1'],
            [metric_row(name, results[name]) for name in ['traditional', 'E1', 'E2', 'E3', 'E4']], [113, 95, 95, 95, 95]),
        Spacer(1, 13), table(['實驗', 'A', 'B', 'Best epoch', '相對 E1 的 F1 差'],
            [[n, '是' if n in ('E2', 'E4') else '否', '是' if n in ('E3', 'E4') else '否',
              results[n]['best_epoch'], f"{100*(results[n]['macro_f1']-results['E1']['macro_f1']):+.2f} 個百分點"]
             for n in ['E1', 'E2', 'E3', 'E4']], [82, 55, 55, 110, 191]),
        subheading('最終選擇與解讀'),
        para(f"選擇 {chosen}，best epoch={final['best_epoch']}，Macro F1={pct(final['macro_f1'])}。相對 baseline E1 的差為 {100*(final['macro_f1']-results['E1']['macro_f1']):+.2f} 個百分點。判斷依完整四組控制變因結果，沒有預設 A+B 必然最好。"),
        para(f"A only、B only、A+B 相對 E1 的 Macro F1 分別為 {100*(results['E2']['macro_f1']-results['E1']['macro_f1']):+.2f}、{100*(results['E3']['macro_f1']-results['E1']['macro_f1']):+.2f}、{100*(results['E4']['macro_f1']-results['E1']['macro_f1']):+.2f} 個百分點。此固定訓練預算下，資料擴增略有收益；Dropout 單獨或與擴增併用未提高驗證 F1，因此最終選擇只保留 A。"),
        para('單一 seed、單一驗證切分的差異不代表統計顯著。資料擴增與 Dropout 都改變訓練模式下的難度，因此 train/validation accuracy 的差距不能直接全部歸因於過擬合。'),
        table(['模型', '參數數量', 'Checkpoint', '訓練時間'],
              [[n, f"{results[n].get('parameter_count', 0):,}" if n != 'traditional' else 'PCA + support vectors',
                mb(results[n]['checkpoint_bytes']), f"{results[n]['training_seconds']:.1f} s"]
               for n in ['traditional', 'E1', 'E2', 'E3', 'E4']], [95, 145, 110, 143]),
        PageBreak(), heading('改善幅度與 E4 退步診斷'),
        subheading('同一影像的 E1 → E2 配對比較'),
        para(f"E1 錯 {paired['E1_wrong']} 張、E2 錯 {paired['E2_wrong']} 張；E2 修正 {len(paired['corrected'])} 張、同時新增 {len(paired['new_errors'])} 張錯誤，淨多 1 張正確。共同錯誤仍有 {len(paired['shared_errors'])} 張。Macro F1 增加 {100*(results['E2']['macro_f1']-results['E1']['macro_f1']):.2f} 個百分點，是有限改善，不能解讀為全面解決混淆。"),
        figure(root / paired['figure'], '固定 Validation、各組最佳 checkpoint。綠字為正確、紅字為錯分；括號是該預測類別的信心。', max_height=205),
        para('VAL_0097 可見暗色塊狀區，VAL_0236 可見細長亮直線；E2 將兩者由 pitted_surface 修正為 patches / scratches。VAL_0082 的細長暗紋較淡，卻從 E1 正確的 inclusion 變為 pitted_surface。這些是影像與預測的對照觀察；尚無證據將修正或退步直接歸因於特定旋轉／翻轉樣式。'),
        subheading('E4：最佳結果與最後一次更新應分開看'),
        table(['E4 checkpoint', 'Train Acc', 'Val Acc', 'Val loss', 'Val Macro F1'],
            [[f"Epoch {row['epoch']} ({label})", pct(row['train_accuracy']), pct(row['val_accuracy']),
              f"{row['val_loss']:.4f}", pct(row['val_macro_f1'])]
             for label, row in [('best', e4_best), ('last', e4_last)]], [125, 92, 92, 92, 92]),
        Spacer(1, 8), para('A+B 同時提高訓練難度，而 E4 最佳 epoch 的 train accuracy 仍偏低，與正則化較強或固定40 epochs預算不足的假說相容；但不能據此證實欠擬合。第34到40 epoch的 train / validation 表現同時退步、validation loss升高，亦顯示後段更新波動，不能只用「尚未收斂」解釋。未做低 Dropout、延長訓練或不同 learning rate 的控制實驗，故保留這些解釋為待驗證假說。')]
    for first, second in [('E1', 'E2'), ('E3', 'E4')]:
        story += [PageBreak(), heading(f'學習曲線 | {first}、{second}')]
        for name in (first, second):
            history = read(root / 'experiments' / name / 'history.json')
            last = history[-1]
            story += [figure(root / 'experiments' / name / 'training_curves.png',
                f"{name}：40 epochs；最終 epoch train accuracy={pct(last['train_accuracy'])}、validation accuracy={pct(last['val_accuracy'])}；實際提交使用 best epoch={results[name]['best_epoch']}。", max_height=220)]
        if first == 'E1':
            story += [para('E1 訓練 loss 持續降低後，validation loss 仍有明顯波動；E2 也在第40 epoch出現驗證退步。完整40 epochs均有保留，但提交模型選用E2第37 epoch，避免把最後一次更新當成最佳泛化表現。')]
        else:
            story += [para('加入 Dropout 的 E3/E4 在 train 模式會隨機移除 classifier activations，而 validation 不會；因此 validation accuracy 高於 train 並不矛盾。E4 的驗證 F1 低於 E2，顯示本次固定設定中兩種改良疊加沒有額外收益。')]
        story += [para('曲線保留全部訓練 epoch；最佳 checkpoint 依 Macro F1 而非最後 epoch。驗證波動與訓練模式差異均應一併解讀。')]
    story += [PageBreak(), heading('Confusion Matrix | 傳統與 Baseline'),
        figure(root / 'experiments/traditional/confusion_matrix.png', 'PCA + RBF SVM，Validation 270 張。', max_height=285),
        figure(root / 'experiments/E1/confusion_matrix.png', 'E1 baseline CNN，Validation 270 張。', max_height=285),
        PageBreak(), heading(f'最終 Improved CNN | {chosen}'),
        figure(root / f'experiments/{chosen}/confusion_matrix.png', '列為真實類別，欄為預測類別。', max_height=335),
        table(['類別', 'Precision', 'Recall', 'F1', 'Support'], per_class_rows(final['per_class']), [160, 85, 85, 85, 78]),
        Spacer(1, 10), para('主要錯誤：' + top_confusions(final['confusion_matrix'], classes)),
        PageBreak(), heading('失敗案例與部署分析')]
    failure_path = root / f'experiments/{chosen}/failure_cases.png'
    if failure_path.exists():
        story += [figure(failure_path, '依錯誤預測信心由高至低選例；每張列出檔名、GT、prediction。', max_height=360)]
    inference = read(root / 'experiments/deployment_benchmark.json')['models'][0]
    story += [para('鋼材瑕疵的局部紋理、方向與對比可能重疊。此處以錯誤類別與影像作定性觀察，不能把影像相似性當成已驗證的因果解釋。各類 recall 反映哪些缺陷較容易漏分，應與單一整體 Accuracy 一起檢視。'),
        para('全部6張錯分均列於上圖。VAL_0159/0160 的密集點狀紋理伴隨較粗暗色紋路，被分成 crazing；VAL_0082/0083 的 inclusion 呈細長且低對比的暗紋，被分成 pitted_surface。VAL_0177 偏暗，VAL_0144 的紋理較細密；這些是可見的影像特徵觀察，未證明其必然造成錯分。pitted_surface recall=41/45，為最終模型最低的類別召回率。'),
        para(f"最終模型共 {final['parameter_count']:,} parameters，checkpoint {mb(final['checkpoint_bytes'])}。單張完整分類流程平均 {inference['mean_ms']:.3f} ms，P95 {inference['p95_ms']:.3f} ms，裝置 {inference['device']}。全270張 Validation、六類各45張，batch=1、FP32，20次暖機後同步CUDA；計時含CPU前處理、CPU/GPU傳輸、forward及分類結果回CPU，不含讀圖與模型載入。"),
        para('此階段輸出影像級分類，無法定位瑕疵位置。部署選擇需再與 Stage 2 的分類精度、模型大小，以及 Stage 3 的定位需求一起比較；本階段沒有測量產線相機、PLC 或連續傳輸的完整延遲。'),
        PageBreak()] + sources(1, ['PyTorch / Torchvision： https://docs.pytorch.org/vision/0.21/'])
    build(root / 'stage1_report.pdf', story)


def stage2():
    root = SUB / 'Team07_Stage2'
    data = read(root / 'experiments/summary.json')
    results = data['experiments']
    final = next(r for r in results if r['experiment'] == data['final_experiment'])
    benchmark = read(root / 'experiments/deployment_benchmark.json')
    selected_latency = next(row for row in benchmark['models'] if row['experiment'] == final['experiment'])
    resnet_ft_latency = next(row for row in benchmark['models'] if row['experiment'] == 'resnet18__full_fine_tuning')
    mobile_fe = next(r for r in results if r['experiment'] == 'mobilenet_v3_small__feature_extraction')
    mobile_ft = next(r for r in results if r['experiment'] == 'mobilenet_v3_small__full_fine_tuning')
    weak_correct = min(read(root / 'experiments/mobilenet_v3_small__feature_extraction/predictions.json'),
                       key=lambda row: max(row['probabilities']))
    aliases = {'resnet18__feature_extraction': 'R18-FE', 'resnet18__full_fine_tuning': 'R18-FT',
               'mobilenet_v3_small__feature_extraction': 'MNV3-FE', 'mobilenet_v3_small__full_fine_tuning': 'MNV3-FT'}
    story = cover('Stage 2 | Transfer Learning', '兩個架構家族 × 兩種策略的完整比較')
    story += dataset_section() + [PageBreak(), heading('模型、預訓練與訓練策略'),
        para('Architecture A：ResNet18；Architecture B：MobileNetV3-Small。兩者屬不同 architecture family。均使用 Torchvision 官方 ImageNet-1K V1 權重，未使用針對 FactoryVision／鋼材瑕疵 fine-tune 的網路 checkpoint。最後分類輸出皆換成六類。'),
        table(['策略', 'ResNet18', 'MobileNetV3-Small'], [
            ['Feature extraction (FE)', '凍結全部 backbone；僅訓練 fc', '凍結 features；訓練整個 classifier，包含原有 hidden Linear 與新六類輸出'],
            ['Full fine-tuning (FT)', '所有 layers 可訓練', '所有 layers 可訓練'],
            ['BatchNorm', 'FE 保持 backbone eval；FT 更新統計', 'FE 保持 backbone eval；FT 更新統計']], [116, 188, 189]),
        Spacer(1, 13), table(['公平比較項目', '設定'], [
            ['資料 / seed', '固定 1,260 / 270 split；每組重新設定 seed=42'],
            ['輸入', '灰階轉 RGB；直接 Resize(224,224)，不裁切、不加強化'],
            ['Normalization', 'ImageNet mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]'],
            ['訓練預算', '15 epochs，每組 batch=32'],
            ['Optimizer', 'AdamW，weight_decay=0.0001；head lr=0.001；FT backbone lr=0.0001'],
            ['Scheduler', 'CosineAnnealingLR，T_max=15'],
            ['選模', '最高 Validation Macro F1 → 較小 checkpoint → 較少可訓練參數 → 實驗名稱排序']], [140, 353]),
        para('Resize 採相同直接縮放而非預設權重 transform 的中心裁切，目的是保留完整瑕疵區域。兩種策略各跑在兩個架構上，沒有把架構差異與策略差異混在單一比較。'),
        PageBreak(), heading('2×2 完整實驗結果'),
        table(['實驗', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1'],
              [metric_row(aliases[r['experiment']], r['metrics']) for r in results], [113, 95, 95, 95, 95]),
        Spacer(1, 14), table(['實驗', 'Total params', 'Trainable', 'Checkpoint', 'Best epoch'],
             [[aliases[r['experiment']], f"{r['parameters']:,}", f"{r['trainable_parameters']:,}", mb(r['checkpoint_bytes']), r['best_epoch']]
               for r in results], [98, 109, 109, 100, 77]),
        subheading('最終模型決策'),
        para(f"選擇 {final['architecture']} / {final['strategy']}，Validation Macro F1={pct(final['metrics']['macro_f1'])}，Accuracy={pct(final['metrics']['accuracy'])}。stage2_model.pth 是該組 best epoch={final['best_epoch']} 的本地 checkpoint；此決策與 config.json、summary.json 一致。"),
        para(f"R18-FT、MNV3-FE、MNV3-FT 的最佳 checkpoint 均在270張驗證影像全對。先以較小 checkpoint 選 MobileNet；FE 與 FT 仍同分、同大小，再以較少可訓練參數選 FE：{mobile_fe['trainable_parameters']:,} 對 {mobile_ft['trainable_parameters']:,}。僅當前三項完全相同才依名稱排序。"),
        para('FE 將可訓練參數限制於分類器，可降低更新成本；FT 能調整底層與高層特徵，但參數更新更多。實際收益需依上述完整 2×2 數字判斷，不能僅由網路大小推定。')]
    for architecture in ('resnet18', 'mobilenet_v3_small'):
        fe = next(r for r in results if r['architecture'] == architecture and r['strategy'] == 'feature_extraction')
        ft = next(r for r in results if r['architecture'] == architecture and r['strategy'] == 'full_fine_tuning')
        story += [para(f"{architecture}：FT 相對 FE 的 Macro F1 改變為 {100*(ft['metrics']['macro_f1']-fe['metrics']['macro_f1']):+.2f} 個百分點；可訓練參數從 {fe['trainable_parameters']:,} 增至 {ft['trainable_parameters']:,}。")]
    for architecture in ('resnet18', 'mobilenet_v3_small'):
        story += [PageBreak(), heading(f'Learning Curves | {architecture}')]
        for r in results:
            if r['architecture'] == architecture:
                story += [figure(root / 'experiments' / r['experiment'] / 'curves.png',
                    f"{aliases[r['experiment']]}：完整15 epochs；best epoch {r['best_epoch']}。", max_height=220)]
        story += [para('train 指標在訓練模式逐 batch 計算；validation 為 eval 模式。MobileNet 原生 classifier Dropout、FT 時 BatchNorm 統計都會影響曲線差距。所有 checkpoint 由 validation Macro F1 選擇；未用 validation 反向傳播。')]
        if architecture == 'mobilenet_v3_small':
            first_fe = read(root / 'experiments' / mobile_fe['experiment'] / 'history.json')[0]
            first_ft = read(root / 'experiments' / mobile_ft['experiment'] / 'history.json')[0]
            story += [para('兩組 MobileNet 均完成15 epochs；FE 在第2 epoch、FT在第7 epoch首次取得Macro F1=1.0，故各保留該次checkpoint。最後第15 epoch兩者各有1張錯分，與所交best checkpoint的零錯誤不同；報告數字使用實際提交模型。'),
                para(f"第1 epoch：FE 的 validation F1={pct(first_fe['val_macro_f1'])}、loss={first_fe['val_loss']:.4f}；FT 為 {pct(first_ft['val_macro_f1'])}、loss={first_ft['val_loss']:.4f}。FT 初期波動可能與 backbone / BatchNorm 同時適應新資料有關，FE 則保留預訓練統計；尚未做單獨凍結BN的控制實驗，不能宣稱BN是已證實原因。")]
    story += [PageBreak(), heading('最終模型的分類表現'),
        figure(root / 'evaluation/confusion_matrix.png', '由獨立 evaluate.py 重新載入 stage2_model.pth 計算。', max_height=345),
        table(['類別', 'Precision', 'Recall', 'F1', 'Support'], per_class_rows(final['metrics']['per_class']), [160, 85, 85, 85, 78]),
        PageBreak(), heading('錯誤案例與工程取捨')]
    if (root / 'evaluation/failure_cases.png').exists():
        story += [figure(root / 'evaluation/failure_cases.png', '最終模型的實際錯分，依信心高至低選取。', max_height=300)]
    story += [para('主要混淆：' + top_confusions(final['metrics']['confusion_matrix'],
        ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']))]
    if final['metrics']['accuracy'] == 1.0:
        story += [para('本次選中的模型在270張Validation沒有錯分，因此沒有最終模型的失敗影像。模型與epoch是在此驗證集上選擇，且固定split原有一組重複；100%只描述這批驗證資料，尚未取得獨立Hidden data成績。四組中只有R18-FE的最佳checkpoint仍有2張錯分，其餘三組的最佳checkpoint均全對。'),
            figure(root / 'experiments/resnet18__feature_extraction/failure_cases.png',
                   '補充：未選中候選模型 R18-FE 的2張實際錯分，並非最終 MNV3-FE 的錯誤。', max_height=235),
            para('R18-FE 將 VAL_0083 的 inclusion 判為 pitted_surface（0.762），將 VAL_0228 的 scratches 判為 inclusion（0.635）。前者可見低對比細長暗紋，後者是面積小的細長亮痕；這是影像觀察而非已驗證的錯誤成因。兩張在所選 MobileNet FE 均分類正確。'),
            subheading('全對仍有決策邊界接近的影像'),
            para(f"最終模型信心最低的正確例為 {Path(weak_correct['file']).name}：GT / prediction 均是 inclusion，分數 {weak_correct['probabilities'][1]:.4f}，第二候選 scratches 為 {weak_correct['probabilities'][5]:.4f}。這是正確但兩類分數接近的觀測，並非失敗例；驗證全對不表示每張影像都具有同樣大的決策餘裕，也不等於信心已校準。")]
    else:
        story += [para('錯分反映部分瑕疵紋理與灰階對比的相似性；高信心錯誤也顯示 confidence 並非可靠的正確性保證。此為單一 seed 的驗證分析，沒有以少數案例宣稱已確認模型的因果機制。')]
    story += [PageBreak(), heading('訓練成本與一致計時比較'),
        table(['實驗', '可訓練參數', '完整15 epochs時間', 'Checkpoint'],
            [[aliases[r['experiment']], f"{r['trainable_parameters']:,}", f"{r['training_seconds']:.2f} s", mb(r['checkpoint_bytes'])]
             for r in results], [112, 135, 136, 110]),
        Spacer(1, 8), para(f"MNV3-FE 比同架構 FT 少更新 {mobile_ft['trainable_parameters']-mobile_fe['trainable_parameters']:,} 個參數（{pct(1-mobile_fe['trainable_parameters']/mobile_ft['trainable_parameters'])}），本次完整訓練流程少 {mobile_ft['training_seconds']-mobile_fe['training_seconds']:.2f} 秒。時間包含15 epochs訓練、每輪驗證與最佳checkpoint寫入，排除模型／資料準備及最後重新評估；是本次紀錄，沒有重複測量，不能視為跨裝置固定效益。FE與FT推論圖相同，較少更新參數不代表推論必然更快。"),
        subheading('同一驗證樣本、同一部署計時範圍'),
        table(['實驗', '平均 / ms', 'P50 / ms', 'P95 / ms'],
            [[aliases[row['experiment']], f"{row['mean_ms']:.3f}", f"{row['p50_ms']:.3f}", f"{row['p95_ms']:.3f}"]
             for row in benchmark['models']], [160, 111, 111, 111]),
        Spacer(1, 8), para(f"裝置 {benchmark['models'][0]['device']}。每組使用同270張 Validation、六類各45張，224×224、batch=1、FP32；20次暖機後同步CUDA。從記憶體影像開始，計時包含CPU前處理、CPU/GPU傳輸、forward及分類結果回CPU；排除讀圖及模型載入。保留逐張延遲與影像ID於 experiments/deployment_benchmark.json，避免只重複單張樣本。"),
        para(f"最終模型 {final['parameters']:,} parameters，checkpoint {mb(final['checkpoint_bytes'])}。本表提供兩種架構與四組訓練結果在相同輸入尺寸、相同計時範圍下的實測成本；相機擷取、網路與PLC不在測量範圍。"),
        para(f"本機 R18-FT 平均 {resnet_ft_latency['mean_ms']:.3f} ms，MNV3-FE 為 {selected_latency['mean_ms']:.3f} ms；較小模型沒有在此GPU上換得較短延遲。兩者驗證F1同為1.0，本次選MNV3-FE依據是較小權重及較少更新參數，而非宣稱它最快。"),
        para('若產線只要求影像分類，最終模型依實測 Macro F1 決定；若受儲存／記憶體限制，可由表中模型大小評估精度成本，但本報告未擅自設定業主的最低可接受精度或硬體限制。若需要缺陷位置，分類結果仍需 Stage 3 的 detector 支援。'),
        PageBreak()] + sources(2, [f"{architecture} pretrained：{next(r['pretrained_source'] for r in results if r['architecture'] == architecture)}" for architecture in ('resnet18', 'mobilenet_v3_small')] + [
            'ResNet18：https://docs.pytorch.org/vision/0.21/models/resnet.html',
            'MobileNetV3：https://docs.pytorch.org/vision/0.21/models/mobilenetv3.html'])
    build(root / 'stage2_report.pdf', story)


def final_report():
    r1, r2, r3 = [SUB / f'Team07_Stage{n}' for n in (1, 2, 3)]
    s1, s2, s3 = [read(root / 'experiments/summary.json') for root in (r1, r2, r3)]
    c1, c2, c3 = [read(root / 'config.json') for root in (r1, r2, r3)]
    e2 = read(r2 / 'evaluation/metrics.json')
    f1 = s1['selected_model']
    f2 = next(r for r in s2['experiments'] if r['experiment'] == s2['final_experiment'])
    story = cover('Final Report', '三階段結果、失敗分析與部署決策')
    story += [heading('1. Problem & Dataset'),
        para('FactoryVision 的目標是建立智慧工廠鋼材表面瑕疵辨識與定位系統。Stage 1 以傳統方法和自行設計 CNN 建立分類基準並做消融；Stage 2 比較架構家族與遷移學習策略；Stage 3 偵測瑕疵類別與 bounding box。本報告統整同一批交付檔中的實際結果，模型與各 Stage checkpoint 完全一致。'),
        table(['資料', 'Train', 'Validation', '標註'], [
            ['Classification', '1,260', '270', '六類影像 label；每類210 / 45'],
            ['Detection', '1,260', '270', '2,924 / 641 bounding boxes']], [135, 90, 90, 178]),
        Spacer(1, 12), para('共用助教指定 Master Split，原圖200×200，seed=42。資料完整性檢查：ZIP CRC通過、所有3,060張PNG可解碼、VOC／YOLO框座標相符。沒有 Hidden data，不報告虛構的測試成績。'),
        subheading('評估界線'), para('分類以 Macro F1 選模，偵測以 mAP@0.5 選模。各 Stage 只用 Training 更新模型；Validation 用於選 epoch、分析及比較，因此這些數值是模型選擇集的結果，不能視為未見測試資料的獨立估計。'),
        para('指定 split 原已含一組完全相同影像：train/patches/TRN_0504.png 與 validation/patches/VAL_0101.png。遵守固定 split 未刪改，並揭露其可能造成的樂觀偏差。六類包含 crazing、inclusion、patches、pitted_surface、rolled_in_scale、scratches。'),
        PageBreak(), heading('2. Stage 1'),
        para('傳統基準採PCA100+RBF SVM；自製SurfaceCNN從零訓練。A是翻轉與小角度旋轉的basic augmentation，B是classifier Dropout0.5。四組以相同40 epochs、seed42、batch64及Adam0.001完成。'),
        table(['實驗', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1'],
            [metric_row(n, s1['experiments'][n]) for n in ['traditional', 'E1', 'E2', 'E3', 'E4']], [113, 95, 95, 95, 95]),
        Spacer(1, 12), para(f"最終 improved CNN 為 {s1['selected_experiment']}，Macro F1={pct(f1['macro_f1'])}，best epoch={f1['best_epoch']}。E2、E3、E4 相對E1的F1差分別為 " +
            '、'.join(f"{100*(s1['experiments'][n]['macro_f1']-s1['experiments']['E1']['macro_f1']):+.2f} 個百分點" for n in ['E2', 'E3', 'E4']) + '。'),
        figure(r1 / f"experiments/{s1['selected_experiment']}/training_curves.png", '所選 Improved CNN 的完整訓練曲線。', max_height=230),
        para('沒有預設兩種改良疊加必定最好。最終選模使用完整 ablation 的實測結果；曲線需考慮訓練 augmentation／dropout 和驗證eval模式的差異。'),
        para('E1錯7張、E2錯6張：擴增修正VAL_0097、VAL_0236，卻新增VAL_0082錯分，淨增加1張正確，共同錯5張。這是小幅、單次驗證集改善，不能視為統計顯著；三張變動影像已列於Stage1配對分析。E4在epoch34的train/val accuracy為92.30%/95.19%，最後降至90.56%/90.74%；正則化強度與更新波動是待驗證假說。'),
        PageBreak(), heading('3. Stage 2'),
        para('不同家族ResNet18與MobileNetV3-Small，各自跑Feature Extraction與Full Fine-tuning。使用官方ImageNet-1K V1權重，固定224×224、15epochs、batch32、seed42及相同evaluation流程。FE凍結backbone與BN統計；FT更新全部參數。'),
        table(['架構 / 策略', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1'],
            [metric_row(r['architecture'] + ' / ' + ('FE' if r['strategy']=='feature_extraction' else 'FT'), r['metrics']) for r in s2['experiments']], [145, 87, 87, 87, 87]),
        Spacer(1, 12), table(['架構 / 策略', '參數', '可訓練參數', 'Checkpoint'],
            [[r['architecture']+' / '+('FE' if r['strategy']=='feature_extraction' else 'FT'), f"{r['parameters']:,}", f"{r['trainable_parameters']:,}", mb(r['checkpoint_bytes'])] for r in s2['experiments']], [169, 110, 110, 104]),
        Spacer(1, 12), para(f"最終選 {f2['architecture']} / {f2['strategy']}，Macro F1={pct(f2['metrics']['macro_f1'])}。依序比較最高MacroF1、較小checkpoint、較少可訓練參數，最後以名稱排序固定平手。"),
        para('MobileNet FE/FT同為100%且檔案同大；FE更新596,998參數，FT更新1,524,006。本次15epochs迴圈（含每epoch驗證與checkpoint儲存）FE為44.80秒、FT48.44秒；ResNet FE/FT為48.42/71.46秒。選FE有較低更新成本，訓練時間僅為單次同機紀錄，不當作跨硬體保證。'),
        PageBreak(), heading('4. Stage 3'),
        para('以官方COCO預訓練YOLO11n進行60epochs訓練，320×320、batch16、AdamW、seed42。依完整Validation mAP@0.5選epoch，confidence=0.25、NMS IoU=0.7；輸出為原圖xyxy_pixel。'),
        table(['驗證指標', '實測值'], [
            ['mAP@0.5', pct(s3['official_metrics']['map50'])],
            ['mAP@0.5:0.95（補充）', pct(s3['official_metrics']['map50_95'])],
            ['官方 Macro P / R（max-F1 operating point）', pct(s3['official_metrics']['macro_precision_at_max_f1'])+' / '+pct(s3['official_metrics']['macro_recall_at_max_f1'])],
            ['固定 confidence=0.25 的 Micro P / R', pct(s3['fixed_threshold_metrics']['micro']['precision'])+' / '+pct(s3['fixed_threshold_metrics']['micro']['recall'])],
            ['選中 epoch', s3['selection']['selected_epoch']]], [310, 183]),
        Spacer(1, 8), figure(r3 / 'experiments/figures/per_class_ap50.png', '六類 Validation AP@0.5。', max_height=240),
        para('AP最低confidence為0.001，依confidence排序計算P/R曲線；固定0.25的P/R另以同類別、IoU>=0.5、一對一matching統計。兩種operating point定義不同，不混為同一個precision／recall。'),
        PageBreak(), heading('5. Failure Analysis'),
        subheading('分類錯誤'),
        para('Stage 1：'+top_confusions(f1['confusion_matrix'], c1['class_order'])),
        para('Stage 2：'+top_confusions(f2['metrics']['confusion_matrix'], c2['class_order'])),
        para('Stage1錯分以pitted_surface最多（Recall=41/45）；VAL_0159/0160密集點狀紋理伴粗暗紋，被判為crazing。Stage2最終模型雖270/270正確，VAL_0084仍僅以0.517對0.470分開inclusion與scratches，完美驗證分數不等於無不確定性。這些影像觀察不是因果證明，也不是獨立測試保證。'),
        subheading('偵測錯誤與定位'),
        para(f"Stage3在固定0.25 confidence下共有{len(s3['failure_cases'])}個選列的真實失敗案例；完整GT／Prediction對照與逐例分析位於stage3_report.pdf。FP表示多餘或錯類的預測，FN表示未成功匹配的GT，定位不佳是重疊存在但未達IoU門檻。"),
        figure(r3 / next(case['figure'] for case in s3['failure_cases'] if '0145' in case['image']), 'VAL_0145：左GT、右Prediction；右側偵測成功，左側弱紋理漏檢。', max_height=220),
        para('VAL_0145漏檢區有同類IoU=0.851、confidence=0.180的候選，被0.25門檻濾除。VAL_0046左側縱向inclusion被拆短並產生競爭配對；VAL_0077左側亮線被判為scratches但該區無提供的GT，依現有標註計FP，不斷言背景或漏標。五例完整量化與影像見Stage3。'),
        PageBreak(), heading('6. Deployment / Engineering Trade-off')]
    lat1 = read(r1 / 'experiments/deployment_benchmark.json')['models'][0]
    lat2 = next(row for row in read(r2 / 'experiments/deployment_benchmark.json')['models'] if row['experiment'] == s2['final_experiment'])
    lat3 = read(r3 / 'experiments/deployment_benchmark.json')['models'][0]
    story += [table(['模型 / 輸入', '主要驗證指標', '推論權重', '平均 / P95 (ms)'], [
        ['Stage1 '+s1['selected_experiment']+' / 128', 'Macro F1 '+pct(f1['macro_f1']), mb(f1['checkpoint_bytes']), f"{lat1['mean_ms']:.3f} / {lat1['p95_ms']:.3f}"],
        ['Stage2 MobileNet / 224', 'Macro F1 '+pct(f2['metrics']['macro_f1']), mb(f2['checkpoint_bytes']), f"{lat2['mean_ms']:.3f} / {lat2['p95_ms']:.3f}"],
        ['Stage3 YOLO11n / 320', 'mAP@0.5 '+pct(s3['official_metrics']['map50']), mb(s3['checkpoint_size_bytes']), f"{lat3['mean_ms']:.3f} / {lat3['p95_ms']:.3f}"]], [130, 130, 100, 133]),
        Spacer(1, 10), para('同一RTX3060、batch=1、FP32、20次暖機、同步CUDA；全部270個相同影像ID，六類各45張，分別保留原分類灰階／偵測RGB版本。統一從記憶體BGR影像計時至結果回CPU，含前處理、傳輸、forward及分類分數/argmax或偵測NMS，排除磁碟I/O與載入模型。原圖版本、輸入尺寸及輸出任務不同，因此反映應用流程成本，不能解讀為純架構速度比。'),
        para(f"Stage3訓練快照{mb(s3['checkpoint_export']['training_snapshot_bytes'])}保留供追溯，正式推論權重{mb(s3['checkpoint_size_bytes'])}已移除optimizer且重載指標一致。Stage1/2權重存FP32，Stage3 EMA存FP16，計時全以FP32執行；檔案大小不是GPU執行記憶體用量。"),
        subheading('漏檢代價決定能否上線'),
        table(['confidence', 'TP / FP / FN', 'Precision', 'Recall'],
              [[str(row['confidence_threshold']), f"{row['micro']['tp']} / {row['micro']['fp']} / {row['micro']['fn']}", pct(row['micro']['precision']), pct(row['micro']['recall'])]
               for row in s3['threshold_sweep'] if row['confidence_threshold'] in (0.05, 0.25, 0.5)], [95, 184, 107, 107]),
        Spacer(1, 9), para('0.25門檻漏掉195/641個GT（30.42%），crazing僅找回24/102（Recall23.53%、AP39.95%）。降低到0.05可把整體Recall提高至88.77%，卻有1,363個FP，無法只靠降低門檻就宣稱解決問題。現有結果不足以支持直接自動化放行，真實工廠暫緩直接部署。'),
        PageBreak(), heading('7. Conclusion')]
    classification_choice = ('Stage1 '+s1['selected_experiment']) if (f1['macro_f1'], -f1['checkpoint_bytes']) >= (f2['metrics']['macro_f1'], -f2['checkpoint_bytes']) else ('Stage2 '+f2['architecture']+' / '+f2['strategy'])
    story += [para(f'影像級分類選擇 {classification_choice}：在現有Validation精度平手時，以較小權重及較少更新參數作明確取捨。Stage3 YOLO11n是本作業的定位候選，但其整體30.42%漏檢率與crazing低召回使真實工廠應暫緩直接部署。分類100%並不能補償定位漏檢，也沒有驗證兩者級聯後的可靠性。'),
        para('本次完成傳統分類基準、四組CNN消融、四組遷移學習比較與一個完整detector。各Stage的README提供訓練與評估指令，實驗目錄保存設定、圖表與checkpoint。'),
        para('限制包括單一seed、驗證集同時用於選模、固定split的一組原始重複影像，以及尚無隱藏測試集／實際產線驗收。這些限制影響能對泛化與部署下結論的範圍。'),
        para(f"YOLO11n完整記憶體推論約{1000/lat3['mean_ms']:.1f}張/秒，未包含相機擷取、通訊或PLC，不能當成產線吞吐承諾。本次未實測更大detector，無法判定增加模型成本是否值得；沒有以未執行的實驗支持部署結論。"),
        subheading('交付對應'), table(['項目', '檔案'], [
            ['Stage1', 'Team07_Stage1.zip / stage1_model.pth / stage1_report.pdf'],
            ['Stage2', 'Team07_Stage2.zip / stage2_model.pth / stage2_report.pdf'],
            ['Stage3', 'Team07_Stage3.zip / stage3_model.pt / stage3_report.pdf'],
            ['整合', 'Team07_Final_Report.pdf']], [110, 383]),
        Spacer(1, 14), para('資料與規定來源：課程FactoryVision_Final_Project.pdf與iLearning作業頁。模型來源：Torchvision官方ImageNet-1K weights、Ultralytics官方COCO YOLO11n；完整URL列於各Stage報告與README。', 'small')]
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    build(output / 'Team07_Final_Report.pdf', story)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['1', '2', '3', 'final', 'all'], required=True)
    args = parser.parse_args()
    if args.stage in ('1', 'all'):
        stage1()
    if args.stage in ('2', 'all'):
        stage2()
    if args.stage in ('3', 'all'):
        from build_stage3_report import main as build_stage3
        build_stage3()
    if args.stage in ('final', 'all'):
        final_report()
