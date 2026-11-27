"""Local report authoring helpers; not a dependency of submitted model code."""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
BLUE = colors.HexColor('#16425B')
TEAL = colors.HexColor('#187D8D')
MUTED = colors.HexColor('#526474')
pdfmetrics.registerFont(TTFont('CJK', 'C:/Windows/Fonts/msjh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJKB', 'C:/Windows/Fonts/msjhbd.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJKB', italic='CJK', boldItalic='CJKB')

STYLES = {
    'title': ParagraphStyle('title', fontName='CJKB', fontSize=26, leading=37, textColor=BLUE, spaceAfter=22, wordWrap='CJK'),
    'subtitle': ParagraphStyle('subtitle', fontName='CJK', fontSize=14, leading=22, textColor=MUTED, spaceAfter=14, wordWrap='CJK'),
    'h1': ParagraphStyle('h1', fontName='CJKB', fontSize=17, leading=25, textColor=BLUE, spaceBefore=4, spaceAfter=12, wordWrap='CJK'),
    'h2': ParagraphStyle('h2', fontName='CJKB', fontSize=12, leading=18, textColor=TEAL, spaceBefore=10, spaceAfter=7, wordWrap='CJK'),
    'body': ParagraphStyle('body', fontName='CJK', fontSize=9.6, leading=15.8, spaceAfter=8, wordWrap='CJK'),
    'small': ParagraphStyle('small', fontName='CJK', fontSize=8, leading=12, textColor=MUTED, spaceAfter=5, wordWrap='CJK'),
    'cell': ParagraphStyle('cell', fontName='CJK', fontSize=8, leading=12, wordWrap='CJK'),
    'header': ParagraphStyle('header', fontName='CJKB', fontSize=8, leading=12, textColor=colors.white, wordWrap='CJK'),
}


def para(text, style='body'):
    return Paragraph(escape(str(text)).replace('\n', '<br/>'), STYLES[style])


def heading(text):
    return para(text, 'h1')


def subheading(text):
    return para(text, 'h2')


def table(headers, rows, widths=None):
    data = [[para(c, 'header') for c in headers]] + [[para(c, 'cell') for c in row] for row in rows]
    item = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    item.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F5F7')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, 0), 0.4, BLUE),
        ('LINEBELOW', (0, -1), (-1, -1), 0.4, colors.HexColor('#CCD9DF')),
    ]))
    return item


def figure(path, caption='', max_width=493, max_height=340):
    image = Image(str(path))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = 'CENTER'
    return KeepTogether([image, Spacer(1, 5), para(caption, 'small'), Spacer(1, 8)])


def cover(stage, subtitle):
    return [Spacer(1, 55), para('FactoryVision', 'subtitle'),
        para('智慧工廠鋼材表面\n瑕疵辨識與定位系統', 'title'),
        para(stage, 'h1'), para(subtitle, 'subtitle'), Spacer(1, 35),
        table(['封面資料', '內容'], [['組別', '第七組 / Team07'],
             ['組員姓名 / 學號', '姓名：________________　學號：________________\n姓名：________________　學號：________________\n姓名：________________　學號：________________\n姓名：________________　學號：________________'],
             ['實驗日期', '2026-11-20'], ['課程', '電腦視覺']], [130, 363]),
        Spacer(1, 23), para('所有數值均來自本次實際訓練與固定 Validation split。未使用 Hidden data；未將 Validation 納入訓練。', 'small'),
        PageBreak()]


def dataset_section():
    return [heading('資料與評估原則'),
        para('使用助教提供的 FactoryVision Public Dataset。Classification 與 Detection 共用 Master Split：Train 1,260 張、Validation 270 張。六類各有 210 張訓練、45 張驗證影像；原圖均為 200×200。'),
        table(['類別順序', '名稱'], [[str(i), c] for i, c in enumerate([
            'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches'])], [100, 393]),
        Spacer(1, 12), para('主要實驗 seed=42。分類以 Macro F1 選模，並完整報 Accuracy、Macro Precision、Macro Recall、Confusion Matrix；偵測以 mAP@0.5 為主要指標。Validation 用於模型與超參數選擇，以下結果不是獨立測試集的泛化估計。'),
        subheading('資料限制'),
        para('原始固定 split 包含一組完全相同影像：classification/train/patches/TRN_0504.png 與 classification/validation/patches/VAL_0101.png。本實驗依作業要求保留原 split，未移動、刪除或合併資料；此既有重複可能使驗證表現略為樂觀。'),
        para('VOC 類名 rolled-in_scale 對應指定名稱 rolled_in_scale；Detection 共有 2,924 個訓練框、641 個驗證框。Hidden data 未提供，沒有推估其結果。')]


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#CCD9DF'))
    canvas.line(50, 41, A4[0]-50, 41)
    canvas.setFont('CJK', 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(50, 28, 'FactoryVision | Team07')
    canvas.drawRightString(A4[0]-50, 28, str(doc.page))


def build(path, story):
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=50, leftMargin=50,
        topMargin=49, bottomMargin=58, title=Path(path).stem, author='Team07')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def pct(value):
    return f'{value * 100:.2f}%'


def mb(value):
    return f'{value / (1024 * 1024):.2f} MiB'
