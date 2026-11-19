import json
import os

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ISP = r'C:\Users\ravi8\Downloads\isp'
SCR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ISP, 'presentation', 'Federated_Unlearning_Audit.pptx')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

NAVY, TEAL, CORAL = RGBColor(0x14, 0x21, 0x3D), RGBColor(0x2A, 0x9D, 0x8F), RGBColor(0xE7, 0x6F, 0x51)
INK, MUTED, TINT, WHITE = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xEE, 0xF4, 0xF6), RGBColor(0xFF, 0xFF, 0xFF)
SAND, GREY = RGBColor(0xF4, 0xA2, 0x61), RGBColor(0x9C, 0xA3, 0xAF)
HEAD, BODY = 'Cambria', 'Calibri'

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(6)
        segs = p if isinstance(p, tuple) else ((p, {}),) if isinstance(p, str) else p
        if isinstance(p, str):
            segs = ((p, {}),)
        for s, o in segs:
            r = para.add_run()
            r.text = s
            r.font.size = Pt(o.get('size', size))
            r.font.bold = o.get('bold', bold)
            r.font.italic = o.get('italic', False)
            r.font.name = o.get('font', font)
            r.font.color.rgb = o.get('color', color)
    return tb


def title(slide, t, sub=None, dark=False):
    text(slide, 0.6, 0.4, 12.1, 0.9, t, size=34, bold=True, font=HEAD, color=WHITE if dark else NAVY, name='Title')
    if sub:
        text(slide, 0.6, 1.2, 12.1, 0.5, sub, size=16, color=GREY if dark else MUTED, name='Subtitle')


def card(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.5)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def badge(slide, x, y, d, label, fill=TEAL, size=16):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, WHITE, BODY
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def style_chart(ch, legend=False, size=12):
    ch.font.size = Pt(size)
    ch.font.name = BODY
    ch.font.color.rgb = MUTED
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def footer(slide, s, dark=False):
    text(slide, 0.6, 6.95, 12.1, 0.35, s, size=10, color=GREY if dark else MUTED)


# ---------------------------------------------------------------- data
S = json.load(open(os.path.join(ISP, 'results_final', 'final_summary.json')))['summary']
ORDER = ['original', 'retrain', 'fedosd_hard', 'soft_fedosd', 'plain_ga', 'posttrain_plain', 'posttrain_proj']
LAB = {'original': 'Original', 'retrain': 'Retrain (gold)', 'fedosd_hard': 'FedOSD hard',
       'soft_fedosd': 'Soft FedOSD (ours)', 'plain_ga': 'Plain GA', 'posttrain_plain': 'E3 post-train',
       'posttrain_proj': 'E4 post-train (proj.)'}
rows = [r for s in (31, 32, 33) for r in json.load(open(os.path.join(ISP, 'results_final', f'seed{s}', 'relearn.json')))]

# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, 0.8, 1.6, 11.5, 0.5, 'FEDERATED UNLEARNING FOR VISION MODELS', size=16, bold=True, color=TEAL)
text(s, 0.8, 2.2, 11.5, 1.8, 'What did the model forget? Auditing and visualising a "right to be forgotten" request',
     size=40, bold=True, font=HEAD, color=WHITE)
text(s, 0.8, 4.3, 11.5, 0.6, 'Soft FedOSD on CIFAR-10, with MIA, Grad-CAM, distance-to-retrain and relearning audits',
     size=18, color=GREY)
text(s, 0.8, 5.9, 11.5, 0.5, '[Your name]  ·  [Course / date]', size=16, color=WHITE)
for i, c in enumerate((TEAL, SAND, CORAL)):
    badge(s, 10.9 + i * 0.55, 6.0, 0.35, '', fill=c)
notes(s, 'Title. One sentence: we simulate a client asking to be forgotten, unlearn it, and then check '
         'with four independent audits whether the model really forgot.')

# ---------------------------------------------------------------- 2 brief -> delivered
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'The brief, and what we delivered', 'Every item of the problem statement is implemented and evaluated on 3 final seeds')
items = [('FL pipeline on an image dataset', 'FedAvg, 10 clients, CIFAR-10, non-IID (2 classes per client)'),
         ('Simulate a client forget request', 'One client is backdoored; it asks to be forgotten'),
         ('Implement an unlearning method', 'FedOSD + our Soft FedOSD; gradient-ascent baseline; retrain as gold standard'),
         ('Membership inference before / after', 'Logistic-regression attacker, class-balanced, 5 splits'),
         ('Grad-CAM before vs after', 'Attention maps + trigger-energy and SSIM vs retrain'),
         ('Interactive demo or notebook', 'Gradio app ("Request to be forgotten") + results notebook')]
for i, (h, d) in enumerate(items):
    col, row = i % 2, i // 2
    x, y = 0.6 + col * 6.2, 1.95 + row * 1.6
    card(s, x, y, 5.9, 1.35)
    badge(s, x + 0.25, y + 0.38, 0.6, '✓', fill=TEAL, size=20)
    text(s, x + 1.05, y + 0.18, 4.7, 0.45, h, size=17, bold=True, color=NAVY)
    text(s, x + 1.05, y + 0.62, 4.7, 0.7, d, size=14, color=INK)
footer(s, 'Extras beyond the brief: L2/KL distance to retrain, relearning test, pre-registered protocol with no test-set leakage.')
notes(s, 'Map each bullet of the problem statement to what we built. Mention that we used CIFAR-10 rather than '
         'CelebA/CIFAR-100 (the brief says e.g.).')

# ---------------------------------------------------------------- 3 pipeline
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'End-to-end pipeline', 'From federated training to a forget request to four independent audits')
steps = [('1', 'Federated training', 'FedAvg, 100 rounds; client X carries a hidden backdoor'),
         ('2', 'Forget request', 'Client X asks to be removed ("right to be forgotten")'),
         ('3', 'Unlearning', '100 rounds of projected ascent on client X only'),
         ('4', 'Post-training', '100 rounds on retained clients to recover accuracy'),
         ('5', 'Audit', 'ASR · MIA · Grad-CAM · distance to retrain · relearning')]
w, gap = 2.25, 0.21
for i, (n, h, d) in enumerate(steps):
    x = 0.6 + i * (w + gap)
    card(s, x, 2.3, w, 3.2, fill=CORAL if i == 2 else TINT)
    badge(s, x + w / 2 - 0.35, 2.55, 0.7, n, fill=NAVY if i != 2 else WHITE, size=20)
    if i == 2:
        s.shapes[-1].text_frame.paragraphs[0].runs[0].font.color.rgb = CORAL
    fg = WHITE if i == 2 else NAVY
    text(s, x + 0.15, 3.45, w - 0.3, 0.8, h, size=18, bold=True, color=fg, align=PP_ALIGN.CENTER)
    text(s, x + 0.15, 4.2, w - 0.3, 1.2, d, size=14, color=WHITE if i == 2 else INK, align=PP_ALIGN.CENTER)
    if i < 4:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + w + 0.02), Inches(3.8), Inches(0.17), Inches(0.25))
        a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background()
text(s, 0.6, 5.85, 12.1, 0.8, [(('Gold standard: ', {'bold': True, 'color': NAVY}),
                                ('retrain from scratch on the 9 retained clients (300 rounds = same total budget). '
                                 'A perfect unlearning method should be indistinguishable from it.', {}))], size=15)
notes(s, 'Walk left to right. Step 3 is where our method lives.')

# ---------------------------------------------------------------- 4 setup
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Setup: a backdoor makes forgetting measurable', 'If the forget client is truly removed, its secret trigger must stop working')
stats = [('10', 'clients, 2 CIFAR-10 classes each'), ('1', 'forget client with a 4×4 patch trigger'),
         ('84%', 'attack success rate before unlearning')]
for i, (v, l) in enumerate(stats):
    y = 1.95 + i * 1.55
    text(s, 0.6, y, 2.6, 1.0, v, size=54, bold=True, font=HEAD, color=CORAL if i == 2 else TEAL)
    text(s, 3.1, y + 0.3, 3.4, 0.9, l, size=16, color=INK)
card(s, 6.9, 1.95, 5.8, 4.55)
text(s, 7.2, 2.15, 5.3, 4.2, [
    (('Data  ', {'bold': True, 'color': NAVY}), ('CIFAR-10, 10 clients, pathological non-IID split', {})),
    (('Model  ', {'bold': True, 'color': NAVY}), ('small CNN (2 conv + 3 FC); ResNet-18 skipped for compute', {})),
    (('Trigger  ', {'bold': True, 'color': NAVY}), ('white 4×4 square, bottom-right; flips the label to the target class', {})),
    (('Check  ', {'bold': True, 'color': NAVY}), ('without the trigger the target rate is only 6-8% → the trigger causes it', {})),
    (('Splits  ', {'bold': True, 'color': NAVY}), ('each client: train / validation / test; validation only for every choice', {})),
], size=15)
footer(s, 'An earlier random-pattern trigger was found never to be learned (same rate with and without it); it was replaced and the gate was added.')
notes(s, 'Why a backdoor: it gives a crisp, measurable signal of what the forget client taught the model.')

# ---------------------------------------------------------------- 5 methods
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Methods compared', 'Our contribution: a soft, closed-form version of FedOSD\'s gradient projection')
cards = [('Retrain', 'Gold standard. Train from scratch without the forget client.', NAVY),
         ('Plain gradient ascent', 'Push the loss of the forget client up. No protection for other clients.', GREY),
         ('FedOSD (hard)', 'Ascend for the forget client, but project out any direction that hurts retained clients.', TEAL),
         ('Soft FedOSD (ours)', 'Replace the hard constraint by a penalty λ and solve in closed form; optional adaptive client weights η.', CORAL)]
for i, (h, d, c) in enumerate(cards):
    x = 0.6 + i * 3.08
    card(s, x, 1.95, 2.85, 2.9, fill=TINT, line=c if i == 3 else None)
    badge(s, x + 0.25, 2.2, 0.5, str(i + 1), fill=c, size=16)
    text(s, x + 0.2, 2.85, 2.5, 0.6, h, size=18, bold=True, color=c if i == 3 else NAVY)
    text(s, x + 0.2, 3.45, 2.5, 1.4, d, size=14)
card(s, 0.6, 5.15, 12.1, 1.45, fill=NAVY)
text(s, 0.9, 5.3, 11.5, 1.2, [
    (('Soft FedOSD update   ', {'bold': True, 'color': SAND}),
     ('d = g_u − Âᵀ (W⁻¹/λ + ÂÂᵀ)⁻¹ Â g_u', {'font': 'Cambria Math', 'color': WHITE, 'size': 20})),
    (('g_u: unlearning gradient · Â: retained-client gradients · W: client weights (uniform, or adaptive by accuracy deficit) · '
      'λ → ∞ recovers the hard projection', {'color': GREY, 'size': 13}),)], size=16)
footer(s, 'Post-training variants: E3 = plain recovery training, E4 = projected recovery (FedOSD default).')
notes(s, 'Key intuition: hard FedOSD never lets the update hurt a retained client; the soft version allows a small, '
         'weighted amount of harm in exchange for faster forgetting.')

# ---------------------------------------------------------------- 6 protocol
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'No data leakage: a pre-registered protocol', 'Every choice made on validation data of tuning seeds; final seeds run exactly once')
phases = [('Pre-register', 'Hypothesis, grid and decision rule written and committed before any run', NAVY),
          ('Tune  ·  seeds 101-103', '3 splits × 3 weightings + λ sweep; validation accuracy only (33 runs)', TEAL),
          ('Decide', 'Rule applied by a script: uniform weights, λ = 100', SAND),
          ('Report  ·  seeds 31-33', 'All methods run once; test split read only here', CORAL)]
for i, (h, d, c) in enumerate(phases):
    x = 0.6 + i * 3.08
    badge(s, x + 1.1, 1.95, 0.65, str(i + 1), fill=c, size=20)
    text(s, x, 2.75, 2.85, 0.5, h, size=17, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    text(s, x + 0.1, 3.25, 2.65, 1.3, d, size=14, align=PP_ALIGN.CENTER)
    if i < 3:
        ln = s.shapes.add_connector(1, Inches(x + 1.95), Inches(2.27), Inches(x + 3.9), Inches(2.27))
        ln.line.color.rgb = GREY; ln.line.width = Pt(1.5)
card(s, 0.6, 4.85, 12.1, 1.75)
text(s, 0.9, 5.0, 11.6, 1.5, [
    (('Safeguards  ', {'bold': True, 'color': NAVY}),
     ('code refuses any validation→test fallback · forbidden seeds rejected · every change disclosed with its timing', {})),
    (('Amendments (disclosed)  ', {'bold': True, 'color': NAVY}),
     ('A: trigger replaced after an audit · B: gate threshold relaxed after seeing gate values, before any unlearning result', {})),
], size=15)
notes(s, 'This is how we can claim the numbers are honest: nothing about the method was tuned on the reported seeds.')

# ---------------------------------------------------------------- 7 tuning decision
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Tuning result: adaptive weighting did not pass the rule', 'Worst-client gain from adaptive weights, validation, seeds 101-103')
cd = CategoryChartData()
cd.categories = ['pathological', 'Dirichlet 0.5', 'Dirichlet 0.1']
cd.add_series('Floor gain (pp)', (2.67, 0.08, 2.28))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.85), Inches(6.3), Inches(4.6), cd)
ch = gf.chart
style_chart(ch)
pl = ch.plots[0]
pl.gap_width = 70
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '+0.00" pp"', False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size, pl.data_labels.font.color.rgb = Pt(14), INK
for i, pt in enumerate(pl.series[0].points):
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = CORAL if i == 2 else TEAL
ch.value_axis.has_major_gridlines = True
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
ch.value_axis.maximum_scale = 3.5
ch.has_title = True
ch.chart_title.text_frame.text = 'G = floor(η = 1) − floor(η = 0)'
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
card(s, 7.3, 1.95, 5.4, 4.5)
text(s, 7.6, 2.15, 4.9, 4.2, [
    (('Rule: ', {'bold': True, 'color': NAVY}), ('adaptive wins only if G(Dir 0.1) ≥ 2 pp AND G(Dir 0.1) > G(pathological)', {})),
    (('Result: ', {'bold': True, 'color': NAVY}), ('2.28 ≥ 2 ✓   but   2.28 < 2.67 ✗  →  uniform weights', {})),
    (('λ choice: ', {'bold': True, 'color': NAVY}), ('λ = 5 / 20 / 100 gave retained mean 37.9 / 41.6 / 43.2%, all with ASR < 1% → λ = 100', {})),
    (('Note: ', {'bold': True, 'color': CORAL}), ('the pathological split turned out harsher than Dir 0.1, so a +2-3 pp gain on harsh splits is a real but unconfirmed signal', {'italic': True})),
], size=15)
notes(s, 'Explain that the rule was fixed in advance, so we accept its verdict even though adaptive weighting looked mildly helpful.')

# ---------------------------------------------------------------- 8 headline
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
title(s, 'Headline: the backdoor is gone, utility is restored', 'Final seeds 31-33, test split, mean over 3 seeds', dark=True)
big = [('84% → 1.8%', 'attack success rate\nafter Soft FedOSD', CORAL),
       ('51.6%', 'clean accuracy after post-training\n(original model: 49.3%)', TEAL),
       ('≈', 'Soft FedOSD vs hard FedOSD\n42.2% vs 42.7% clean accuracy', SAND)]
for i, (v, l, c) in enumerate(big):
    x = 0.6 + i * 4.1
    text(s, x, 2.3, 3.9, 1.3, v, size=40 if i == 0 else 50, bold=True, font=HEAD, color=c, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.BOTTOM)
    text(s, x, 3.7, 3.9, 1.2, l, size=16, color=WHITE, align=PP_ALIGN.CENTER)
text(s, 0.6, 5.4, 12.1, 1.0, 'Honest result: at the chosen λ the soft relaxation behaves like the hard projection, so it does not '
     'beat FedOSD here. We report this null result rather than tune it away on the test seeds.', size=16, color=GREY,
     align=PP_ALIGN.CENTER)
notes(s, 'Three numbers to remember. Be upfront that our variant matches but does not beat hard FedOSD.')

# ---------------------------------------------------------------- 9 chart: ASR + clean acc
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Forgetting vs utility across methods', 'Backdoor ASR (lower is better) and clean retained accuracy (higher is better)')
cd = CategoryChartData()
cd.categories = [LAB[m] for m in ORDER]
cd.add_series('Backdoor ASR (%)', [round(S[m]['asr'][0], 1) for m in ORDER])
cd.add_series('Clean accuracy (%)', [round(S[m]['clean_acc'][0], 1) for m in ORDER])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.8), Inches(8.4), Inches(4.9), cd)
ch = gf.chart
style_chart(ch, legend=True, size=11)
pl = ch.plots[0]
pl.gap_width, pl.overlap = 60, -10
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0.0', False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(10)
for ser, c in zip(pl.series, (CORAL, TEAL)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = c
ch.value_axis.maximum_scale = 100
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
card(s, 9.3, 1.95, 3.4, 4.6)
text(s, 9.5, 2.1, 3.0, 4.4, [
    (('All methods ', {'bold': True, 'color': NAVY}), ('cut ASR from 84% to < 2%.', {})),
    (('Plain GA ', {'bold': True, 'color': NAVY}), ('overshoots: clean accuracy drops to 34%.', {})),
    (('Post-training ', {'bold': True, 'color': NAVY}), ('recovers utility (51.6%) with ASR near retrain.', {})),
    (('E3 = E4: ', {'bold': True, 'color': NAVY}), ('projection during recovery made no difference.', {})),
], size=14)
footer(s, 'Retrain reaches 59.6% clean accuracy partly because it trains 300 rounds from scratch.')
notes(s, 'Point at Plain GA as the reason projection matters, and at E3/E4 being equal.')

# ---------------------------------------------------------------- 10 table
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Full results (mean ± std, 3 seeds)', 'Test split of final seeds 31-33')
cols = [('Model', None), ('Clean acc', 'clean_acc'), ('ASR', 'asr'), ('Forget held-out acc', 'heldout_acc'),
        ('Worst client', 'floor_test'), ('MIA AUC', 'mia_attack_auc_mean'), ('KL to retrain', 'distance_mean_kl_retrain_to_model')]
tbl = s.shapes.add_table(len(ORDER) + 1, len(cols), Inches(0.6), Inches(1.85), Inches(12.1), Inches(4.6)).table
widths = [2.6, 1.55, 1.45, 1.85, 1.6, 1.45, 1.6]
for j, w in enumerate(widths):
    tbl.columns[j].width = Inches(w)
for j, (h, _) in enumerate(cols):
    c = tbl.cell(0, j); c.text = h
    c.fill.solid(); c.fill.fore_color.rgb = NAVY
    r = c.text_frame.paragraphs[0].runs[0]; r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(13), True, WHITE, BODY
for i, m in enumerate(ORDER, start=1):
    for j, (h, k) in enumerate(cols):
        c = tbl.cell(i, j)
        if k is None:
            c.text = LAB[m]
        else:
            mu, sd = S[m][k][0], S[m][k][1]
            c.text = 'n/a' if mu != mu else (f'{mu:.3f} ± {sd:.3f}' if k.startswith(('mia', 'distance')) else f'{mu:.1f} ± {sd:.1f}')
        c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xFD, 0xE8, 0xE2) if m == 'soft_fedosd' else (TINT if i % 2 else WHITE)
        r = c.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.name, r.font.color.rgb = Pt(12), BODY, INK
        r.font.bold = m == 'soft_fedosd' or j == 0
footer(s, 'Worst client = lowest test accuracy among the 9 retained clients. Full metric list: results_final/final_summary.csv')
notes(s, 'Reference slide; do not read it out. Highlight the coral row.')

# ---------------------------------------------------------------- 11 MIA
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Membership inference: no signal to remove', 'Attack AUC on forget-client training images vs its unseen held-out images')
cd = CategoryChartData()
cd.categories = [LAB[m] for m in ORDER]
cd.add_series('MIA AUC', [round(S[m]['mia_attack_auc_mean'][0], 3) for m in ORDER])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.8), Inches(7.0), Inches(4.8), cd)
ch = gf.chart
style_chart(ch, size=12)
pl = ch.plots[0]
pl.gap_width = 60
pl.has_data_labels = True
pl.data_labels.number_format, pl.data_labels.number_format_is_linked = '0.000', False
pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
pl.data_labels.font.size = Pt(11)
pl.series[0].format.fill.solid(); pl.series[0].format.fill.fore_color.rgb = TEAL
ch.value_axis.minimum_scale, ch.value_axis.maximum_scale = 0.4, 0.6
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
ch.category_axis.reverse_order = True
text(s, 8.0, 1.95, 4.7, 1.2, '≈ 0.50', size=60, bold=True, font=HEAD, color=CORAL)
text(s, 8.0, 3.2, 4.7, 0.5, 'AUC for every model, even before unlearning', size=16, bold=True, color=NAVY)
text(s, 8.0, 3.9, 4.7, 2.6, [
    'The original model does not overfit the forget client: 47.7% accuracy on its training images vs 48.4% on unseen ones.',
    'With no membership signal before unlearning, MIA cannot certify that unlearning removed one. We report this rather than overclaim.'],
    size=15)
notes(s, 'Important honesty point: a common mistake is to claim MIA ≈ 0.5 after unlearning as proof of forgetting; '
         'here it was already 0.5 before.')

# ---------------------------------------------------------------- 12 Grad-CAM
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Grad-CAM: where the model looks', 'Triggered forget-client images; red box = trigger region (seed 31)')
im = Image.open(os.path.join(ISP, 'results_final', 'seed31', 'audit_gradcam.png'))
crop = os.path.join(SCR, 'cam_crop.png')
im.crop((65, 82, 1460, 995)).save(crop)
pic_w = 7.3
pic_h = pic_w * (995 - 82) / (1460 - 65)
labels = ['Original', 'Soft FedOSD', 'FedOSD hard', 'Retrain']
for i, l in enumerate(labels):
    text(s, 0.6 + i * pic_w / 4, 1.75, pic_w / 4, 0.4, l, size=13, bold=True, color=CORAL if i == 1 else NAVY, align=PP_ALIGN.CENTER)
s.shapes.add_picture(crop, Inches(0.6), Inches(2.15), Inches(pic_w), Inches(pic_h))
card(s, 8.3, 1.95, 4.4, 4.7)
text(s, 8.55, 2.1, 3.95, 4.5, [
    (('Original: ', {'bold': True, 'color': NAVY}), ('heat sits on the trigger patch, the model\'s shortcut to the backdoor label.', {})),
    (('Retrain: ', {'bold': True, 'color': NAVY}), ('heat on the object; only 3% of CAM energy in the patch.', {})),
    (('After unlearning: ', {'bold': True, 'color': NAVY}), ('the backdoor prediction is gone, yet 44-49% of the remaining CAM energy still falls in the patch.', {})),
    (('Reading: ', {'bold': True, 'color': CORAL}), ('behaviour removed, some trigger sensitivity remains (CAMs here are weak, so treat as indicative).', {'italic': True})),
], size=14)
notes(s, 'The left column is the "before", the middle two are "after" for the two FedOSD variants, right is the gold standard.')

# ---------------------------------------------------------------- 13 relearning
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Relearning test: does the backdoor come back?', 'Attacker fine-tunes on 32 triggered images; backdoor ASR per step, mean of 3 seeds')
xy = XyChartData()
show = [('original', GREY), ('retrain', NAVY), ('fedosd_hard', TEAL), ('soft_fedosd', CORAL), ('plain_ga', SAND)]
for m, _ in show:
    ser = xy.add_series(LAB[m])
    for k in range(0, 21):
        ser.add_data_point(k, float(np.mean([r['asr'] for r in rows if r['model'] == m and r['round'] == k])))
gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(0.6), Inches(1.8), Inches(7.6), Inches(4.9), xy)
ch = gf.chart
style_chart(ch, legend=True, size=11)
for ser, (m, c) in zip(ch.plots[0].series, show):
    ser.format.line.color.rgb = c
    ser.format.line.width = Pt(3.5 if m in ('soft_fedosd', 'retrain') else 2)
    ser.smooth = False
ch.value_axis.maximum_scale, ch.value_axis.minimum_scale = 100, 0
ch.category_axis.maximum_scale, ch.category_axis.minimum_scale = 20, 0
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE7, 0xEB)
ch.category_axis.has_title = True
ch.category_axis.axis_title.text_frame.text = 'relearning step'
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
card(s, 8.5, 1.95, 4.2, 4.7)
text(s, 8.75, 2.1, 3.75, 4.5, [
    (('Even retrain ', {'bold': True, 'color': NAVY}), ('(never saw the trigger) learns it in 1-2 steps: the patch is very easy to learn.', {})),
    (('Unlearned models ', {'bold': True, 'color': NAVY}), ('start slower, overtake retrain by step ~5, reach 99-100% vs 96%.', {})),
    (('Calibration: ', {'bold': True, 'color': CORAL}), ('a pre-registered weaker-attack grid found no attack that a clean model resists, so relearning cannot certify forgetting for this trigger.', {})),
], size=14)
footer(s, 'Future work: a harder-to-learn trigger (blended / low-contrast) would make the relearning audit discriminative.')
notes(s, 'Be precise: the test is not powerful enough here, and we showed that with a pre-registered calibration on tuning seeds.')

# ---------------------------------------------------------------- 14 demo
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, 'Interactive demo', 'Gradio app: pick a model and an image, then "Request to be forgotten"')
shot = r'C:\Users\ravi8\AppData\Local\Temp\claude\a3d1621303a65fe6\5452a042-4c43-4d2c-80aa-bca28d060788\images\1.png'
sim = Image.open(shot).crop((0, 110, 1920, 1020))
sp = os.path.join(SCR, 'demo_crop.png'); sim.save(sp)
w = 8.2
s.shapes.add_picture(sp, Inches(0.6), Inches(1.9), Inches(w), Inches(w * 910 / 1920))
card(s, 9.1, 1.95, 3.6, 4.7)
text(s, 9.3, 2.1, 3.2, 4.5, [
    (('Shows ', {'bold': True, 'color': NAVY}), ('Grad-CAM for the original, the selected model and retrain, side by side.', {})),
    (('Predictions ', {'bold': True, 'color': NAVY}), ('and confidences per model; audit table; relearning curve.', {})),
    (('Here: ', {'bold': True, 'color': CORAL}), ('the original model predicts "airplane" (the backdoor label) with 0.999 confidence because of the patch.', {})),
    (('[Add your screenshots of Soft FedOSD and Retrain on the same image]', {'italic': True, 'color': MUTED}),),
], size=14)
footer(s, 'Run: python -m demo.app --bundle bundles/seed31 --results results_final/seed31  (served locally; notebook: notebooks/results.ipynb)')
notes(s, 'Live demo if possible; otherwise use the screenshots.')

# ---------------------------------------------------------------- 15 conclusions
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
title(s, 'Conclusions', dark=True)
concl = [('1', 'Federated unlearning removes the backdoor\'s effect (ASR 84% → < 2%) while post-training restores utility.', TEAL),
         ('2', 'Soft FedOSD matches hard FedOSD at the chosen λ but does not beat it: a pre-registered null result.', CORAL),
         ('3', 'Audits disagree on "fully forgotten": MIA has no signal; Grad-CAM and relearning show residual trigger sensitivity.', SAND),
         ('4', 'Projection matters during unlearning (plain GA collapses), not during recovery (E3 = E4).', TEAL)]
for i, (n, t, c) in enumerate(concl):
    y = 1.6 + i * 1.05
    badge(s, 0.6, y + 0.05, 0.55, n, fill=c, size=16)
    text(s, 1.4, y, 11.3, 0.9, t, size=18, color=WHITE)
text(s, 0.6, 5.95, 12.1, 0.9, [
    (('Limitations & next steps: ', {'bold': True, 'color': SAND}),
     ('small CNN on CIFAR-10 (not ResNet-18 / CIFAR-100); 3 seeds; pretraining only 100 rounds; '
      'harder triggers for a discriminative relearning audit; adaptive weighting on more seeds.', {'color': GREY}))], size=14)
notes(s, 'End on what we learned, including the negative results; they are the most defensible part.')

prs.save(OUT)
print(OUT)
