# Demo guide: "Federated unlearning: what did the model forget?"

The demo is a Gradio app. It lets you look inside the models before and after a client is
forgotten. It trains nothing live: every model was trained beforehand and is loaded from an
audit bundle.

## Start it

On the GPU server, with the venv active:
```bash
cd ~/ravi_isp/isp/audit_code
CUDA_VISIBLE_DEVICES=3 python -m demo.app --bundle ../bundles/seed31 --results ../results_final/seed31
```
On your laptop, open a tunnel and keep the window open:
```powershell
ssh -L 7860:localhost:7860 ds404_user8@10.8.1.106
```
Then go to http://localhost:7860. The app is visible only to you. To show another seed, change
both paths, for example `seed32`.

## The story it tells

One of the 10 clients (the *forget client*) secretly taught the model a **backdoor**. A small
white square in the bottom-right corner (the *trigger*) makes the model predict one fixed
class (the *target*, e.g. airplane), whatever the image shows. That client then asks to be
forgotten. If unlearning works, the trigger stops working and the model looks at the object
again, as a model retrained without that client does.

## Controls (top row)

| Control | What it does |
|---|---|
| **Model** | Which model fills the middle panel (see the model list below). |
| **Image group** | Which set of images to browse (see the image groups below). |
| **Image** (slider 0-23) | Picks one of 24 fixed images from that group. The same number always gives the same image, so you can compare models on one image. |
| **Request to be forgotten** | Switches the middle panel to the unlearned model (Soft FedOSD, our method) and shows which client was forgotten. It loads the precomputed result; it does not run unlearning live. |

### Models

| Name in the menu | Meaning |
|---|---|
| Original (before forgetting) | The federated model trained with all 10 clients, backdoor included. |
| Retrain (gold standard) | Trained from scratch without the forget client. The ideal outcome. |
| FedOSD (hard) | Unlearning with the hard gradient projection (baseline method). |
| Soft FedOSD (ours) | Our method: soft, closed-form projection (lambda = 100). |
| Plain gradient ascent | Naive unlearning without projection. It also destroys useful knowledge. |
| Post-training, no projection (E3) | Soft/hard unlearning followed by plain recovery training. |
| Post-training, projected (E4) | The same, with projected recovery training (FedOSD default). |

### Image groups

| Group | Images | What to look for |
|---|---|---|
| **Forget client + trigger** | The forget client's own training images, with the trigger added | **The main group.** The original model is fooled; a forgotten model should not be. |
| **Trigger test** | Other clients' test images with the trigger added | Does the backdoor work on unseen images? (The usual "attack success rate".) |
| **Retained clients (test)** | Normal test images of the other clients | Is the model still useful after unlearning? |
| **Forget client held-out (clean)** | The forget client's images the model never trained on | Did unlearning also destroy knowledge of those classes? (Plain GA does.) |

## Outputs

**Three Grad-CAM panels**
- **Original** (left), **Selected model** (middle), **Retrain (reference)** (right), all on the
  same image.
- Grad-CAM colours the pixels that pushed the model towards the class in the label line below:
  red/yellow means strong evidence, blue means none.
- For triggered images, that class is the **backdoor target**. So a hot spot on the bottom-right
  square means "the trigger is what made me say airplane".

**Label line** (under the panels)
- For triggered forget-client images it shows the **true class** and the **backdoor label**. The
  backdoor label is the class Grad-CAM explains.
- For clean images it shows the true class.

**Prediction table**
- What each of the three models predicts for this image, with its confidence (0-1).
- A forgotten model should **not** predict the backdoor label on triggered images.

**Audit metrics table**
- Pre-computed numbers for the original, selected and retrain models (seed of the bundle).
- `mia_attack_auc_mean`: membership-inference attack AUC. 0.5 means the attacker cannot tell
  whether the forget client's images were used in training. In our results it is about 0.5
  even before unlearning, so it carries no signal here.
- `distance_mean_kl_retrain_to_model`: how different the model's outputs are from retrain's.
  Lower means closer to the gold standard.
- `distance_l2_param_to_retrain`: weight-space distance to retrain. It is about 7.2 for every
  model, so it is not informative.
- `gradcam_<group>_energy_in_trigger_mean`: the share of Grad-CAM heat inside the 4x4 trigger
  square (0-1). The original is high, retrain is about 0.03.
- `gradcam_<group>_ssim_vs_retrain_mean`: how similar the heatmaps look to retrain's
  (1 = identical).
- `..._pearson_undefined_frac`: the share of images whose heatmap was blank, so no correlation
  could be computed.

**Relearning curve** (bottom)
- An attacker fine-tunes each model on 32 triggered images; the curve shows how fast the
  backdoor returns.
- Panels: `asr` (backdoor success), `forget_acc` (accuracy on the forget client's clean images),
  `mia_auc`.
- The selected model is drawn with a thick line.
- Even retrain learns the trigger within 1-2 steps, so this test cannot prove forgetting for
  this trigger (see RESULTS_SOFT_FEDOSD.md, section 5).

**Full Grad-CAM audit figure** (collapsible)
- The complete grid of heatmaps for all image groups and models, as saved by the audit.

## A 2-minute walkthrough for a presentation

1. **Model = Original**, **group = Forget client + trigger**, image 0.
   - Left panel: heat on the corner square.
   - Prediction table: the backdoor label (e.g. airplane) with about 0.99 confidence.
   - Say: *"the model was taught a shortcut by one client."*
2. Click **Request to be forgotten**. The middle panel switches to Soft FedOSD.
   - Its prediction is no longer the backdoor label.
   - Compare with retrain on the right, which looks at the object.
3. Move the **Image** slider through 2-3 images to show it is not a single lucky example.
4. **Model = Plain gradient ascent**, **group = Forget client held-out (clean)**.
   - It now gets these images wrong too.
   - Say: *"naive unlearning over-forgets; projection prevents that."*
5. **Group = Retained clients (test)** with Soft FedOSD.
   - Predictions are still sensible.
   - Say: *"the other clients keep their accuracy."*
6. Point at the audit table and the relearning curve for the honest caveats.
   - MIA ≈ 0.5 already before unlearning.
   - Some trigger sensitivity remains.
