# 2026-09-26更正：区分NRHints新采集场景与复用的DNL Cat/Pixiu

此前以下审计准确描述了NRHints论文的新采集流程，但把该流程直接用于本地Cat/Pixiu是不充分的。
NRHints README将Cat与CatSmall（Cat on Decor）分列，原文§4说明复用了4个DNL场景。
DNL作者项目及原文§4展示并列出Cat/Pixiu，用双DSLR采集，采用gamma2.2近似线性化，并假设没有其他主要光源。
所以不能用NRHints新数据的Sony A7II+iPhone/S-log证明本地Cat/Pixiu存在S-log响应错误。
当前gamma2.2有相关来源依据；这不等于已经核验本地PNG的逐文件处理链或精确辐射校准。

[原DNL采集§4](https://gao-duan.github.io/publications/neuralrelighting/DeferredNeuralLighting_low_resolution.pdf) ·
[NRHints数据表](https://github.com/iamNCJ/NRHints#data-and-models) ·
[本轮诊断](experiments/quality_diagnosis_20260926.md)

---

# Historical audit: NRHints new-capture protocol

This is a primary-source audit of the **real capture/data protocol**. It does
not audit SSD-GS or reproduce the NRHints method. The paper PDF was saved at
[`docs/sources/nrhints-sig23.pdf`](sources/nrhints-sig23.pdf) for local page
checking. Sources examined:

- [NRHints paper PDF](https://nrhints.github.io/pdfs/nrhints-sig23.pdf), p. 5,
  §3.4 “Data Acquisition”, and p. 6, §4 “Results”.
- [Official NRHints repository README](https://raw.githubusercontent.com/iamNCJ/NRHints/main/README.md),
  data/training/testing notes.
- [Author version, §3.4–§3.5](https://arxiv.org/html/2308.13404v1), for the
  later camera-viewpoint optimization description.

## Facts supported by the sources

- The real setup uses a handheld **Sony A7II** as the primary camera and an
  **iPhone 13 Pro** as the secondary camera. The secondary camera captures the
  scene with its colocated flash enabled; those secondary-camera images are
  used to calibrate the light-source position. A checkerboard is placed under
  the scene to aid camera calibration. The paper says the acquisition takes
  about ten minutes, with video recorded by both cameras and 500–1,000 frames
  randomly selected as training data (paper p. 5, §3.4).
- The paper explicitly says: “The video is captured using S-log encoding to
  minimize overexposure.” (paper p. 5, §3.4). This establishes an encoding/
  capture choice, not a supplied inverse camera-response curve or radiometric
  calibration. The sources do not establish that released real RGB values are
  linear radiance, nor do they state exposure locking, white-balance handling,
  per-frame exposure metadata, or tone-curve inversion.
- The paper describes the illumination as a flash light colocated with the
  second camera and parameterizes the relighting basis with a point light. It
  gives no flash model, emitter size, angular emission profile, radiant power,
  spectral calibration, or numerical error bound showing that the real flash
  is an ideal point source (paper p. 4–5, §§3.1 and 3.4).
- The sources require varied viewpoints and light positions so that shadows
  and interreflections are observed. They do not say that room/environmental
  illumination was switched off, measured, or subtracted. The real protocol
  therefore does not prove a zero-ambient capture.
- The paper gives checkerboard-assisted camera calibration and light-position
  calibration from the secondary-camera images, but no numerical camera/light
  position accuracy, uncertainty, or independent calibration benchmark.

## Camera optimization and evaluation boundary

- The author version (§3.5) explicitly describes jointly optimizing the
  representation and small per-view camera corrections during training. It
  uses corrections of the form `R = ΔR R0` and `t = Δt + ΔR t0`, assumes the
  initial calibration error is small, and reduces the correction-transform
  learning rate. The official README later recommends enabling camera
  optimization for real scenes because it can reduce blur; the README presents
  this as a real-scene training option added after the paper submission.
- This is evidence for **training-view** pose optimization. Neither the paper,
  author version, nor README states that held-out evaluation/test poses were
  optimized against their ground-truth pixels. No such test-pose fitting should
  be inferred from the training camera-optimization description.
- For the synthetic benchmark, the paper states that PSNR/SSIM/LPIPS use 100
  test images different from 500 training images (paper p. 6, §4). For the
  captured scenes, the examined sources report the scenes and results but do
  not specify a complete real train/test frame-count or held-out-pose fitting
  protocol. That part remains unknown from these sources.
- The author version’s real-scene ablation reports viewpoint optimization with
  PSNR 34.72, SSIM 0.9762, LPIPS 0.0695 versus 33.62, 0.9719, 0.0794 without
  it (Table 2). These are training-protocol ablation results; they do not
  establish evaluation-pose calibration against held-out ground truth.

## Consequence for a real-scene direct-light model

The official real-capture description supports using estimated camera/light
geometry and treating a colocated flash as a point-light approximation. It
does **not** establish linear radiometry, absolute flash intensity, or zero
ambient illumination. Consequently, a hard direct-light factor proportional to
`1/r²` is an explicit model assumption for the real data, not a source-verified
physical contract. Unknown tone response/exposure and unknown ambient light can
also be absorbed by an uncalibrated global scale in practice, but that does not
make the result radiometrically calibrated or prove that the ambient term is
zero. Any real-scene claim should label this assumption and keep pose fitting
restricted to training views unless a separate evaluation protocol is
provided.

## Uncertainties and limits

“Not specified” above means absent from the examined paper, author-version
sections, and official README; it is not evidence that the quantity was absent
from the physical capture or from unpublished metadata. This audit did not read
the SSD method, did not use test images, and did not alter dataset files.
