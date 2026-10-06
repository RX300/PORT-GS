# Task: draw the method-overview figure (TikZ) for a CVPR paper

You are producing ONE figure for a CVPR 2027 submission about "LiSA: Light-Space Atlases for
Relightable Gaussian Splatting". Write files ONLY inside this directory:
`/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/paper/codex/pipeline/`.
Do not modify, move or delete anything else on disk. Do not run any GPU job or training.

## Deliverables
1. `pipeline.tex` – a bare `\begin{tikzpicture} ... \end{tikzpicture}` (no preamble, no figure
   environment) that the paper will `\input` inside `figure*` with `\resizebox{\textwidth}{!}{...}`.
   Image paths inside it must be relative to the paper root
   `/workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/paper/latex/`, e.g.
   `figures/decomp_hotdog/atlas_depth.png`.
2. `standalone.tex` – a `\documentclass[border=2pt]{standalone}` wrapper that loads `tikz`
   (+ needed libraries), `amsmath`, `amssymb`, `times`, sets
   `\graphicspath{{../../latex/}}` and `\input{pipeline}`, so it compiles from this directory.
3. `pipeline.pdf` and `pipeline.png` (200 dpi) compiled previews.
4. `NOTES.md` – 5 lines: what you drew and any assumptions.

Compile with: `export PATH=/workspace/ubuntu2004_cuda12_1/utils/tinytex/.TinyTeX/bin/x86_64-linux:$PATH; pdflatex -interaction=nonstopmode standalone.tex`
(TinyTeX is installed there; if a TikZ library is missing use `tlmgr install <pkg>`). For the PNG,
use `pdftoppm -png -r 200` if available, otherwise python3 with `fitz`/`pdf2image` if available, otherwise skip PNG.
Iterate until it compiles with no errors and looks clean (look at the PNG yourself).

## Visual style
- Final printed width = 6.875 in (CVPR two-column text width); target height 2.1–2.5 in.
  Design at that physical size: text 7–8 pt equivalent (`\footnotesize` / `\scriptsize`), sans-serif
  labels (`\sffamily`), math in normal math font. No text smaller than ~6.5 pt after scaling.
- Restrained academic palette: light pass = warm orange (e.g. `orange!70!black` lines,
  `orange!8` fills), camera pass = blue (`blue!60!black`, `blue!6`), learned networks = green
  rounded boxes (`green!45!black`, `green!8`), equations in plain black. Thin lines (0.6–0.8 pt),
  rounded corners 2 pt, arrows `-{Stealth[length=4pt]}`. No drop shadows, no gradients, no 3D effects.
- Thumbnails: square images, about 0.48–0.6 in on the printed page, thin gray border.

## Content (left → right, two lanes + a schedule strip)

**Top lane – light pass (orange).**
(a) Box "3D Gaussians" with a small math line `$\{\mu_i,\Sigma_i,o_i,\mathbf b_i,\mathbf f_i\}$`.
(b) Arrow labeled "rasterize from the light $\mathbf p$" into
(c) "Light-space atlas" showing two thumbnails side by side:
   `figures/decomp_hotdog/atlas_depth.png` (caption under it: "depth moments $M_1,M_2$") and
   `figures/decomp_hotdog/atlas_flux.png` (caption: "flux features $\Phi$"),
   with the defining equation under the pair:
   `$\mathbf A(u)=\sum_i w_i(u)\,[\boldsymbol\phi_i,\,z_i,\,z_i^2,\,1]$`.
   A tiny green network box "$\phi$-MLP" attached to (a)→(c) arrow, label
   `$\boldsymbol\phi_i=\mathrm{softplus}(\mathrm{MLP}(\mathbf f_i,\mathbf b_i,|\mathbf n_i\!\cdot\!\mathbf l_i|))$` may be omitted if crowded; keep at least the box name.
(d) Arrow "blur + $\downarrow 2$" into "Pyramid $\mathbf A_0,\dots,\mathbf A_{K-1}$": draw 3 stacked
   squares of decreasing size (use `figures/decomp_hotdog/atlas_flux.png` for the largest and
   `figures/decomp_hotdog/atlas_flux_level3.png` for the smaller ones), offset diagonally.

**Bottom lane – camera pass (blue).**
(e) Box "Receivers $\mathbf x$" with sub-label "Gaussian centers (training) / pixels (refinement)".
(f) Arrow "project into light: $(u(\mathbf x), z(\mathbf x))$" going UP into the pyramid (d), and a
   return arrow from (d) DOWN into
(g) "Gather" box listing per-level statistics in one compact line:
   `$\mathbf s_k=[M_{0,k},\,t_k,\,\delta_k,\,\log s_k,\,V_k]$, $\ \Phi_k$` and under it the moment test
   `$V_k=1-M_{0,k}\,\Phi_{\mathcal N}(t_k-\beta)$`.
(h) Three green network boxes stacked vertically to the right of (g):
   - "visibility $g$": `$V=\sigma(\mathrm{logit}\,V_0+g(\mathbf S,\mathbf f,\mathbf n\!\cdot\!\mathbf l))$`
   - "transfer $h$": `$L_{\mathrm{tr}}=\sum_k \mathbf W_k\,\Phi_k$`
   - "material": `$\rho,\ s=\mathrm{MLP}(\mathbf f,\mathbf n,\mathbf l,\mathbf v,\mathbf h,\mathbf r)$`
   Each with a small thumbnail to its right: `figures/decomp_hotdog/visibility.png`,
   `figures/decomp_hotdog/transfer.png`, `figures/decomp_hotdog/local.png`.
(i) A merge into the shading equation (boxed, black):
   `$L(\mathbf x)=E(\mathbf x)\big[\,V\rho+V_0\,s+\textstyle\sum_k\mathbf W_k\Phi_k\big]$, $\ E=I/\|\mathbf p-\mathbf x\|^2$`
   and an arrow to the final thumbnail `figures/decomp_hotdog/ours.png` labeled "render".

**Bottom strip – training schedule (full width, thin).** A horizontal timeline with three
proportional segments (8k | 22k | 8k iterations) labeled:
- "I. Geometry warm-up (0–8k): network-free point-light model; target exponent $1/\gamma\!\to\!1$"
- "II. Joint training (8k–30k): linear radiance, shade Gaussians, densify"
- "III. Appearance refinement (+8k): frozen geometry, shade pixels, display-domain loss"
Use three light fills (gray!10 / blue!8 / green!8) and small text.

Keep the whole thing uncluttered: if space is tight, shorten labels rather than shrinking fonts.
Prefer `positioning`, `fit`, `calc`, `arrows.meta`, `backgrounds` libraries only.
