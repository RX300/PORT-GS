# LiSA 中文与日文译稿

翻译对象为 [英文源文件](../latex/) 中截至 2026-10-06 的当前英文修订稿：
**LiSA: Light-Space Atlases for Relightable Gaussian Splatting**，包括完整正文及补充材料。

| 语言 | 正文 PDF | 补充材料 PDF | 可编辑源文件 |
|---|---|---|---|
| 简体中文 | [正文](zh-CN/build/main.pdf) | [补充材料](zh-CN/build/supp.pdf) | [main.tex](zh-CN/main.tex)、[supp.tex](zh-CN/supp.tex) |
| 日本語 | [本文](ja/build/main.pdf) | [補足資料](ja/build/supp.pdf) | [main.tex](ja/main.tex)、[supp.tex](ja/supp.tex) |

正文、图注、表头、表内说明、可编辑流程图文字和 E1–E7 实验协议均已翻译。
公式、量化结果、场景和方法名称、引用键及原文的限定条件保持一致；参考文献保留原始发表语言。
红色待开展实验与待补结果标记也保留，分别显示为中文“待开展／待补”和日文“未実施／未定”。
译稿按各语言的自然长度排版，不压缩到英文投稿稿件的页数。

英文源文件、英文 PDF、生成表格脚本和实验结果未改动。翻译表格位于各语言自己的 tables/ 中，
不由原稿的 make_tables.py 覆盖。图像、参考文献和官方模板资源复用原稿；CJK 字体保存在
assets/fonts/，附带 SIL Open Font License，不修改系统字体或 Conda 环境。

## 重新编译

沿用项目已经安装的 TinyTeX，以 XeLaTeX 编译。每种语言先编译正文，再编译补充材料；
后者通过 build/main.aux 读取该语言正文的交叉引用。

~~~bash
export PATH=/workspace/ubuntu2004_cuda12_1/utils/tinytex/.TinyTeX/bin/x86_64-linux:$PATH
cd /workspace/ubuntu2004_cuda12_1/projects/3dgs_relighting/PORT-GS/paper/translations/zh-CN
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build supp.tex
~~~

日文版进入 ../ja 后执行相同两条编译命令。请保留相对于本目录的英文 ../../latex/ 资源和共享字体。
这是含多个源文件和图像的 LaTeX 项目。此次内置编辑器编译器返回环境目录不可用，
因此最终 PDF 使用现有 TinyTeX 成功生成，未安装或升级 TeX 环境。

## 已完成核对

- 对两种语言各 26 个内容源文件核对公式、引用、标签、图像路径、输入路径、章节和待办标记数量。
- 逐表核对图注中的量化信息及表内所有数值，未发现差异；见 [verification.json](verification.json)。
- 中文另作独立语义复核；日文统一润色学术用语与留出数据的表达。
- 四份 PDF 均成功编译，无缺失字符、未解析交叉引用或超出行宽的警告；已检查全部页面排版。
- 中文正文 11 页、补充材料 6 页；日文正文 13 页、补充材料 6 页（正文页数包含参考文献）。

后续英文稿有内容更新时，应同步修改两种译稿；本次译稿不是随英文自动生成的文件。
