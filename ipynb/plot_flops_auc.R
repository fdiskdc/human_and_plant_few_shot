# ==============================================================================
# plot_flops_auc.R - FLOPs vs AUC 散点/柱状图 / FLOPs vs AUC scatter/bar plot
# ==============================================================================
#
# 读取 data/flops_auc.csv（包含 model, auc, auc_err, flops, flops_err），对各基线
# （RNAFold / Random / Sequential / Fully Connected / Empty）做 AUC 柱状图和
# FLOPs 散点图，使用莫兰迪配色与 Times New Roman 字体。
# Reads data/flops_auc.csv (model, auc, auc_err, flops, flops_err) and produces
# AUC bar charts and FLOPs scatter plots for each baseline (RNAFold / Random /
# Sequential / Fully Connected / Empty) using a Morandi palette and Times New Roman.
#
# 功能模块 / Modules:
# - read.csv: 读取 flops_auc.csv / Load flops_auc.csv
# - chart1: AUC 柱状图（含误差棒）/ AUC bar chart with error bars
# - chart2: FLOPs 散点 / FLOPs scatter
#
# 输入 / Inputs:
# - data/flops_auc.csv: 模型 AUC + FLOPs 数据 / Per-model AUC and FLOPs
#
# 输出 / Outputs:
# - output/flops_auc_*.png 与 *.pdf: AUC / FLOPs 图表 / AUC / FLOPs figures
#
# 数据流 / Data Flow:
# 1. 读 flops_auc.csv / Load CSV
# 2. 翻译模型名为英文 / Translate model names to English
# 3. 构造 AUC 柱状图 + FLOPs 散点 / Build AUC + FLOPs plots
# 4. ggsave 写 PNG/PDF / Save PNG/PDF
#
# 相关文件 / Related Files:
# - 调用 / Calls: ggplot2, dplyr, showtext
# - 被调用 / Called by: FLOPs 对比实验流水线 / FLOPs comparison experiment pipeline
#
# 使用示例 / Usage Example:
#     Rscript ipynb/plot_flops_auc.R
#
# ==============================================================================

library(ggplot2)
library(dplyr)
library(showtext)

showtext_auto()

# ── Read data ────────────────────────────────────────────────────────────
df <- read.csv("data/flops_auc.csv", stringsAsFactors = FALSE)

colnames(df) <- c("model", "auc", "auc_err", "flops", "flops_err")

# ── Translate model names to English ──────────────────────────────────────
df$model <- factor(df$model, levels = df$model)
df$model_en <- factor(
  case_when(
    df$model == "RNAFold"           ~ "RNAFold",
    df$model == "随机生成"           ~ "Random",
    df$model == "顺序边"             ~ "Sequential",
    df$model == "全连接"             ~ "Fully Connected",
    df$model == "空"                 ~ "Empty",
    TRUE                             ~ df$model
  ),
  levels = c("RNAFold", "Random", "Sequential", "Fully Connected", "Empty")
)

df$flops_gflops <- df$flops / 1e9
df$flops_err_gflops <- df$flops_err / 1e9

# ── Shared palette & labels ──────────────────────────────────────────────
morandi_colors <- c(
  "RNAFold"          = "#A89B8C",
  "Random"           = "#8FA3A8",
  "Sequential"       = "#C2956B",
  "Fully Connected"  = "#9B8EA8",
  "Empty"            = "#A8B5A0"
)

pd <- position_dodge(width = 0.75)

font_family <- "Times New Roman"
base_sz <- 18

# ══════════════════════════════════════════════════════════════════════════
# Chart 1: AUC (×100, percentage style)
# ══════════════════════════════════════════════════════════════════════════
p_auc <- ggplot(df, aes(x = model_en, y = auc * 100, fill = model_en)) +
  geom_col(position = pd, width = 0.65, color = "white", linewidth = 0.3) +
  geom_errorbar(
    aes(ymin = (auc - auc_err) * 100, ymax = (auc + auc_err) * 100),
    position = pd,
    width = 0.2,
    linewidth = 0.5
  ) +
  geom_text(
    aes(label = sprintf("%.2f", auc * 100)),
    position = pd,
    vjust = -1.2,
    size = 5.5,
    family = font_family
  ) +
  scale_fill_manual(values = morandi_colors, guide = "none") +
  scale_y_continuous(
    limits = c(0, 105),
    breaks = seq(65, 100, by = 5),
    expand = c(0, 0)
  ) +
  coord_cartesian(ylim = c(65, 105)) +
  labs(x = NULL, y = "AUC (%)") +
  theme_bw(base_size = base_sz, base_family = font_family) +
  theme(
    legend.position      = "none",
    axis.title           = element_text(size = 18),
    axis.text.x          = element_text(size = 18, angle = 0, hjust = 0.5, vjust = 0.5),
    axis.text.y          = element_text(size = 18),
    panel.grid.major.x   = element_blank(),
    panel.grid.minor     = element_blank(),
    plot.margin          = margin(10, 15, 10, 10)
  )

# ══════════════════════════════════════════════════════════════════════════
# Chart 2: FLOPs (Giga FLOPs)
# ══════════════════════════════════════════════════════════════════════════
p_flops <- ggplot(df, aes(x = model_en, y = flops_gflops, fill = model_en)) +
  geom_col(position = pd, width = 0.65, color = "white", linewidth = 0.3) +
  geom_errorbar(
    aes(ymin = flops_gflops - flops_err_gflops, ymax = flops_gflops + flops_err_gflops),
    position = pd,
    width = 0.2,
    linewidth = 0.5
  ) +
  geom_text(
    aes(label = sprintf("%.2f", flops_gflops)),
    position = pd,
    vjust = -1.2,
    size = 5.5,
    family = font_family
  ) +
  scale_fill_manual(values = morandi_colors, guide = "none") +
  scale_y_continuous(
    limits = c(0, 26),
    breaks = seq(0, 25, by = 5),
    expand = c(0, 0)
  ) +
  coord_cartesian(ylim = c(0, 26)) +
  labs(x = NULL, y = "FLOPs (Giga)") +
  theme_bw(base_size = base_sz, base_family = font_family) +
  theme(
    legend.position      = "none",
    axis.title           = element_text(size = 18),
    axis.text.x          = element_text(size = 18, angle = 0, hjust = 0.5, vjust = 0.5),
    axis.text.y          = element_text(size = 18),
    panel.grid.major.x   = element_blank(),
    panel.grid.minor     = element_blank(),
    plot.margin          = margin(10, 15, 10, 10)
  )

# ══════════════════════════════════════════════════════════════════════════
# Chart 2b: FLOPs flipped (y-axis reversed)
# ══════════════════════════════════════════════════════════════════════════
p_flops_flipped <- p_flops + scale_y_reverse()

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════

dir.create("png4", showWarnings = FALSE)

# PDF: disable showtext so text remains editable (not converted to paths)
showtext_auto(FALSE)
ggsave("png4/flops_auc_auc.pdf",            p_auc,          width = 10, height = 4, device = cairo_pdf)
ggsave("png4/flops_auc_flops.pdf",          p_flops,        width = 10, height = 5, device = cairo_pdf)
ggsave("png4/flops_auc_flops_flipped.pdf",  p_flops_flipped, width = 10, height = 5, device = cairo_pdf)

# PNG: re-enable showtext for bitmap rendering
showtext_auto(TRUE)
ggsave("png4/flops_auc_auc.png",            p_auc,          width = 10, height = 4, dpi = 300)
ggsave("png4/flops_auc_flops.png",          p_flops,        width = 10, height = 5, dpi = 300)
ggsave("png4/flops_auc_flops_flipped.png",  p_flops_flipped, width = 10, height = 5, dpi = 300)

cat("Done: AUC and FLOPs charts saved to png4/.\n")
