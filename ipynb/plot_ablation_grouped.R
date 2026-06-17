# ==============================================================================
# plot_ablation_grouped.R - 消融 AUC + Params 双带分组柱图 / Ablation AUC + Params dual-band grouped bars
# ==============================================================================
#
# 读取 data/ablation_results.csv，把 AUC 与 参数量 映射到同一 Y 轴的不同带
# (AUC 在 [0.77, 1.0]，Params 压缩到 [0.70, 0.77]) 形成上下双带分组柱状图，
# 用 4 种莫兰迪色区分 query_type。
# Reads data/ablation_results.csv and maps AUC and parameter count onto the same
# Y axis in two bands (AUC in [0.77, 1.0], Params squeezed into [0.70, 0.77]) to
# form a dual-band grouped bar chart; uses 4 Morandi colors for query_type.
#
# 功能模块 / Modules:
# - scale_p / unscale_p: Params -> AUC 坐标映射 / Params->AUC band mapping
# - 双 geom_col: AUC 上带 + Params 下带 / Two geom_col bands
# - 二次 Y 轴 (right): 真实 Params 数值 / Right axis with real Params ticks
#
# 输入 / Inputs:
# - data/ablation_results.csv: 消融结果 CSV / Ablation results
#
# 输出 / Outputs:
# - output/ablation_grouped.png / .pdf: 双带柱图 / Dual-band bar chart
#
# 数据流 / Data Flow:
# 1. 读 CSV / Load CSV
# 2. 构造 scale_p 映射 / Build scale_p mapper
# 3. 画双 geom_col + 二次 Y 轴 / Plot two geom_cols + secondary axis
# 4. ggsave / Save
#
# 相关文件 / Related Files:
# - 调用 / Calls: ggplot2, dplyr, tidyr, scales, extrafont
# - 被调用 / Called by: 消融实验流水线 / Ablation experiment pipeline
#
# 使用示例 / Usage Example:
#     Rscript ipynb/plot_ablation_grouped.R
#
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(extrafont)

# Register Times New Roman (one-time import already done)
font_import(paths = "/usr/share/fonts/truetype/msttcorefonts/", prompt = FALSE)
loadfonts(device = "pdf")

# ── Read data ──────────────────────────────────────────────────────────
df <- read.csv("data/ablation_results.csv", stringsAsFactors = FALSE)

df$group_query_dim <- factor(df$group_query_dim, levels = c(128, 256, 512, 1001))
df$query_type <- factor(df$query_type, levels = c("1query", "4query", "12query", "fullattn"))

# ── Morandi palette (4 muted tones) ───────────────────────────────────
morandi_colors <- c(
  "1query"   = "#A89B8C",  # warm taupe
  "4query"   = "#8FA3A8",  # muted teal
  "12query"  = "#C2956B",  # soft terracotta
  "fullattn" = "#9B8EA8"   # dusty lavender
)

# ── Axis scaling: map params onto AUC scale for dual-axis display ─────
param_lo <- min(df$total_params)     # 4608
param_hi <- max(df$total_params)     # 1792080
bottom_band <- c(0.70, 0.77)         # params occupy lower band

scale_p   <- function(p) bottom_band[1] + (p - param_lo) / (param_hi - param_lo) * diff(bottom_band)
unscale_p <- function(y) param_lo + (y - bottom_band[1]) / diff(bottom_band) * (param_hi - param_lo)

df$auc_y <- df$best_avg_auc
df$par_y <- scale_p(df$total_params)

# Right-axis tick positions (mapped onto the shared y scale)
par_ticks_raw <- c(0, 5e5, 1e6, 1.5e6, param_hi)
par_ticks_lab <- c("0", "0.5M", "1.0M", "1.5M", "1.8M")
par_ticks_y   <- scale_p(par_ticks_raw)

# ── Dodge position ─────────────────────────────────────────────────────
pd <- position_dodge(width = 0.75)

# ── Build plot ─────────────────────────────────────────────────────────
p <- ggplot(df, aes(x = group_query_dim, group = query_type)) +

  # ── Upper panel: AUC bars ────────────────────────────────────────────
  geom_col(aes(y = auc_y, fill = query_type), position = pd, width = 0.65) +

  # AUC value labels above bars
  geom_text(
    aes(y = auc_y, label = sprintf("%.3f", best_avg_auc)),
    position = pd, vjust = -0.4, size = 3.8,
    family = "Times New Roman"
  ) +

  # ── Lower panel: Params bars (mapped into [0.70, 0.77]) ─────────────
  geom_col(aes(y = par_y, fill = query_type), position = pd, width = 0.65) +

  # Params labels inside bars
  geom_text(
    aes(y = par_y, label = ifelse(
      total_params < 1e4,
      sprintf("%.1fK", total_params / 1e3),
      sprintf("%.2fM", total_params / 1e6)
    )),
    position = pd, vjust = 1.5, size = 2.6,
    family = "Times New Roman", colour = "white"
  ) +

  # ── Separator between the two bands ─────────────────────────────────
  geom_hline(yintercept = 0.775, linetype = "longdash", colour = "grey55", linewidth = 0.45) +

  # ── Band labels ─────────────────────────────────────────────────────
  annotate("text", x = 0.55, y = 0.89, label = "AUC Performance",
           size = 5.2, fontface = "italic", family = "Times New Roman",
           colour = "grey30", hjust = 0) +
  annotate("text", x = 0.55, y = 0.735, label = "Parameters",
           size = 5.2, fontface = "italic", family = "Times New Roman",
           colour = "grey30", hjust = 0) +

  # ── Scales ──────────────────────────────────────────────────────────
  scale_fill_manual(
    values = morandi_colors,
    name = "Query Type",
    labels = c("1 Query", "4 Query", "12 Query", "Full Attention")
  ) +

  scale_y_continuous(
    name = "Best Average AUC",
    breaks = seq(0.70, 1.00, by = 0.05),
    labels = sprintf("%.2f", seq(0.70, 1.00, by = 0.05)),
    sec.axis = sec_axis(
      transform = ~ unscale_p(.),
      name = "Total Parameters",
      breaks = par_ticks_y,
      labels = par_ticks_lab
    )
  ) +

  scale_x_discrete(
    name = "Query Dimension (group_query_dim)",
    labels = c("128 dim", "256 dim", "512 dim", "1001 dim")
  ) +

  # ── Clip to desired y range ─────────────────────────────────────────
  coord_cartesian(ylim = c(0.70, 1.005)) +

  # ── Theme ───────────────────────────────────────────────────────────
  theme_bw(base_size = 18, base_family = "Times New Roman") +
  theme(
    legend.position    = "top",
    legend.title       = element_text(size = 14, face = "bold"),
    legend.text        = element_text(size = 13),
    legend.key.size    = unit(0.8, "cm"),
    legend.margin      = margin(t = 0, b = 5),

    axis.title.y.left  = element_text(size = 16, margin = margin(r = 8)),
    axis.title.y.right = element_text(size = 16, margin = margin(l = 8)),
    axis.title.x       = element_text(size = 16, margin = margin(t = 8)),
    axis.text           = element_text(size = 14),
    axis.ticks          = element_line(linewidth = 0.4),

    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(linewidth = 0.6),
    plot.margin        = margin(t = 10, r = 15, b = 10, l = 10)
  )

# ── Save outputs ──────────────────────────────────────────────────────
ggsave("png4/ablation_grouped_bar.pdf", p, width = 10, height = 7, dpi = 300,
       device = cairo_pdf)
ggsave("png4/ablation_grouped_bar.png", p, width = 10, height = 7, dpi = 300,
       device = "png", type = "cairo")

cat("Done: png4/ablation_grouped_bar.pdf and .png saved.\n")
