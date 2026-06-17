# ==============================================================================
# plot_ablation_grouped_v2.R - 消融 AUC + Params 双图版 (v2) / Ablation AUC + Params two-chart v2
# ==============================================================================
#
# v2 版本：把 AUC 与 Params 分别画成两张独立的 ggplot 面板（共享 query_type 颜色），
# 便于论文中并排排版；相比 v1 不再共用 Y 轴。
# Version 2: draws AUC and Params as two independent ggplot panels that share the
# query_type color mapping, easier to lay out side-by-side in papers. Drops the
# shared Y axis used in v1.
#
# 功能模块 / Modules:
# - p_auc: AUC × 100 柱状图 (Times New Roman) / AUC bar chart
# - p_params: log10(Params) 柱状图 / log10 params bar chart
# - showtext: 字体加载 / Font loading
#
# 输入 / Inputs:
# - data/ablation_results.csv: 消融结果 / Ablation results
#
# 输出 / Outputs:
# - output/ablation_grouped_v2_*.png / .pdf: 两张并排面板 / Two side-by-side panels
#
# 数据流 / Data Flow:
# 1. 读 CSV / Load CSV
# 2. factor 化 / Factorize
# 3. 画 p_auc / p_params / Build p_auc / p_params
# 4. ggsave / Save
#
# 相关文件 / Related Files:
# - 调用 / Calls: ggplot2, dplyr, showtext
# - 被调用 / Called by: 消融实验流水线 / Ablation experiment pipeline
#
# 使用示例 / Usage Example:
#     Rscript ipynb/plot_ablation_grouped_v2.R
#
# ==============================================================================

library(ggplot2)
library(dplyr)
library(showtext)

showtext_auto()

# ── Read data ────────────────────────────────────────────────────────────
df <- read.csv("data/ablation_results.csv", stringsAsFactors = FALSE)

df$group_query_dim <- factor(df$group_query_dim, levels = c(128, 256, 512, 1001))
df$query_type      <- factor(df$query_type, levels = c("1query", "4query", "12query", "fullattn"))

# ── Shared palette & labels ──────────────────────────────────────────────
morandi_colors <- c(
  "1query"   = "#A89B8C",
  "4query"   = "#8FA3A8",
  "12query"  = "#C2956B",
  "fullattn" = "#9B8EA8"
)

query_labels <- c(
  "1query"   = "1 Query",
  "4query"   = "4 Query",
  "12query"  = "12 Query",
  "fullattn" = "Full Attn"
)

x_labels <- c("128" = "128 dim", "256" = "256 dim", "512" = "512 dim", "1001" = "1001 dim")

pd <- position_dodge(width = 0.75)

font_family <- "Times New Roman"
base_sz <- 18

# ══════════════════════════════════════════════════════════════════════════
# Chart 1: Best Average AUC (×100)
# ══════════════════════════════════════════════════════════════════════════
p_auc <- ggplot(df, aes(x = group_query_dim, y = best_avg_auc * 100, fill = query_type)) +
  geom_col(position = pd, width = 0.65, color = "white", linewidth = 0.3) +
  geom_text(
    aes(label = sprintf("%.2f", best_avg_auc * 100)),
    position = pd,
    vjust = -0.3,
    angle = 45,
    size = 6.3,
    family = font_family
  ) +
  scale_fill_manual(values = morandi_colors, labels = query_labels) +
  scale_y_continuous(
    limits = c(0, 110),
    breaks = seq(40, 100, by = 10),
    expand = c(0, 0)
  ) +
  scale_x_discrete(labels = x_labels) +
  coord_cartesian(ylim = c(40, 110)) +
  labs(x = "Query Dimension", y = "Best Average AUC", fill = "Query Type") +
  theme_bw(base_size = base_sz, base_family = font_family) +
  theme(
    legend.position      = "top",
    legend.title         = element_text(size = 18, face = "bold"),
    legend.text          = element_text(size = 18),
    legend.key.size      = unit(0.8, "cm"),
    axis.title           = element_text(size = 18),
    axis.text            = element_text(size = 18),
    panel.grid.major.x   = element_blank(),
    panel.grid.minor     = element_blank(),
    plot.margin          = margin(10, 15, 10, 10)
  )

# ══════════════════════════════════════════════════════════════════════════
# Chart 2: Total Parameters (log10 scale, upward bars)
# ══════════════════════════════════════════════════════════════════════════
param_label <- function(x) {
  ifelse(x >= 1e8, sprintf("%.0fM", x / 1e6),
  ifelse(x >= 1e6, sprintf("%.1fM", x / 1e6),
  ifelse(x >= 1e4, sprintf("%.0fK", x / 1e3),
         sprintf("%.1fK", x / 1e3))))
}

p_params <- ggplot(df, aes(x = group_query_dim, y = log10(total_params), fill = query_type)) +
  geom_col(position = pd, width = 0.65, color = "white", linewidth = 0.3) +
  geom_text(
    aes(label = param_label(total_params)),
    position = pd,
    vjust = -0.3,
    angle = 45,
    size = 6.3,
    family = font_family
  ) +
  scale_fill_manual(values = morandi_colors, labels = query_labels) +
  scale_y_continuous(
    limits = c(0, 9),
    breaks = 3:8,
    labels = c("1K", "10K", "100K", "1M", "10M", "100M"),
    expand = c(0, 0)
  ) +
  scale_x_discrete(labels = x_labels) +
  coord_cartesian(ylim = c(3, 8.8)) +
  labs(x = "Query Dimension", y = "Total Parameters", fill = "Query Type") +
  theme_bw(base_size = base_sz, base_family = font_family) +
  theme(
    legend.position      = "top",
    legend.title         = element_text(size = 18, face = "bold"),
    legend.text          = element_text(size = 18),
    legend.key.size      = unit(0.8, "cm"),
    axis.title           = element_text(size = 18),
    axis.text            = element_text(size = 18),
    panel.grid.major.x   = element_blank(),
    panel.grid.minor     = element_blank(),
    plot.margin          = margin(10, 15, 10, 10)
  )

# ══════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════

# PDF: disable showtext so text remains editable (not converted to paths)
showtext_auto(FALSE)
ggsave("png4/ablation_grouped_auc.pdf",    p_auc,    width = 10, height = 4, device = cairo_pdf)
ggsave("png4/ablation_grouped_params.pdf", p_params, width = 10, height = 4, device = cairo_pdf)

# PNG: re-enable showtext for bitmap rendering
showtext_auto(TRUE)
ggsave("png4/ablation_grouped_auc.png",    p_auc,    width = 10, height = 5, dpi = 300)
ggsave("png4/ablation_grouped_params.png", p_params, width = 10, height = 5, dpi = 300)

cat("Done: AUC and Params saved separately.\n")
