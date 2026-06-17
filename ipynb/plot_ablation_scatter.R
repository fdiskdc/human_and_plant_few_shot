# ==============================================================================
# plot_ablation_scatter.R - 消融实验 Params-FLOPs-AUC 散点 / Ablation Params-FLOPs-AUC scatter
# ==============================================================================
#
# 读取 data/ablation_results.csv，将 (query_type, group_query_dim) 组合画在
# log-log 总参数量-FLOPs 平面上，点的大小映射 best_avg_auc，颜色映射 query_type，
# 形状映射 group_query_dim；用 geom_text_repel 标注点。
# Reads data/ablation_results.csv and plots each (query_type, group_query_dim)
# configuration on a log-log total-params vs FLOPs plane; point size encodes
# best_avg_auc, color encodes query_type, shape encodes group_query_dim; labels
# use geom_text_repel to avoid overlap.
#
# 功能模块 / Modules:
# - geom_point (color=query_type, shape=dim, size=AUC): 三重编码散点 / Triple-encoded scatter
# - geom_text_repel: 名称标注 / Label repel
# - scale_x_log10 / scale_y_log10: 双对数坐标 / Log-log axes
#
# 输入 / Inputs:
# - data/ablation_results.csv: 消融实验 (params, flops, auc, query_type, dim)
#
# 输出 / Outputs:
# - output/ablation_scatter.png 与 .pdf: 散点图 / Scatter plot
#
# 数据流 / Data Flow:
# 1. 读 ablation_results.csv / Load CSV
# 2. factor 化 query_type / group_query_dim / Factorize
# 3. ggplot + 3 重编码 + repel / Triple-encode + repel
# 4. ggsave / Save
#
# 相关文件 / Related Files:
# - 调用 / Calls: ggplot2, dplyr, scales, ggrepel
# - 被调用 / Called by: 消融实验流水线 / Ablation experiment pipeline
#
# 使用示例 / Usage Example:
#     Rscript ipynb/plot_ablation_scatter.R
#
# ==============================================================================

library(ggplot2)
library(dplyr)
library(scales)
library(ggrepel)

# ── Read data ────────────────────────────────────────────────────────────
df <- read.csv("data/ablation_results.csv", stringsAsFactors = FALSE)

df$group_query_dim <- factor(df$group_query_dim, levels = c(128, 256, 512, 1001))
df$query_type <- factor(df$query_type, levels = c("1query", "4query", "12query", "fullattn"))

# ── Morandi palette ──────────────────────────────────────────────────────
morandi_colors <- c(
  "1query"   = "#B5A99D",
  "4query"   = "#8FA3A8",
  "12query"  = "#C2956B",
  "fullattn" = "#9B8EA8"
)

# ── Shape mapping for dimension ──────────────────────────────────────────
dim_shapes <- c("128" = 16, "256" = 17, "512" = 15, "1001" = 18)

# ── Build plot ───────────────────────────────────────────────────────────
p <- ggplot(df, aes(x = total_params, y = flops)) +

  # bubbles: size = AUC
  geom_point(
    aes(colour = query_type, shape = group_query_dim, size = best_avg_auc),
    alpha = 0.88, stroke = 0.4
  ) +

  # labels with repulsion
  geom_text_repel(
    aes(label = name),
    size = 4.2, family = "Times New Roman",
    colour = "grey30", fontface = "italic",
    segment.colour = "grey75", segment.size = 0.25,
    max.overlaps = 20, box.padding = 0.6, point.padding = 0.3
  ) +

  # Scales
  scale_colour_manual(
    values = morandi_colors,
    name  = "Query Type",
    labels = c("1 Query", "4 Query", "12 Query", "Full Attention")
  ) +
  scale_shape_manual(
    values = dim_shapes,
    name   = "Hidden Dim",
    labels = c("128", "256", "512", "1001")
  ) +
  scale_size_continuous(
    name   = "Best Avg AUC",
    range  = c(2.5, 8),
    breaks = c(0.75, 0.80, 0.85, 0.90, 0.95),
    labels = function(x) sprintf("%.2f", x)
  ) +
  scale_x_log10(
    name   = "Total Parameters",
    breaks = 10^c(3, 4, 5, 6, 7, 8),
    labels = c("1K", "10K", "100K", "1M", "10M", "100M")
  ) +
  scale_y_log10(
    name   = "FLOPs",
    breaks = 10^c(6, 7, 8, 9, 10, 11, 12),
    labels = c("1M", "10M", "100M", "1G", "10G", "100G", "1T")
  ) +

  # Theme
  theme_bw(base_size = 18, base_family = "Times New Roman") +
  theme(
    legend.position      = "right",
    legend.title         = element_text(size = 15, face = "bold"),
    legend.text          = element_text(size = 14),
    legend.key.size      = unit(0.65, "cm"),
    legend.spacing.y     = unit(0.25, "cm"),
    legend.margin        = margin(l = 6, r = 6),

    axis.title           = element_text(size = 18),
    axis.title.y         = element_text(margin = margin(r = 8)),
    axis.title.x         = element_text(margin = margin(t = 8)),
    axis.text            = element_text(size = 16, colour = "grey20"),

    panel.grid.major     = element_line(colour = "grey92", linewidth = 0.3),
    panel.grid.minor     = element_blank(),
    panel.border         = element_rect(linewidth = 0.6, colour = "grey30"),
    plot.margin          = margin(t = 12, r = 15, b = 12, l = 12)
  ) +
  guides(
    colour = guide_legend(order = 1, override.aes = list(size = 4.5)),
    shape  = guide_legend(order = 2, override.aes = list(size = 4.5)),
    size   = guide_legend(order = 3)
  )

# ── Save ──────────────────────────────────────────────────────────────────
ggsave("png4/ablation_scatter.pdf", p,
       width = 11, height = 7.5, dpi = 300, device = cairo_pdf)
ggsave("png4/ablation_scatter.png", p,
       width = 11, height = 7.5, dpi = 300, device = "png", type = "cairo")

cat("Done: png4/ablation_scatter.pdf and .png saved.\n")
