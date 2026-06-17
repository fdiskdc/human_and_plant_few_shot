# ==============================================================================
# auc_stability_comparison.R - AUC 随序列长度的稳定性 dumbbell 图 / AUC stability dumbbell across lengths
# ==============================================================================
#
# 对 TreeX 与 MultiRM 在序列长度 51 / 101 / 1001 上的 AUC 做"哑铃图 (dumbbell plot)"：
# 每个长度一对点+一条连接线，凸显两者差距随序列长度的变化（MultiRM 退化）。
# Draws a dumbbell plot of TreeX vs MultiRM AUC at sequence lengths 51 / 101 / 1001,
# with a connecting line per length, highlighting that MultiRM degrades rapidly
# while TreeX remains stable.
#
# 功能模块 / Modules:
# - data.frame: 内嵌硬编码 6 个 AUC 值 / Inline 6 AUC values
# - geom_line + geom_point: 哑铃图 / Dumbbell plot
# - scale_color_manual: 莫兰迪双色 (冷灰蓝 vs 暖粉陶) / Two Morandi colors
#
# 输入 / Inputs:
# - (无外部输入) / No external inputs (data is inline)
#
# 输出 / Outputs:
# - ipynb/len_comp/auc_stability_comparison.png (+ .pdf): 哑铃图 / Dumbbell plot
#
# 数据流 / Data Flow:
# 1. 构造 data.frame (Length, Model, AUC) / Build data.frame
# 2. factor 化 Length / Factor Length
# 3. ggplot 哑铃图 / Plot dumbbell
# 4. ggsave / Save
#
# 相关文件 / Related Files:
# - 调用 / Calls: ggplot2, dplyr, scales
# - 被调用 / Called by: 序列长度对比流水线 / Length-comparison pipeline
#
# 使用示例 / Usage Example:
#     Rscript ipynb/len_comp/auc_stability_comparison.R
#
# ==============================================================================

library(ggplot2)
library(dplyr)
library(scales)

# Create output directory if not exists
output_dir <- "ipynb/len_comp/"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Create data
data <- data.frame(
  Length = c(51, 51, 101, 101, 1001, 1001),
  Model = rep(c("TreeX", "MultiRM"), 3),
  AUC = c(93.94, 84.53, 93.98, 76.22, 96.07, 57.33)
)

# Convert Length to ordered factor
data$Length <- factor(data$Length, levels = c(51, 101, 1001))

# Morandi palette colors
treex_color <- "#6D8299"   # 冷灰蓝
multirm_color <- "#C68B77" # 暖粉陶

# Create dumbbell plot
p <- ggplot(data, aes(x = Length, y = AUC, color = Model, group = Length)) +
  # Connecting lines (dumbbell effect)
  geom_line(aes(group = Length), color = "gray70", size = 0.8, linetype = "solid") +
  # Points
  geom_point(aes(color = Model), size = 4, shape = 16) +
  # Morandi color scale
  scale_color_manual(
    values = c("TreeX" = treex_color, "MultiRM" = multirm_color),
    labels = c("TreeX", "MultiRM")
  ) +
  # Y-axis scale focusing on the relevant range
  scale_y_continuous(
    limits = c(50, 100),
    expand = expansion(mult = c(0, 0))
  ) +
  # Minimal theme
  theme_minimal(base_size = 14) +
  theme(
    # Remove extra grid lines
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.ontop = FALSE,
    # Axis styling
    axis.line.x = element_line(color = "gray50", linewidth = 0.5),
    axis.line.y = element_line(color = "gray50", linewidth = 0.5),
    axis.ticks = element_line(color = "gray50"),
    # Legend
    legend.position = "top",
    legend.title = element_blank(),
    legend.background = element_rect(fill = NA, color = NA),
    # Plot margins
    plot.margin = margin(20, 20, 20, 20)
  ) +
  # Labels
  labs(
    x = "Sequence Length",
    y = "AUC (%)"
  )

# Save as PNG (300 dpi)
ggsave(
  filename = file.path(output_dir, "auc_stability_comparison.png"),
  plot = p,
  width = 6,
  height = 5,
  dpi = 300,
  bg = "white"
)

# Save as PDF (vector format)
ggsave(
  filename = file.path(output_dir, "auc_stability_comparison.pdf"),
  plot = p,
  width = 6,
  height = 5,
  device = "pdf"
)

print("Plot saved to:")
print(file.path(output_dir, "auc_stability_comparison.png"))
print(file.path(output_dir, "auc_stability_comparison.pdf"))
