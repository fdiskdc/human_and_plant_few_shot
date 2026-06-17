# ==============================================================================
# generate_plot.R - 参数量 vs AUC 气泡图 / Parameter-count vs AUC bubble plot
# ==============================================================================
#
# 读取 para_comp/ 下三份 Excel（51.xlsx, 101.xlsx, 1001.xlsx），把不同序列长度
# 下的 (Model, PARA, AUC) 合并，用颜色编码 Sequence_Length、用形状编码 Model
# 画 莫兰迪色 气泡图（log X 轴）。
# Reads the three Excel files (51.xlsx, 101.xlsx, 1001.xlsx) under para_comp/,
# merges (Model, PARA, AUC) across sequence lengths, and plots a Morandi-colored
# bubble chart with color = sequence length and shape = model (log-x axis).
#
# 功能模块 / Modules:
# - read_excel x3: 读 51/101/1001 序列长度数据 / Load three sequence-length tables
# - bind_rows: 合并三份数据 / Concatenate
# - ggplot + geom_point: 气泡图 / Bubble chart
#
# 输入 / Inputs:
# - 51.xlsx, 101.xlsx, 1001.xlsx: 三组序列长度下的模型参数与 AUC / Per-length tables
#
# 输出 / Outputs:
# - bubble_chart_R.png / .pdf: 气泡图 / Bubble chart
#
# 数据流 / Data Flow:
# 1. 读三份 xlsx / Read 3 xlsx
# 2. mutate Sequence_Length / bind_rows / Add length & merge
# 3. ggplot 画气泡 / Plot bubble
# 4. ggsave / Save
#
# 相关文件 / Related Files:
# - 调用 / Calls: ggplot2, readxl, dplyr, scales
# - 被调用 / Called by: 参数量对比流水线 / Parameter-comparison pipeline
#
# 使用示例 / Usage Example:
#     Rscript ipynb/para_comp/generate_plot.R
#
# ==============================================================================

#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ggplot2)
  library(readxl)
  library(dplyr)
  library(scales)
})

# Morandi color palette for sequence length
MORANDI_COLORS <- c(
  "51" = "#94a7ae",   # 灰蓝 (gray blue)
  "101" = "#b99d8e",  # 灰褐 (gray brown)
  "1001" = "#a6bba1"  # 豆绿 (bean green)
)

# Shape palette for models
MODEL_SHAPES <- c(
  "Hierarchy Attention" = 16,  # circle
  "LSTM" = 15,                  # square
  "CNN" = 17,                   # triangle
  "Self-Attention" = 18         # diamond
)

# Read the three Excel files
df51 <- read_excel("51.xlsx", sheet = "Sheet1")
df101 <- read_excel("101.xlsx", sheet = "Sheet1")
df1001 <- read_excel("1001.xlsx", sheet = "Sheet1")

# Rename first column and add sequence length
df51 <- df51 %>%
  rename(Model = 1) %>%
  mutate(Sequence_Length = factor(51))

df101 <- df101 %>%
  rename(Model = 1) %>%
  mutate(Sequence_Length = factor(101))

df1001 <- df1001 %>%
  rename(Model = 1) %>%
  mutate(Sequence_Length = factor(1001))

# Merge all data
df <- bind_rows(df51, df101, df1001)

# Ensure numeric types
df$PARA <- as.numeric(df$PARA)
df$AUC <- as.numeric(df$AUC)

# Ensure Model is a factor
df$Model <- factor(df$Model, levels = c("Hierarchy Attention", "LSTM", "CNN", "Self-Attention"))

print("Merged data:")
print(df)

# Create the bubble chart
# Use shape for models, color/fill for sequence length
p <- ggplot(df, aes(x = PARA, y = AUC, color = Sequence_Length, shape = Model)) +
  geom_point(size = 6, alpha = 0.85, stroke = 1) +
  scale_x_log10(
    breaks = c(1.7e6, 2.0e6, 2.3e6),
    labels = scales::comma_format()
  ) +
  scale_color_manual(values = MORANDI_COLORS, name = "Sequence Length") +
  scale_shape_manual(values = MODEL_SHAPES, name = "Model") +
  labs(
    title = "Model Performance vs Parameter Count",
    subtitle = "Across Different Sequence Lengths",
    x = "Parameters (log scale)",
    y = "AUC (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray40"),
    axis.title = element_text(face = "bold", size = 13),
    axis.text = element_text(color = "gray30"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.5, color = "gray85"),
    panel.background = element_rect(fill = "#fafafa", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10),
    legend.background = element_rect(fill = "white", color = "#cccccc", linewidth = 0.5),
    legend.box = "horizontal"
  ) +
  coord_cartesian(ylim = c(60, 100)) +
  guides(
    color = guide_legend(override.aes = list(size = 4)),
    shape = guide_legend(override.aes = list(size = 4))
  )

# Save as high-resolution PNG
ggsave(
  "bubble_chart_R.png",
  plot = p,
  dpi = 300,
  width = 10,
  height = 7,
  bg = "white"
)

# Save as PDF
ggsave(
  "bubble_chart_R.pdf",
  plot = p,
  width = 10,
  height = 7,
  bg = "white"
)

message("Bubble chart saved as 'bubble_chart_R.png' and 'bubble_chart_R.pdf'")
