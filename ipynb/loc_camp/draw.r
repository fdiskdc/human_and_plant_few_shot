# 加载必要的包
library(ggplot2)
library(dplyr)
library(tidyr)

# 1. 创建输出目录
output_dir <- "ipynb/loc_camp"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 2. 数据配置项：TreeX读取CSV，其余从数组读取
file_list <- list(
  "Attention Pooling" = "rgcnformer_loc.csv", 
#   "CNN Only" = c(0.85, 0.82, 0.88, 0.90, 0.86, 0.79, 0.87, 0.84, 0.83),
  "Avg Pooling" = c(0.3286, 0.2309, 0.4976, 0.3222, 0.1268, 0.2342, 0.5155, 0.3918, 0.2343,0.2898, 0.1609, 0.3769),
  "Max Pooling" = c(0.4079, 0.1912, 0.4567, 0.3994, 0.1202, 0.2787, 0.4293, 0.3864, 0.1569, 0.2799, 0.1715, 0.3051)
)

# 3. 数据读取与预处理
df_treex <- read.csv(file_list[["Attention Pooling"]])
class_names <- df_treex$Name

# 将模型的数据拼接到基础数据框中
df <- data.frame(
  Class = class_names,
  `Attention_Pooling` = df_treex$Top.1,
  `Avg_Pooling` = file_list[["Avg Pooling"]],
  `Max_Pooling` = file_list[["Max Pooling"]]
)

# 计算每个模型的平均值(Avg)，并作为新的一行加入
avg_row <- data.frame(
  Class = "Avg",
  `Attention_Pooling` = mean(df$Attention_Pooling, na.rm = TRUE),
  `Avg_Pooling` = mean(df$Avg_Pooling, na.rm = TRUE),
  `Max_Pooling` = mean(df$Max_Pooling, na.rm = TRUE)
)
df <- rbind(df, avg_row)

# 4. 数据格式转换：宽表转长表，适配 ggplot2 绘图
df_long <- df %>%
  pivot_longer(cols = c("Attention_Pooling", "Avg_Pooling", "Max_Pooling"),
               names_to = "Model",
               values_to = "Recall") %>%
  mutate(Model = gsub("_", " ", Model)) # 把列名中的下划线还原为空格

# 调整 Class 顺序：取消反转，使 Avg 在最前面，其他从左向右排列
df_long$Class <- factor(df_long$Class, levels = c("Avg", class_names))

# 5. 莫兰迪配色配置
morandi_colors <- c(
  "Attention Pooling"       = "#b99d8e",  # 莫兰迪·灰褐 (突出自己的模型)
  "Avg Pooling" = "#94a7ae",  # 莫兰迪·灰蓝
  "Max Pooling" = "#a6bba1"   # 莫兰迪·豆绿
)

# 6. 绘制多柱竖直棒棒糖图
p <- ggplot(df_long, aes(x = Class, y = Recall, color = Model, group = Model)) +
  # 【修改核心】: 使用 geom_linerange 完美替代 geom_segment，解决错位问题
  geom_linerange(aes(ymin = 0, ymax = Recall), 
                 position = position_dodge(width = 0.7), size = 0.8) +
  # 绘制棒棒糖的“糖果”
  geom_point(aes(shape = Class), position = position_dodge(width = 0.7), size = 3.5) +
  # 应用莫兰迪配色
  scale_color_manual(values = morandi_colors) +
  scale_shape_manual(values = c(
    "Avg" = 16, "Am" = 15, "Atol" = 17, "Cm" = 18,
    "Gm" = 19, "Tm" = 20, "Y" = 21, "ac4C" = 22,
    "m1A" = 23, "m5C" = 24, "m6A" = 25, "m6Am" = 3, "m7G" = 4
  )) +
  guides(shape = "none") +
  # 调整主题与标签
  theme_minimal(base_size = 18) +
  labs(
    x = "修饰类别",           
    y = "top-1 recall",       
    color = "Model Settings"
  ) +
  theme(
    legend.position = "top",                  
    # 倾斜 X 轴文本，防止长类别名称重叠
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", size = 18),
    # 去除垂直网格线，保留水平网格线方便对齐 Y 轴数值
    panel.grid.major.x = element_blank(),     
    panel.grid.minor = element_blank()
  )

# 7. 保存文件
pdf_path <- file.path(output_dir, "lollipop_comp_chart.pdf")
png_path <- file.path(output_dir, "lollipop_comp_chart.png")

ggsave(pdf_path, plot = p, width = 10, height = 6, device = "pdf")
ggsave(png_path, plot = p, width = 10, height = 6, dpi = 300, device = "png")

cat("绘制成功！棍子与端点已完美对齐，图像已保存至:", output_dir, "\n")