# 加载必要的包
library(ggplot2)
library(dplyr)
library(tidyr)

# 1. 创建输出目录
output_dir <- "ipynb/loc_camp"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 2. 数据配置项：TreeX读取CSV，其余从数组读取
file_list <- list(
  "TreeX" = "rgcnformer_loc.csv", 
#   "CNN Only" = c(0.85, 0.82, 0.88, 0.90, 0.86, 0.79, 0.87, 0.84, 0.83),
  "GNN Only" = c(0.6286, 0.7506, 0.4604, 0.7372, 0.7327, 0.5452, 0.4604, 0.7797, 0.6636,0.87, 0.84, 0.83)
)

# 3. 数据读取与预处理
df_treex <- read.csv(file_list[["TreeX"]])
class_names <- df_treex$Name

# 将模型的数据拼接到基础数据框中
df <- data.frame(
  Class = class_names,
  `TreeX` = df_treex$Top.1,  
  `GNN_Only` = file_list[["GNN Only"]]
)

# 计算每个模型的平均值(Avg)，并作为新的一行加入
avg_row <- data.frame(
  Class = "Avg",
  `TreeX` = mean(df$TreeX, na.rm = TRUE),
  `GNN_Only` = mean(df$GNN_Only, na.rm = TRUE)
)
df <- rbind(df, avg_row)

# 4. 数据格式转换：宽表转长表，适配 ggplot2 绘图
df_long <- df %>%
  pivot_longer(cols = c("TreeX", "GNN_Only"),
               names_to = "Model",
               values_to = "Recall") %>%
  mutate(Model = gsub("_", " ", Model)) # 把列名中的下划线还原为空格

# 调整 Class 顺序：取消反转，使 Avg 在最前面，其他从左向右排列
df_long$Class <- factor(df_long$Class, levels = c("Avg", class_names))

# 5. 莫兰迪配色配置
morandi_colors <- c(
  "TreeX"    = "#b99d8e",  # 莫兰迪·灰褐 (突出自己的模型)
  "CNN Only" = "#94a7ae",  # 莫兰迪·灰蓝
  "GNN Only" = "#a6bba1"   # 莫兰迪·豆绿
)

# 6. 绘制多柱竖直棒棒糖图
p <- ggplot(df_long, aes(x = Class, y = Recall, color = Model, group = Model)) +
  # 【修改核心】: 使用 geom_linerange 完美替代 geom_segment，解决错位问题
  geom_linerange(aes(ymin = 0, ymax = Recall), 
                 position = position_dodge(width = 0.7), size = 0.8) +
  # 绘制棒棒糖的“糖果”
  geom_point(position = position_dodge(width = 0.7), size = 3.5) +
  # 应用莫兰迪配色
  scale_color_manual(values = morandi_colors) +
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