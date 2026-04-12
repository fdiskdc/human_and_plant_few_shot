#!/usr/bin/env Rscript
# plot_zero_fewshot_analysis.R
# Generates publication-quality figures from Python-exported CSV data.
# This script handles both plant and 3-generation (gen3) data visualizations.
#
# Usage:
#   Rscript plot_zero_fewshot_analysis.R --input_dir <path> --output_dir <path>
#   conda run -n learn-new Rscript plot_zero_fewshot_analysis.R --input_dir <path> --output_dir <path>
#
# Dependencies: ggplot2, dplyr, tidyr, readr, scales, forcats, viridis

# Check and install ggrastr if not available
if (!requireNamespace("ggrastr", quietly = TRUE)) {
  message("Installing ggrastr package for rasterizing points in PDF...")
  tryCatch({
    install.packages("ggrastr", repos = "https://cloud.r-project.org")
  }, error = function(e) {
    message("Warning: Failed to install ggrastr: ", e$message)
    message("Will use standard plotting without rasterization")
  })
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(forcats)
  library(viridis)
  # Try to load ggrastr, but don't fail if not available
  has_ggrastr <- FALSE
  tryCatch({
    library(ggrastr)
    has_ggrastr <- TRUE
    message("ggrastr loaded successfully - points will be rasterized in PDF")
  }, error = function(e) {
    message("ggrastr not available - using standard plotting")
  })
})

# Helper function to optionally rasterize geom_point
rasterise_or_not <- function(geom_obj, dpi = 300) {
  if (exists("has_ggrastr") && has_ggrastr) {
    return(rasterise(geom_obj, dpi = dpi))
  } else {
    return(geom_obj)
  }
}

# ── Morandi palette (matches Python) ──────────────────────────────────────────
MORANDI_CLASS_COLORS <- c(
  "Y"   = "#B58A83",   # Dusty rose
  "m5C" = "#9AAA91",   # Sage green
  "m6A" = "#8EA3B0"    # Dusty blue
)
MORANDI_SPECIES_COLORS <- c(
  "Human" = "#8E8A84",  # Warm grey
  "Plant" = "#A69C87"   # Taupe
)
MORANDI_NEUTRAL   <- "#C7C0B7"  # Neutral beige
MORANDI_GRID      <- "#E7E0D8"  # Light grid
MORANDI_TEXT      <- "#6E675F"  # Text color
MORANDI_TITLE     <- "#4A4540"  # Title color
MORANDI_SPINE     <- "#D7CFC4"  # Spine/axis color
MORANDI_WARM_ACCENT <- "#C4A898"  # Warm accent (for gain+)
MORANDI_COOL_ACCENT <- "#A8B8C4"  # Cool accent (for gain-)
PLOT_BG           <- "#FBF8F3"  # Plot background
CLASS_ORDER       <- c("Y", "m5C", "m6A")
SHOT_COUNTS       <- c(0, 1, 5, 10)
SHOT_LABELS       <- c("0-shot", "1-shot", "5-shot", "10-shot")

# ── Theme ─────────────────────────────────────────────────────────────────────
theme_paper <- function(base_size = 11, base_family = "") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      plot.background    = element_rect(fill = PLOT_BG, color = NA),
      panel.background   = element_rect(fill = PLOT_BG, color = NA),
      panel.grid.major   = element_line(color = MORANDI_GRID, linewidth = 0.35),
      panel.grid.minor   = element_blank(),
      axis.ticks         = element_line(color = MORANDI_SPINE, linewidth = 0.3),
      axis.text          = element_text(color = MORANDI_TEXT, size = rel(0.9)),
      axis.title         = element_text(color = MORANDI_TEXT, face = "bold", size = rel(0.95)),
      legend.background  = element_rect(fill = PLOT_BG, color = NA),
      legend.key         = element_rect(fill = PLOT_BG, color = NA),
      legend.text        = element_text(color = MORANDI_TEXT, size = rel(0.85)),
      legend.title       = element_text(color = MORANDI_TITLE, face = "bold", size = rel(0.9)),
      plot.title         = element_text(face = "bold", hjust = 0.5, color = MORANDI_TITLE, size = rel(1.1)),
      plot.subtitle      = element_text(hjust = 0.5, color = MORANDI_TEXT, size = rel(0.85)),
      strip.text         = element_text(face = "bold", color = MORANDI_TITLE, size = rel(0.9)),
      strip.background   = element_rect(fill = "#EDE8E0", color = MORANDI_SPINE, linewidth = 0.3),
      plot.margin        = margin(10, 12, 8, 8)
    )
}

# ── Argument parsing ──────────────────────────────────────────────────────────
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  input_dir  <- "."
  output_dir <- "."
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--input_dir" && i < length(args)) {
      input_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output_dir" && i < length(args)) {
      output_dir <- args[i + 1]
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  list(input_dir = input_dir, output_dir = output_dir)
}

# ── Save figure helper ────────────────────────────────────────────────────────
save_figure <- function(fig, base_path, width = 10, height = 6, dpi = 300) {
  ggsave(paste0(base_path, ".png"), fig, width = width, height = height, dpi = dpi, bg = PLOT_BG)
  ggsave(paste0(base_path, ".pdf"), fig, width = width, height = height, dpi = dpi, bg = PLOT_BG)
}

# ── Figure 1: Zero-shot alignment bar chart ───────────────────────────────────
plot_zeroshot_alignment <- function(alignment_csv, layer_name, output_dir) {
  df <- read_csv(alignment_csv, show_col_types = FALSE) %>%
    mutate(class_name = factor(class_name, levels = CLASS_ORDER))

  if (nrow(df) == 0) {
    message("  [WARN] No alignment data found, skipping Figure 1.")
    return(invisible(NULL))
  }

  # Long format for compactness (human vs plant)
  df_compact <- df %>%
    select(class_name, human_compactness, plant_compactness) %>%
    pivot_longer(
      cols = c(human_compactness, plant_compactness),
      names_to = "species_metric",
      values_to = "compactness"
    ) %>%
    mutate(species_metric = ifelse(species_metric == "human_compactness", "Human", "Plant"))

  # Panel A: centroid distance
  p1 <- ggplot(df, aes(x = class_name, y = centroid_distance, fill = class_name)) +
    geom_col(width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.3f", centroid_distance)), vjust = -0.4, size = 3, color = MORANDI_TEXT) +
    scale_fill_manual(values = MORANDI_CLASS_COLORS) +
    labs(title = "A. Centroid Distance", y = "Euclidean Distance", x = NULL) +
    theme_paper() +
    theme(legend.position = "none")

  # Panel B: centroid cosine
  p2 <- ggplot(df, aes(x = class_name, y = centroid_cosine, fill = class_name)) +
    geom_col(width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.3f", centroid_cosine)), vjust = -0.4, size = 3, color = MORANDI_TEXT) +
    scale_fill_manual(values = MORANDI_CLASS_COLORS) +
    labs(title = "B. Centroid Cosine Similarity", y = "Cosine", x = NULL) +
    theme_paper() +
    theme(legend.position = "none")

  # Panel C: plant-to-human NN distance
  p3 <- ggplot(df, aes(x = class_name, y = plant_to_human_nn_distance, fill = class_name)) +
    geom_col(width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.3f", plant_to_human_nn_distance)), vjust = -0.4, size = 3, color = MORANDI_TEXT) +
    scale_fill_manual(values = MORANDI_CLASS_COLORS) +
    labs(title = "C. Plant → Human NN Distance", y = "Mean NN Distance", x = NULL) +
    theme_paper() +
    theme(legend.position = "none")

  # Panel D: compactness comparison
  p4 <- ggplot(df_compact, aes(x = class_name, y = compactness, fill = species_metric)) +
    geom_col(position = position_dodge(width = 0.65), width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(
      aes(label = sprintf("%.3f", compactness), group = species_metric),
      position = position_dodge(width = 0.65), vjust = -0.4, size = 3, color = MORANDI_TEXT
    ) +
    scale_fill_manual(values = MORANDI_SPECIES_COLORS) +
    labs(title = "D. Intra-class Compactness", y = "Mean Intra-class Distance", x = NULL, fill = "Species") +
    theme_paper()

  # Save individual panels (always, for flexibility)
  base <- file.path(output_dir, paste0("figure1_zeroshot_alignment_bar_", layer_name))
  save_figure(p1, paste0(base, "_centroid_distance"), width = 5, height = 4)
  save_figure(p2, paste0(base, "_centroid_cosine"), width = 5, height = 4)
  save_figure(p3, paste0(base, "_nn_distance"), width = 5, height = 4)
  save_figure(p4, paste0(base, "_compactness"), width = 6.5, height = 4)

  # Composite using gridExtra (widely available)
  composite <- tryCatch({
    library(gridExtra)
    grid.arrange(p1, p2, p3, p4, ncol = 2, top = "Zero-shot Feature Alignment Metrics")
  }, error = function(e) {
    message("  [INFO] gridExtra not available; composite not saved.")
    return(NULL)
  })

  if (!is.null(composite)) {
    ggsave(paste0(base, ".png"), composite, width = 10, height = 8, dpi = 300, bg = PLOT_BG)
    ggsave(paste0(base, ".pdf"), composite, width = 10, height = 8, dpi = 300, bg = PLOT_BG)
    message("  [OK] Figure 1 composite saved: ", base)
  }
  message("  [OK] Figure 1 panels saved: ", base)
}

# ── Figure 2: Few-shot trajectory line chart ──────────────────────────────────
plot_fewshot_trajectory <- function(trajectory_csv, layer_name, output_dir) {
  df <- read_csv(trajectory_csv, show_col_types = FALSE) %>%
    mutate(
      class_name = factor(class_name, levels = CLASS_ORDER),
      shot = as.numeric(shot)
    )

  if (nrow(df) == 0) {
    message("  [WARN] No trajectory data found, skipping Figure 2.")
    return(invisible(NULL))
  }

  metrics_to_plot <- c(
    "mean_distance_to_human_centroid",
    "compactness",
    "separation_margin",
    "separation_ratio"
  )
  metric_labels <- c(
    "mean_distance_to_human_centroid" = "Dist → Human Centroid",
    "compactness"                     = "Compactness",
    "separation_margin"               = "Separation Margin",
    "separation_ratio"                = "Separation Ratio"
  )

  df_long <- df %>%
    select(class_name, shot, all_of(metrics_to_plot)) %>%
    pivot_longer(
      cols = all_of(metrics_to_plot),
      names_to = "metric",
      values_to = "value"
    ) %>%
    mutate(metric = factor(metric, levels = metrics_to_plot, labels = metric_labels))

  p <- ggplot(df_long, aes(x = shot, y = value, color = class_name, group = class_name)) +
    geom_line(linewidth = 0.9, alpha = 0.85) +
    geom_point(size = 3, alpha = 0.95, shape = 16) +
    geom_point(size = 3, alpha = 0.95, shape = 1, color = MORANDI_SPINE, stroke = 0.3) +
    facet_wrap(~ metric, scales = "free_y", ncol = 2) +
    scale_color_manual(values = MORANDI_CLASS_COLORS) +
    scale_x_continuous(breaks = SHOT_COUNTS, labels = SHOT_LABELS) +
    labs(
      title = "Few-shot Trajectory Metrics",
      x = "Shot Count",
      y = "Value",
      color = "Class"
    ) +
    theme_paper()

  base <- file.path(output_dir, paste0("figure2_fewshot_trajectory_line_", layer_name))
  save_figure(p, base, width = 10, height = 7)
  message("  [OK] Figure 2 saved: ", base)
}

# ── Figure 3: Gain heatmap ────────────────────────────────────────────────────
plot_gain_heatmap <- function(trajectory_csv, layer_name, output_dir) {
  df <- read_csv(trajectory_csv, show_col_types = FALSE) %>%
    mutate(class_name = factor(class_name, levels = CLASS_ORDER))

  if (nrow(df) == 0) {
    message("  [WARN] No trajectory data found, skipping Figure 3.")
    return(invisible(NULL))
  }

  gain_metrics <- c(
    "distance_to_human_centroid_gain",
    "compactness_gain",
    "separation_margin_gain",
    "separation_ratio_gain"
  )
  gain_labels <- c(
    "distance_to_human_centroid_gain" = "Dist→Human Gain",
    "compactness_gain"                = "Compactness Gain",
    "separation_margin_gain"          = "Sep Margin Gain",
    "separation_ratio_gain"           = "Sep Ratio Gain"
  )

  df_long <- df %>%
    select(class_name, shot, all_of(gain_metrics)) %>%
    pivot_longer(
      cols = all_of(gain_metrics),
      names_to = "metric",
      values_to = "gain"
    ) %>%
    mutate(
      metric = factor(metric, levels = gain_metrics, labels = gain_labels),
      label = sprintf("%.3f", gain),
      gain_sign = ifelse(gain >= 0, "Positive", "Negative")
    )

  # Compute symmetric limits for diverging color scale
  max_abs_gain <- max(abs(df_long$gain), na.rm = TRUE)
  limits <- c(-max_abs_gain, max_abs_gain)

  p <- ggplot(df_long, aes(x = metric, y = factor(shot, labels = SHOT_LABELS), fill = gain)) +
    geom_tile(color = PLOT_BG, linewidth = 1) +
    geom_text(aes(label = label, color = gain_sign), size = 3.2, fontface = "bold") +
    facet_wrap(~ class_name, nrow = 1) +
    scale_fill_gradient2(
      low = MORANDI_COOL_ACCENT, mid = "#F0EBE3", high = MORANDI_WARM_ACCENT,
      midpoint = 0, name = "Gain", limits = limits,
      oob = scales::squish
    ) +
    scale_color_manual(
      values = c("Positive" = MORANDI_TITLE, "Negative" = MORANDI_TITLE),
      guide = "none"
    ) +
    labs(
      title = "Few-shot Gain Heatmap",
      x = NULL,
      y = "Shot Count"
    ) +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = rel(0.85)))

  base <- file.path(output_dir, paste0("figure3_gain_heatmap_", layer_name))
  save_figure(p, base, width = 12, height = 5.5)
  message("  [OK] Figure 3 saved: ", base)
}

# ── Figure 4: Joint UMAP (optional, but implemented) ──────────────────────────
plot_joint_umap <- function(umap_csv, layer_name, output_dir) {
  df <- read_csv(umap_csv, show_col_types = FALSE) %>%
    mutate(
      class_name = factor(class_name, levels = c(CLASS_ORDER, "Unknown")),
      species = factor(species, levels = c("Human", "Plant"))
    )

  if (nrow(df) == 0) {
    message("  [WARN] No UMAP data found, skipping Figure 4.")
    return(invisible(NULL))
  }

  # Extended Morandi palette for UMAP
  umap_class_colors <- c(MORANDI_CLASS_COLORS, "Unknown" = MORANDI_NEUTRAL)

  # By class - rasterize points for PDF editing efficiency
  p1 <- ggplot(df, aes(x = umap_x, y = umap_y, color = class_name)) +
    rasterise_or_not(geom_point(size = 1.0, alpha = 0.65, shape = 16)) +
    stat_ellipse(aes(group = class_name), type = "norm", linetype = "dashed",
                 linewidth = 0.4, alpha = 0.5, show.legend = FALSE) +
    scale_color_manual(values = umap_class_colors) +
    labs(
      title = paste0("Joint UMAP by Class (", layer_name, ")"),
      x = "UMAP 1", y = "UMAP 2", color = "Class"
    ) +
    theme_paper()

  # By species - rasterize points for PDF editing efficiency
  p2 <- ggplot(df, aes(x = umap_x, y = umap_y, color = species)) +
    rasterise_or_not(geom_point(size = 1.0, alpha = 0.65, shape = 16)) +
    stat_ellipse(aes(group = species), type = "norm", linetype = "dashed",
                 linewidth = 0.4, alpha = 0.5, show.legend = FALSE) +
    scale_color_manual(values = MORANDI_SPECIES_COLORS) +
    labs(
      title = paste0("Joint UMAP by Species (", layer_name, ")"),
      x = "UMAP 1", y = "UMAP 2", color = "Species"
    ) +
    theme_paper()

  base1 <- file.path(output_dir, paste0("figure4_joint_umap_by_class_", layer_name))
  base2 <- file.path(output_dir, paste0("figure4_joint_umap_by_species_", layer_name))
  save_figure(p1, base1, width = 8, height = 6)
  save_figure(p2, base2, width = 8, height = 6)
  message("  [OK] Figure 4 saved: ", base1, " and ", base2)

  # Composite using gridExtra
  composite <- tryCatch({
    library(gridExtra)
    grid.arrange(p1, p2, ncol = 2, top = paste0("Joint UMAP (", layer_name, ")"))
  }, error = function(e) NULL)
  if (!is.null(composite)) {
    base3 <- file.path(output_dir, paste0("figure4_joint_umap_composite_", layer_name))
    ggsave(paste0(base3, ".png"), composite, width = 14, height = 6, dpi = 300, bg = PLOT_BG)
    ggsave(paste0(base3, ".pdf"), composite, width = 14, height = 6, dpi = 300, bg = PLOT_BG)
    message("  [OK] Figure 4 composite saved: ", base3)
  }
}

# ── Supplementary: Sample count bar chart ─────────────────────────────────────
plot_sample_counts <- function(sample_csv, layer_name, output_dir) {
  df <- read_csv(sample_csv, show_col_types = FALSE) %>%
    mutate(class_name = factor(class_name, levels = CLASS_ORDER))

  if (nrow(df) == 0) {
    message("  [WARN] No sample count data found, skipping supplementary figure.")
    return(invisible(NULL))
  }

  df_long <- df %>%
    select(class_name, human_samples, plant_samples) %>%
    pivot_longer(
      cols = c(human_samples, plant_samples),
      names_to = "species",
      values_to = "count"
    ) %>%
    mutate(species = ifelse(species == "human_samples", "Human", "Plant"))

  # Extended palette for species (slightly varied)
  species_colors_extended <- c(
    "Human" = "#9B958E",  # Slightly darker warm grey
    "Plant" = "#B5A88E"   # Slightly lighter taupe
  )

  p <- ggplot(df_long, aes(x = class_name, y = count, fill = species)) +
    geom_col(position = position_dodge(width = 0.65), width = 0.55, alpha = 0.9,
             color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(
      aes(label = scales::comma(count), group = species),
      position = position_dodge(width = 0.65),
      vjust = -0.4, size = 3.5, color = MORANDI_TEXT, fontface = "bold"
    ) +
    scale_fill_manual(values = species_colors_extended) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(
      title = paste0("Supplementary: Sample Counts (", layer_name, ")"),
      x = NULL,
      y = "Number of Samples",
      fill = "Species"
    ) +
    theme_paper()

  base <- file.path(output_dir, paste0("supplementary_sample_counts_", layer_name))
  save_figure(p, base, width = 7, height = 5)
  message("  [OK] Supplementary figure saved: ", base)
}

# ── Gen3: Sample count bar chart ─────────────────────────────────────────────
plot_gen3_sample_counts <- function(sample_csv, layer_name, output_dir) {
  df <- read_csv(sample_csv, show_col_types = FALSE)

  if (nrow(df) == 0) {
    message("  [WARN] No gen3 sample count data found, skipping gen3 figure.")
    return(invisible(NULL))
  }

  # Extended palette for 12 classes
  gen3_class_colors <- c(
    "#B58A83", "#9AAA91", "#8EA3B0", "#C4A898", "#A8B8C4", "#D7CFC4",
    "#9B958E", "#B5A88E", "#8E8A84", "#A69C87", "#C7C0B7", "#E7E0D8"
  )
  names(gen3_class_colors) <- df$class_name

  p <- ggplot(df, aes(x = reorder(class_name, -positive_samples), y = positive_samples, fill = class_name)) +
    geom_col(width = 0.65, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(
      aes(label = scales::comma(positive_samples)),
      vjust = -0.4, size = 3.5, color = MORANDI_TEXT, fontface = "bold"
    ) +
    scale_fill_manual(values = gen3_class_colors) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(
      title = paste0("Gen3 Sample Counts (", layer_name, ")"),
      x = NULL,
      y = "Number of Positive Samples",
      fill = "Class"
    ) +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  base <- file.path(output_dir, paste0("gen3_sample_counts_", layer_name))
  save_figure(p, base, width = 10, height = 6)
  message("  [OK] Gen3 sample counts saved: ", base)
}

# ── Gen3: UMAP visualization ────────────────────────────────────────────────
plot_gen3_umap <- function(umap_csv, layer_name, output_dir) {
  df <- read_csv(umap_csv, show_col_types = FALSE)

  if (nrow(df) == 0) {
    message("  [WARN] No gen3 UMAP data found, skipping gen3 UMAP figure.")
    return(invisible(NULL))
  }

  # Extended palette for 12 classes
  gen3_class_colors <- c(
    "#B58A83", "#9AAA91", "#8EA3B0", "#C4A898", "#A8B8C4", "#D7CFC4",
    "#9B958E", "#B5A88E", "#8E8A84", "#A69C87", "#C7C0B7", "#E7E0D8"
  )
  names(gen3_class_colors) <- unique(df$class_name)

  # Rasterize points for PDF editing efficiency
  p <- ggplot(df, aes(x = umap_x, y = umap_y, color = class_name)) +
    rasterise_or_not(geom_point(size = 1.0, alpha = 0.65, shape = 16)) +
    stat_ellipse(aes(group = class_name), type = "norm", linetype = "dashed",
                 linewidth = 0.4, alpha = 0.5, show.legend = FALSE) +
    scale_color_manual(values = gen3_class_colors) +
    labs(
      title = paste0("Gen3 UMAP by Class (", layer_name, ")"),
      x = "UMAP 1", y = "UMAP 2", color = "Class"
    ) +
    theme_paper()

  base <- file.path(output_dir, paste0("gen3_umap_by_class_", layer_name))
  save_figure(p, base, width = 9, height = 7)
  message("  [OK] Gen3 UMAP saved: ", base)
}

# ── Gen3: Zero-shot alignment bar chart ──────────────────────────────────────
plot_gen3_zeroshot_alignment <- function(alignment_csv, layer_name, output_dir) {
  df <- read_csv(alignment_csv, show_col_types = FALSE)

  if (nrow(df) == 0) {
    message("  [WARN] No gen3 alignment data found, skipping gen3 alignment figure.")
    return(invisible(NULL))
  }

  # Filter for target classes only (Y, m5C, m6A)
  target_classes <- c("Y", "m5C", "m6A")
  df <- df %>% filter(class_name %in% target_classes)

  if (nrow(df) == 0) {
    message("  [WARN] No target class data found in gen3 alignment, skipping.")
    return(invisible(NULL))
  }

  df <- df %>% mutate(class_name = factor(class_name, levels = CLASS_ORDER))

  # Panel A: centroid distance
  p1 <- ggplot(df, aes(x = class_name, y = centroid_distance, fill = class_name)) +
    geom_col(width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.3f", centroid_distance)), vjust = -0.4, size = 3, color = MORANDI_TEXT) +
    scale_fill_manual(values = MORANDI_CLASS_COLORS) +
    labs(title = "A. Centroid Distance (Gen3 vs Human)", y = "Euclidean Distance", x = NULL) +
    theme_paper() +
    theme(legend.position = "none")

  # Panel B: centroid cosine
  p2 <- ggplot(df, aes(x = class_name, y = centroid_cosine, fill = class_name)) +
    geom_col(width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.3f", centroid_cosine)), vjust = -0.4, size = 3, color = MORANDI_TEXT) +
    scale_fill_manual(values = MORANDI_CLASS_COLORS) +
    labs(title = "B. Centroid Cosine Similarity (Gen3 vs Human)", y = "Cosine", x = NULL) +
    theme_paper() +
    theme(legend.position = "none")

  # Panel C: gen3-to-human NN distance
  p3 <- ggplot(df, aes(x = class_name, y = gen3_to_human_nn_distance, fill = class_name)) +
    geom_col(width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(aes(label = sprintf("%.3f", gen3_to_human_nn_distance)), vjust = -0.4, size = 3, color = MORANDI_TEXT) +
    scale_fill_manual(values = MORANDI_CLASS_COLORS) +
    labs(title = "C. Gen3 → Human NN Distance", y = "Mean NN Distance", x = NULL) +
    theme_paper() +
    theme(legend.position = "none")

  # Panel D: compactness comparison
  df_compact <- df %>%
    select(class_name, human_compactness, gen3_compactness) %>%
    pivot_longer(
      cols = c(human_compactness, gen3_compactness),
      names_to = "species_metric",
      values_to = "compactness"
    ) %>%
    mutate(species_metric = ifelse(species_metric == "human_compactness", "Human", "Gen3"))

  p4 <- ggplot(df_compact, aes(x = class_name, y = compactness, fill = species_metric)) +
    geom_col(position = position_dodge(width = 0.65), width = 0.55, alpha = 0.9, color = MORANDI_SPINE, linewidth = 0.2) +
    geom_text(
      aes(label = sprintf("%.3f", compactness), group = species_metric),
      position = position_dodge(width = 0.65), vjust = -0.4, size = 3, color = MORANDI_TEXT
    ) +
    scale_fill_manual(values = c("Human" = "#8E8A84", "Gen3" = "#A69C87")) +
    labs(title = "D. Intra-class Compactness (Gen3 vs Human)", y = "Mean Intra-class Distance", x = NULL, fill = "Species") +
    theme_paper()

  # Save individual panels
  base <- file.path(output_dir, paste0("gen3_zeroshot_alignment_bar_", layer_name))
  save_figure(p1, paste0(base, "_centroid_distance"), width = 5, height = 4)
  save_figure(p2, paste0(base, "_centroid_cosine"), width = 5, height = 4)
  save_figure(p3, paste0(base, "_nn_distance"), width = 5, height = 4)
  save_figure(p4, paste0(base, "_compactness"), width = 6.5, height = 4)

  # Composite using gridExtra
  composite <- tryCatch({
    library(gridExtra)
    grid.arrange(p1, p2, p3, p4, ncol = 2, top = "Gen3 Zero-shot Feature Alignment Metrics")
  }, error = function(e) {
    message("  [INFO] gridExtra not available; composite not saved.")
    return(NULL)
  })

  if (!is.null(composite)) {
    ggsave(paste0(base, ".png"), composite, width = 10, height = 8, dpi = 300, bg = PLOT_BG)
    ggsave(paste0(base, ".pdf"), composite, width = 10, height = 8, dpi = 300, bg = PLOT_BG)
    message("  [OK] Gen3 alignment composite saved: ", base)
  }
  message("  [OK] Gen3 alignment panels saved: ", base)
}

# ── Gen3: Joint UMAP visualization ──────────────────────────────────────────
plot_gen3_joint_umap <- function(umap_csv, layer_name, output_dir) {
  df <- read_csv(umap_csv, show_col_types = FALSE)

  if (nrow(df) == 0) {
    message("  [WARN] No gen3 joint UMAP data found, skipping gen3 joint UMAP figure.")
    return(invisible(NULL))
  }

  # Filter for target classes only
  target_classes <- c("Y", "m5C", "m6A")
  df <- df %>% filter(class_name %in% target_classes)

  if (nrow(df) == 0) {
    message("  [WARN] No target class data found in gen3 joint UMAP, skipping.")
    return(invisible(NULL))
  }

  # By class - rasterize points for PDF editing efficiency
  p1 <- ggplot(df, aes(x = umap_x, y = umap_y, color = class_name)) +
    rasterise_or_not(geom_point(size = 1.0, alpha = 0.65, shape = 16)) +
    stat_ellipse(aes(group = class_name), type = "norm", linetype = "dashed",
                 linewidth = 0.4, alpha = 0.5, show.legend = FALSE) +
    scale_color_manual(values = MORANDI_CLASS_COLORS) +
    labs(
      title = paste0("Gen3/Human Joint UMAP by Class (", layer_name, ")"),
      x = "UMAP 1", y = "UMAP 2", color = "Class"
    ) +
    theme_paper()

  # By species - rasterize points for PDF editing efficiency
  species_colors <- c("Human" = "#8E8A84", "Gen3" = "#A69C87")
  p2 <- ggplot(df, aes(x = umap_x, y = umap_y, color = species)) +
    rasterise_or_not(geom_point(size = 1.0, alpha = 0.65, shape = 16)) +
    stat_ellipse(aes(group = species), type = "norm", linetype = "dashed",
                 linewidth = 0.4, alpha = 0.5, show.legend = FALSE) +
    scale_color_manual(values = species_colors) +
    labs(
      title = paste0("Gen3/Human Joint UMAP by Species (", layer_name, ")"),
      x = "UMAP 1", y = "UMAP 2", color = "Species"
    ) +
    theme_paper()

  base1 <- file.path(output_dir, paste0("gen3_joint_umap_by_class_", layer_name))
  base2 <- file.path(output_dir, paste0("gen3_joint_umap_by_species_", layer_name))
  save_figure(p1, base1, width = 8, height = 6)
  save_figure(p2, base2, width = 8, height = 6)
  message("  [OK] Gen3 joint UMAP saved: ", base1, " and ", base2)

  # Composite using gridExtra
  composite <- tryCatch({
    library(gridExtra)
    grid.arrange(p1, p2, ncol = 2, top = paste0("Gen3/Human Joint UMAP (", layer_name, ")"))
  }, error = function(e) NULL)
  if (!is.null(composite)) {
    base3 <- file.path(output_dir, paste0("gen3_joint_umap_composite_", layer_name))
    ggsave(paste0(base3, ".png"), composite, width = 14, height = 6, dpi = 300, bg = PLOT_BG)
    ggsave(paste0(base3, ".pdf"), composite, width = 14, height = 6, dpi = 300, bg = PLOT_BG)
    message("  [OK] Gen3 joint UMAP composite saved: ", base3)
  }
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  args <- parse_args()
  input_dir  <- args$input_dir
  output_dir <- args$output_dir

  if (!dir.exists(input_dir)) {
    stop("Input directory does not exist: ", input_dir)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  message("Input dir:  ", input_dir)
  message("Output dir: ", output_dir)

  # Detect layer name from filenames
  layer_name <- "gcn_output"
  json_files <- list.files(input_dir, pattern = "zeroshot_alignment_metrics_.*\\.csv", full.names = TRUE)
  if (length(json_files) > 0) {
    m <- regmatches(json_files[1], regexpr("zeroshot_alignment_metrics_([^/]+)\\.csv", json_files[1]))
    if (length(m) > 0) {
      layer_name <- sub("zeroshot_alignment_metrics_", "", sub("\\.csv$", "", m[1]))
    }
  }
  message("Detected layer name: ", layer_name)

  # Locate CSV files
  alignment_csv   <- file.path(input_dir, paste0("zeroshot_alignment_metrics_", layer_name, ".csv"))
  trajectory_csv  <- file.path(input_dir, paste0("fewshot_trajectory_metrics_", layer_name, ".csv"))
  umap_csv        <- file.path(input_dir, paste0("zeroshot_joint_umap_points_", layer_name, ".csv"))
  sample_csv      <- file.path(input_dir, paste0("sample_count_summary_", layer_name, ".csv"))
  gen3_sample_csv <- file.path(input_dir, paste0("gen3_sample_count_summary_", layer_name, ".csv"))
  gen3_umap_csv   <- file.path(input_dir, paste0("gen3_umap_points_", layer_name, ".csv"))
  gen3_alignment_csv <- file.path(input_dir, paste0("gen3_alignment_metrics_", layer_name, ".csv"))
  gen3_joint_umap_csv <- file.path(input_dir, paste0("gen3_joint_umap_points_", layer_name, ".csv"))

  # Generate figures
  message("\n── Generating Figure 1: Zero-shot alignment bar chart ──")
  if (file.exists(alignment_csv)) {
    plot_zeroshot_alignment(alignment_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", alignment_csv)
  }

  message("\n── Generating Figure 2: Few-shot trajectory line chart ──")
  if (file.exists(trajectory_csv)) {
    plot_fewshot_trajectory(trajectory_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", trajectory_csv)
  }

  message("\n── Generating Figure 3: Gain heatmap ──")
  if (file.exists(trajectory_csv)) {
    plot_gain_heatmap(trajectory_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", trajectory_csv)
  }

  message("\n── Generating Figure 4: Joint UMAP ──")
  if (file.exists(umap_csv)) {
    plot_joint_umap(umap_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", umap_csv)
  }

  message("\n── Generating Supplementary: Sample count bar chart ──")
  if (file.exists(sample_csv)) {
    plot_sample_counts(sample_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", sample_csv)
  }

  message("\n── Generating Gen3: Sample count bar chart ──")
  if (file.exists(gen3_sample_csv)) {
    plot_gen3_sample_counts(gen3_sample_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", gen3_sample_csv)
  }

  message("\n── Generating Gen3: UMAP visualization ──")
  if (file.exists(gen3_umap_csv)) {
    plot_gen3_umap(gen3_umap_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", gen3_umap_csv)
  }

  message("\n── Generating Gen3: Zero-shot alignment bar chart ──")
  if (file.exists(gen3_alignment_csv)) {
    plot_gen3_zeroshot_alignment(gen3_alignment_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", gen3_alignment_csv)
  }

  message("\n── Generating Gen3: Joint UMAP visualization ──")
  if (file.exists(gen3_joint_umap_csv)) {
    plot_gen3_joint_umap(gen3_joint_umap_csv, layer_name, output_dir)
  } else {
    message("  [WARN] File not found: ", gen3_joint_umap_csv)
  }

  message("\n✓ All R plots completed. Output directory: ", output_dir)
}

main()
