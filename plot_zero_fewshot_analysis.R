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
  dpi <- max(300, as.numeric(dpi))
  if (exists("has_ggrastr") && has_ggrastr) {
    return(rasterise(geom_obj, dpi = dpi))
  } else {
    return(geom_obj)
  }
}

# ── High-Contrast Color Palette (matches Python constants) ─────────────────────
# Primary colors for real points, secondary for synthetic points
HIGH_CONTRAST_MOD_COLORS <- list(
  "Y" = list(
    "primary" = "#e4852b",
    "secondary" = "#fae41e"
  ),
  "m5C" = list(
    "primary" = "#3d4092",
    "secondary" = "#6ac6e9"
  ),
  "m6A" = list(
    "primary" = "#0f82bf",
    "secondary" = "#b96497"
  )
)

# Gen3-specific colors (Gen3 is treated as different modification from Plant)
HIGH_CONTRAST_GEN3_COLORS <- list(
  "m6A" = list(
    "primary" = "#0a8648",    # Green for real Gen3 m6A points
    "secondary" = "#83bd55"    # Light green for synthetic Gen3 m6A points
  )
)

# Species colors
HIGH_CONTRAST_SPECIES_COLORS <- list(
  "Human" = list(
    "primary" = "#e4852b",
    "secondary" = "#fae41e"
  ),
  "Plant" = list(
    "primary" = "#0f82bf",
    "secondary" = "#6ac6e9"
  ),
  "Gen3" = list(
    "primary" = "#0a8648",
    "secondary" = "#83bd55"
  )
)

# Point style parameters (matches Python POINT_STYLE_PARAMS)
REAL_POINT_SIZE <- 4.5
SYNTHETIC_POINT_SIZE <- 1.8
REAL_POINT_ALPHA <- 0.32
SYNTHETIC_POINT_ALPHA <- 0.08

# ── Morandi palette (kept for backward compatibility) ─────────────────────────
MORANDI_CLASS_COLORS <- c(
  "Y"   = "#e4852b",
  "m5C" = "#3d4092",
  "m6A" = "#0f82bf"
)
MORANDI_SPECIES_COLORS <- c(
  "Human" = "#e4852b",
  "Plant" = "#0f82bf",
  "Gen3" = "#0a8648"
)
MORANDI_NEUTRAL   <- "#C7C0B7"
MORANDI_GRID      <- "#E7E0D8"
MORANDI_TEXT      <- "#6E675F"
MORANDI_TITLE     <- "#4A4540"
MORANDI_SPINE     <- "#D7CFC4"
MORANDI_WARM_ACCENT <- "#e4852b"
MORANDI_COOL_ACCENT <- "#6ac6e9"
PLOT_BG           <- "#FBF8F3"
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

# ── Helper: Get high contrast color ───────────────────────────────────────────
get_high_contrast_color <- function(source_group, color_type = "primary") {
  # source_group format: "Human_m6A", "Plant_m6A", "Gen3_m6A", "Y", "m5C", "m6A"
  
  # First, determine if it's a species-based or modification-based group
  if (grepl("_", source_group)) {
    parts <- strsplit(source_group, "_")[[1]]
    if (length(parts) >= 2) {
      species <- parts[1]
      mod <- paste(parts[2:length(parts)], collapse = "_")  # Handle potential "_" in modification names
    } else {
      species <- parts[1]
      mod <- ""
    }
  } else {
    species <- ""
    mod <- source_group
  }
  
  # Check for Gen3 modification-specific colors first
  if (!is.null(gen3_mod <<- get0("gen3_override", envir = globalenv())) && 
      mod == "m6A" && species == "Gen3" &&
      mod %in% names(HIGH_CONTRAST_GEN3_COLORS)) {
    return(HIGH_CONTRAST_GEN3_COLORS[[mod]][[color_type]])
  }
  
  # Check species colors
  if (species %in% names(HIGH_CONTRAST_SPECIES_COLORS)) {
    return(HIGH_CONTRAST_SPECIES_COLORS[[species]][[color_type]])
  }
  
  # Check modification colors
  if (mod %in% names(HIGH_CONTRAST_MOD_COLORS)) {
    return(HIGH_CONTRAST_MOD_COLORS[[mod]][[color_type]])
  }
  
  # Fallback
  return("#999999")
}

# ── Argument parsing ──────────────────────────────────────────────────────────
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  input_dir  <- "."
  output_dir <- "."
  use_high_contrast <- TRUE
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
  dpi <- max(300, as.numeric(dpi))
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

# ── Figure 4: Joint UMAP (Enhanced with synthetic points) ────────────────────
plot_joint_umap <- function(umap_csv, layer_name, output_dir) {
  df <- read_csv(umap_csv, show_col_types = FALSE)

  if (nrow(df) == 0) {
    message("  [WARN] No UMAP data found, skipping Figure 4.")
    return(invisible(NULL))
  }

  # Check if is_synthetic column exists
  has_synthetic <- "is_synthetic" %in% colnames(df)
  
  # Split real and synthetic points
  if (has_synthetic) {
    df_real <- df %>% filter(is_synthetic == 0 | is_synthetic == "0" | is_synthetic == FALSE)
    df_synth <- df %>% filter(is_synthetic == 1 | is_synthetic == "1" | is_synthetic == TRUE)
  } else {
    df_real <- df
    df_synth <- df[FALSE, ]  # Empty dataframe
  }
  
  message(sprintf("  [INFO] Real points: %d, Synthetic points: %d", nrow(df_real), nrow(df_synth)))

  # Extended Morandi palette for UMAP (fallback)
  umap_class_colors <- c(MORANDI_CLASS_COLORS, "Unknown" = MORANDI_NEUTRAL)

  # By class - plot real points first, then synthetic
  p1 <- ggplot()
  
  # Add real points if any
  if (nrow(df_real) > 0) {
    # Create color mapping for real points based on source_group
    real_colors <- sapply(df_real$source_group, function(sg) {
      if (!is.na(sg) && sg != "") {
        get_high_contrast_color(sg, "primary")
      } else {
        umap_class_colors[as.character(df_real$class_name[which(!is.na(df_real$source_group) & df_real$source_group == sg)[1]])]
      }
    })
    real_colors[is.na(real_colors)] <- "#999999"
    
    p1 <- p1 + rasterise_or_not(geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = REAL_POINT_SIZE,
      alpha = REAL_POINT_ALPHA,
      shape = 16  # Circle
    )) +
      scale_color_manual(values = umap_class_colors)
  }
  
  # Add synthetic points if any
  if (nrow(df_synth) > 0) {
    synth_colors <- sapply(df_synth$source_group, function(sg) {
      if (!is.na(sg) && sg != "") {
        get_high_contrast_color(sg, "secondary")
      } else {
        "#999999"
      }
    })
    synth_colors[is.na(synth_colors)] <- "#999999"
    
    p1 <- p1 + rasterise_or_not(geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = SYNTHETIC_POINT_SIZE,
      alpha = SYNTHETIC_POINT_ALPHA,
      shape = 16  # Circle
    )) +
      scale_color_manual(values = umap_class_colors)
  }
  
  if (nrow(df_real) > 0 || nrow(df_synth) > 0) {
    p1 <- p1 +
      labs(
        title = paste0("Joint UMAP by Class (", layer_name, ")"),
        x = "UMAP 1", y = "UMAP 2", color = "Class"
      ) +
      theme_paper()
  }
  
  # By species - plot real points first, then synthetic
  p2 <- ggplot()
  
  # Add real points if any
  if (nrow(df_real) > 0) {
    real_species_colors <- sapply(df_real$source_group, function(sg) {
      if (!is.na(sg) && sg != "" && grepl("_", sg)) {
        species <- strsplit(sg, "_")[[1]][1]
        if (species %in% names(HIGH_CONTRAST_SPECIES_COLORS)) {
          return(HIGH_CONTRAST_SPECIES_COLORS[[species]]$primary)
        }
      }
      MORANDI_SPECIES_COLORS[as.character(df_real$species[1])]
    })
    real_species_colors[is.na(real_species_colors)] <- "#999999"
    
    p2 <- p2 + rasterise_or_not(geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = species),
      size = REAL_POINT_SIZE,
      alpha = REAL_POINT_ALPHA,
      shape = 16  # Circle
    )) +
      scale_color_manual(values = MORANDI_SPECIES_COLORS)
  }
  
  # Add synthetic points if any
  if (nrow(df_synth) > 0) {
    synth_species_colors <- sapply(df_synth$source_group, function(sg) {
      if (!is.na(sg) && sg != "" && grepl("_", sg)) {
        species <- strsplit(sg, "_")[[1]][1]
        if (species %in% names(HIGH_CONTRAST_SPECIES_COLORS)) {
          return(HIGH_CONTRAST_SPECIES_COLORS[[species]]$secondary)
        }
      }
      "#999999"
    })
    synth_species_colors[is.na(synth_species_colors)] <- "#999999"
    
    p2 <- p2 + rasterise_or_not(geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = species),
      size = SYNTHETIC_POINT_SIZE,
      alpha = SYNTHETIC_POINT_ALPHA,
      shape = 16  # Circle
    )) +
      scale_color_manual(values = MORANDI_SPECIES_COLORS)
  }
  
  if (nrow(df_real) > 0 || nrow(df_synth) > 0) {
    p2 <- p2 +
      labs(
        title = paste0("Joint UMAP by Species (", layer_name, ")"),
        x = "UMAP 1", y = "UMAP 2", color = "Species"
      ) +
      theme_paper()
  }

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

# ── Figure 4b: Per-modification UMAP ─────────────────────────────────────────
plot_joint_umap_per_mod <- function(input_dir, layer_name, output_dir) {
  mod_files <- list.files(
    input_dir,
    pattern = paste0("^zeroshot_joint_umap_points_", layer_name, "_(Y|m5C|m6A)\\.csv$"),
    full.names = TRUE
  )

  if (length(mod_files) == 0) {
    message("  [WARN] No per-modification UMAP CSV files found, skipping per-modification UMAP.")
    return(invisible(NULL))
  }
  
  for (mod_file in mod_files) {
    mod <- sub(paste0("^.*zeroshot_joint_umap_points_", layer_name, "_"), "", mod_file)
    mod <- sub("\\.csv$", "", mod)
    df_mod <- read_csv(mod_file, show_col_types = FALSE)
    
    if (nrow(df_mod) == 0) next
    
    # Check if is_synthetic column exists
    has_synthetic <- "is_synthetic" %in% colnames(df_mod)
    
    # Split real and synthetic points
    if (has_synthetic) {
      df_real <- df_mod %>% filter(is_synthetic == 0 | is_synthetic == "0" | is_synthetic == FALSE)
      df_synth <- df_mod %>% filter(is_synthetic == 1 | is_synthetic == "1" | is_synthetic == TRUE)
    } else {
      df_real <- df_mod
      df_synth <- df_mod[FALSE, ]
    }
    
    # Get colors for this modification
    mod_primary <- if (mod %in% names(HIGH_CONTRAST_MOD_COLORS)) {
      HIGH_CONTRAST_MOD_COLORS[[mod]]$primary
    } else {
      "#999999"
    }
    mod_secondary <- if (mod %in% names(HIGH_CONTRAST_MOD_COLORS)) {
      HIGH_CONTRAST_MOD_COLORS[[mod]]$secondary
    } else {
      "#999999"
    }
    
    p <- ggplot()
    
    # Add real points
    if (nrow(df_real) > 0) {
      p <- p + rasterise_or_not(geom_point(
        data = df_real,
        aes(x = umap_x, y = umap_y, color = source_group),
        size = REAL_POINT_SIZE,
        alpha = REAL_POINT_ALPHA,
        shape = 16
      ))
    }
    
    # Add synthetic points
    if (nrow(df_synth) > 0) {
      p <- p + rasterise_or_not(geom_point(
        data = df_synth,
        aes(x = umap_x, y = umap_y, color = source_group),
        size = SYNTHETIC_POINT_SIZE,
        alpha = SYNTHETIC_POINT_ALPHA,
        shape = 16
      ))
    }
    
    # Build color scale for this modification's groups
    if (nrow(df_real) > 0 || nrow(df_synth) > 0) {
      all_groups <- c(df_real$source_group, df_synth$source_group)
      all_groups <- unique(all_groups[!is.na(all_groups)])
      
      group_colors <- sapply(all_groups, function(sg) {
        parts <- strsplit(as.character(sg), "_")[[1]]
        species <- parts[1]
        if (species == "Human" && "Human" %in% names(HIGH_CONTRAST_SPECIES_COLORS)) {
          return(HIGH_CONTRAST_SPECIES_COLORS$Human$primary)
        } else if (species == "Plant" && "Plant" %in% names(HIGH_CONTRAST_SPECIES_COLORS)) {
          return(HIGH_CONTRAST_SPECIES_COLORS$Plant$primary)
        }
        mod_primary
      })
      
      p <- p +
        scale_color_manual(
          values = setNames(group_colors, all_groups),
          guide = guide_legend(title = "Group", override.aes = list(size = 3))
        ) +
        labs(
          title = paste0("Joint UMAP - ", mod, " (", layer_name, ")"),
          x = "UMAP 1", y = "UMAP 2"
        ) +
        theme_paper()
      
      base <- file.path(output_dir, paste0("figure4_joint_umap_", mod, "_", layer_name))
      save_figure(p, base, width = 8, height = 6)
      message("  [OK] Per-mod UMAP saved: ", base)
    }
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
    "Human" = "#9B958E",
    "Plant" = "#B5A88E"
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

  # Check if is_synthetic column exists
  has_synthetic <- "is_synthetic" %in% colnames(df)
  
  # Split real and synthetic points
  if (has_synthetic) {
    df_real <- df %>% filter(is_synthetic == 0 | is_synthetic == "0" | is_synthetic == FALSE)
    df_synth <- df %>% filter(is_synthetic == 1 | is_synthetic == "1" | is_synthetic == TRUE)
  } else {
    df_real <- df
    df_synth <- df[FALSE, ]
  }

  # Extended palette for 12 classes
  gen3_class_colors <- c(
    "#B58A83", "#9AAA91", "#8EA3B0", "#C4A898", "#A8B8C4", "#D7CFC4",
    "#9B958E", "#B5A88E", "#8E8A84", "#A69C87", "#C7C0B7", "#E7E0D8"
  )
  
  p <- ggplot()
  
  # Add real points
  if (nrow(df_real) > 0) {
    p <- p + rasterise_or_not(geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = REAL_POINT_SIZE,
      alpha = REAL_POINT_ALPHA,
      shape = 16
    )) +
      scale_color_manual(values = gen3_class_colors)
  }
  
  # Add synthetic points
  if (nrow(df_synth) > 0) {
    p <- p + rasterise_or_not(geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = SYNTHETIC_POINT_SIZE,
      alpha = SYNTHETIC_POINT_ALPHA,
      shape = 16
    )) +
      scale_color_manual(values = gen3_class_colors)
  }
  
  if (nrow(df_real) > 0 || nrow(df_synth) > 0) {
    p <- p +
      labs(
        title = paste0("Gen3 UMAP by Class (", layer_name, ")"),
        x = "UMAP 1", y = "UMAP 2", color = "Class"
      ) +
      theme_paper()
  }

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
    scale_fill_manual(values = c(
      "Human" = HIGH_CONTRAST_SPECIES_COLORS$Human$primary,
      "Gen3" = HIGH_CONTRAST_SPECIES_COLORS$Gen3$primary
    )) +
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
  
  # Check if is_synthetic column exists
  has_synthetic <- "is_synthetic" %in% colnames(df)
  
  # Split real and synthetic points
  if (has_synthetic) {
    df_real <- df %>% filter(is_synthetic == 0 | is_synthetic == "0" | is_synthetic == FALSE)
    df_synth <- df %>% filter(is_synthetic == 1 | is_synthetic == "1" | is_synthetic == TRUE)
  } else {
    df_real <- df
    df_synth <- df[FALSE, ]
  }
  
  message(sprintf("  [INFO] Gen3 joint UMAP - Real points: %d, Synthetic points: %d", nrow(df_real), nrow(df_synth)))

  # By class - real points first, then synthetic
  p1 <- ggplot()
  
  gen3_class_colors <- c(
    "Y" = HIGH_CONTRAST_MOD_COLORS$Y$primary,
    "m5C" = HIGH_CONTRAST_MOD_COLORS$m5C$primary,
    "m6A" = HIGH_CONTRAST_GEN3_COLORS$m6A$primary
  )
  gen3_class_synth_colors <- c(
    "Y" = HIGH_CONTRAST_MOD_COLORS$Y$secondary,
    "m5C" = HIGH_CONTRAST_MOD_COLORS$m5C$secondary,
    "m6A" = HIGH_CONTRAST_GEN3_COLORS$m6A$secondary
  )

  if (nrow(df_synth) > 0) {
    p1 <- p1 + rasterise_or_not(geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = SYNTHETIC_POINT_SIZE,
      alpha = SYNTHETIC_POINT_ALPHA,
      shape = 16
    ))
  }
  
  if (nrow(df_real) > 0) {
    p1 <- p1 + rasterise_or_not(geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = REAL_POINT_SIZE,
      alpha = REAL_POINT_ALPHA,
      shape = 16
    ))
  }
  
  if (nrow(df_real) > 0 || nrow(df_synth) > 0) {
    p1 <- p1 + scale_color_manual(values = gen3_class_colors)
  }
  
  if (nrow(df_real) > 0 || nrow(df_synth) > 0) {
    p1 <- p1 +
      labs(
        title = paste0("Gen3/Human Joint UMAP by Class (", layer_name, ")"),
        x = "UMAP 1", y = "UMAP 2", color = "Class"
      ) +
      theme_paper()
  }

  # By species - real points first, then synthetic
  species_colors_enhanced <- c(
    "Human" = HIGH_CONTRAST_SPECIES_COLORS$Human$primary,
    "Gen3" = HIGH_CONTRAST_SPECIES_COLORS$Gen3$primary
  )
  
  p2 <- ggplot()
  
  if (nrow(df_real) > 0) {
    p2 <- p2 + rasterise_or_not(geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = species),
      size = REAL_POINT_SIZE,
      alpha = REAL_POINT_ALPHA,
      shape = 16
    )) +
      scale_color_manual(values = species_colors_enhanced)
  }
  
  if (nrow(df_synth) > 0) {
    synth_colors <- sapply(df_synth$source_group, function(sg) {
      if (!is.na(sg) && sg != "" && grepl("_", sg)) {
        species <- strsplit(sg, "_")[[1]][1]
        if (species %in% names(HIGH_CONTRAST_SPECIES_COLORS)) {
          return(HIGH_CONTRAST_SPECIES_COLORS[[species]]$secondary)
        }
      }
      "#999999"
    })
    
    p2 <- p2 + rasterise_or_not(geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = species),
      size = SYNTHETIC_POINT_SIZE,
      alpha = SYNTHETIC_POINT_ALPHA,
      shape = 16
    )) +
      scale_color_manual(values = species_colors_enhanced)
  }
  
  if (nrow(df_real) > 0 || nrow(df_synth) > 0) {
    p2 <- p2 +
      labs(
        title = paste0("Gen3/Human Joint UMAP by Species (", layer_name, ")"),
        x = "UMAP 1", y = "UMAP 2", color = "Species"
      ) +
      theme_paper()
  }

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

  message("\n── Generating Figure 4b: Per-modification UMAP ──")
  if (dir.exists(input_dir)) {
    plot_joint_umap_per_mod(input_dir, layer_name, output_dir)
  } else {
    message("  [WARN] Input directory not found: ", input_dir)
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
