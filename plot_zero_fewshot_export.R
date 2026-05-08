#!/usr/bin/env Rscript
# plot_zero_fewshot_export.R
# Generates publication-quality figures from Python-exported CSV/JSON data.
#
# Usage:
#   conda run -n learn-new Rscript plot_zero_fewshot_export.R --input_dir <path> --output_dir <path>
#
# Dependencies: ggplot2, dplyr, tidyr, readr, scales, forcats, viridis

# Check and install required packages if not available
required_packages <- c("ggplot2", "dplyr", "tidyr", "readr", "scales", "forcats")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing ", pkg, "...")
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
}

# Optional packages
has_ggrastr <- requireNamespace("ggrastr", quietly = TRUE)
has_gridExtra <- requireNamespace("gridExtra", quietly = TRUE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(forcats)
})

if (has_ggrastr) {
  library(ggrastr)
  message("ggrastr loaded - points will be rasterized in PDF")
}

# ── Paper-Ready High-Contrast Color Palette ──────────────────────────────────
PAPER_CLASS_COLORS <- c(
  "m6A" = "#0f82bf",
  "m5C" = "#3d4092",
  "Y"   = "#e4852b"
)

PAPER_CLASS_SYNTH_COLORS <- c(
  "m6A" = "#b96497",
  "m5C" = "#6ac6e9",
  "Y"   = "#fae41e"
)

PAPER_SPECIES_COLORS <- c(
  "Human"  = "#e4852b",
  "Plant"  = "#0f82bf",
  "Gen3"   = "#0a8648"
)

PAPER_SPECIES_SYNTH_COLORS <- c(
  "Human"  = "#fae41e",
  "Plant"  = "#6ac6e9",
  "Gen3"   = "#83bd55"
)

# Point size and alpha parameters
REAL_POINT_SIZE <- 1.15
SYNTHETIC_POINT_SIZE <- 0.12
REAL_POINT_ALPHA <- 0.11
SYNTHETIC_POINT_ALPHA <- 0.004
PLANT_SIZE_MULTIPLIER <- 2.0

# Plot styling
PLOT_BG <- "#FFFFFF"
PANEL_BG <- "#FFFFFF"
GRID_COLOR <- "#F1F1F1"
TEXT_COLOR <- "#333333"
TITLE_COLOR <- "#000000"

CLASS_ORDER <- c("Y", "m5C", "m6A")

# ── Paper Theme ───────────────────────────────────────────────────────────────
theme_paper <- function(base_size = 11) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      plot.background    = element_rect(fill = PLOT_BG, color = NA),
      panel.background   = element_rect(fill = PANEL_BG, color = NA),
      panel.grid.major   = element_line(color = GRID_COLOR, linewidth = 0.3),
      panel.grid.minor   = element_blank(),
      axis.ticks         = element_line(color = "#CCCCCC", linewidth = 0.3),
      axis.text          = element_text(color = TEXT_COLOR, size = rel(0.85)),
      axis.title         = element_text(color = TEXT_COLOR, face = "bold", size = rel(0.9)),
      legend.background  = element_rect(fill = PLOT_BG, color = NA),
      legend.key         = element_rect(fill = PLOT_BG, color = NA),
      legend.text        = element_text(color = TEXT_COLOR, size = rel(0.8)),
      legend.title       = element_text(color = TEXT_COLOR, face = "bold", size = rel(0.85)),
      plot.title         = element_text(face = "bold", hjust = 0, color = TITLE_COLOR, size = rel(1.1)),
      plot.subtitle      = element_text(hjust = 0, color = TEXT_COLOR, size = rel(0.85)),
      strip.text         = element_text(face = "bold", color = TEXT_COLOR, size = rel(0.85)),
      strip.background   = element_rect(fill = "#E8E8E8", color = "#CCCCCC", linewidth = 0.3),
      plot.margin        = margin(10, 12, 10, 10),
      legend.margin      = margin(6, 6, 6, 6)
    )
}

# ── Safe CSV Reader with Robust Type Handling ─────────────────────────────────
safe_read_csv <- function(file_path, ...) {
  if (!file.exists(file_path)) {
    warning("File not found: ", file_path)
    return(NULL)
  }
  
  tryCatch({
    # Read with character encoding, then clean up
    df <- read_csv(file_path, show_col_types = FALSE, ...)

    # Clean up shot and reference_type columns that might be malformed from older exports.
    if ("shot" %in% colnames(df)) {
      df$shot <- as.character(df$shot)
      df$shot <- sapply(df$shot, function(x) {
        if (is.na(x) || x == "") return("NA")
        x <- trimws(x)
        # New exports should already be scalar strings such as 0/1/5/10/NA.
        # Older broken exports wrote the whole array into each row, e.g. "['NA' 'NA' ... '10']".
        # Keep a sentinel so downstream code can fail with a clear message instead of silently
        # collapsing to a single bogus facet.
        if (grepl("^\\[.*\\]$", x) && grepl("'", x)) {
          return("__INVALID_ARRAY_EXPORT__")
        }
        if (grepl(",", x, fixed = TRUE)) {
          vals <- strsplit(x, ",")[[1]]
          return(trimws(vals[1]))
        }
        return(x)
      })
    }
    
    if ("reference_type" %in% colnames(df)) {
      df$reference_type <- as.character(df$reference_type)
      df$reference_type <- sapply(df$reference_type, function(x) {
        if (is.na(x) || x == "") return("unknown")
        if (grepl(",", x, fixed = TRUE)) {
          vals <- strsplit(x, ",")[[1]]
          return(vals[1])
        }
        return(x)
      })
    }
    
    # Clean up is_synthetic to be consistent
    if ("is_synthetic" %in% colnames(df)) {
      df$is_synthetic <- as.character(df$is_synthetic)
      df$is_synthetic[df$is_synthetic %in% c("0", "FALSE", "False", "false")] <- "0"
      df$is_synthetic[df$is_synthetic %in% c("1", "TRUE", "True", "true")] <- "1"
    }
    
    return(df)
  }, error = function(e) {
    warning("Error reading ", file_path, ": ", e$message)
    return(NULL)
  })
}

# ── Split Real vs Synthetic Points ────────────────────────────────────────────
split_real_synth <- function(df) {
  df_real <- df %>% filter(is_synthetic == "0" | is_synthetic == 0)
  df_synth <- df %>% filter(is_synthetic == "1" | is_synthetic == 1)
  list(real = df_real, synth = df_synth)
}

# ── Get Color for Source Group ─────────────────────────────────────────────────
get_source_group_color <- function(source_group, is_synthetic = FALSE) {
  if (is.null(source_group) || is.na(source_group)) return("#999999")
  
  parts <- strsplit(as.character(source_group), "_")[[1]]
  if (length(parts) >= 2) {
    species <- parts[1]
    mod <- paste(parts[2:length(parts)], collapse = "_")
    
    if (species == "Gen3") {
      if (is_synthetic) return("#83bd55") else return("#0a8648")
    } else if (species == "Human") {
      if (is_synthetic) return("#fae41e") else return("#e4852b")
    } else if (species == "Plant") {
      if (is_synthetic) return("#6ac6e9") else return("#0f82bf")
    }
  }
  
  # Just modification name
  mod <- source_group
  if (is_synthetic) {
    return(PAPER_CLASS_SYNTH_COLORS[mod])
  } else {
    return(PAPER_CLASS_COLORS[mod])
  }
}

# ── Build Class Color Scale ───────────────────────────────────────────────────
build_class_color_scale <- function(use_synth = FALSE) {
  if (use_synth) {
    scale_color_manual(values = PAPER_CLASS_SYNTH_COLORS, guide = "none")
  } else {
    scale_color_manual(values = PAPER_CLASS_COLORS, 
                      name = "Modification",
                      guide = guide_legend(override.aes = list(size = 4, alpha = 0.7)))
  }
}

# ── Build Species Color Scale ────────────────────────────────────────────────
build_species_color_scale <- function(use_synth = FALSE) {
  if (use_synth) {
    scale_color_manual(values = PAPER_SPECIES_SYNTH_COLORS, guide = "none")
  } else {
    scale_color_manual(values = PAPER_SPECIES_COLORS, 
                      name = "Species",
                      guide = guide_legend(override.aes = list(size = 4, alpha = 0.7)))
  }
}

# ── Generic UMAP Plot by Class ────────────────────────────────────────────────
plot_umap_by_class <- function(df, title = "UMAP", subtitle = "", 
                               use_species_colors = FALSE,
                               split_by_shot = FALSE,
                               highlight_plant = FALSE) {
  splits <- split_real_synth(df)
  df_real <- splits$real
  df_synth <- splits$synth
  
  p <- ggplot()
  
  # Add synthetic points first (lighter, smaller)
  if (nrow(df_synth) > 0) {
    if (use_species_colors) {
      p <- p + geom_point(
        data = df_synth,
        aes(x = umap_x, y = umap_y, color = species),
        size = SYNTHETIC_POINT_SIZE, alpha = SYNTHETIC_POINT_ALPHA, shape = 16
      ) + build_species_color_scale(use_synth = TRUE)
    } else {
      p <- p + geom_point(
        data = df_synth,
        aes(x = umap_x, y = umap_y, color = class_name),
        size = SYNTHETIC_POINT_SIZE, alpha = SYNTHETIC_POINT_ALPHA, shape = 16
      ) + build_class_color_scale(use_synth = TRUE)
    }
  }
  
  # Add real points (prominent)
  if (nrow(df_real) > 0) {
    if (highlight_plant) {
      # Plant points are larger
      df_real_plant <- df_real %>% filter(species == "Plant")
      df_real_other <- df_real %>% filter(species != "Plant")
      
      if (nrow(df_real_other) > 0) {
        p <- p + geom_point(
          data = df_real_other,
          aes(x = umap_x, y = umap_y, color = if(use_species_colors) species else class_name),
          size = REAL_POINT_SIZE, alpha = REAL_POINT_ALPHA, shape = 16
        )
      }
      if (nrow(df_real_plant) > 0) {
        p <- p + geom_point(
          data = df_real_plant,
          aes(x = umap_x, y = umap_y, color = if(use_species_colors) species else class_name),
          size = REAL_POINT_SIZE * PLANT_SIZE_MULTIPLIER, alpha = REAL_POINT_ALPHA, shape = 16
        )
      }
    } else {
      p <- p + geom_point(
        data = df_real,
        aes(x = umap_x, y = umap_y, color = if(use_species_colors) species else class_name),
        size = REAL_POINT_SIZE, alpha = REAL_POINT_ALPHA, shape = 16
      )
    }
    
    if (use_species_colors) {
      p <- p + build_species_color_scale()
    } else {
      p <- p + build_class_color_scale()
    }
  }
  
  # Add facet by shot if requested
  if (split_by_shot && "shot" %in% colnames(df)) {
    df_filt <- df %>% filter(!is.na(shot) & shot != "NA")
    if (nrow(df_filt) > 0) {
      shot_order <- c("0", "1", "5", "10")
      shot_labels <- c("0" = "0-shot", "1" = "1-shot", "5" = "5-shot", "10" = "10-shot")
      df_filt$shot <- factor(df_filt$shot, levels = shot_order, labels = shot_labels)
      p <- p + facet_wrap(~ shot, ncol = 2)
    }
  }
  
  p <- p +
    coord_equal() +
    labs(title = title, subtitle = subtitle, x = "UMAP 1", y = "UMAP 2") +
    theme_paper() +
    theme(
      panel.grid.major = element_line(color = GRID_COLOR, linewidth = 0.22),
      panel.grid.minor = element_blank()
    )
  
  return(p)
}

# ── Generic UMAP Plot by Species ──────────────────────────────────────────────
plot_umap_by_species <- function(df, title = "UMAP", subtitle = "") {
  plot_umap_by_class(df, title = title, subtitle = subtitle, use_species_colors = TRUE)
}

# ── Argument Parsing ──────────────────────────────────────────────────────────
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  input_dir  <- "output/zero_fewshot_export_latest"
  output_dir <- "output/figures"
  dpi <- 300
  
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--input_dir" && i < length(args)) {
      input_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--output_dir" && i < length(args)) {
      output_dir <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--dpi" && i < length(args)) {
      dpi <- as.numeric(args[i + 1])
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  list(input_dir = input_dir, output_dir = output_dir, dpi = dpi)
}

# ── Save Figure Helper ─────────────────────────────────────────────────────────
save_figure <- function(fig, base_path, width = 8, height = 6, dpi = 300, formats = c("png", "pdf")) {
  dpi <- max(300, as.numeric(dpi))
  
  # Ensure directory exists
  dir.create(dirname(base_path), recursive = TRUE, showWarnings = FALSE)
  
  if ("png" %in% formats) {
    tryCatch({
      ggsave(paste0(base_path, ".png"), fig,
             width = width, height = height, dpi = dpi, bg = PLOT_BG)
      message("Saved: ", base_path, ".png")
    }, error = function(e) {
      warning("Failed to save PNG: ", e$message)
    })
  }
  
  if ("pdf" %in% formats) {
    tryCatch({
      ggsave(paste0(base_path, ".pdf"), fig,
             width = width, height = height, dpi = dpi, bg = PLOT_BG, device = "pdf")
      message("Saved: ", base_path, ".pdf")
    }, error = function(e) {
      warning("Failed to save PDF: ", e$message)
    })
  }
}

# ── Figure 1: Zero-shot Alignment Bar Chart ───────────────────────────────────
plot_zeroshot_alignment <- function(input_dir, output_dir, layer_name) {
  csv_file <- file.path(input_dir, paste0("zeroshot_alignment_metrics_", layer_name, ".csv"))
  
  if (!file.exists(csv_file)) {
    message("[WARN] Zero-shot alignment CSV not found: ", csv_file)
    return(NULL)
  }
  
  df <- safe_read_csv(csv_file)
  if (is.null(df) || nrow(df) == 0) {
    message("[WARN] No data in zero-shot alignment CSV")
    return(NULL)
  }
  
  df <- df %>% mutate(class_name = factor(class_name, levels = CLASS_ORDER))
  
  message("  Generating zero-shot alignment bar charts...")
  
  # Panel A: Centroid Distance
  p1 <- ggplot(df, aes(x = class_name, y = centroid_distance, fill = class_name)) +
    geom_col(width = 0.6, alpha = 0.9, color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", centroid_distance)), 
              vjust = -0.5, size = 3.5, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = PAPER_CLASS_COLORS) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "A. Centroid Distance (Human vs Plant)", 
         x = NULL, y = "Euclidean Distance") +
    theme_paper() +
    theme(legend.position = "none")
  
  # Panel B: Centroid Cosine Similarity
  p2 <- ggplot(df, aes(x = class_name, y = centroid_cosine, fill = class_name)) +
    geom_col(width = 0.6, alpha = 0.9, color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", centroid_cosine)), 
              vjust = -0.5, size = 3.5, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = PAPER_CLASS_COLORS) +
    scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
    labs(title = "B. Centroid Cosine Similarity", 
         x = NULL, y = "Cosine Similarity") +
    theme_paper() +
    theme(legend.position = "none")
  
  # Panel C: Plant-to-Human NN Distance
  p3 <- ggplot(df, aes(x = class_name, y = plant_to_human_nn_distance, fill = class_name)) +
    geom_col(width = 0.6, alpha = 0.9, color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", plant_to_human_nn_distance)), 
              vjust = -0.5, size = 3.5, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = PAPER_CLASS_COLORS) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "C. Plant → Human Nearest Neighbor Distance", 
         x = NULL, y = "Mean NN Distance") +
    theme_paper() +
    theme(legend.position = "none")
  
  # Panel D: Compactness Comparison
  df_compact <- df %>%
    select(class_name, human_compactness, plant_compactness) %>%
    pivot_longer(cols = c(human_compactness, plant_compactness),
                 names_to = "metric", values_to = "value") %>%
    mutate(
      species = ifelse(metric == "human_compactness", "Human", "Plant"),
      species = factor(species, levels = c("Human", "Plant"))
    )
  
  p4 <- ggplot(df_compact, aes(x = class_name, y = value, fill = species)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.9, 
             color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", value), group = species),
              position = position_dodge(width = 0.7), vjust = -0.5, 
              size = 3, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = c("Human" = PAPER_SPECIES_COLORS["Human"], 
                                  "Plant" = PAPER_SPECIES_COLORS["Plant"])) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "D. Intra-class Compactness Comparison", 
         x = NULL, y = "Mean Intra-class Distance", fill = "Species") +
    theme_paper()
  
  # Save individual panels
  base <- file.path(output_dir, paste0("fig1_zeroshot_alignment_", layer_name))
  save_figure(p1, paste0(base, "_centroid_distance"), width = 5, height = 4)
  save_figure(p2, paste0(base, "_cosine"), width = 5, height = 4)
  save_figure(p3, paste0(base, "_nn_distance"), width = 5, height = 4)
  save_figure(p4, paste0(base, "_compactness"), width = 6, height = 4)
  
  # Composite figure
  if (has_gridExtra) {
    tryCatch({
      composite <- grid.arrange(p1, p2, p3, p4, ncol = 2, 
                                top = "Zero-shot Feature Alignment Metrics")
      save_figure(composite, base, width = 10, height = 8)
    }, error = function(e) {
      warning("Failed to create composite figure: ", e$message)
    })
  }
  
  message("  [OK] Zero-shot alignment figures saved")
}

# ── Figure 2: Few-shot Trajectory Line Chart ──────────────────────────────────
plot_fewshot_trajectory <- function(input_dir, output_dir, layer_name) {
  csv_file <- file.path(input_dir, paste0("fewshot_trajectory_metrics_", layer_name, ".csv"))
  
  if (!file.exists(csv_file)) {
    message("[WARN] Few-shot trajectory CSV not found: ", csv_file)
    return(NULL)
  }
  
  df <- safe_read_csv(csv_file)
  if (is.null(df) || nrow(df) == 0) {
    message("[WARN] No data in few-shot trajectory CSV")
    return(NULL)
  }
  
  df <- df %>% mutate(
    class_name = factor(class_name, levels = CLASS_ORDER),
    shot = as.numeric(shot)
  )
  
  message("  Generating few-shot trajectory line charts...")
  
  metrics <- c("mean_distance_to_human_centroid", "compactness", 
               "separation_margin", "separation_ratio")
  metric_labels <- c(
    "mean_distance_to_human_centroid" = "Distance → Human Centroid",
    "compactness"                     = "Compactness",
    "separation_margin"              = "Separation Margin",
    "separation_ratio"               = "Separation Ratio"
  )
  
  df_long <- df %>%
    select(class_name, shot, all_of(metrics)) %>%
    pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric, levels = metrics, labels = metric_labels))
  
  p <- ggplot(df_long, aes(x = shot, y = value, color = class_name, group = class_name)) +
    geom_line(linewidth = 1, alpha = 0.9) +
    geom_point(size = 3.5, alpha = 1, shape = 16) +
    geom_point(size = 3.5, alpha = 1, shape = 1, color = "#666666", stroke = 0.5) +
    facet_wrap(~ metric, scales = "free_y", ncol = 2) +
    scale_color_manual(values = PAPER_CLASS_COLORS) +
    scale_x_continuous(breaks = c(0, 1, 5, 10), labels = c("0-shot", "1-shot", "5-shot", "10-shot")) +
    labs(title = "Few-shot Trajectory Metrics",
         x = "Shot Count", y = "Value", color = "Modification") +
    theme_paper()
  
  base <- file.path(output_dir, paste0("fig2_fewshot_trajectory_", layer_name))
  save_figure(p, base, width = 10, height = 7)
  
  message("  [OK] Few-shot trajectory figure saved")
}

# ── Figure 3: Gain Heatmap ────────────────────────────────────────────────────
plot_gain_heatmap <- function(input_dir, output_dir, layer_name) {
  csv_file <- file.path(input_dir, paste0("fewshot_trajectory_metrics_", layer_name, ".csv"))
  
  if (!file.exists(csv_file)) {
    message("[WARN] Few-shot trajectory CSV not found: ", csv_file)
    return(NULL)
  }
  
  df <- safe_read_csv(csv_file)
  if (is.null(df) || nrow(df) == 0) {
    message("[WARN] No data in few-shot trajectory CSV")
    return(NULL)
  }
  
  df <- df %>% mutate(class_name = factor(class_name, levels = CLASS_ORDER))
  
  message("  Generating gain heatmap...")
  
  gain_metrics <- c("distance_to_human_centroid_gain", "compactness_gain",
                    "separation_margin_gain", "separation_ratio_gain")
  gain_labels <- c(
    "distance_to_human_centroid_gain" = "Dist→Human Gain",
    "compactness_gain"                = "Compactness Gain",
    "separation_margin_gain"          = "Sep Margin Gain",
    "separation_ratio_gain"           = "Sep Ratio Gain"
  )
  
  df_long <- df %>%
    select(class_name, shot, all_of(gain_metrics)) %>%
    pivot_longer(cols = all_of(gain_metrics), names_to = "metric", values_to = "gain") %>%
    mutate(
      metric = factor(metric, levels = gain_metrics, labels = gain_labels),
      shot_label = factor(shot, levels = c(0, 1, 5, 10), labels = c("0-shot", "1-shot", "5-shot", "10-shot"))
    )
  
  # Compute symmetric limits for diverging scale
  max_abs <- max(abs(df_long$gain), na.rm = TRUE)
  limits <- c(-max_abs, max_abs)
  
  p <- ggplot(df_long, aes(x = metric, y = shot_label, fill = gain)) +
    geom_tile(color = PLOT_BG, linewidth = 1) +
    geom_text(aes(label = sprintf("%.2f", gain), color = ifelse(gain >= 0, "pos", "neg")),
              size = 3.5, fontface = "bold") +
    facet_wrap(~ class_name, nrow = 1) +
    scale_fill_gradient2(
      low = "#6ac6e9", mid = "#F0F0F0", high = "#e4852b",
      midpoint = 0, name = "Gain", limits = limits,
      oob = scales::squish
    ) +
    scale_color_manual(values = c("pos" = "#333333", "neg" = "#333333"), guide = "none") +
    labs(title = "Few-shot Gain Heatmap", x = NULL, y = "Shot Count") +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1, size = rel(0.8)))
  
  base <- file.path(output_dir, paste0("fig3_gain_heatmap_", layer_name))
  save_figure(p, base, width = 12, height = 5)
  
  message("  [OK] Gain heatmap saved")
}

# ── Figure 4: Zero-shot Joint UMAP ───────────────────────────────────────────
plot_zeroshot_joint_umap <- function(input_dir, output_dir, layer_name) {
  csv_file <- file.path(input_dir, paste0("zeroshot_joint_umap_points_", layer_name, ".csv"))
  
  if (!file.exists(csv_file)) {
    message("[WARN] Zero-shot joint UMAP CSV not found: ", csv_file)
    return(NULL)
  }
  
  df <- safe_read_csv(csv_file)
  if (is.null(df) || nrow(df) == 0) {
    message("[WARN] No data in zero-shot joint UMAP CSV")
    return(NULL)
  }
  
  message("  Generating zero-shot joint UMAP plots...")
  
  splits <- split_real_synth(df)
  n_real <- nrow(splits$real)
  n_synth <- nrow(splits$synth)
  message(sprintf("    Real points: %d, Synthetic points: %d", n_real, n_synth))
  
  # By Class UMAP
  p1 <- plot_umap_by_class(df, title = "Zero-shot Joint UMAP by Modification", 
                           use_species_colors = FALSE)
  
  # By Species UMAP
  p2 <- plot_umap_by_species(df, title = "Zero-shot Joint UMAP by Species")
  
  # Save individual UMAP plots
  base <- file.path(output_dir, paste0("fig4_zeroshot_joint_umap_", layer_name))
  save_figure(p1, paste0(base, "_by_class"), width = 7, height = 6)
  save_figure(p2, paste0(base, "_by_species"), width = 7, height = 6)
  
  # Composite
  if (has_gridExtra) {
    tryCatch({
      composite <- grid.arrange(p1, p2, ncol = 2,
                               top = paste0("Zero-shot Joint UMAP (", layer_name, ")"))
      save_figure(composite, base, width = 12, height = 6)
    }, error = function(e) {
      warning("Failed to create composite figure: ", e$message)
    })
  }
  
  message("  [OK] Zero-shot joint UMAP figures saved")
}

# ── Figure 5: Per-modification UMAP ─────────────────────────────────────────
plot_per_mod_umap <- function(input_dir, output_dir, layer_name) {
  mod_files <- list.files(
    input_dir,
    pattern = paste0("^zeroshot_joint_umap_points_", layer_name, "_(Y|m5C|m6A)\\.csv$"),
    full.names = TRUE
  )
  
  if (length(mod_files) == 0) {
    message("[WARN] No per-modification UMAP CSV files found")
    return(NULL)
  }
  
  message("  Generating per-modification UMAP plots...")
  
  for (mod_file in mod_files) {
    mod <- sub(paste0("^.*zeroshot_joint_umap_points_", layer_name, "_"), "", mod_file)
    mod <- sub("\\.csv$", "", mod)
    
    df <- safe_read_csv(mod_file)
    if (is.null(df) || nrow(df) == 0) {
      message("  [SKIP] Empty or invalid file: ", mod_file)
      next
    }
    
    p <- plot_umap_by_species(df, 
                               title = paste0("Zero-shot UMAP - ", mod, " (", layer_name, ")"))
    
    base <- file.path(output_dir, paste0("fig5_umap_", mod, "_", layer_name))
    save_figure(p, base, width = 7, height = 6)
  }
  
  message("  [OK] Per-modification UMAP figures saved")
}

# ── Figure 6: Few-shot Trajectory UMAP ────────────────────────────────────────
plot_fewshot_trajectory_umap <- function(input_dir, output_dir, layer_name) {
  traj_files <- list.files(
    input_dir,
    pattern = paste0("^fewshot_trajectory_umap_points_.*_", layer_name, "\\.csv$"),
    full.names = TRUE
  )
  
  if (length(traj_files) == 0) {
    message("[WARN] No few-shot trajectory UMAP CSV files found")
    return(NULL)
  }
  
  message("  Generating few-shot trajectory UMAP plots...")
  
  for (traj_file in traj_files) {
    class_name <- sub(paste0("^.*fewshot_trajectory_umap_points_"), "", traj_file)
    class_name <- sub(paste0("_", layer_name, "\\.csv$"), "", class_name)
    
    df <- safe_read_csv(traj_file)
    if (is.null(df) || nrow(df) == 0) {
      message("  [SKIP] Empty or invalid file: ", traj_file)
      next
    }

    if ("shot" %in% colnames(df) && any(df$shot == "__INVALID_ARRAY_EXPORT__", na.rm = TRUE)) {
      stop(
        paste0(
          "Invalid few-shot CSV export detected for ", basename(traj_file), ". ",
          "The shot column contains whole-array strings from an older Python export. ",
          "Please rerun zero_shot_fewshot_extract_only.py to regenerate the few-shot CSVs, ",
          "then run this R script again."
        )
      )
    }

    # Repeat human reference points across all shot panels so each panel keeps the same anchor cloud.
    if ("shot" %in% colnames(df) && "species" %in% colnames(df)) {
      df_human <- df %>% filter(species == "Human")
      df_nonhuman <- df %>% filter(species != "Human")
      if (nrow(df_human) > 0) {
        shot_levels <- c("0", "1", "5", "10")
        df_human_expanded <- bind_rows(lapply(shot_levels, function(s) {
          tmp <- df_human
          tmp$shot <- s
          tmp
        }))
        df <- bind_rows(df_human_expanded, df_nonhuman)
      }
    }
    
    message(sprintf("    Processing %s: %d points", class_name, nrow(df)))
    
    # Create plot with shot facets for few-shot trajectory
    p <- plot_umap_by_class(df, 
                            title = paste0("Few-shot Trajectory UMAP: ", class_name, " (", layer_name, ")"),
                            use_species_colors = TRUE,
                            highlight_plant = TRUE,
                            split_by_shot = TRUE)
    
    base <- file.path(output_dir, paste0("fewshot_trajectory_", class_name, "_", layer_name))
    save_figure(p, base, width = 10, height = 8, formats = c("pdf"))
  }
  
  message("  [OK] Few-shot trajectory UMAP figures saved")
}

# ── Figure 7: Gen3 Alignment Bar Chart ────────────────────────────────────────
plot_gen3_alignment <- function(input_dir, output_dir, layer_name) {
  csv_file <- file.path(input_dir, paste0("gen3_alignment_metrics_", layer_name, ".csv"))
  
  if (!file.exists(csv_file)) {
    message("[WARN] Gen3 alignment CSV not found: ", csv_file)
    return(NULL)
  }
  
  df <- safe_read_csv(csv_file)
  if (is.null(df) || nrow(df) == 0) {
    message("[WARN] No data in Gen3 alignment CSV")
    return(NULL)
  }
  
  df <- df %>% filter(class_name %in% CLASS_ORDER) %>%
    mutate(class_name = factor(class_name, levels = CLASS_ORDER))
  
  message("  Generating Gen3 alignment bar charts...")
  
  # Panel A: Centroid Distance
  p1 <- ggplot(df, aes(x = class_name, y = centroid_distance, fill = class_name)) +
    geom_col(width = 0.6, alpha = 0.9, color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", centroid_distance)), 
              vjust = -0.5, size = 3.5, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = PAPER_CLASS_COLORS) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "A. Centroid Distance (Gen3 vs Human)", 
         x = NULL, y = "Euclidean Distance") +
    theme_paper() +
    theme(legend.position = "none")
  
  # Panel B: Cosine Similarity
  p2 <- ggplot(df, aes(x = class_name, y = centroid_cosine, fill = class_name)) +
    geom_col(width = 0.6, alpha = 0.9, color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", centroid_cosine)), 
              vjust = -0.5, size = 3.5, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = PAPER_CLASS_COLORS) +
    scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1))) +
    labs(title = "B. Centroid Cosine Similarity", 
         x = NULL, y = "Cosine Similarity") +
    theme_paper() +
    theme(legend.position = "none")
  
  # Panel C: NN Distance
  p3 <- ggplot(df, aes(x = class_name, y = gen3_to_human_nn_distance, fill = class_name)) +
    geom_col(width = 0.6, alpha = 0.9, color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", gen3_to_human_nn_distance)), 
              vjust = -0.5, size = 3.5, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = PAPER_CLASS_COLORS) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "C. Gen3 → Human Nearest Neighbor Distance", 
         x = NULL, y = "Mean NN Distance") +
    theme_paper() +
    theme(legend.position = "none")
  
  # Panel D: Compactness
  df_compact <- df %>%
    select(class_name, human_compactness, gen3_compactness) %>%
    pivot_longer(cols = c(human_compactness, gen3_compactness),
                 names_to = "metric", values_to = "value") %>%
    mutate(
      species = ifelse(metric == "human_compactness", "Human", "Gen3"),
      species = factor(species, levels = c("Human", "Gen3"))
    )
  
  p4 <- ggplot(df_compact, aes(x = class_name, y = value, fill = species)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.9, 
             color = "#666666", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", value), group = species),
              position = position_dodge(width = 0.7), vjust = -0.5, 
              size = 3, color = TEXT_COLOR, fontface = "bold") +
    scale_fill_manual(values = c("Human" = PAPER_SPECIES_COLORS["Human"], 
                                  "Gen3" = PAPER_SPECIES_COLORS["Gen3"])) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(title = "D. Intra-class Compactness (Gen3 vs Human)", 
         x = NULL, y = "Mean Intra-class Distance", fill = "Species") +
    theme_paper()
  
  # Save individual panels
  base <- file.path(output_dir, paste0("fig7_gen3_alignment_", layer_name))
  save_figure(p1, paste0(base, "_centroid_distance"), width = 5, height = 4)
  save_figure(p2, paste0(base, "_cosine"), width = 5, height = 4)
  save_figure(p3, paste0(base, "_nn_distance"), width = 5, height = 4)
  save_figure(p4, paste0(base, "_compactness"), width = 6, height = 4)
  
  # Composite
  if (has_gridExtra) {
    tryCatch({
      composite <- grid.arrange(p1, p2, p3, p4, ncol = 2, 
                                top = "Gen3 Zero-shot Feature Alignment Metrics")
      save_figure(composite, base, width = 10, height = 8)
    }, error = function(e) {
      warning("Failed to create composite figure: ", e$message)
    })
  }
  
  message("  [OK] Gen3 alignment figures saved")
}

# ── Figure 8: Gen3 UMAP (Standalone) ─────────────────────────────────────────
plot_gen3_umap <- function(input_dir, output_dir, layer_name) {
  csv_file <- file.path(input_dir, paste0("gen3_umap_points_", layer_name, ".csv"))
  
  if (!file.exists(csv_file)) {
    message("[WARN] Gen3 UMAP CSV not found: ", csv_file)
    return(NULL)
  }
  
  df <- safe_read_csv(csv_file)
  if (is.null(df) || nrow(df) == 0) {
    message("[WARN] No data in Gen3 UMAP CSV")
    return(NULL)
  }
  
  message("  Generating Gen3 UMAP plot...")
  
  # Gen3 specific colors (m6A is primary Gen3 color, other classes use class colors)
  gen3_colors <- c(
    "m6A" = PAPER_SPECIES_COLORS["Gen3"],
    "m5C" = PAPER_CLASS_COLORS["m5C"],
    "Y"   = PAPER_CLASS_COLORS["Y"],
    "Unknown" = "#888888"
  )
  gen3_synth_colors <- c(
    "m6A" = PAPER_SPECIES_SYNTH_COLORS["Gen3"],
    "m5C" = PAPER_CLASS_SYNTH_COLORS["m5C"],
    "Y"   = PAPER_CLASS_SYNTH_COLORS["Y"],
    "Unknown" = "#888888"
  )
  
  splits <- split_real_synth(df)
  df_real <- splits$real
  df_synth <- splits$synth
  
  p <- ggplot()
  
  # Add synthetic points first
  if (nrow(df_synth) > 0) {
    p <- p + geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = SYNTHETIC_POINT_SIZE, alpha = SYNTHETIC_POINT_ALPHA, shape = 16
    ) + scale_color_manual(values = gen3_synth_colors, guide = "none")
  }
  
  # Add real points
  if (nrow(df_real) > 0) {
    p <- p + geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = REAL_POINT_SIZE, alpha = REAL_POINT_ALPHA, shape = 16
    ) + scale_color_manual(values = gen3_colors, 
                           name = "Modification",
                           guide = guide_legend(override.aes = list(size = 4, alpha = 0.7)))
  }
  
  p <- p +
    labs(title = paste0("Gen3 UMAP by Class (", layer_name, ")"),
         x = "UMAP 1", y = "UMAP 2") +
    theme_paper()
  
  base <- file.path(output_dir, paste0("fig8_gen3_umap_", layer_name))
  save_figure(p, base, width = 8, height = 6)
  
  message("  [OK] Gen3 UMAP figure saved")
}

# ── Figure 9: Gen3/Human Joint UMAP ──────────────────────────────────────────
plot_gen3_joint_umap <- function(input_dir, output_dir, layer_name) {
  csv_file <- file.path(input_dir, paste0("gen3_joint_umap_points_", layer_name, ".csv"))
  
  if (!file.exists(csv_file)) {
    message("[WARN] Gen3 joint UMAP CSV not found: ", csv_file)
    return(NULL)
  }
  
  df <- safe_read_csv(csv_file)
  if (is.null(df) || nrow(df) == 0) {
    message("[WARN] No data in Gen3 joint UMAP CSV")
    return(NULL)
  }
  
  df <- df %>% filter(class_name %in% CLASS_ORDER)
  
  message("  Generating Gen3/Human joint UMAP plots...")
  
  splits <- split_real_synth(df)
  df_real <- splits$real
  df_synth <- splits$synth
  
  message(sprintf("    Real: %d, Synthetic: %d", nrow(df_real), nrow(df_synth)))
  
  # By Class - Gen3 uses species color for m6A
  gen3_class_colors <- c(
    "m6A" = PAPER_SPECIES_COLORS["Gen3"],
    "m5C" = PAPER_CLASS_COLORS["m5C"],
    "Y"   = PAPER_CLASS_COLORS["Y"]
  )
  gen3_class_synth_colors <- c(
    "m6A" = PAPER_SPECIES_SYNTH_COLORS["Gen3"],
    "m5C" = PAPER_CLASS_SYNTH_COLORS["m5C"],
    "Y"   = PAPER_CLASS_SYNTH_COLORS["Y"]
  )
  
  p1 <- ggplot()
  
  if (nrow(df_synth) > 0) {
    p1 <- p1 + geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = SYNTHETIC_POINT_SIZE, alpha = SYNTHETIC_POINT_ALPHA * 0.5, shape = 16
    ) + scale_color_manual(values = gen3_class_synth_colors, guide = "none")
  }
  
  if (nrow(df_real) > 0) {
    p1 <- p1 + geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = class_name),
      size = REAL_POINT_SIZE * 0.8, alpha = REAL_POINT_ALPHA, shape = 16  # Smaller points for denser plot
    ) + scale_color_manual(values = gen3_class_colors, 
                           name = "Class",
                           guide = guide_legend(override.aes = list(size = 4, alpha = 0.7)))
  }
  
  p1 <- p1 +
    labs(title = paste0("Gen3/Human Joint UMAP by Class (", layer_name, ")"),
         x = "UMAP 1", y = "UMAP 2") +
    theme_paper()
  
  # By Species
  p2 <- ggplot()
  
  if (nrow(df_synth) > 0) {
    p2 <- p2 + geom_point(
      data = df_synth,
      aes(x = umap_x, y = umap_y, color = species),
      size = SYNTHETIC_POINT_SIZE, alpha = SYNTHETIC_POINT_ALPHA * 0.5, shape = 16
    ) + scale_color_manual(values = PAPER_SPECIES_SYNTH_COLORS, guide = "none")
  }
  
  if (nrow(df_real) > 0) {
    p2 <- p2 + geom_point(
      data = df_real,
      aes(x = umap_x, y = umap_y, color = species),
      size = REAL_POINT_SIZE * 0.8, alpha = REAL_POINT_ALPHA, shape = 16
    ) + scale_color_manual(values = PAPER_SPECIES_COLORS, 
                           name = "Species",
                           guide = guide_legend(override.aes = list(size = 4, alpha = 0.7)))
  }
  
  p2 <- p2 +
    labs(title = paste0("Gen3/Human Joint UMAP by Species (", layer_name, ")"),
         x = "UMAP 1", y = "UMAP 2") +
    theme_paper()
  
  # Save only the final requested composite PDF
  base <- file.path(output_dir, paste0("gen3_joint_umap_", layer_name))
  if (has_gridExtra) {
    tryCatch({
      composite <- grid.arrange(p1, p2, ncol = 2,
                                top = paste0("Gen3/Human Joint UMAP (", layer_name, ")"))
      save_figure(composite, base, width = 14, height = 6, formats = c("pdf"))
    }, error = function(e) {
      warning("Failed to create composite figure: ", e$message)
      save_figure(p1, base, width = 8, height = 6, formats = c("pdf"))
    })
  } else {
    save_figure(p1, base, width = 8, height = 6, formats = c("pdf"))
  }
  
  message("  [OK] Gen3/Human joint UMAP figures saved")
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  args <- parse_args()
  input_dir <- args$input_dir
  output_dir <- args$output_dir
  dpi <- args$dpi
  
  if (!dir.exists(input_dir)) {
    stop("Input directory does not exist: ", input_dir)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  message(paste(rep("=", 70), collapse = ""))
  message("PLOT ZERO/FEWSHOT EXPORT - Paper-Ready Visualization")
  message(paste(rep("=", 70), collapse = ""))
  message("Input dir:  ", input_dir)
  message("Output dir: ", output_dir)
  message("DPI:        ", dpi)
  
  # Detect layer name from files
  layer_name <- "gcn_output"
  csv_files <- list.files(input_dir, pattern = "zeroshot_alignment_metrics_.*\\.csv", full.names = TRUE)
  if (length(csv_files) > 0) {
    m <- regmatches(csv_files[1], regexpr("zeroshot_alignment_metrics_([^/]+)\\.csv", csv_files[1]))
    if (length(m) > 0) {
      layer_name <- sub("zeroshot_alignment_metrics_", "", sub("\\.csv$", "", m[1]))
    }
  }
  message("Detected layer name: ", layer_name)
  
  # Track errors
  errors <- list()
  
  # Generate all figures with error trapping
  message("\n── Requested Figure A: Few-shot Trajectory UMAP PDFs ──")
  tryCatch({
    plot_fewshot_trajectory_umap(input_dir, output_dir, layer_name)
  }, error = function(e) {
    errors[[length(errors) + 1]] <<- paste("Fewshot UMAP:", e$message)
    message("  [ERROR] ", e$message)
  })

  message("\n── Requested Figure B: Gen3/Human Joint UMAP PDF ──")
  tryCatch({
    plot_gen3_joint_umap(input_dir, output_dir, layer_name)
  }, error = function(e) {
    errors[[length(errors) + 1]] <<- paste("Gen3 joint UMAP:", e$message)
    message("  [ERROR] ", e$message)
  })
  
  # Report summary
  message(paste(rep("=", 70), collapse = ""))
  if (length(errors) == 0) {
    message("ALL FIGURES GENERATED SUCCESSFULLY!")
  } else {
    message("COMPLETED WITH ", length(errors), " ERROR(S):")
    for (err in errors) {
      message("  - ", err)
    }
  }
  message("Output directory: ", output_dir)
  message(paste(rep("=", 70), collapse = ""))
  
  # Save error log if any
  if (length(errors) > 0) {
    writeLines(unlist(errors), file.path(output_dir, "r_plot_errors.log"))
  }
}

main()
