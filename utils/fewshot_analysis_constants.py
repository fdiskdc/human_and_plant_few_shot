"""
Few-shot Analysis Constants

This module contains all constants used in the few-shot analysis pipeline.
"""

# Target classes and names
TARGET_CLASSES = [5, 8, 9]
CLASS_NAMES = ['Y', 'm5C', 'm6A']
CLASS_NAME_MAP = dict(zip(TARGET_CLASSES, CLASS_NAMES))
SHOT_COUNTS = [0, 1, 5, 10]

# Spatial motif constants
SEQ_LENGTH = 1001
NUCLEOTIDES = ['A', 'C', 'G', 'U']
MIN_REL_POS = -1000
MAX_REL_POS = 1000
REL_RANGE = (MAX_REL_POS - MIN_REL_POS) + 1  # 2001 positions
CENTER_IDX = -MIN_REL_POS
NUC_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# UMAP color pool
UMAP_COLOR_POOL = [
    '#0f82bf',
    '#6ac6e9',
    '#3d4092',
    '#e92633',
    '#e4852b',
    '#fae41e',
    '#0a8648',
    '#83bd55',
    '#b96497',
]

# High-contrast color palette for enhanced visualization
# Primary (real points) and secondary (synthetic points) colors for each modification
HIGH_CONTRAST_MOD_COLORS = {
    'Y': {
        'primary': '#e92633',
        'secondary': '#fae41e',
    },
    'm5C': {
        'primary': '#3d4092',
        'secondary': '#6ac6e9',
    },
    'm6A': {
        'primary': '#0f82bf',
        'secondary': '#b96497',
    },
}

# Gen3 specific colors (Gen3 is treated as different modification from Plant)
HIGH_CONTRAST_GEN3_COLORS = {
    'm6A': {
        'primary': '#0a8648',
        'secondary': '#83bd55',
    },
}

# High-contrast species colors
HIGH_CONTRAST_SPECIES_COLORS = {
    'Human': {
        'primary': '#e4852b',
        'secondary': '#fae41e',
    },
    'Plant': {
        'primary': '#0f82bf',
        'secondary': '#6ac6e9',
    },
    'Gen3': {
        'primary': '#0a8648',
        'secondary': '#83bd55',
    },
}

# Visual enhancement interpolation parameters
INTERPOLATION_PARAMS = {
    'n_synthetic_per_point': 12,
    'jitter_strength': 0.008,
    'n_neighbors_for_interpolation': 5,
}

# Point style parameters
POINT_STYLE_PARAMS = {
    'real_point_size_ratio': 0.22,
    'synthetic_point_size_ratio': 0.08,
    'real_point_alpha': 0.32,
    'synthetic_point_alpha': 0.08,
}

# Morandi color palette (kept for backward compatibility)
MORANDI_CLASS_COLORS = {
    'Y': '#e4852b',
    'm5C': '#3d4092',
    'm6A': '#0f82bf',
}
MORANDI_SPECIES_COLORS = {
    'Human': '#e4852b',
    'Plant': '#0f82bf',
    'Gen3': '#0a8648',
}
MORANDI_NEUTRAL = '#C7C0B7'
MORANDI_GRID = '#E7E0D8'

# Morandi Liquid Color Scheme for spatial motif
MORANDI_COLORS = {
    'A': (95/255, 158/255, 160/255, 0.80),   # Cadet Blue
    'C': (188/255, 143/255, 143/255, 0.75),  # Rosy Brown
    'G': (143/255, 188/255, 143/255, 0.80),  # Dark Sea Green
    'U': (218/255, 165/255, 32/255, 0.78)    # Golden Rod
}

# Default checkpoint path
DEFAULT_CHECKPOINT_PATH = 'logs/old/rna_classification_20260129_164810/checkpoints/epoch_030.pt'
DEFAULT_CONFIG_PATH = 'json/plant_single.json'
