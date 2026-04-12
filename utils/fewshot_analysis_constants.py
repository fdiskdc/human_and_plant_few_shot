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

# Morandi color palette
MORANDI_CLASS_COLORS = {
    'Y': '#B58A83',
    'm5C': '#9AAA91',
    'm6A': '#8EA3B0',
}
MORANDI_SPECIES_COLORS = {
    'Human': '#8E8A84',
    'Plant': '#A69C87',
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
