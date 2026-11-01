"""MUSeg label ids, in the order used by the released annotations."""

CLASS_NAMES = [
    "background",
    "person",
    "cable",
    "tube",
    "indicator",
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "rail",
    "support equipment",
    "door",
    "tools and materials",
    "rescue equipment",
    "container",
    "metal fixture",
    "anchor",
]

NUM_CLASSES = len(CLASS_NAMES)

# Distinct colors for qualitative dumps. Index 0 is background.
PALETTE = [
    (0, 0, 0),
    (220, 20, 60),
    (255, 215, 0),
    (30, 144, 255),
    (50, 205, 50),
    (255, 140, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 105, 180),
    (154, 205, 50),
    (255, 69, 0),
    (70, 130, 180),
    (255, 20, 147),
    (0, 191, 255),
    (218, 165, 32),
    (127, 255, 212),
]
