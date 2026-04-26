#!/usr/bin/env python3
"""
Generate SPECTRA app icon with radial scan lines design.
Creates a 1024x1024 PNG with amber radial lines from center on dark background.
"""

from PIL import Image, ImageDraw
import math

# SPECTRA color palette (from HTML design)
BG_DARK = (13, 12, 10)  # #0d0c0a
AMBER = (212, 168, 67)  # #d4a843
AMBER_DIM = (138, 110, 42)  # #8a6e2a

# Icon size
SIZE = 1024
CENTER = SIZE // 2

# Create image
img = Image.new('RGB', (SIZE, SIZE), BG_DARK)
draw = ImageDraw.Draw(img, 'RGBA')

# Draw subtle radial glow
for glow_i in range(3):
    glow_radius = 420 - (glow_i * 80)
    glow_opacity = 6 - (glow_i * 2)
    glow_color = AMBER + (glow_opacity,)
    bbox = [
        CENTER - glow_radius,
        CENTER - glow_radius,
        CENTER + glow_radius,
        CENTER + glow_radius
    ]
    draw.ellipse(bbox, fill=glow_color)

# Draw radial lines (LiDAR scan rays)
num_rays = 36  # Number of rays emanating from center
max_length = 480  # Maximum length of rays
line_width = 6  # Uniform line width

for i in range(num_rays):
    angle = (i / num_rays) * 2 * math.pi

    # Calculate start and end points of ray
    start_x = CENTER + math.cos(angle) * 45  # Start from edge of center dot
    start_y = CENTER + math.sin(angle) * 45
    end_x = CENTER + math.cos(angle) * max_length
    end_y = CENTER + math.sin(angle) * max_length

    # Vary opacity - some rays brighter than others for depth effect
    base_opacity = 200 if i % 3 == 0 else 140 if i % 2 == 0 else 100

    # Draw the ray as a gradient by drawing multiple segments
    segments = 30
    for seg in range(segments):
        progress = seg / segments

        # Calculate segment start and end
        start_dist = 45 + (max_length - 45) * progress
        end_dist = 45 + (max_length - 45) * (progress + 1/segments)

        seg_start_x = CENTER + math.cos(angle) * start_dist
        seg_start_y = CENTER + math.sin(angle) * start_dist
        seg_end_x = CENTER + math.cos(angle) * end_dist
        seg_end_y = CENTER + math.sin(angle) * end_dist

        # Fade opacity as we go outward
        segment_opacity = int(base_opacity * (1 - progress * 0.65))

        color = AMBER + (segment_opacity,)
        draw.line(
            [(seg_start_x, seg_start_y), (seg_end_x, seg_end_y)],
            fill=color,
            width=line_width  # Uniform width
        )

# Draw central bright dot (LiDAR origin point)
# Create radial gradient effect with multiple circles
for r in range(40, 0, -1):
    # Gradient from full amber to transparent
    progress = r / 40.0
    base_opacity = int(255 * (1 - progress * 0.3))
    color = AMBER + (base_opacity,)
    bbox = [
        CENTER - r,
        CENTER - r,
        CENTER + r,
        CENTER + r
    ]
    draw.ellipse(bbox, fill=color)

# Add extra glow around center
for glow_r in range(41, 60):
    glow_progress = (glow_r - 41) / 19.0
    glow_opacity = int(220 * (1 - glow_progress))
    color = AMBER + (glow_opacity,)
    bbox = [
        CENTER - glow_r,
        CENTER - glow_r,
        CENTER + glow_r,
        CENTER + glow_r
    ]
    draw.ellipse(bbox, outline=color, width=1)

# Save the icon
output_path = 'SPECTRALive/SPECTRALive/Assets.xcassets/AppIcon.appiconset/AppIcon.png'
img.save(output_path, 'PNG')
print(f"✅ App icon generated: {output_path}")
print(f"   Size: {SIZE}x{SIZE}px")
print(f"   Style: Radial amber scan lines on dark background")
