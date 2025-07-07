"""
Command-line interface for pixel8.
"""

import argparse
import sys
from pathlib import Path

from .core import create_pixel_line_art


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate pixel art from images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pixel8 input.jpg 1.5
  pixel8 input.jpg 2.0 --output output.png --no-show
        """
    )
    
    parser.add_argument(
        "image_path",
        help="Path to the input image"
    )
    
    parser.add_argument(
        "pixelation_factor",
        type=float,
        help="Control for pixelation threshold. Higher = fewer pixels"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output image"
    )
    
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the result"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.image_path).exists():
        print(f"Error: Input file '{args.image_path}' not found")
        sys.exit(1)
    
    # Validate pixelation factor
    if args.pixelation_factor <= 0:
        print("Error: pixelation_factor must be positive")
        sys.exit(1)
    
    try:
        create_pixel_line_art(
            image_path=args.image_path,
            pixelation_factor=args.pixelation_factor,
            output_path=args.output,
            show_result=not args.no_show
        )
        
        if args.output:
            print(f"Pixel art saved to: {args.output}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 