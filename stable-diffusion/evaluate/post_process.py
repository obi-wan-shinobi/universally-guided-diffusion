from PIL import __version__ as PILLOW_VERSION, Image
import sys

def split_image(input_path: str, out1_path: str, out2_path: str) -> None:
    """
    Open an image, print its dimensions, split it horizontally into two equal parts,
    and save each part to the specified output paths.
    """
    # Load image
    img = Image.open(input_path)
    
    # Get dimensions
    width, height = img.size
    print(f"Image size: {width} x {height}")
    
    # Compute midpoint for horizontal split
    mid = width // 2
    
    # Crop left and right halves
    left = img.crop((0, 0, mid, height))
    right = img.crop((mid, 0, width, height))
    
    # Save each half
    left.save(out1_path)
    right.save(out2_path)

if __name__ == "__main__":
    
    # Expect: python post_process.py <input_image> <output_left> <output_right>
    if len(sys.argv) != 4:
        print("Usage: python split.py <input_image> <output_left> <output_right>")
        sys.exit(1)
    
    input_image, output_left, output_right = sys.argv[1], sys.argv[2], sys.argv[3]
    split_image(input_image, output_left, output_right)