import numpy as np
import cv2
from pathlib import Path

def create_adversarial_set(base_dir: Path):
    """
    Generates 'dirty' test cases to challenge the Bulletproof detection logic.
    """
    test_dir = base_dir / "ADVERSARIAL_TEST"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # CASE 1: Keyword Detection (Filename)
    # Should be skipped instantly by pre-parallelization filter
    fake_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(test_dir / "eksik_veri_1990.png"), fake_img)
    cv2.imwrite(str(test_dir / "arizali_analog_2005.png"), fake_img)
    cv2.imwrite(str(test_dir / "iptal_kayit.png"), fake_img)
    
    # CASE 2: Ink Density Detection (Blank Page)
    # Valid name, but nearly zero ink. Should be skipped at density layer.
    cv2.imwrite(str(test_dir / "clean_temp_1920_01.png"), fake_img)
    
    # CASE 3: Signal Variance Detection (Fake Noise/Flat Line)
    # Image has 'ink' but it's just a perfectly straight horizontal line (noise/edge).
    noisy_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    cv2.line(noisy_img, (0, 500), (1000, 500), (0, 0, 255), 2) # Red horizontal line (std=0)
    cv2.imwrite(str(test_dir / "flat_line_precip_1950.png"), noisy_img)
    
    # CASE 4: Small Random Noise (Dust)
    # A few random dots, not enough for a trace.
    dusty_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    for _ in range(20):
        cv2.circle(dusty_img, (np.random.randint(0, 1000), np.random.randint(0, 1000)), 2, (0, 0, 255), -1)
    cv2.imwrite(str(test_dir / "dusty_obs_1960.png"), dusty_img)

    # CASE 5: High Density Horizontal Noise (Grid remnants or thick border)
    # This will pass the density check but should fail the Variance check.
    thick_line_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    for y in [495, 500, 505, 510, 515]: # 5 thick black horizontal lines
        cv2.line(thick_line_img, (0, y), (1000, y), (0, 0, 0), 3)
    cv2.imwrite(str(test_dir / "grid_noise_temp_1977.png"), thick_line_img)

    print(f"Adversarial test set created at {test_dir}")

if __name__ == "__main__":
    root = Path("/Users/yasinkaya/Hackhaton")
    create_adversarial_set(root)
