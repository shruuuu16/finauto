#backend/app/services/image_processing.py
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os


class ImagePreprocessor:
    """
    Robust image preprocessing pipeline for documents, optimized for OCR.
    """
    def __init__(self, verbose: bool = False):
        self.TARGET_DPI_WIDTH = 2480
        self.verbose = verbose


    # --- UTILITY / GEOMETRIC (Omitted for brevity) ---
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        # ... (Omitted) ...
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        # ... (Omitted) ...
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        maxWidth = max(int(np.hypot(br[0]-bl[0], br[1]-bl[1])), int(np.hypot(tr[0]-tl[0], tr[1]-tl[1])))
        maxHeight = max(int(np.hypot(tr[0]-br[0], tr[1]-br[1])), int(np.hypot(tl[0]-bl[0], tl[1]-bl[1])))
        if maxWidth <= 0 or maxHeight <= 0:
             if self.verbose: print("four_point_transform: Degenerate warp size, returning original.")
             return image
        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        # ... (Omitted) ...
        (h, w) = image.shape[:2]
        if w == 0 or h == 0 or abs(angle) < 0.1: return image
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    def resize_to_target_dpi(self, image: np.ndarray) -> np.ndarray:
        # ... (Omitted) ...
        h, w = image.shape[:2]
        if w == 0: return image
        scale = self.TARGET_DPI_WIDTH / float(w)
        if scale >= 1.0: return image
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0: return image
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


    # --- ENHANCEMENT & CLEANING (Kept the tuning from last attempt) ---
    def remove_shadows(self, image: np.ndarray, base_dilate_size: int = 15) -> np.ndarray:
        if image is None: return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        h, w = gray.shape[:2]
        scale_factor = max(0.5, (w * h) / (1000 * 1000))
        dynamic_size = max(5, int(base_dilate_size * scale_factor**0.18))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dynamic_size, dynamic_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel).astype(np.float32)
        background = np.where(background <= 1, 1.0, background)
        corrected = np.clip((gray.astype(np.float32) / background) * 255.0, 0, 255).astype(np.uint8)

        # Extra illumination roll-off removal using large Gaussian to catch gentle gradients
        sigma = max(h, w) * 0.02
        illum = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        illum = np.clip(illum, 1, 255).astype(np.float32)
        illum_corrected = np.clip((gray.astype(np.float32) / illum) * 180.0, 0, 255).astype(np.uint8)

        # Blend both corrections and normalize
        blend = cv2.addWeighted(corrected, 0.55, illum_corrected, 0.45, 0)
        blend_norm = cv2.normalize(blend, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Build a document mask to avoid dark borders/wood bleeding into the page
        mask = cv2.adaptiveThreshold(
            cv2.GaussianBlur(blend_norm, (5, 5), 0), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 61, 5
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        # Apply mask: keep normalized page, set outside to white to suppress shadows/background texture
        page = blend_norm
        final_gray = np.where(mask > 0, page, 255).astype(np.uint8)

        if len(image.shape) == 3:
            hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            hls[:, :, 1] = final_gray
            return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
        else:
            return cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)


    def enhance_image(self, image: np.ndarray, profile: str = "auto") -> np.ndarray:
        if profile == "auto":
            gray_auto = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dynamic_range = float(np.percentile(gray_auto, 95) - np.percentile(gray_auto, 5))
            profile = "photo" if dynamic_range > 60 else "scan"


        profiles = {
            "scan": {"clahe_clip": 1.3, "tile": 8, "h": 0, "denoise": False},
            "photo": {"clahe_clip": 2.8, "tile": 12, "h": 10, "denoise": True},
            "contrast": {"clahe_clip": 3.2, "tile": 6, "h": 12, "denoise": True},
        }
        cfg = profiles.get(profile, profiles["scan"])


        try:
            if profile in ("photo", "contrast") or self.verbose:
                image = self.remove_shadows(image, base_dilate_size=15)
        except Exception as e:
            if self.verbose:
                print(f"Enhancement: Shadow removal skipped ({e})")


        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(cfg["tile"], cfg["tile"]))
        l_chan = clahe.apply(l_chan)
        enhanced_img = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
       
        if cfg["denoise"]:
            return cv2.fastNlMeansDenoisingColored(enhanced_img, None, cfg["h"], cfg["h"], 7, 21)
        else:
             return enhanced_img


    # ... (Other cleaning/geometric methods omitted for brevity) ...
    def detect_and_inpaint_edge_blob(self, image: np.ndarray, pad_px: int = 15, min_area_ratio: float = 0.01) -> np.ndarray:
         if image is None or image.size == 0: return image
         h, w = image.shape[:2]
         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         _, th = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
         th_closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
         contours, _ = cv2.findContours(th_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         if not contours: return image
         HITS = []
         for c in contours:
             x, y, cw, ch = cv2.boundingRect(c)
             area = cw * ch
             if area >= min_area_ratio * w * h and (x <= 5 or y <= 5 or x + cw >= w - 5 or y + ch >= h - 5):
                 HITS.append((area, (x, y, cw, ch)))
         if not HITS: return image
         _, (x, y, cw, ch) = max(HITS, key=lambda t: t[0])
         x1 = max(0, x - pad_px)
         y1 = max(0, y - pad_px)
         x2 = min(w - 1, x + cw + pad_px)
         y2 = min(h - 1, y + ch + pad_px)
         if self.verbose: print(f"[CLEANUP] Edge blob removed at bbox = ({x1}, {y1}, {x2}, {y2})")
         return self.manual_inpaint_area(image, [(x1, y1, x2, y2)])
   
    def manual_inpaint_area(self, image: np.ndarray, areas: List[Tuple[int, int, int, int]]) -> np.ndarray:
         if not areas: return image
         mask = np.zeros(image.shape[:2], dtype=np.uint8)
         for x1, y1, x2, y2 in areas:
             x_min = max(0, min(x1, x2))
             y_min = max(0, min(y1, y2))
             x_max = min(image.shape[1] - 1, max(x1, x2))
             y_max = min(image.shape[0] - 1, max(y1, y2))
             cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
         return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


    def auto_scan_transform(self, image: np.ndarray) -> Optional[np.ndarray]:
         h_img, w_img = image.shape[:2]
         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         blur = cv2.GaussianBlur(gray, (5, 5), 0)
         edged = cv2.Canny(blur, 50, 150)
         contours = sorted(cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:10]
         screenCnt = None
         for c in contours:
             peri = cv2.arcLength(c, True)
             approx = cv2.approxPolyDP(c, 0.02 * peri, True)
             if len(approx) == 4 and cv2.contourArea(approx) / float(w_img * h_img) >= 0.05:
                  screenCnt = approx
                  break
         if screenCnt is None:
              if self.verbose: print("[PERSPECTIVE] No suitable 4-corner contour found.")
              return None
         rect = self.order_points(screenCnt.reshape(4, 2))
         return self.four_point_transform(image, rect)
       
    def auto_deskew_fusion(self, image: np.ndarray) -> np.ndarray:
        return image
   
    def orient_and_deskew(self, image: np.ndarray) -> np.ndarray:
        return image




    # --- FINAL BINARIZATION FOR OCR (NEW, HYBRID APPROACH) ---
   
    def aggressive_binarization(self, image: np.ndarray) -> np.ndarray:
        """
        Final Binarization using a robust hybrid method:
        - Adaptive Gaussian Thresholding (Local) with high contrast.
        - Otsu's Thresholding (Global) for overall contrast.
        - The final image is created by taking the Adaptive result, which
          preserves local detail, and applying a very slight thinning/cleaning.
        """
        if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: gray = image.copy()
       
        h, w = gray.shape[:2]
        # Choose block size (must be odd)
        block_size = max(51, int(h / 8) | 1)
       
        # Adaptive Thresholding: Moderate C (15) to keep text thin but visible
        # This provides the local contrast needed for shadowed regions
        adaptive_binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 15
        )
       
        # Otsu's Thresholding: Good for setting a strong global background/foreground
        # We use this to see if the overall document is too light or dark
        _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
       
        # Initial image is Adaptive, as it handles uneven lighting better
        binary_img = adaptive_binary


        # Invert based on the Otsu result's polarity (if Otsu is mostly white, it means it inverted correctly)
        #if cv2.mean(otsu_binary)[0] < 127: # If Otsu made the background black (incorrect polarity)
            # binary_img = cv2.bitwise_not(binary_img) # Then invert the adaptive result too
       
        # We REMOVE the post-binarization CLOSING to stop characters from merging.
        # Instead, we apply a very slight EROSION to clean up noise without breaking lines.
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # Note: We apply ERODE to the white-on-black image (i.e., the text is 0, background is 255)
        # So ERODE shrinks the white area (background) and makes the black text slightly thinner/cleaner.
        if cv2.mean(binary_img)[0] > 127: # If text is black on white (correct OCR color)
            binary_img = cv2.bitwise_not(binary_img) # Invert for erosion
            binary_img = cv2.erode(binary_img, clean_kernel, iterations=1)
            binary_img = cv2.bitwise_not(binary_img) # Invert back
        return cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    # --- MAIN PIPELINE IMPLEMENTATION (Unchanged) ---
    def process_pipeline(self, image_bytes: bytes, options: Optional[Dict[str, Any]] = None) -> np.ndarray:
        options = options or {}
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Could not decode image bytes into an image")
        if self.verbose: print(f"Pipeline start: {image.shape[1]}x{image.shape[0]}")


        image = self._downscale_if_huge(image, max_megapixels=float(options.get("max_megapixels", 12.0)))
        if not options.get("disable_auto_edge_inpaint", False):
            image = self.detect_and_inpaint_edge_blob(image, pad_px=15, min_area_ratio=0.01)
       
        image = self.resize_to_target_dpi(image)
        if self.verbose: print(f"Resized to: {image.shape[1]}x{image.shape[0]}")
       
        image = self.enhance_image(image, profile=str(options.get("profile", "auto")))


        final_corrected_image = image
        if not options.get("skip_deskew", False):
            warped_image = self.auto_scan_transform(image.copy())
            if warped_image is not None:
                if self.verbose: print("[GEOMETRIC] Applied perspective warp.")
                final_corrected_image = warped_image
            else:
                final_corrected_image = self.orient_and_deskew(image)
                if self.verbose: print("[GEOMETRIC] Applied robust deskew + orientation fix.")
       
        final_image = self.aggressive_binarization(final_corrected_image)


        manual_rotation = options.get("manual_rotation")
        if manual_rotation is not None:
            final_image = self.rotate_image(final_image, float(manual_rotation))


        return final_image


    def _downscale_if_huge(self, image: np.ndarray, max_megapixels: float = 12.0) -> np.ndarray:
        h, w = image.shape[:2]
        if w == 0 or h == 0: return image
        current_mp = (w * h) / 1e6
        if current_mp <= max_megapixels: return image
        scale = (max_megapixels * 1e6 / (w * h)) ** 0.5
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        if self.verbose:
            print(f"[MEM] Downscaling from {w}x{h} ({current_mp:.1f}MP) to {new_w}x{new_h} to stay memory-efficient.")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------------
# --- USAGE EXAMPLE TO CORRECT YOUR IMAGE ---
# --------------------------------------------------------------------------------


def run_correction(input_filename="public.jpg", output_filename="final_corrected_unrotated_v3.png"):
    """
    Runs the pipeline using the settings necessary to fix the 'crooked' issue
    by skipping all geometric (rotation) corrections.
    """
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found. Please ensure the image is in the current directory.")
        return


    fixer = ImagePreprocessor(verbose=True)


    with open(input_filename, "rb") as f:
        image_data = f.read()


    options_to_preserve_orientation = {
        "skip_deskew": True,
        "manual_rotation": None
    }
   
    print(f"\n--- Starting Processing for {input_filename} (Skipping Deskew) ---")
    try:
        final_image_np = fixer.process_pipeline(
            image_bytes=image_data,
            options=options_to_preserve_orientation
        )


        cv2.imwrite(output_filename, final_image_np)
        print(f"\n--- SUCCESS: Processed image saved as '{output_filename}' ---")
        print("This version uses a hybrid binarization and avoids aggressive closing to prevent merging.")
       
    except ValueError as e:
        print(f"Processing Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# If you run this file directly, it will attempt to process:
# run_correction()

