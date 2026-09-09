# Optional mask refinement with a small SAM model.
#
# Off by default. At 0.98 confidence the raw YOLO masks were already tight
# enough that refining changed little; this exists for imagery where the
# segmenter is less certain.

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from typing import Tuple, Optional, Union, List
import warnings
import urllib.request

try:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    MOBILESAM_AVAILABLE = True
except ImportError:
    MOBILESAM_AVAILABLE = False
    
try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
except ImportError:
    FASTSAM_AVAILABLE = False


class LightweightSAMRefiner:
    
    def __init__(
        self,
        model_type: str = 'mobilesam',
        device: str = 'cuda',
        checkpoint_path: Optional[str] = None,
        use_point_prompts: bool = True,
        use_box_prompts: bool = True,
        post_process: bool = True,
        close_kernel_size: int = 5,
        open_kernel_size: int = 5,
        min_mask_area: int = 100,
        max_mask_area_ratio: float = 0.95,
        verbose: bool = True
    ):

        self.model_type = model_type.lower()
        self.device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
        self.use_point_prompts = use_point_prompts
        self.use_box_prompts = use_box_prompts
        self.post_process = post_process
        self.close_kernel_size = close_kernel_size
        self.open_kernel_size = open_kernel_size
        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio
        self.verbose = verbose
        
        # Timing statistics
        self.total_inference_time = 0.0
        self.inference_count = 0
        
        # Initialize model
        self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path: Optional[str] = None):
        
        if self.model_type == 'mobilesam':
            if not MOBILESAM_AVAILABLE:
                raise ImportError(
                    "MobileSAM not installed"
                )
            
            # Download MobileSAM checkpoint if not provided
            if checkpoint_path is None:
                checkpoint_path = 'mobile_sam.pt'
                if not Path(checkpoint_path).exists():
                    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
                    urllib.request.urlretrieve(url, checkpoint_path)
            
            # Load MobileSAM
            sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            
            if self.verbose:
                print(f"MobileSAM loaded from {checkpoint_path}")
                
        elif self.model_type == 'fastsam':
            if not FASTSAM_AVAILABLE:
                raise ImportError(
                    "FastSAM not installed"
                )
            
            # Download FastSAM checkpoint if not provided
            if checkpoint_path is None:
                checkpoint_path = 'FastSAM-s.pt'
                
            # Load FastSAM (will auto-download if not found)
            self.predictor = FastSAM(checkpoint_path)
            if self.verbose:
                print(f"FastSAM loaded (model: {checkpoint_path})")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Use 'mobilesam' or 'fastsam'")
        
        # Print memory usage
        if self.verbose and torch.cuda.is_available() and 'cuda' in self.device:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        
    
    def _extract_prompts_from_mask(
        self, 
        mask: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int]]:

        # Ensure binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # Get centroid (point prompt)
        centroid = None
        point_coords = None
        if self.use_point_prompts:
            moments = cv2.moments(mask)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                centroid = (cx, cy)
                point_coords = np.array([[cx, cy]], dtype=np.float32)
        
        # Get bounding box
        box_coords = None
        if self.use_box_prompts:
            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)
            if np.any(rows) and np.any(cols):
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                box_coords = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
        
        return point_coords, box_coords, centroid
    
    def _run_mobilesam(
        self, 
        image: np.ndarray, 
        point_coords: Optional[np.ndarray],
        box_coords: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:

        try:
            # Set image
            self.predictor.set_image(image)
            
            # Prepare prompts
            point_labels = np.array([1], dtype=np.int32) if point_coords is not None else None
            
            # Run prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_coords,
                multimask_output=False  # Single mask output
            )
            
            if masks is not None and len(masks) > 0:
                # Take the first (and only) mask
                mask = masks[0].astype(np.uint8) * 255
                return mask
            else:
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"MobileSAM inference failed: {e}")
            return None
    
    def _run_fastsam(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray],
        box_coords: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        try:
            # FastSAM inference
            results = self.predictor(
                image,
                device=self.device,
                retina_masks=True,
                imgsz=1024,
                conf=0.4,
                iou=0.9,
            )
            
            if len(results) == 0 or results[0].masks is None:
                return None
            
            # Get prompt processor
            from ultralytics.models.fastsam import FastSAMPrompt
            prompt_process = FastSAMPrompt(image, results, device=self.device)
            
            # Apply prompts (prefer box, then point)
            if box_coords is not None:
                # FastSAM expects [x1, y1, x2, y2] format
                ann = prompt_process.box_prompt(bbox=box_coords.tolist())
            elif point_coords is not None:
                # Use point prompt
                ann = prompt_process.point_prompt(
                    points=[point_coords[0].tolist()],
                    pointlabel=[1]
                )
            else:
                # No prompts, use all masks (should not happen)
                ann = prompt_process.everything_prompt()
            
            if ann is not None and len(ann) > 0:
                # Convert to binary mask
                mask = (ann[0] > 0).astype(np.uint8) * 255
                return mask
            else:
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"FastSAM inference failed: {e}")
            return None
    
    def _post_process_mask(
        self, 
        mask: np.ndarray, 
        roi_shape: Tuple[int, int]
    ) -> np.ndarray:

        if not self.post_process:
            return mask
        
        try:
            # Ensure binary
            mask = (mask > 127).astype(np.uint8) * 255
            
            # Morphological closing to fill holes
            if self.close_kernel_size > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (self.close_kernel_size, self.close_kernel_size)
                )
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Morphological opening to smooth boundaries
            if self.open_kernel_size > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.open_kernel_size, self.open_kernel_size)
                )
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Filter by area, keep largest component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            
            if num_labels > 1:  # Background is label 0
                # Get areas of all components (excluding background)
                areas = stats[1:, cv2.CC_STAT_AREA]
                
                if len(areas) > 0:
                    # Keep only the largest component
                    largest_label = np.argmax(areas) + 1  # +1 because we excluded background
                    largest_area = areas[np.argmax(areas)]
                    
                    # Check if largest component is reasonable size
                    roi_area = roi_shape[0] * roi_shape[1]
                    area_ratio = largest_area / roi_area
                    
                    if largest_area >= self.min_mask_area and area_ratio <= self.max_mask_area_ratio:
                        mask = (labels == largest_label).astype(np.uint8) * 255
                    else:
                        # Mask failed quality checks
                        return np.zeros_like(mask)
                else:
                    return np.zeros_like(mask)
            
            return mask
            
        except Exception as e:
            if self.verbose:
                print(f"Post-processing failed: {e}")
            return mask
    
    def refine_with_sam(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        seed_point: Optional[Tuple[int, int]] = None,
        fallback_to_original: bool = True
    ) -> np.ndarray:
        
        start_time = time.time()
        
        # Convert to RGB if needed (MobileSAM expects RGB)
        if self.model_type == 'mobilesam':
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR, convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image  # FastSAM handles BGR
        
        # Extract prompts from mask
        point_coords, box_coords, centroid = self._extract_prompts_from_mask(mask)
        
        # Override with provided seed point if given
        if seed_point is not None:
            point_coords = np.array([[seed_point[0], seed_point[1]]], dtype=np.float32)
        
        # Check if we have any prompts
        if point_coords is None and box_coords is None:
            if self.verbose:
                print("No valid prompts extracted from mask")
            return mask if fallback_to_original else np.zeros_like(mask)
        
        # Run SAM inference
        if self.model_type == 'mobilesam':
            refined_mask = self._run_mobilesam(image_rgb, point_coords, box_coords)
        elif self.model_type == 'fastsam':
            refined_mask = self._run_fastsam(image_rgb, point_coords, box_coords)
        else:
            refined_mask = None
        
        # Fallback if SAM failed
        if refined_mask is None or refined_mask.sum() == 0:
            if self.verbose:
                print("SAM produced no mask, using fallback")
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.inference_count += 1
            return mask if fallback_to_original else np.zeros_like(mask)
        
        # Post-process
        refined_mask = self._post_process_mask(refined_mask, mask.shape)
        
        # Final fallback check
        if refined_mask.sum() == 0:
            if self.verbose:
                print("SAM mask failed quality checks, using fallback")
            refined_mask = mask if fallback_to_original else refined_mask
        
        # Update timing stats
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.inference_count += 1
        
        if self.verbose:
            print(f"SAM refinement: {inference_time*1000:.1f}ms")
        
        return refined_mask
    
    def get_avg_inference_time(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return (self.total_inference_time / self.inference_count) * 1000
    
    def get_gpu_memory_usage(self) -> Tuple[float, float]:

        if torch.cuda.is_available() and 'cuda' in self.device:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return allocated, reserved
        return 0.0, 0.0
    
    def compare_refinement_methods(
        self,
        image: np.ndarray,
        yolo_mask: np.ndarray,
        region_growing_func: callable,
        seed_point: Optional[Tuple[int, int]] = None,
        tolerance: int = 15,
        save_path: Optional[str] = None,
        show_image: bool = False
    ) -> np.ndarray:

        # Get SAM refinement
        sam_start = time.time()
        sam_mask = self.refine_with_sam(image, yolo_mask, seed_point=seed_point)
        sam_time = time.time() - sam_start
        
        # Get region growing refinement
        rg_start = time.time()
        rg_mask = region_growing_func(image, yolo_mask, seed_point=seed_point, tolerance=tolerance)
        rg_time = time.time() - rg_start
        
        # Create visualization
        vis = image.copy()
        
        # Draw original YOLO mask (blue outline)
        yolo_contours, _ = cv2.findContours(
            (yolo_mask > 127).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, yolo_contours, -1, (255, 0, 0), 2)  # Blue
        
        # Draw region growing result (green outline)
        rg_contours, _ = cv2.findContours(
            (rg_mask > 127).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, rg_contours, -1, (0, 255, 0), 2)  # Green
        
        # Draw SAM result (red outline)
        sam_contours, _ = cv2.findContours(
            (sam_mask > 127).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, sam_contours, -1, (0, 0, 255), 2)  # Red
        
        # Add legend
        legend_height = 120
        legend = np.ones((legend_height, vis.shape[1], 3), dtype=np.uint8) * 255
        
        cv2.putText(legend, "BLUE: Original YOLO", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(legend, f"GREEN: Region Growing ({rg_time*1000:.1f}ms)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(legend, f"RED: {self.model_type.upper()} ({sam_time*1000:.1f}ms)", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Combine visualization and legend
        comparison = np.vstack([vis, legend])
        
        
        # GPU memory if available
        if torch.cuda.is_available() and 'cuda' in self.device:
            allocated, reserved = self.get_gpu_memory_usage()
            print(f"\nGPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
        
        # Save if requested
        if save_path is not None:
            cv2.imwrite(save_path, comparison)
            print(f"Comparison saved to: {save_path}")
        
        # Show if requested
        if show_image:
            cv2.imshow("Refinement Comparison", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return comparison


# Convenience function to maintain backward compatibility
def refine_with_lightweight_sam(
    image: np.ndarray,
    mask: np.ndarray,
    model_type: str = 'mobilesam',
    device: str = 'cuda',
    seed_point: Optional[Tuple[int, int]] = None,
    **kwargs
) -> np.ndarray:
    refiner = LightweightSAMRefiner(model_type=model_type, device=device, verbose=False, **kwargs)
    return refiner.refine_with_sam(image, mask, seed_point=seed_point)

