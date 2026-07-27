import pandas as pd
import numpy as np
import shutil
import json
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any, Tuple
import os
import traceback
import sys
import urllib.parse
import cv2

# Set up logger
logger = logging.getLogger(__name__)


class QCManager:

    def __init__(
        self,
        measurements_folder: str,
        output_folder: str,
        new_quadrant_images: Optional[List[str]] = None,
        min_cells_threshold: int = 10,
        max_cells_threshold: int = 5000,
        port: int = 8888
    ):

        self.measurements_folder = Path(measurements_folder)
        self.output_folder = Path(output_folder)
        self.qc_folder = self.output_folder / "quality_control"
        self.new_quadrant_images = new_quadrant_images or []
        self.min_cells_threshold = min_cells_threshold
        self.max_cells_threshold = max_cells_threshold
        self.port = port
        
        # Ensure directories exist
        self.qc_folder.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.qc_summary = []
        self.images_to_delete = []
        self.deleted_images = []
        
        # Store deleted image names for tracking
        self._deleted_names = set()
    
    def run_quality_control(self, auto_delete: bool = False, interactive: bool = False) -> pd.DataFrame:

        print(f"auto_delete: {auto_delete}")
        print(f"interactive: {interactive}")
        print(f"new_quadrant_images: {self.new_quadrant_images}")
        
        logger.info("Running quality control...")
        
        # Find all _centers.jpg files
        centers_files = list(self.measurements_folder.glob("*_centers.jpg"))
        print(f"Found {len(centers_files)} _centers.jpg files in measurements folder")
        
        if not centers_files:
            logger.info("No _centers.jpg files found for quality control")
            return pd.DataFrame()
        
        # Filter to only new images if specified
        filtered_files = self._filter_new_images(centers_files)
        print(f"Filtered to {len(filtered_files)} new images")
        
        if not filtered_files:
            logger.info("No new images to review for quality control")
            return pd.DataFrame()
        
        logger.info(f"Reviewing {len(filtered_files)} images for quality control")
        print(f"Images to review: {[f.stem.replace('_centers', '') for f in filtered_files]}")
        
        # Process each image
        for centers_path in filtered_files:
            self._process_image(centers_path)
        
        print(f"Processed {len(self.qc_summary)} images, {len(self.images_to_delete)} flagged as bad")
        
        # Create summary DataFrame
        qc_df = pd.DataFrame(self.qc_summary)
        if not qc_df.empty:
            qc_df.to_csv(self.qc_folder / "qc_summary.csv", index=False)
            print(f"QC summary saved to {self.qc_folder / 'qc_summary.csv'}")
        
        # Create HTML report
        if not qc_df.empty:
            print("Creating HTML report...")
            self._create_html_report(qc_df)
        else:
            print("No data for HTML report")
        
        # Auto-delete if requested
        if auto_delete:
            print("Auto-delete enabled - deleting bad images...")
            self._delete_bad_images()
            # Update qc_df after deletion
            qc_df = pd.DataFrame(self.qc_summary)
        
        # Interactive mode - open web server for manual deletion
        print(f"\nDEBUG: Checking interactive mode:")
        print(f"interactive: {interactive}")
        print(f"auto_delete: {auto_delete}")
        print(f"Condition: interactive and not auto_delete = {interactive and not auto_delete}")
        
        if interactive and not auto_delete:
            print("Starting interactive server...")
            self._start_interactive_server(qc_df)
        else:
            print(f"NOT starting interactive server")
            if not interactive:
                print(" Reason: interactive=False")
            if auto_delete:
                print(" Reason: auto_delete=True")
        
        # Print summary
        self._print_summary()
        
        return qc_df
    
    def _filter_new_images(self, centers_files: List[Path]) -> List[Path]:

        if not self.new_quadrant_images:
            print("No new_quadrant_images provided - returning all files")
            return centers_files
        
        # DEBUG: Print what we're filtering
        print(f"new_quadrant_images: {self.new_quadrant_images}")
        
        # The names in new_quadrant_images should already be the full base names
        new_names = set(self.new_quadrant_images)
        
        print(f"Filtering to {len(new_names)} new image names: {list(new_names)[:5]}...")
        
        filtered = []
        for f in centers_files:
            base_name = f.stem.replace('_centers', '')
            if base_name in new_names:
                filtered.append(f)
                print(f"Matched: {base_name}")
        
        print(f"Found {len(filtered)} matching _centers.jpg files")
        return filtered
    
    def _process_image(self, centers_path: Path) -> None:

        base_name = centers_path.stem.replace('_centers', '')
        
        # Check if measurement CSV exists
        csv_path = self.measurements_folder / f"{base_name}_measurements.csv"
        if not csv_path.exists():
            logger.warning(f"Measurement CSV not found for {base_name}")
            return
        
        # Get cell count from CSV
        try:
            df = pd.read_csv(csv_path)
            n_cells = len(df)
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
            return
        
        # Determine if this is a bad image
        is_bad, reason = self._check_quality(n_cells)
        
        # Copy files to QC folder for reference
        self._copy_to_qc_folder(centers_path, csv_path)
        
        # Store summary
        self.qc_summary.append({
            'image_name': base_name,
            'n_cells': n_cells,
            'is_bad': is_bad,
            'reason': reason,
            'centers_image': str(centers_path),
            'csv_path': str(csv_path)
        })
        
        if is_bad:
            self.images_to_delete.append({
                'image_name': base_name,
                'reason': reason,
                'centers_path': centers_path,
                'csv_path': csv_path
            })
    
    def _check_quality(self, n_cells: int) -> Tuple[bool, str]:

        reasons = []
        is_bad = False
        
        if n_cells < self.min_cells_threshold:
            is_bad = True
            reasons.append(f"low_cells ({n_cells} < {self.min_cells_threshold})")
        
        if n_cells > self.max_cells_threshold:
            is_bad = True
            reasons.append(f"too_many_cells ({n_cells} > {self.max_cells_threshold})")
        
        return is_bad, '; '.join(reasons) if reasons else 'good'
    
    def _copy_to_qc_folder(self, centers_path: Path, csv_path: Path) -> None:

        # Copy centers image
        dest_img = self.qc_folder / centers_path.name
        shutil.copy2(centers_path, dest_img)
        
        # Copy measurement CSV
        dest_csv = self.qc_folder / csv_path.name
        shutil.copy2(csv_path, dest_csv)
    
    def _delete_image(self, image_name: str) -> bool:

        # Check if already deleted
        if image_name in self._deleted_names:
            print(f"Image {image_name} already deleted")
            return True
        
        deleted = False
        deleted_files = []
        
        # Look for files using glob patterns
        print(f"\nSearching for files to delete: {image_name}")
        
        # 1. Look for _centers.jpg in measurements folder
        centers_patterns = [
            self.measurements_folder / f"{image_name}_centers.jpg",
            self.measurements_folder / f"{image_name}_centers.png",
            self.measurements_folder / f"{image_name}_centers.jpeg",
        ]
        
        for f in centers_patterns:
            if f.exists():
                try:
                    f.unlink()
                    deleted_files.append(str(f))
                    deleted = True
                    print(f"Deleted: {f.name}")
                except Exception as e:
                    print(f"Failed to delete {f.name}: {e}")
        
        # 2. Look for measurement CSV
        csv_path = self.measurements_folder / f"{image_name}_measurements.csv"
        if csv_path.exists():
            try:
                csv_path.unlink()
                deleted_files.append(str(csv_path))
                deleted = True
                print(f"Deleted: {csv_path.name}")
            except Exception as e:
                print(f"Failed to delete {csv_path.name}: {e}")
        
        # 3. Look for quadrant images in data/chosen_results
        quadrant_dir = Path("data/chosen_results")
        if quadrant_dir.exists():
            for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
                quad_path = quadrant_dir / f"{image_name}{ext}"
                if quad_path.exists():
                    try:
                        quad_path.unlink()
                        deleted_files.append(str(quad_path))
                        deleted = True
                        print(f"Deleted quadrant: {quad_path.name}")
                    except Exception as e:
                        print(f"Failed to delete {quad_path.name}: {e}")
        
        if deleted:
            self._deleted_names.add(image_name)
            self.deleted_images.append(image_name)
            
            # Remove from new_quadrant_images if present
            if hasattr(self, 'new_quadrant_images'):
                self.new_quadrant_images = [
                    img for img in self.new_quadrant_images 
                    if Path(img).stem != image_name
                ]
            
            # Remove from qc_summary
            self.qc_summary = [
                s for s in self.qc_summary 
                if s['image_name'] != image_name
            ]
            
            # Remove from images_to_delete
            self.images_to_delete = [
                i for i in self.images_to_delete 
                if i['image_name'] != image_name
            ]
            
            print(f"Successfully deleted: {image_name} ({len(deleted_files)} files)")
        else:
            print(f"No files found for: {image_name}")
            print(f"   Searched in:")
            print(f"     - {self.measurements_folder}/{image_name}_centers.*")
            print(f"     - {self.measurements_folder}/{image_name}_measurements.csv")
            print(f"     - data/chosen_results/{image_name}.*")
        
        return deleted
    
    def _delete_bad_images(self) -> None:

        if not self.images_to_delete:
            return
        
        logger.info(f"Deleting {len(self.images_to_delete)} bad images...")
        
        for img_info in self.images_to_delete:
            self._delete_image(img_info['image_name'])
    
    def _create_html_report(self, qc_df: pd.DataFrame) -> None:

        if qc_df.empty:
            return
        
        # Get all filenames in QC folder for exact matching
        qc_files = set(os.listdir(self.qc_folder))
        print(f"\nFound {len(qc_files)} files in QC folder")
        
        html_content = self._generate_html_content(qc_df)
        
        report_path = self.qc_folder / "qc_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"QC HTML report saved to: {report_path}")
        print(f"HTML report saved to: {report_path}")
    
    def _generate_html_content(self, qc_df: pd.DataFrame) -> str:

        total = len(qc_df)
        good = len(qc_df[qc_df['is_bad'] == False])
        bad = len(qc_df[qc_df['is_bad'] == True])
        deleted = len(self.deleted_images)
        
        # Get all filenames in QC folder for exact matching
        qc_files = set(os.listdir(self.qc_folder))
        print(f"\nFound {len(qc_files)} files in QC folder")
        
        # Build a mapping of image names to actual filenames
        image_file_map = {}
        for _, row in qc_df.iterrows():
            img_name = row['image_name']
            
            # Find the actual filename in the QC folder
            actual_file = None
            
            # Check if the exact expected file exists
            expected = f"{img_name}_centers.jpg"
            if expected in qc_files:
                actual_file = expected
            else:
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    test_name = f"{img_name}_centers{ext}"
                    if test_name in qc_files:
                        actual_file = test_name
                        break
                
                # If still not found, try without _centers
                if actual_file is None:
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        test_name = f"{img_name}{ext}"
                        if test_name in qc_files:
                            actual_file = test_name
                            break
                
                # Look for any file that starts with this name and contains _centers
                if actual_file is None:
                    for f in qc_files:
                        if f.startswith(img_name) and "_centers" in f:
                            actual_file = f
                            print(f"   Found via pattern match: {actual_file}")
                            break
            
            # If still not found, use the expected name (will show placeholder)
            if actual_file is None:
                print(f"No matching file found for: {img_name}")
                # List files that might be close
                for f in qc_files:
                    if img_name.replace(' ', '') in f.replace(' ', ''):
                        print(f"Possible match: {f}")
                actual_file = expected
            
            # Don't URL encode - use the exact filename
            image_file_map[img_name] = actual_file
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>QC Report - Cell Segmentation Review</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }}
                .card {{ 
                    background: white; 
                    border-radius: 8px; 
                    overflow: hidden; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    opacity: 0;
                    transform: translateY(20px);
                    transition: opacity 0.3s ease, transform 0.3s ease;
                }}
                .card.visible {{
                    opacity: 1;
                    transform: translateY(0);
                }}
                .card img {{ 
                    width: 100%; 
                    height: auto; 
                    display: block;
                    min-height: 200px;
                    background: #f0f0f0;
                }}
                .card-info {{ padding: 10px; }}
                .card-info .name {{ font-weight: bold; font-size: 14px; }}
                .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-top: 5px; }}
                .badge.good {{ background: #4CAF50; color: white; }}
                .badge.bad {{ background: #f44336; color: white; }}
                .badge.info {{ background: #2196F3; color: white; }}
                .filters {{ margin-bottom: 15px; }}
                .filters button {{ 
                    margin-right: 5px; 
                    padding: 5px 10px; 
                    border: 1px solid #ddd; 
                    border-radius: 4px; 
                    cursor: pointer;
                    background: white;
                }}
                .filters button.active {{ background: #2196F3; color: white; border-color: #2196F3; }}
                .instructions {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ffc107; }}
                .cell-count {{ font-size: 12px; color: #666; margin-top: 5px; }}
                .btn-delete {{ 
                    background: #f44336; 
                    color: white; 
                    border: none; 
                    padding: 5px 10px; 
                    border-radius: 4px; 
                    cursor: pointer; 
                    font-size: 12px; 
                }}
                .btn-delete:hover {{ background: #d32f2f; }}
                .btn-deleted {{ 
                    background: #888; 
                    color: white; 
                    border: none; 
                    padding: 5px 10px; 
                    border-radius: 4px; 
                    font-size: 12px; 
                    cursor: not-allowed; 
                }}
                .status-msg {{ margin-left: 10px; font-size: 12px; }}
                .status-msg.success {{ color: green; }}
                .status-msg.error {{ color: red; }}
                
                /* Loading placeholder styles */
                .loading-placeholder {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                    background-size: 200% 100%;
                    animation: loading 1.5s infinite;
                    color: #999;
                    font-family: Arial;
                    font-size: 14px;
                }}
                
                @keyframes loading {{
                    0% {{ background-position: 200% 0; }}
                    100% {{ background-position: -200% 0; }}
                }}
                
                /* Image counter */
                .image-counter {{
                    position: sticky;
                    top: 0;
                    background: rgba(255,255,255,0.95);
                    padding: 10px 20px;
                    z-index: 100;
                    border-bottom: 1px solid #ddd;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    flex-wrap: wrap;
                    gap: 10px;
                }}
                
                .image-counter .stats {{
                    font-size: 14px;
                    color: #666;
                }}
                
                .image-counter .progress {{
                    font-size: 14px;
                    font-weight: bold;
                    color: #2196F3;
                }}
                
                /* Progress bar */
                .progress-bar {{
                    width: 200px;
                    height: 6px;
                    background: #e0e0e0;
                    border-radius: 3px;
                    overflow: hidden;
                    display: inline-block;
                    vertical-align: middle;
                    margin: 0 10px;
                }}
                
                .progress-bar .fill {{
                    height: 100%;
                    background: #4CAF50;
                    transition: width 0.3s ease;
                    width: 0%;
                    border-radius: 3px;
                }}
                
                /* Load more button */
                .load-more {{
                    text-align: center;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .load-more button {{
                    padding: 10px 30px;
                    font-size: 16px;
                    background: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background 0.3s ease;
                }}
                .load-more button:hover {{
                    background: #1976D2;
                }}
                .load-more button:disabled {{
                    background: #ccc;
                    cursor: not-allowed;
                }}
                
                /* Error state for images */
                .image-error {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                    background: #fee;
                    color: #c00;
                    font-family: Arial;
                    font-size: 14px;
                    text-align: center;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Cell Segmentation Quality Control</h1>
            
            <div class="instructions">
                <strong>Instructions:</strong> Review the cell center images below.
                <ul>
                    <li><strong>Good</strong> = Clear, evenly distributed cell centers</li>
                    <li><strong>Bad</strong> = Too few cells, too many cells, or obvious segmentation errors</li>
                    <li>Click <strong>Delete</strong> to remove an image</li>
                    <li>Deleted images will be removed from the dataset</li>
                    <li>Images load as you scroll (lazy loading for performance)</li>
                </ul>
            </div>
            
            <div class="summary">
                <p><strong>Total Images:</strong> {total}</p>
                <p><strong>Good:</strong> {good} | <strong>Bad:</strong> {bad} | <strong>Deleted:</strong> {deleted}</p>
            </div>
            
            <div class="filters">
                <button class="active" onclick="filterImages('all')">All</button>
                <button onclick="filterImages('good')">Good Only</button>
                <button onclick="filterImages('bad')">Bad Only</button>
                <button onclick="filterImages('deleted')">Deleted Only</button>
            </div>
            
            <div class="image-counter">
                <span class="stats">
                    Showing <span id="visible-count">0</span> / {total} images
                </span>
                <span class="progress">
                    <span class="progress-bar">
                        <span class="fill" id="progress-fill"></span>
                    </span>
                    <span id="loading-status">Loading...</span>
                </span>
            </div>
            
            <div class="grid" id="image-grid">
        """
        
        # Add cards for each image with lazy loading
        for idx, (_, row) in enumerate(qc_df.iterrows()):
            img_name = row['image_name']
            n_cells = row['n_cells']
            reason = row['reason']
            is_deleted = img_name in self.deleted_images
            
            # Get the actual filename from the map (not URL encoded)
            actual_file = image_file_map.get(img_name, f"{img_name}_centers.jpg")
            
            status = 'good' if not row['is_bad'] else 'bad'
            if is_deleted:
                status = 'deleted'
            
            # Determine badge
            if is_deleted:
                badge_text = 'Deleted'
                badge_class = 'info'
            elif 'low_cells' in reason:
                badge_text = f'Low cells ({n_cells})'
                badge_class = 'bad'
            elif 'too_many_cells' in reason:
                badge_text = f'Too many cells ({n_cells})'
                badge_class = 'bad'
            else:
                badge_text = f'Good ({n_cells} cells)'
                badge_class = 'good'
            
            # Button
            if is_deleted:
                btn_html = '<span class="btn-deleted">Deleted</span>'
            else:
                btn_html = f'<button class="btn-delete" onclick="deleteImage(\'{img_name}\')">Delete</button>'
            
            # Create a unique ID for this card
            card_id = f"card-{img_name}"
            placeholder_id = f"placeholder-{img_name}"
            img_id = f"img-{img_name}"
            
            html += f"""
                <div class="card" data-status="{status}" id="{card_id}" data-index="{idx}">
                    <div class="loading-placeholder" id="{placeholder_id}">
                        Loading image {idx+1} of {total}...
                    </div>
                    <img 
                        data-src="{actual_file}" 
                        alt="{img_name}" 
                        class="lazy-image"
                        id="{img_id}"
                        style="display:none;"
                        onerror="handleImageError('{img_name}', '{placeholder_id}', '{img_id}')">
                    <div class="card-info">
                        <div class="name">{img_name}</div>
                        <div>
                            <span class="badge {badge_class}">{badge_text}</span>
                        </div>
                        <div class="cell-count">Cells: {n_cells}</div>
                        <div style="margin-top: 8px;">
                            {btn_html}
                            <span class="status-msg" id="status-{img_name}"></span>
                        </div>
                    </div>
                </div>
            """
        
        html += f"""
            </div>
            
            <div class="load-more">
                <button id="load-more-btn" onclick="loadAllImages()">Load All Images</button>
            </div>
            
            <script>
                // Global variables
                let loadedCount = 0;
                const totalImages = {total};
                let observer = null;
                let allLoaded = false;
                
                // Handle image load errors
                function handleImageError(imgName, placeholderId, imgId) {{
                    const placeholder = document.getElementById(placeholderId);
                    const img = document.getElementById(imgId);
                    
                    if (placeholder) {{
                        placeholder.innerHTML = 'Image not found<br><span style="font-size:12px;color:#999;">' + imgName + '</span>';
                        placeholder.style.background = '#fee';
                        placeholder.style.animation = 'none';
                    }}
                    if (img) {{
                        img.style.display = 'none';
                    }}
                    
                    // Still count as loaded for progress
                    loadedCount++;
                    updateCounter();
                }}
                
                // Update the counter and progress bar
                function updateCounter() {{
                    document.getElementById('visible-count').textContent = loadedCount;
                    
                    const progress = (loadedCount / totalImages) * 100;
                    document.getElementById('progress-fill').style.width = Math.min(progress, 100) + '%';
                    
                    if (loadedCount >= totalImages) {{
                        document.getElementById('loading-status').textContent = 'All loaded!';
                        document.getElementById('load-more-btn').disabled = true;
                        document.getElementById('load-more-btn').textContent = 'All images loaded';
                        allLoaded = true;
                        if (observer) {{
                            observer.disconnect();
                        }}
                    }} else {{
                        document.getElementById('loading-status').textContent = 
                            `Loading... ${{Math.round(progress)}}%`;
                    }}
                }}
                
                // Lazy loading with Intersection Observer
                function initLazyLoading() {{
                    observer = new IntersectionObserver((entries) => {{
                        entries.forEach(entry => {{
                            if (entry.isIntersecting && !allLoaded) {{
                                const card = entry.target;
                                const img = card.querySelector('.lazy-image');
                                const placeholder = card.querySelector('.loading-placeholder');
                                
                                if (img && img.dataset && img.dataset.src) {{
                                    // Create a new Image object to load the image
                                    const tempImg = new Image();
                                    tempImg.onload = function() {{
                                        // Image loaded successfully
                                        img.src = img.dataset.src;
                                        img.style.display = 'block';
                                        if (placeholder) {{
                                            placeholder.style.display = 'none';
                                        }}
                                        card.classList.add('visible');
                                        loadedCount++;
                                        updateCounter();
                                        
                                        // Remove the data-src to prevent re-loading
                                        img.dataset.src = '';
                                        observer.unobserve(card);
                                    }};
                                    tempImg.onerror = function() {{
                                        // Image failed to load
                                        handleImageError(
                                            img.alt || 'unknown',
                                            placeholder ? placeholder.id : '',
                                            img.id
                                        );
                                        card.classList.add('visible');
                                        observer.unobserve(card);
                                    }};
                                    
                                    // Start loading the image
                                    tempImg.src = img.dataset.src;
                                }}
                            }}
                        }});
                    }}, {{
                        rootMargin: '100px', // Load images 100px before they enter viewport
                        threshold: 0.01
                    }});
                    
                    // Observe all cards
                    document.querySelectorAll('.card').forEach(card => {{
                        observer.observe(card);
                    }});
                    
                    // Initial counter update
                    updateCounter();
                }}
                
                // Load all remaining images at once
                function loadAllImages() {{
                    if (allLoaded) return;
                    
                    const btn = document.getElementById('load-more-btn');
                    btn.disabled = true;
                    btn.textContent = 'Loading all...';
                    
                    document.querySelectorAll('.card:not(.visible)').forEach(card => {{
                        const img = card.querySelector('.lazy-image');
                        const placeholder = card.querySelector('.loading-placeholder');
                        
                        if (img && img.dataset && img.dataset.src) {{
                            const tempImg = new Image();
                            tempImg.onload = function() {{
                                img.src = img.dataset.src;
                                img.style.display = 'block';
                                if (placeholder) {{
                                    placeholder.style.display = 'none';
                                }}
                                card.classList.add('visible');
                                loadedCount++;
                                updateCounter();
                                img.dataset.src = '';
                                if (observer) {{
                                    observer.unobserve(card);
                                }}
                            }};
                            tempImg.onerror = function() {{
                                handleImageError(
                                    img.alt || 'unknown',
                                    placeholder ? placeholder.id : '',
                                    img.id
                                );
                                card.classList.add('visible');
                                if (observer) {{
                                    observer.unobserve(card);
                                }}
                            }};
                            tempImg.src = img.dataset.src;
                        }}
                    }});
                    
                    setTimeout(() => {{
                        if (loadedCount >= totalImages) {{
                            btn.textContent = 'All images loaded';
                        }} else {{
                            btn.disabled = false;
                            btn.textContent = `Load remaining ${{totalImages - loadedCount}} images`;
                        }}
                    }}, 1000);
                }}
                
                // Filter images by status
                function filterImages(status) {{
                    const cards = document.querySelectorAll('.card');
                    cards.forEach(card => {{
                        if (status === 'all' || card.dataset.status === status) {{
                            card.style.display = 'block';
                        }} else {{
                            card.style.display = 'none';
                        }}
                    }});
                    
                    document.querySelectorAll('.filters button').forEach(btn => btn.classList.remove('active'));
                    event.target.classList.add('active');
                }}
                
                // Delete an image
                function deleteImage(imageName) {{
                    if (!confirm(`Are you sure you want to delete ${{imageName}}? This cannot be undone.`)) {{
                        return;
                    }}
                    
                    const statusSpan = document.getElementById(`status-${{imageName}}`);
                    statusSpan.textContent = 'Deleting...';
                    statusSpan.className = 'status-msg';
                    
                    fetch('/delete', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ image_name: imageName }})
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        const card = document.getElementById(`card-${{imageName}}`);
                        if (!card) return;
                        
                        const badge = card.querySelector('.badge');
                        const btn = card.querySelector('.btn-delete');
                        
                        if (data.success) {{
                            statusSpan.textContent = 'Deleted!';
                            statusSpan.className = 'status-msg success';
                            if (badge) {{
                                badge.textContent = 'Deleted';
                                badge.className = 'badge info';
                            }}
                            if (btn) {{
                                btn.outerHTML = '<span class="btn-deleted">Deleted</span>';
                            }}
                            card.dataset.status = 'deleted';
                            
                            // Refresh the page after a short delay to show updated state
                            setTimeout(() => {{
                                location.reload();
                            }}, 500);
                        }} else {{
                            statusSpan.textContent = (data.message || 'Failed');
                            statusSpan.className = 'status-msg error';
                        }}
                    }})
                    .catch(error => {{
                        statusSpan.textContent = 'Error deleting';
                        statusSpan.className = 'status-msg error';
                        console.error('Error:', error);
                    }});
                }}
                
                // Initialize lazy loading when page loads
                document.addEventListener('DOMContentLoaded', function() {{
                    initLazyLoading();
                }});
                
                // Re-initialize when filter changes (for dynamically shown/hidden cards)
                const originalFilter = filterImages;
                filterImages = function(status) {{
                    originalFilter(status);
                    // Re-observe cards that become visible
                    if (observer) {{
                        document.querySelectorAll('.card:not(.visible)').forEach(card => {{
                            if (card.style.display !== 'none') {{
                                observer.observe(card);
                            }}
                        }});
                    }}
                }};
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _start_interactive_server(self, qc_df: pd.DataFrame) -> None:

        print("STARTING INTERACTIVE SERVER")
        
        if qc_df.empty:
            print("No images to review. Please run QC first.")
            return
        
        # Print debug info
        print(f"Total images: {len(qc_df)}")
        print(f"Bad images: {len(self.images_to_delete)}")
        print(f"QC folder: {self.qc_folder}")
        print(f"HTML report: {self.qc_folder / 'qc_report.html'}")
        
        # Check if HTML report exists
        report_path = self.qc_folder / "qc_report.html"
        if not report_path.exists():
            print(f"HTML report not found at: {report_path}")
            self._create_html_report(qc_df)
        
        # Try multiple import paths
        server_module = None
        try:
            # Try relative import first (when running from pipeline)
            from code.quality_control.qc_server import start_qc_server, stop_qc_server
            server_module = True
            print("Imported qc_server from code.quality_control")
        except ImportError as e:
            print(f"Import from code.quality_control failed: {e}")
            try:
                # Try direct import (when running script directly)
                sys.path.insert(0, str(Path(__file__).parent))
                from qc_server import start_qc_server, stop_qc_server
                server_module = True
                print("Imported qc_server from local directory")
            except ImportError as e2:
                print(f"Local import failed: {e2}")
                try:
                    # Try adding parent directory
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from quality_control.qc_server import start_qc_server, stop_qc_server
                    server_module = True
                    print("Imported qc_server from quality_control")
                except ImportError as e3:
                    print(f"All imports failed: {e3}")
        
        if server_module is None:
            print(f"\nCould not import QC server module.")
            print("Please make sure qc_server.py is in the same directory.")
            print(f"Current file: {__file__}")
            print(f"Looking for: qc_server.py in {Path(__file__).parent}")
            return
        
        # Change to QC folder so relative paths work
        original_dir = os.getcwd()
        os.chdir(self.qc_folder)
        
        print(f"\nServing from: {self.qc_folder}")
        print(f"Found {len(qc_df)} images to review")
        print(f"HTML report: {report_path}")
        print(f"Exists: {report_path.exists()}")
        
        server = None
        try:
            # Start the server - this sets the global _QC_MANAGER
            print("Starting server...")
            server = start_qc_server(self, port=self.port)
            
            # Keep running until interrupted
            print("Server running. Press Ctrl+C to stop.")
            server.serve_forever()
            
        except KeyboardInterrupt:
            print("\n\nStopping QC server...")
        except Exception as e:
            print(f"\nError: {e}")
            traceback.print_exc()
        finally:
            if server:
                try:
                    stop_qc_server(server)
                except:
                    pass
            os.chdir(original_dir)
    
    def _print_summary(self) -> None:

        total = len(self.qc_summary)
        good = len([r for r in self.qc_summary if not r['is_bad']])
        bad = len(self.images_to_delete)
        deleted = len(self.deleted_images)
        
        print("QUALITY CONTROL SUMMARY")
        print(f"Total images reviewed: {total}")
        print(f"Good images: {good}")
        print(f"Bad images flagged: {bad}")
        print(f"Deleted images: {deleted}")
        
        if self.deleted_images:
            print("\nDeleted images:")
            for img in self.deleted_images:
                print(f"  - {img}")
        
        remaining_bad = [img for img in self.images_to_delete if img['image_name'] not in self.deleted_images]
        if remaining_bad:
            print("\nBad images (not deleted):")
            for img in remaining_bad:
                print(f"  - {img['image_name']}: {img['reason']}")
        
        print(f"\nQC Report: {self.qc_folder / 'qc_report.html'}")
    
    def get_deleted_images(self) -> List[str]:
        return self.deleted_images
    
    def get_bad_images(self) -> List[Dict[str, Any]]:
        return self.images_to_delete

def run_quality_control(
    measurements_folder: str,
    output_folder: str,
    new_quadrant_images: Optional[List[str]] = None,
    auto_delete: bool = False,
    interactive: bool = False,
    min_cells_threshold: int = 10,
    max_cells_threshold: int = 5000
) -> QCManager:

    qc_manager = QCManager(
        measurements_folder=measurements_folder,
        output_folder=output_folder,
        new_quadrant_images=new_quadrant_images,
        min_cells_threshold=min_cells_threshold,
        max_cells_threshold=max_cells_threshold
    )
    
    qc_manager.run_quality_control(auto_delete=auto_delete, interactive=interactive)
    
    return qc_manager


if __name__ == "__main__":
    # Test the QC Manager with sample data
    print("Testing QC Manager")
    
    # Use absolute paths
    measurements_folder = Path("d:/jagdeep/results/measurements_all")
    output_folder = Path("d:/jagdeep/results")
    
    # Create a test QC manager
    qc = QCManager(
        measurements_folder=str(measurements_folder),
        output_folder=str(output_folder),
        min_cells_threshold=10,
        max_cells_threshold=5000
    )
    
    # Check if there are real images to process
    centers_files = list(measurements_folder.glob("*_centers.jpg"))
    
    if centers_files:
        print(f"\nFound {len(centers_files)} _centers.jpg files in measurements folder")
        print("Using real data for QC testing")
        
        # Run QC with real data
        qc.run_quality_control(auto_delete=False, interactive=True)
    else:
        print("\nNo _centers.jpg files found. Creating test files...")
        
        # Create test files in the actual directories
        import cv2
        import numpy as np
        
        test_images = [
            {'name': 'test_good_1', 'n_cells': 50, 'is_bad': False},
            {'name': 'test_good_2', 'n_cells': 30, 'is_bad': False},
            {'name': 'test_bad_1', 'n_cells': 5, 'is_bad': True},
            {'name': 'test_bad_2', 'n_cells': 3, 'is_bad': True},
        ]
        
        chosen_dir = Path("data/chosen_results")
        chosen_dir.mkdir(parents=True, exist_ok=True)
        
        for test_img in test_images:
            name = test_img['name']
            n_cells = test_img['n_cells']
            is_bad = test_img['is_bad']
            
            # Create a fake image
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            for i in range(n_cells):
                x = np.random.randint(10, 190)
                y = np.random.randint(10, 190)
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            
            # Save to measurements_all
            centers_path = measurements_folder / f"{name}_centers.jpg"
            cv2.imwrite(str(centers_path), img)
            
            # Create measurement CSV
            df = pd.DataFrame({
                'cell_id': list(range(n_cells)),
                'x_pixels': np.random.randint(10, 190, n_cells),
                'y_pixels': np.random.randint(10, 190, n_cells),
                'area_pixels': np.random.randint(50, 200, n_cells),
            })
            csv_path = measurements_folder / f"{name}_measurements.csv"
            df.to_csv(csv_path, index=False)
            
            # Save quadrant image
            quad_path = chosen_dir / f"{name}.jpg"
            cv2.imwrite(str(quad_path), img)
            
            # Add to QC summary
            reason = 'good' if not is_bad else f'low_cells ({n_cells} < 10)'
            qc.qc_summary.append({
                'image_name': name,
                'n_cells': n_cells,
                'is_bad': is_bad,
                'reason': reason,
                'centers_image': str(centers_path),
                'csv_path': str(csv_path)
            })
            
            if is_bad:
                qc.images_to_delete.append({
                    'image_name': name,
                    'reason': reason,
                    'centers_path': centers_path,
                    'csv_path': csv_path
                })
            
            print(f"  Created: {name} ({n_cells} cells)")
        
        # Create HTML report and run
        qc_df = pd.DataFrame(qc.qc_summary)
        qc._create_html_report(qc_df)
        qc._start_interactive_server(qc_df)