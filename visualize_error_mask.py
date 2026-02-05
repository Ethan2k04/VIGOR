"""
Error Mask Visualization Tool
Visualize reprojection error masks saved as .npz files
"""
import os
import numpy as np
import gradio as gr
import cv2
from typing import List, Tuple, Optional


def load_error_mask(npz_file) -> Tuple[Optional[np.ndarray], str]:
    """
    Load error mask from .npz file.
    
    Returns:
        error_mask: (S, Hp, Wp) array or None if failed
        info_text: Information about the loaded mask
    """
    if npz_file is None:
        return None, "Please upload an .npz file"
    
    try:
        # Handle both file path and Gradio file object
        if isinstance(npz_file, str):
            file_path = npz_file
        else:
            file_path = npz_file.name
        
        # Load .npz file
        data = np.load(file_path)
        
        if "error_mask" not in data:
            available_keys = list(data.keys())
            return None, f"Error: 'error_mask' key not found. Available keys: {available_keys}"
        
        error_mask = data["error_mask"]
        
        # Validate shape
        if error_mask.ndim != 3:
            return None, f"Error: Expected 3D array (S, Hp, Wp), got shape {error_mask.shape}"
        
        S, Hp, Wp = error_mask.shape
        
        # Compute statistics
        valid_mask = np.isfinite(error_mask) & (error_mask >= 0)
        valid_errors = error_mask[valid_mask]
        
        if valid_errors.size > 0:
            mean_err = float(np.mean(valid_errors))
            max_err = float(np.max(valid_errors))
            min_err = float(np.min(valid_errors))
            std_err = float(np.std(valid_errors))
            median_err = float(np.median(valid_errors))
        else:
            mean_err = max_err = min_err = std_err = median_err = 0.0
        
        info = (
            f"✓ Successfully loaded error mask\n"
            f"File: {os.path.basename(file_path)}\n"
            f"Shape: {S} frames × {Hp}×{Wp} patches\n"
            f"Valid patches: {valid_errors.size:,} / {error_mask.size:,}\n"
            f"\n"
            f"Statistics (pixels):\n"
            f"  Mean:   {mean_err:.4f}\n"
            f"  Median: {median_err:.4f}\n"
            f"  Std:    {std_err:.4f}\n"
            f"  Min:    {min_err:.4f}\n"
            f"  Max:    {max_err:.4f}\n"
        )
        
        return error_mask, info
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def error_to_heatmap(
    error_2d: np.ndarray, 
    vmin: Optional[float] = None, 
    vmax: Optional[float] = None,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Convert 2D error map to RGB heatmap.
    
    Args:
        error_2d: (H, W) error values
        vmin: Minimum value for colormap (None = auto)
        vmax: Maximum value for colormap (None = auto)
        colormap: OpenCV colormap constant
    
    Returns:
        heatmap_rgb: (H, W, 3) RGB uint8 heatmap
    """
    error_2d = np.asarray(error_2d, dtype=np.float32)
    
    # Handle invalid values
    valid_mask = np.isfinite(error_2d) & (error_2d >= 0)
    
    if vmin is None:
        vmin = float(np.min(error_2d[valid_mask])) if valid_mask.any() else 0.0
    if vmax is None:
        vmax = float(np.max(error_2d[valid_mask])) if valid_mask.any() else 1.0
    
    # Avoid division by zero
    if vmax <= vmin:
        vmax = vmin + 1e-6
    
    # Normalize to [0, 255]
    normalized = np.zeros_like(error_2d, dtype=np.float32)
    normalized[valid_mask] = (error_2d[valid_mask] - vmin) / (vmax - vmin)
    normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    
    # Mark invalid regions as black
    normalized[~valid_mask] = 0
    
    # Apply colormap
    heatmap_bgr = cv2.applyColorMap(normalized, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Make invalid regions gray
    heatmap_rgb[~valid_mask] = [50, 50, 50]
    
    return heatmap_rgb


def upsample_patch_to_image(
    patch_map: np.ndarray, 
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """
    Upsample patch-level map to target image size.
    
    Args:
        patch_map: (Hp, Wp) or (Hp, Wp, C) array
        target_size: (width, height) target size
        interpolation: OpenCV interpolation method
    
    Returns:
        upsampled: Upsampled array
    """
    if patch_map.ndim == 2:
        return cv2.resize(patch_map, target_size, interpolation=interpolation)
    elif patch_map.ndim == 3:
        return cv2.resize(patch_map, target_size, interpolation=interpolation)
    else:
        raise ValueError(f"Unexpected patch_map shape: {patch_map.shape}")


def visualize_error_masks(
    npz_file,
    frame_idx: int,
    vmin: float,
    vmax: float,
    colormap_choice: str,
    upscale_factor: int,
    show_grid: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Main visualization function.
    
    Returns:
        patch_heatmap: Patch-level heatmap
        upscaled_heatmap: Upscaled heatmap
        info_text: Information text
    """
    error_mask, info = load_error_mask(npz_file)
    
    if error_mask is None:
        return None, None, info
    
    S, Hp, Wp = error_mask.shape
    
    # Validate frame index
    frame_idx = int(frame_idx)
    if frame_idx < 0 or frame_idx >= S:
        frame_idx = min(max(0, frame_idx), S - 1)
    
    # Get error map for selected frame
    error_2d = error_mask[frame_idx]  # (Hp, Wp)
    
    # Choose colormap
    colormap_dict = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "TURBO": cv2.COLORMAP_TURBO,
    }
    colormap = colormap_dict.get(colormap_choice, cv2.COLORMAP_JET)
    
    # Auto-range if vmin >= vmax
    if vmin >= vmax:
        valid_mask = np.isfinite(error_2d) & (error_2d >= 0)
        if valid_mask.any():
            vmin = float(np.min(error_2d[valid_mask]))
            vmax = float(np.max(error_2d[valid_mask]))
        else:
            vmin = 0.0
            vmax = 1.0
    
    # Create patch-level heatmap
    patch_heatmap = error_to_heatmap(error_2d, vmin=vmin, vmax=vmax, colormap=colormap)
    
    # Add grid overlay if requested
    if show_grid and min(Hp, Wp) <= 200:  # Only show grid for reasonable sizes
        patch_heatmap_grid = patch_heatmap.copy()
        # Draw horizontal lines
        for i in range(0, Hp + 1, max(1, Hp // 20)):
            if i < Hp:
                patch_heatmap_grid[i, :] = [255, 255, 255]
        # Draw vertical lines
        for j in range(0, Wp + 1, max(1, Wp // 20)):
            if j < Wp:
                patch_heatmap_grid[:, j] = [255, 255, 255]
        patch_heatmap = patch_heatmap_grid
    
    # Upscale for better visualization
    upscale_factor = int(upscale_factor)
    target_w = Wp * upscale_factor
    target_h = Hp * upscale_factor
    upscaled_heatmap = upsample_patch_to_image(
        patch_heatmap, 
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Update info with frame-specific stats
    valid_mask = np.isfinite(error_2d) & (error_2d >= 0)
    valid_errors = error_2d[valid_mask]
    
    frame_info = info + (
        f"\n"
        f"Frame {frame_idx} statistics:\n"
        f"  Valid patches: {valid_errors.size} / {error_2d.size}\n"
    )
    
    if valid_errors.size > 0:
        frame_info += (
            f"  Mean:   {np.mean(valid_errors):.4f} px\n"
            f"  Median: {np.median(valid_errors):.4f} px\n"
            f"  Max:    {np.max(valid_errors):.4f} px\n"
        )
    
    frame_info += (
        f"\n"
        f"Colormap range: [{vmin:.2f}, {vmax:.2f}] px\n"
        f"Upscale factor: {upscale_factor}× → {target_w}×{target_h} px"
    )
    
    return patch_heatmap, upscaled_heatmap, frame_info


def create_error_mask_gallery(
    npz_file,
    vmin: float,
    vmax: float,
    colormap_choice: str,
    max_frames: int,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """
    Create gallery showing all frames.
    
    Returns:
        gallery_images: List of (image, caption) tuples
        info_text: Information text
    """
    error_mask, info = load_error_mask(npz_file)
    
    if error_mask is None:
        return [], info
    
    S, Hp, Wp = error_mask.shape
    
    # Limit number of frames to display
    max_frames = int(max_frames)
    if S > max_frames:
        frame_indices = np.linspace(0, S - 1, max_frames).round().astype(int)
        info += f"\n⚠ Showing {max_frames} out of {S} frames"
    else:
        frame_indices = np.arange(S)
    
    # Choose colormap
    colormap_dict = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "TURBO": cv2.COLORMAP_TURBO,
    }
    colormap = colormap_dict.get(colormap_choice, cv2.COLORMAP_JET)
    
    # Auto-range if needed
    if vmin >= vmax:
        valid_mask = np.isfinite(error_mask) & (error_mask >= 0)
        if valid_mask.any():
            vmin = float(np.min(error_mask[valid_mask]))
            vmax = float(np.max(error_mask[valid_mask]))
        else:
            vmin = 0.0
            vmax = 1.0
    
    # Create gallery
    gallery_images = []
    
    for idx in frame_indices:
        error_2d = error_mask[idx]
        heatmap = error_to_heatmap(error_2d, vmin=vmin, vmax=vmax, colormap=colormap)
        
        # Upscale for better visibility
        upscale = max(1, 512 // max(Hp, Wp))
        heatmap_up = upsample_patch_to_image(
            heatmap, 
            (Wp * upscale, Hp * upscale),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Compute frame stats
        valid = np.isfinite(error_2d) & (error_2d >= 0)
        if valid.any():
            mean_err = np.mean(error_2d[valid])
            max_err = np.max(error_2d[valid])
        else:
            mean_err = max_err = 0.0
        
        caption = f"Frame {idx} | Mean: {mean_err:.3f}px | Max: {max_err:.3f}px"
        gallery_images.append((heatmap_up, caption))
    
    return gallery_images, info


# Build Gradio interface
with gr.Blocks(title="Error Mask Visualizer") as demo:
    gr.Markdown(
        """
        # 🔥 Reprojection Error Mask Visualizer
        
        Upload an `.npz` error mask file generated by the evaluation script to visualize reprojection errors.
        
        **File format**: `.npz` with `error_mask` key containing `(S, Hp, Wp)` array
        - S: Number of frames
        - Hp, Wp: Patch grid dimensions
        - Values: Reprojection error in pixels
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload
            npz_input = gr.File(
                label="Upload Error Mask (.npz)",
                file_types=[".npz"],
                type="filepath"
            )
            
            gr.Markdown("### Visualization Settings")
            
            with gr.Row():
                vmin_slider = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    value=0.0,
                    step=0.1,
                    label="Color Range Min (px)",
                )
                vmax_slider = gr.Slider(
                    minimum=0.1,
                    maximum=100.0,
                    value=20.0,
                    step=0.1,
                    label="Color Range Max (px)",
                )
            
            colormap_choice = gr.Dropdown(
                choices=["JET", "HOT", "VIRIDIS", "PLASMA", "INFERNO", "MAGMA", "TURBO"],
                value="JET",
                label="Colormap",
            )
            
            gr.Markdown("### Single Frame View")
            
            frame_idx_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                label="Frame Index",
            )
            
            upscale_slider = gr.Slider(
                minimum=1,
                maximum=32,
                value=8,
                step=1,
                label="Upscale Factor",
            )
            
            show_grid_check = gr.Checkbox(
                value=True,
                label="Show Grid Overlay",
            )
            
            visualize_btn = gr.Button("🔄 Visualize Single Frame", variant="primary")
            
            gr.Markdown("### Gallery View")
            
            max_frames_slider = gr.Slider(
                minimum=4,
                maximum=50,
                value=12,
                step=1,
                label="Max Frames to Display",
            )
            
            gallery_btn = gr.Button("🖼️ Generate Gallery", variant="secondary")
        
        with gr.Column(scale=2):
            info_text = gr.Textbox(
                label="File Information",
                lines=15,
                max_lines=20,
            )
            
            gr.Markdown("### Single Frame Visualization")
            
            with gr.Row():
                patch_output = gr.Image(
                    label="Patch-Level Heatmap (Original)",
                    type="numpy",
                )
                upscaled_output = gr.Image(
                    label="Upscaled Heatmap",
                    type="numpy",
                )
            
            gr.Markdown("### Gallery View (All Frames)")
            
            gallery_output = gr.Gallery(
                label="Error Mask Gallery",
                columns=4,
                rows=3,
                height=600,
                preview=True,
            )
    
    # Event handlers
    npz_input.change(
        fn=load_error_mask,
        inputs=[npz_input],
        outputs=[gr.State(), info_text],
    )
    
    visualize_btn.click(
        fn=visualize_error_masks,
        inputs=[
            npz_input,
            frame_idx_slider,
            vmin_slider,
            vmax_slider,
            colormap_choice,
            upscale_slider,
            show_grid_check,
        ],
        outputs=[patch_output, upscaled_output, info_text],
    )
    
    gallery_btn.click(
        fn=create_error_mask_gallery,
        inputs=[
            npz_input,
            vmin_slider,
            vmax_slider,
            colormap_choice,
            max_frames_slider,
        ],
        outputs=[gallery_output, info_text],
    )
    
    # Examples
    gr.Markdown(
        """
        ---
        ### Usage Tips
        
        1. **Upload** an `.npz` error mask file
        2. **Adjust color range** to highlight different error magnitudes
        3. **Single Frame View**: Visualize one frame at a time with detailed upscaling
        4. **Gallery View**: See all frames at once for temporal patterns
        
        **Color Interpretation**: 
        - 🔵 Blue/Dark = Low error (good)
        - 🟡 Yellow/Orange = Medium error
        - 🔴 Red/Bright = High error (bad)
        - ⬛ Gray = Invalid/sky patches
        """
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Error Mask Visualizer")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )