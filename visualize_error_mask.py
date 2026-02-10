"""
Sampling Mask Visualization Tool
Visualize binary sampling masks showing which patches were selected based on VGGT attention
"""
import os
import numpy as np
import gradio as gr
import cv2
from typing import List, Tuple, Optional


def load_sampling_mask(npz_file) -> Tuple[Optional[np.ndarray], str]:
    """
    Load sampling mask from .npz file.
    
    Returns:
        sampling_mask: (S, Hp, Wp) bool array or None if failed
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
        
        if "sampling_mask" not in data:
            available_keys = list(data.keys())
            return None, f"Error: 'sampling_mask' key not found. Available keys: {available_keys}"
        
        sampling_mask = data["sampling_mask"]
        
        # Validate shape
        if sampling_mask.ndim != 3:
            return None, f"Error: Expected 3D array (S, Hp, Wp), got shape {sampling_mask.shape}"
        
        S, Hp, Wp = sampling_mask.shape
        
        # Compute statistics
        total_patches = S * Hp * Wp
        selected_patches = int(np.sum(sampling_mask))
        selection_rate = (selected_patches / total_patches * 100) if total_patches > 0 else 0.0
        
        # Per-frame statistics
        selected_per_frame = np.sum(sampling_mask, axis=(1, 2))
        mean_selected = float(np.mean(selected_per_frame))
        min_selected = int(np.min(selected_per_frame))
        max_selected = int(np.max(selected_per_frame))
        
        info = (
            f"✓ Successfully loaded sampling mask\n"
            f"File: {os.path.basename(file_path)}\n"
            f"Shape: {S} frames × {Hp}×{Wp} patches\n"
            f"Total patches: {total_patches:,}\n"
            f"Selected patches: {selected_patches:,} ({selection_rate:.2f}%)\n"
            f"\n"
            f"Per-frame statistics:\n"
            f"  Mean selected: {mean_selected:.1f}\n"
            f"  Min selected:  {min_selected}\n"
            f"  Max selected:  {max_selected}\n"
            f"\n"
            f"Interpretation:\n"
            f"  True (white) = Selected patch (high attention)\n"
            f"  False (black) = Not selected patch\n"
        )
        
        return sampling_mask, info
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def mask_to_heatmap(
    mask_2d: np.ndarray,
    selected_color: Tuple[int, int, int] = (0, 255, 0),  # Green for selected
    unselected_color: Tuple[int, int, int] = (50, 50, 50),  # Dark gray for unselected
) -> np.ndarray:
    """
    Convert 2D binary mask to RGB visualization.
    
    Args:
        mask_2d: (H, W) bool array
        selected_color: RGB color for True patches
        unselected_color: RGB color for False patches
    
    Returns:
        heatmap_rgb: (H, W, 3) RGB uint8 heatmap
    """
    mask_2d = np.asarray(mask_2d, dtype=bool)
    H, W = mask_2d.shape
    
    # Create RGB image
    heatmap_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Set colors
    heatmap_rgb[mask_2d] = selected_color
    heatmap_rgb[~mask_2d] = unselected_color
    
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


def visualize_sampling_masks(
    npz_file,
    frame_idx: int,
    selected_color_choice: str,
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
    sampling_mask, info = load_sampling_mask(npz_file)
    
    if sampling_mask is None:
        return None, None, info
    
    S, Hp, Wp = sampling_mask.shape
    
    # Validate frame index
    frame_idx = int(frame_idx)
    if frame_idx < 0 or frame_idx >= S:
        frame_idx = min(max(0, frame_idx), S - 1)
    
    # Get mask for selected frame
    mask_2d = sampling_mask[frame_idx]  # (Hp, Wp) bool
    
    # Choose color scheme
    color_schemes = {
        "Green/Gray": ((0, 255, 0), (50, 50, 50)),
        "White/Black": ((255, 255, 255), (0, 0, 0)),
        "Yellow/Blue": ((255, 255, 0), (0, 0, 128)),
        "Red/Gray": ((255, 0, 0), (50, 50, 50)),
    }
    selected_color, unselected_color = color_schemes.get(selected_color_choice, ((0, 255, 0), (50, 50, 50)))
    
    # Create patch-level heatmap
    patch_heatmap = mask_to_heatmap(mask_2d, selected_color=selected_color, unselected_color=unselected_color)
    
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
    num_selected = int(np.sum(mask_2d))
    total_patches = Hp * Wp
    selection_rate = (num_selected / total_patches * 100) if total_patches > 0 else 0.0
    
    frame_info = info + (
        f"\n"
        f"Frame {frame_idx} statistics:\n"
        f"  Selected patches: {num_selected} / {total_patches} ({selection_rate:.2f}%)\n"
        f"\n"
        f"Upscale factor: {upscale_factor}× → {target_w}×{target_h} px"
    )
    
    return patch_heatmap, upscaled_heatmap, frame_info


def create_sampling_mask_gallery(
    npz_file,
    selected_color_choice: str,
    max_frames: int,
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    """
    Create gallery showing all frames.
    
    Returns:
        gallery_images: List of (image, caption) tuples
        info_text: Information text
    """
    sampling_mask, info = load_sampling_mask(npz_file)
    
    if sampling_mask is None:
        return [], info
    
    S, Hp, Wp = sampling_mask.shape
    
    # Limit number of frames to display
    max_frames = int(max_frames)
    if S > max_frames:
        frame_indices = np.linspace(0, S - 1, max_frames).round().astype(int)
        info += f"\n⚠ Showing {max_frames} out of {S} frames"
    else:
        frame_indices = np.arange(S)
    
    # Choose color scheme
    color_schemes = {
        "Green/Gray": ((0, 255, 0), (50, 50, 50)),
        "White/Black": ((255, 255, 255), (0, 0, 0)),
        "Yellow/Blue": ((255, 255, 0), (0, 0, 128)),
        "Red/Gray": ((255, 0, 0), (50, 50, 50)),
    }
    selected_color, unselected_color = color_schemes.get(selected_color_choice, ((0, 255, 0), (50, 50, 50)))
    
    # Create gallery
    gallery_images = []
    
    for idx in frame_indices:
        mask_2d = sampling_mask[idx]
        heatmap = mask_to_heatmap(mask_2d, selected_color=selected_color, unselected_color=unselected_color)
        
        # Upscale for better visibility
        upscale = max(1, 512 // max(Hp, Wp))
        heatmap_up = upsample_patch_to_image(
            heatmap, 
            (Wp * upscale, Hp * upscale),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Compute frame stats
        num_selected = int(np.sum(mask_2d))
        total_patches = Hp * Wp
        selection_rate = (num_selected / total_patches * 100) if total_patches > 0 else 0.0
        
        caption = f"Frame {idx} | Selected: {num_selected}/{total_patches} ({selection_rate:.1f}%)"
        gallery_images.append((heatmap_up, caption))
    
    return gallery_images, info


# Build Gradio interface
with gr.Blocks(title="Sampling Mask Visualizer") as demo:
    gr.Markdown(
        """
        # 🎯 VGGT Attention-Based Sampling Mask Visualizer
        
        Upload an `.npz` sampling mask file to visualize which patches were selected based on VGGT global attention.
        
        **File format**: `.npz` with `sampling_mask` key containing `(S, Hp, Wp)` bool array
        - S: Number of frames
        - Hp, Wp: Patch grid dimensions
        - Values: True = selected patch (high attention), False = not selected
        
        **According to the paper**: This binary mask represents the geometry-aware sampling strategy,
        where patches are selected based on the top N% of VGGT global attention scores.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload
            npz_input = gr.File(
                label="Upload Sampling Mask (.npz)",
                file_types=[".npz"],
                type="filepath"
            )
            
            gr.Markdown("### Visualization Settings")
            
            color_choice = gr.Dropdown(
                choices=["Green/Gray", "White/Black", "Yellow/Blue", "Red/Gray"],
                value="Green/Gray",
                label="Color Scheme (Selected/Unselected)",
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
                    label="Patch-Level Mask (Original)",
                    type="numpy",
                )
                upscaled_output = gr.Image(
                    label="Upscaled Mask",
                    type="numpy",
                )
            
            gr.Markdown("### Gallery View (All Frames)")
            
            gallery_output = gr.Gallery(
                label="Sampling Mask Gallery",
                columns=4,
                rows=3,
                height=600,
                preview=True,
            )
    
    # Event handlers
    npz_input.change(
        fn=load_sampling_mask,
        inputs=[npz_input],
        outputs=[gr.State(), info_text],
    )
    
    visualize_btn.click(
        fn=visualize_sampling_masks,
        inputs=[
            npz_input,
            frame_idx_slider,
            color_choice,
            upscale_slider,
            show_grid_check,
        ],
        outputs=[patch_output, upscaled_output, info_text],
    )
    
    gallery_btn.click(
        fn=create_sampling_mask_gallery,
        inputs=[
            npz_input,
            color_choice,
            max_frames_slider,
        ],
        outputs=[gallery_output, info_text],
    )
    
    # Examples
    gr.Markdown(
        """
        ---
        ### Usage Tips
        
        1. **Upload** an `.npz` sampling mask file
        2. **Choose color scheme** to visualize selected vs unselected patches
        3. **Single Frame View**: Visualize one frame at a time with detailed upscaling
        4. **Gallery View**: See all frames at once to observe temporal consistency
        
        **Color Interpretation**: 
        - 🟢 Green/Bright = Selected patch (high VGGT attention)
        - ⬛ Gray/Dark = Unselected patch (low attention or filtered out)
        
        **Paper Context**: These masks show the geometry-aware sampling strategy described in Section 4.1,
        where only the top N% patches based on VGGT global attention are used for reprojection error computation.
        """
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Sampling Mask Visualizer")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )