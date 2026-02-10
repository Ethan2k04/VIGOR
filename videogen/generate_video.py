"""
Video Generation System for VGGT-DPO
Generates videos from captions using CausVid (Bidirectional or Autoregressive)
Supports reading from all_captions.json and processing specific prompt index ranges
"""

import json
import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
from omegaconf import OmegaConf
from diffusers.utils import export_to_video

# Import CausVid modules
from causvid.models.wan.bidirectional_inference import BidirectionalInferencePipeline


# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """Load and manage configuration from JSON file"""
    
    def __init__(self, config_path: str = "videogen/config/config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get('model', {})
    
    def get_generation_config(self) -> Dict[str, Any]:
        return self.config.get('generation', {})
    
    def get_input_config(self) -> Dict[str, Any]:
        return self.config.get('input', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        return self.config.get('output', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        return self.config.get('processing', {})


# ============================================================================
# Caption Loader (Modified to read all_captions.json)
# ============================================================================

class CaptionLoader:
    """Load captions from all_captions.json"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.input_config = config_loader.get_input_config()
        self.all_captions = None
        self.flattened_captions = None
    
    def load_all_captions(self) -> Dict[str, List[Dict]]:
        """Load all captions from all_captions.json"""
        input_dir = Path(self.input_config['input_dir'])
        all_captions_file = input_dir / self.input_config.get('all_captions_file', 'all_captions.json')
        
        if not all_captions_file.exists():
            raise FileNotFoundError(f"all_captions.json not found: {all_captions_file}")
        
        logging.info(f"Loading captions from {all_captions_file}")
        
        with open(all_captions_file, 'r', encoding='utf-8') as f:
            self.all_captions = json.load(f)
        
        # Log statistics
        total_captions = sum(len(captions) for captions in self.all_captions.values())
        logging.info(f"Loaded {total_captions} total captions across {len(self.all_captions)} categories")
        for category, captions in self.all_captions.items():
            logging.info(f"  - {category}: {len(captions)} captions")
        
        return self.all_captions
    
    def get_flattened_captions(self) -> List[Dict]:
        """
        Get all captions as a flattened list with global indices
        Each caption will have an additional 'global_index' field
        """
        if self.all_captions is None:
            self.load_all_captions()
        
        if self.flattened_captions is not None:
            return self.flattened_captions
        
        flattened = []
        global_index = 0
        
        for category, captions in self.all_captions.items():
            for caption_data in captions:
                # Add global index and category to each caption
                caption_with_index = caption_data.copy()
                caption_with_index['global_index'] = global_index
                caption_with_index['category'] = category
                flattened.append(caption_with_index)
                global_index += 1
        
        self.flattened_captions = flattened
        logging.info(f"Flattened {len(flattened)} total captions with global indices 0-{len(flattened)-1}")
        
        return flattened
    
    def get_captions_by_range(self, start_index: int, end_index: int) -> List[Dict]:
        """
        Get captions within a specific index range (inclusive)
        
        Args:
            start_index: Starting index (inclusive)
            end_index: Ending index (inclusive)
        
        Returns:
            List of caption dictionaries within the range
        """
        flattened = self.get_flattened_captions()
        
        if start_index < 0 or end_index >= len(flattened):
            raise ValueError(
                f"Index range [{start_index}, {end_index}] out of bounds. "
                f"Valid range: [0, {len(flattened)-1}]"
            )
        
        if start_index > end_index:
            raise ValueError(f"start_index ({start_index}) must be <= end_index ({end_index})")
        
        selected_captions = flattened[start_index:end_index+1]
        
        logging.info(f"Selected {len(selected_captions)} captions from index {start_index} to {end_index}")
        
        return selected_captions
    
    def get_total_caption_count(self) -> int:
        """Get total number of captions"""
        flattened = self.get_flattened_captions()
        return len(flattened)


# ============================================================================
# Video Generator (CausVid)
# ============================================================================

class VideoGenerator:
    """Generate videos using CausVid"""
    
    def __init__(self, config_loader: ConfigLoader, device: str = "cuda:0"):
        self.config_loader = config_loader
        self.device = device
        
        self.model_config = config_loader.get_model_config()
        self.gen_config = config_loader.get_generation_config()
        
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load CausVid pipeline"""
        logging.info(f"Initializing CausVid pipeline on device: {self.device}...")
        
        # 设置当前设备
        if self.device.startswith('cuda'):
            device_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            torch.cuda.set_device(device_id)
        
        # Load CausVid YAML config
        causvid_config_path = self.model_config.get('causvid_config_path', 
                                                     'configs/wan_bidirectional_inference.yaml')
        config = OmegaConf.load(causvid_config_path)
        
        # Initialize pipeline with specified device
        self.pipeline = BidirectionalInferencePipeline(config, device=self.device)
        
        # Load checkpoint
        checkpoint_folder = self.model_config['checkpoint_folder']
        checkpoint_path = os.path.join(checkpoint_folder, "model.pt")
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")['generator']
        self.pipeline.generator.load_state_dict(state_dict)
        
        # Move to device
        self.pipeline = self.pipeline.to(device=self.device, dtype=torch.bfloat16)
        
        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_grad_enabled(False)
        
        logging.info(f"CausVid pipeline loaded successfully on {self.device}")
       
    def generate_video(
        self,
        prompt: str,
        seed: int,
        output_path: str
    ) -> bool:
        """
        Generate a single video from prompt
        
        Args:
            prompt: Text prompt
            seed: Random seed
            output_path: Path to save video
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logging.debug(f"Generating video with seed {seed} on {self.device}")
            logging.debug(f"Prompt: {prompt[:100]}...")
            
            # Get video dimensions from config
            frame_num = self.gen_config.get('frame_num', 21)
            height = self.gen_config.get('height', 60)
            width = self.gen_config.get('width', 104)
            fps = self.gen_config.get('fps', 16)
            
            # Generate random noise on the specified device
            noise = torch.randn(
                1, frame_num, 16, height, width,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                dtype=torch.bfloat16,
                device=self.device
            )
            
            # Run inference
            video = self.pipeline.inference(
                noise=noise,
                text_prompts=[prompt]
            )[0].permute(0, 2, 3, 1).cpu().numpy()
            
            # Save video
            export_to_video(video, output_path, fps=fps)
            
            logging.debug(f"Video saved to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error generating video with seed {seed}:")
            logging.error(f"  Error type: {type(e).__name__}")
            logging.error(f"  Error message: {str(e)}")
            logging.error(f"  Output path: {output_path}")
            
            import traceback
            logging.error("Full traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    logging.error(f"  {line}")
            
            return False


# ============================================================================
# Output Manager (MODIFIED)
# ============================================================================

class OutputManager:
    """Manage output directory structure and metadata"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.output_config = config_loader.get_output_config()
        self.processing_config = config_loader.get_processing_config()
        
        self.output_dir = Path(self.output_config['output_dir'])
        self.save_format = self.output_config.get('save_format', 'mp4')
        self.organize_by_category = self.output_config.get('organize_by_category', True)
        
        self._setup_output_dir()
    
    def _setup_output_dir(self):
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {self.output_dir}")
    
    def get_video_path(
        self,
        global_index: int,
        category: str,
        sample_index: int
    ) -> Path:
        """
        Get output path for a video
        
        Each prompt gets its own folder named prompt{global_index:05d}
        Videos inside are named seed{sample_index:02d}.mp4
        """
        
        if self.organize_by_category:
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
        else:
            category_dir = self.output_dir
        
        # Create prompt-specific folder
        prompt_folder = category_dir / f"prompt{global_index:05d}"
        prompt_folder.mkdir(exist_ok=True)
        
        # Video filename is just the seed number
        filename = f"seed{sample_index:02d}.{self.save_format}"
        
        return prompt_folder / filename
    
    def save_metadata(
        self,
        global_index: int,
        category: str,
        caption_data: Dict,
        results: List[bool],
        seeds: List[int]
    ):
        """
        Save metadata for generated videos
        
        Metadata is saved inside the prompt folder as metadata.json
        """
        
        if not self.processing_config.get('save_metadata', True):
            return
        
        if self.organize_by_category:
            category_dir = self.output_dir / category
        else:
            category_dir = self.output_dir
        
        # Metadata goes inside the prompt folder
        prompt_folder = category_dir / f"prompt{global_index:05d}"
        prompt_folder.mkdir(exist_ok=True)
        
        metadata = {
            'global_index': global_index,
            'category': category,
            'scene_type': caption_data.get('scene_type', ''),
            'caption': caption_data.get('caption', ''),
            'camera_movement': caption_data.get('camera_movement', ''),
            'camera_movement_type': caption_data.get('camera_movement_type', ''),
            'file_path': caption_data.get('file_path', ''),
            'dataset': caption_data.get('dataset', ''),
            'is_video': caption_data.get('is_video', False),
            'samples_generated': sum(results),
            'total_samples': len(results),
            'success_rate': sum(results) / len(results) if results else 0,
            'seeds_used': seeds,
            'video_files': [
                f"seed{i:02d}.{self.save_format}" 
                for i in range(len(results))
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = prompt_folder / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============================================================================
# Generation Pipeline
# ============================================================================

class GenerationPipeline:
    """Main pipeline for video generation"""
    
    def __init__(self, config_path: str, device: str = "cuda:0"):
        self.config_loader = ConfigLoader(config_path)
        self.caption_loader = CaptionLoader(self.config_loader)
        self.video_generator = VideoGenerator(self.config_loader, device)
        self.output_manager = OutputManager(self.config_loader)
        
        self.gen_config = self.config_loader.get_generation_config()
        self.processing_config = self.config_loader.get_processing_config()
    
    def generate_for_single_prompt(
        self,
        caption_data: Dict
    ) -> List[bool]:
        """
        Generate videos for a single prompt with multiple seeds
        
        Args:
            caption_data: Caption dictionary with all metadata
        
        Returns:
            List of success flags for each generated video
        """
        global_index = caption_data['global_index']
        category = caption_data['category']
        prompt = caption_data.get('caption', '')
        
        samples_per_prompt = self.gen_config.get('samples_per_prompt', 10)
        base_seed = self.gen_config.get('base_seed_start', 0)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing prompt {global_index}")
        logging.info(f"Category: {category}")
        logging.info(f"Scene: {caption_data.get('scene_type', 'N/A')}")
        logging.info(f"Camera: {caption_data.get('camera_movement', 'N/A')}")
        logging.info(f"Caption: {prompt[:100]}...")
        logging.info(f"{'='*60}")
        
        results = []
        seeds = []
        
        for sample_idx in tqdm(range(samples_per_prompt), 
                               desc=f"Prompt {global_index}",
                               leave=False):
            # Each prompt uses the same seed sequence (0-9 by default)
            seed = base_seed + sample_idx
            seeds.append(seed)
            
            output_path = self.output_manager.get_video_path(
                global_index, category, sample_idx
            )
            
            success = self.video_generator.generate_video(
                prompt=prompt,
                seed=seed,
                output_path=str(output_path)
            )
            
            results.append(success)
        
        # Save metadata
        self.output_manager.save_metadata(
            global_index, category, caption_data, results, seeds
        )
        
        success_count = sum(results)
        total_count = len(results)
        logging.info(
            f"Prompt {global_index}: {success_count}/{total_count} videos generated successfully"
        )
        
        return results
    
    def generate_by_range(
        self,
        start_index: int,
        end_index: int
    ):
        """
        Generate videos for prompts in a specific index range
        
        Args:
            start_index: Starting prompt index (inclusive)
            end_index: Ending prompt index (inclusive)
        """
        logging.info(f"\n{'#'*60}")
        logging.info(f"Processing prompt range: [{start_index}, {end_index}]")
        logging.info(f"{'#'*60}\n")
        
        # Get captions in range
        captions = self.caption_loader.get_captions_by_range(start_index, end_index)
        
        logging.info(f"Will process {len(captions)} prompts")
        
        # Track statistics
        all_results = {}
        total_success = 0
        total_attempted = 0
        
        # Process each prompt
        for caption_data in tqdm(captions, desc="Overall Progress"):
            try:
                results = self.generate_for_single_prompt(caption_data)
                
                global_index = caption_data['global_index']
                all_results[global_index] = results
                
                total_success += sum(results)
                total_attempted += len(results)
                
            except Exception as e:
                global_index = caption_data.get('global_index', 'unknown')
                logging.error(f"Error processing prompt {global_index}: {e}")
                import traceback
                traceback.print_exc()
                
                if not self.processing_config.get('continue_on_error', True):
                    raise
        
        # Save summary
        self._save_range_summary(start_index, end_index, all_results, captions)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Range [{start_index}, {end_index}] complete!")
        logging.info(f"Total videos generated: {total_success}/{total_attempted}")
        logging.info(f"Success rate: {total_success/total_attempted*100:.2f}%")
        logging.info(f"{'='*60}\n")
    
    def generate_all(self):
        """Generate videos for all prompts"""
        total_count = self.caption_loader.get_total_caption_count()
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing ALL prompts: [0, {total_count-1}]")
        logging.info(f"{'='*60}\n")
        
        self.generate_by_range(0, total_count - 1)
    
    def _save_range_summary(
        self,
        start_index: int,
        end_index: int,
        results: Dict[int, List[bool]],
        captions: List[Dict]
    ):
        """Save summary for a range of prompts"""
        
        summary = {
            'range': {
                'start_index': start_index,
                'end_index': end_index,
                'total_prompts': len(captions)
            },
            'statistics': {
                'total_videos_attempted': sum(len(r) for r in results.values()),
                'total_videos_successful': sum(sum(r) for r in results.values()),
                'prompts_processed': len(results),
                'overall_success_rate': (
                    sum(sum(r) for r in results.values()) / 
                    sum(len(r) for r in results.values())
                ) if results else 0
            },
            'prompt_results': {
                idx: {
                    'success_count': sum(res),
                    'total_count': len(res),
                    'success_rate': sum(res) / len(res) if res else 0
                }
                for idx, res in results.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.output_manager.output_dir / f"summary_range_{start_index}_{end_index}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Range summary saved to {summary_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(f"videogen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate videos from all_captions.json using CausVid'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='videogen/config/config.json',
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--start-index',
        type=int,
        default=None,
        help='Starting prompt index (inclusive). Example: --start-index 0'
    )
    
    parser.add_argument(
        '--end-index',
        type=int,
        default=None,
        help='Ending prompt index (inclusive). Example: --end-index 640'
    )
    
    parser.add_argument(
        '--prompt-index',
        type=int,
        default=None,
        help='Process only a single prompt at this index'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='GPU device to use (e.g., cuda:0, cuda:1, or cpu). Default: cuda:0'
    )

    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show caption statistics and exit'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run: show what would be processed without generating videos'
    )
    
    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    args = parse_args()
    
    # Determine device
    if args.gpu_id is not None:
        device = f"cuda:{args.gpu_id}"
    else:
        device = args.device
    
    # Validate device
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            logging.error("CUDA is not available, falling back to CPU")
            device = "cpu"
        else:
            device_id = int(device.split(':')[1]) if ':' in device else 0
            if device_id >= torch.cuda.device_count():
                logging.error(f"GPU {device_id} not available. Available: 0-{torch.cuda.device_count()-1}")
                return
            gpu_name = torch.cuda.get_device_name(device_id)
            print(f"Using GPU {device_id}: {gpu_name}")
    
    setup_logging()
    
    # Create pipeline
    pipeline = GenerationPipeline(args.config, device=device)
    
    # Show statistics
    if args.show_stats:
        total_count = pipeline.caption_loader.get_total_caption_count()
        all_captions = pipeline.caption_loader.all_captions
        
        print(f"\n{'='*60}")
        print(f"Caption Statistics")
        print(f"{'='*60}")
        print(f"Total prompts: {total_count}")
        print(f"\nBreakdown by category:")
        for category, captions in all_captions.items():
            print(f"  - {category}: {len(captions)} prompts")
        print(f"\nValid index range: [0, {total_count-1}]")
        print(f"{'='*60}\n")
        
        # Show example distribution for multi-machine setup
        if total_count > 0:
            print("Example: Distributing across 4 machines:")
            chunk_size = (total_count + 3) // 4
            for i in range(4):
                start = i * chunk_size
                end = min((i + 1) * chunk_size - 1, total_count - 1)
                print(f"  Machine {i+1}: --start-index {start} --end-index {end} ({end-start+1} prompts)")
            print()
        
        return
    
    # Dry run
    if args.dry_run:
        total_count = pipeline.caption_loader.get_total_caption_count()
        
        if args.prompt_index is not None:
            print(f"\nDry run: Would process prompt {args.prompt_index}")
            captions = pipeline.caption_loader.get_captions_by_range(
                args.prompt_index, args.prompt_index
            )
            for cap in captions:
                print(f"  Index: {cap['global_index']}")
                print(f"  Category: {cap['category']}")
                print(f"  Caption: {cap['caption'][:100]}...")
        
        elif args.start_index is not None and args.end_index is not None:
            print(f"\nDry run: Would process range [{args.start_index}, {args.end_index}]")
            count = args.end_index - args.start_index + 1
            print(f"  Total prompts: {count}")
        
        else:
            print(f"\nDry run: Would process ALL prompts [0, {total_count-1}]")
            print(f"  Total prompts: {total_count}")
        
        print()
        return
    
    # Generate videos
    if args.prompt_index is not None:
        # Process single prompt
        logging.info(f"Processing single prompt at index {args.prompt_index}")
        pipeline.generate_by_range(args.prompt_index, args.prompt_index)
    
    elif args.start_index is not None and args.end_index is not None:
        # Process range
        logging.info(f"Processing range [{args.start_index}, {args.end_index}]")
        pipeline.generate_by_range(args.start_index, args.end_index)
    
    else:
        # Process all
        logging.info("Processing all prompts")
        pipeline.generate_all()
    
    logging.info("\nVideo generation complete!")


if __name__ == '__main__':
    main()