# vggt-dpo/caption/generate_caption.py

"""
Complete Caption Generation System
Generates captions for video/image datasets using Qwen3-VL-8B-Instruct
All functionality in a single file.
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

import torch
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor


# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """Load and manage configuration from JSON file"""
    
    def __init__(self, config_path: str = "config/config.json"):
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
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration"""
        return self.config.get('generation', {})
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset configuration"""
        datasets = self.config.get('datasets', {})
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in config")
        return datasets[dataset_name]
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get('output', {})
    
    def get_scene_config(self, scene_type: str) -> Dict[str, Any]:
        """Get scene configuration"""
        scenes = self.config.get('scenes', {})
        if scene_type not in scenes:
            raise ValueError(f"Scene type '{scene_type}' not found in config")
        return scenes[scene_type]
    
    def get_all_scene_types(self) -> List[str]:
        """Get all available scene types"""
        return list(self.config.get('scenes', {}).keys())
    
    def get_camera_movements(self) -> List[str]:
        """Get camera movement options"""
        return self.config.get('camera_movements', [])
    
    def get_prompt_template(self, scene_type: str) -> str:
        """Get prompt template for scene type"""
        prompts = self.config.get('prompts', {})
        if scene_type not in prompts:
            raise ValueError(f"Prompt for scene type '{scene_type}' not found")
        return prompts[scene_type]
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration values"""
        def update_nested_dict(d: dict, u: dict):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested_dict(self.config, updates)
    
    def save_config(self, output_path: str = None):
        """Save current configuration to file"""
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {save_path}")


# ============================================================================
# Prompt Manager
# ============================================================================

class PromptManager:
    """Manage prompt templates"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def get_prompt(self, scene_type: str, camera_movement: str = None) -> Tuple[str, str]:
        """Get formatted prompt for scene type"""
        
        # Get prompt template
        template = self.config_loader.get_prompt_template(scene_type)
        
        # Get camera movement if not provided
        if camera_movement is None:
            camera_movements = self.config_loader.get_camera_movements()
            camera_movement = random.choice(camera_movements)
        
        # Format prompt with camera movement
        prompt = template.replace('{camera_movement}', camera_movement)
        
        return prompt, camera_movement
    
    def get_random_camera_movement(self) -> str:
        """Get random camera movement"""
        movements = self.config_loader.get_camera_movements()
        return random.choice(movements)


# ============================================================================
# Caption Generator (Model)
# ============================================================================

class CaptionGenerator:
    """Caption generator using Qwen3-VL model"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        """Load Qwen3-VL model and processor"""
        model_config = self.config_loader.get_model_config()
        
        print(f"Loading model: {model_config['model_name']}")
        
        # Parse dtype
        dtype_str = model_config.get('dtype', 'auto')
        if dtype_str == 'auto':
            dtype = torch.bfloat16  # 改为明确指定
        else:
            dtype_map = {
                'bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32
            }
            dtype = dtype_map.get(dtype_str, torch.bfloat16)
        
        # Load model WITHOUT device_map (手动移动到GPU)
        if model_config.get('use_flash_attention', False):
            print("Loading model with flash_attention_2...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_config['model_name'],
                dtype=dtype,
                attn_implementation="flash_attention_2"
                # 移除 device_map="auto"
            )
        else:
            print("Loading model with default attention...")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_config['model_name'],
                dtype=dtype
                # 移除 device_map="auto"
            )
        
        # 手动移动模型到 GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Model moved to GPU")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_config['model_name'])
        
        print("Model loaded successfully")
    
    def _prepare_messages(self, media_path: str, prompt: str, is_video: bool = False) -> List[dict]:
        """Prepare messages for the model"""
        if is_video:
            content = [
                {
                    "type": "video",
                    "video": media_path,
                },
                {"type": "text", "text": prompt}
            ]
        else:
            content = [
                {
                    "type": "image",
                    "image": media_path
                },
                {"type": "text", "text": prompt}
            ]
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def generate_caption(
        self, 
        media_path: str, 
        prompt: str,
        is_video: bool = False
    ) -> str:
        """Generate caption for a single image or video"""
        
        gen_config = self.config_loader.get_generation_config()
        
        # Prepare input
        messages = self._prepare_messages(media_path, prompt, is_video)
        
        # Prepare for inference using apply_chat_template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.get('max_new_tokens', 512),
                temperature=gen_config.get('temperature', 0.7),
                top_p=gen_config.get('top_p', 0.9),
                do_sample=gen_config.get('do_sample', True)
            )
        
        # Trim the input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        # Decode the output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text


# ============================================================================
# Dataset Handler
# ============================================================================

class DatasetHandler:
    """Handle different datasets for caption generation"""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
    
    def get_dataset_samples(
        self,
        dataset_name: str,
        num_samples: int
    ) -> List[str]:
        """Get samples from dataset"""
        
        dataset_config = self.config_loader.get_dataset_config(dataset_name)
        data_path = Path(dataset_config['path'])
        
        if dataset_name == 'realestate10k':
            return self._get_video_samples(dataset_config, data_path, num_samples)
        elif dataset_name == 'gldv2':
            return self._get_image_samples(dataset_config, data_path, num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _get_video_samples(
        self,
        config: dict,
        data_path: Path,
        num_samples: int
    ) -> List[str]:
        """Get video samples"""
        video_dir = data_path / config.get('video_dir', 'videos')
        
        if not video_dir.exists():
            raise ValueError(f"Video directory not found: {video_dir}")
        
        # Get all video files
        video_files = []
        for ext in config.get('file_extensions', ['.mp4']):
            video_files.extend(list(video_dir.glob(f"*{ext}")))
        
        if len(video_files) == 0:
            raise ValueError(f"No video files found in {video_dir}")
        
        # Sample videos
        if len(video_files) < num_samples:
            print(f"Warning: Only {len(video_files)} videos available, requested {num_samples}")
            num_samples = len(video_files)
        
        sampled_videos = random.sample(video_files, num_samples)
        
        return [str(v) for v in sampled_videos]
    
    def _get_image_samples(
        self,
        config: dict,
        data_path: Path,
        num_samples: int
    ) -> List[str]:
        """Get image samples"""
        image_dir = data_path / config.get('image_dir', 'images')
        
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        
        # Get all image files
        image_files = []
        for ext in config.get('file_extensions', ['.jpg', '.png']):
            image_files.extend(list(image_dir.glob(f"*{ext}")))
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {image_dir}")
        
        # Sample images
        if len(image_files) < num_samples:
            print(f"Warning: Only {len(image_files)} images available, requested {num_samples}")
            num_samples = len(image_files)
        
        sampled_images = random.sample(image_files, num_samples)
        
        return [str(img) for img in sampled_images]


# ============================================================================
# Caption Pipeline
# ============================================================================

class CaptionPipeline:
    """Main pipeline for generating captions"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config_loader = ConfigLoader(config_path)
        self.generator = CaptionGenerator(self.config_loader)
        self.prompt_manager = PromptManager(self.config_loader)
        self.dataset_handler = DatasetHandler(self.config_loader)
        
        # Create output directory
        output_config = self.config_loader.get_output_config()
        Path(output_config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def generate_for_scene(self, scene_type: str) -> List[Dict]:
        """Generate captions for a specific scene type"""
        
        scene_config = self.config_loader.get_scene_config(scene_type)
        
        print(f"\n{'='*60}")
        print(f"Processing: {scene_type}")
        print(f"Dataset: {scene_config['dataset']}")
        print(f"Target samples: {scene_config['num_samples']}")
        print(f"Description: {scene_config['description']}")
        print(f"{'='*60}\n")
        
        # Get samples
        samples = self.dataset_handler.get_dataset_samples(
            scene_config['dataset'],
            scene_config['num_samples']
        )
        
        print(f"Found {len(samples)} samples")
        
        # Generate captions
        results = []
        
        for sample_path in tqdm(samples, desc=f"Generating captions for {scene_type}"):
            # Get prompt with random camera movement
            prompt, camera_movement = self.prompt_manager.get_prompt(scene_type)
            
            # Generate caption
            try:
                caption = self.generator.generate_caption(
                    sample_path,
                    prompt,
                    scene_config['is_video']
                )
                
                result = {
                    'file_path': sample_path,
                    'scene_type': scene_type,
                    'dataset': scene_config['dataset'],
                    'camera_movement': camera_movement,
                    'caption': caption,
                    'is_video': scene_config['is_video']
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"\nError processing {sample_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save results
        output_config = self.config_loader.get_output_config()
        output_file = Path(output_config['output_dir']) / f"{scene_type}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                results, 
                f, 
                indent=output_config.get('indent', 2), 
                ensure_ascii=False
            )
        
        print(f"\nSaved {len(results)} captions to {output_file}")
        
        return results
    
    def generate_all(self):
        """Generate captions for all scene types"""
        
        all_scene_types = self.config_loader.get_all_scene_types()
        all_results = {}
        
        for scene_type in all_scene_types:
            results = self.generate_for_scene(scene_type)
            all_results[scene_type] = results
        
        # Save combined results
        output_config = self.config_loader.get_output_config()
        combined_file = Path(output_config['output_dir']) / "all_captions.json"
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(
                all_results, 
                f, 
                indent=output_config.get('indent', 2), 
                ensure_ascii=False
            )
        
        print(f"\n{'='*60}")
        print(f"All captions generated successfully!")
        print(f"Combined results saved to: {combined_file}")
        print(f"Total scenes processed: {sum(len(v) for v in all_results.values())}")
        print(f"{'='*60}\n")
        
        return all_results


# ============================================================================
# Utility Functions
# ============================================================================

def update_dataset_paths(config_path: str):
    """Interactively update dataset paths in config"""
    config_loader = ConfigLoader(config_path)
    
    print("\n=== Update Dataset Paths ===\n")
    
    # Update RealEstate10k path
    realestate_config = config_loader.get_dataset_config('realestate10k')
    current_path = realestate_config['path']
    print(f"Current RealEstate10k path: {current_path}")
    new_path = input("Enter new path (press Enter to keep current): ").strip()
    if new_path:
        config_loader.update_config({
            'datasets': {
                'realestate10k': {'path': new_path}
            }
        })
        print(f"Updated to: {new_path}")
    
    # Update GLDv2 path
    gldv2_config = config_loader.get_dataset_config('gldv2')
    current_path = gldv2_config['path']
    print(f"\nCurrent GLDv2 path: {current_path}")
    new_path = input("Enter new path (press Enter to keep current): ").strip()
    if new_path:
        config_loader.update_config({
            'datasets': {
                'gldv2': {'path': new_path}
            }
        })
        print(f"Updated to: {new_path}")
    
    # Save updated config
    config_loader.save_config()
    print("\nConfiguration updated successfully!\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate captions for video/image datasets using Qwen3-VL'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.json',
        help='Path to configuration JSON file (default: config/config.json)'
    )
    
    parser.add_argument(
        '--scene-type',
        type=str,
        default=None,
        help='Specific scene type to process (if not provided, processes all)'
    )
    
    parser.add_argument(
        '--update-paths',
        action='store_true',
        help='Update dataset paths in config interactively'
    )
    
    parser.add_argument(
        '--list-scenes',
        action='store_true',
        help='List all available scene types and exit'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List scene types
    if args.list_scenes:
        config_loader = ConfigLoader(args.config)
        scene_types = config_loader.get_all_scene_types()
        print("\nAvailable scene types:")
        for st in scene_types:
            scene_config = config_loader.get_scene_config(st)
            print(f"  - {st}: {scene_config['description']}")
        print()
        return
    
    # Update paths if requested
    if args.update_paths:
        update_dataset_paths(args.config)
        return
    
    # Create pipeline
    pipeline = CaptionPipeline(args.config)
    
    # Generate captions
    print("Starting caption generation...")
    
    if args.scene_type:
        # Generate for specific scene type
        results = pipeline.generate_for_scene(args.scene_type)
        print(f"\nGenerated {len(results)} captions for {args.scene_type}")
    else:
        # Generate for all scene types
        results = pipeline.generate_all()
    
    print("\nCaption generation complete!")


if __name__ == '__main__':
    main()