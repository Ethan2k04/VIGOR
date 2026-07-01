import os
import gc
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import torch
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from diffsynth import WanVideoPipeline, ModelManager
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from diffusers.optimization import get_scheduler

from dataset import DPOLatentDataset
from loss import create_loss_strategy, LossOutput


@dataclass
class LoraTrainConfig:
    rank: int = 64
    alpha: float = 128.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q", "k", "v", "o"]
    )


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-5
    max_epochs: int = 1
    accumulate_grad_batches: int = 4
    precision: str = "bf16"
    strategy: str = "auto"
    beta: float = 500.0
    max_steps: int = 20000
    train_strategy: str = 'dpo'
    static_penalty_lambda: float = 0.0
    motion_smoothness_lambda: float = 0.0
    inlier_regression_lambda: float = 0.0
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    sft_warmup_steps: int = 0
    sft_learning_rate: float = 5e-5


@dataclass
class LoggingConfig:
    output_path: str = "./output"
    save_top_k: int = -1
    checkpoint_every_n_steps: Optional[int] = None
    experiment_name: str = "reward_lora"


@dataclass
class ModelConfig:
    dit_path: str = MISSING
    pretrained_lora_path: Optional[str] = None
    sft_lora_path: Optional[str] = None
    inlier_regression_path: Optional[str] = None


@dataclass
class DataConfig:
    metadata_path: str = MISSING
    metric_name: str = "reprojection_euclidean"
    metric_mode: str = "min"
    min_gap: float = 0.0
    metric_threshold: Optional[float] = None
    dataloader_num_workers: int = 1
    batch_size: int = 1


@dataclass
class RewardTrainerConfig:
    training: TrainingConfig = TrainingConfig()
    lora: LoraTrainConfig = LoraTrainConfig()
    logging: LoggingConfig = LoggingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()


class FlowDPOTrainer(pl.LightningModule):
    def __init__(self, config: RewardTrainerConfig):
        super().__init__()
        self.config = config
        self.loss_strategy = create_loss_strategy(
            strategy=config.training.train_strategy,
            beta=config.training.beta,
            static_penalty_lambda=config.training.static_penalty_lambda,
            motion_smoothness_lambda=config.training.motion_smoothness_lambda,
            inlier_regression_lambda=config.training.inlier_regression_lambda,
            inlier_model_path=config.model.inlier_regression_path
        )

        sft_ckpt = config.model.sft_lora_path or config.model.pretrained_lora_path

        # ------------------------------------------------------------------ #
        #  Build ref_pipe: base DiT + SFT LoRA fused, fully frozen.
        #  Fuse directly with diffsynth's load_lora (no PEFT needed; key formats match).
        # ------------------------------------------------------------------ #
        logging.info("Building reference model: base DiT + SFT LoRA fused...")
        ref_model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(config.model.dit_path):
            ref_model_manager.load_models([config.model.dit_path])
        else:
            ref_model_manager.load_models([config.model.dit_path.split(",")])

        self.ref_pipe = WanVideoPipeline.from_model_manager(ref_model_manager)
        self.ref_pipe.scheduler.set_timesteps(1000, training=True)

        # Load the SFT LoRA with diffsynth and fuse it directly into the base weights
        if sft_ckpt:
            ref_model_manager.load_lora(sft_ckpt, lora_alpha=1.0)
            logging.info(f"SFT LoRA fused into reference model from: {sft_ckpt}")
        else:
            logging.warning("No SFT checkpoint; reference model is plain base DiT.")

        # Freeze everything; ref_pipe is used for inference only
        self.ref_pipe.requires_grad_(False)
        ref_trainable = sum(p.numel() for p in self.ref_pipe.parameters() if p.requires_grad)
        assert ref_trainable == 0, f"Reference model should be fully frozen, got {ref_trainable} trainable params!"
        logging.info("Reference model frozen successfully.")

        # ------------------------------------------------------------------ #
        #  Build policy_pipe: add a PEFT LoRA on top of the fused ref_model weights.
        #  This new LoRA starts from random init and is the part DPO trains.
        #  Base weights = SFT-fused weights (same as ref); only the new LoRA is trainable.
        # ------------------------------------------------------------------ #
        logging.info("Building policy model: fused base + new trainable LoRA...")
        policy_model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(config.model.dit_path):
            policy_model_manager.load_models([config.model.dit_path])
        else:
            policy_model_manager.load_models([config.model.dit_path.split(",")])

        self.pipe = WanVideoPipeline.from_model_manager(policy_model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Fuse the SFT LoRA into the base weights first, too (same starting point as ref)
        if sft_ckpt:
            policy_model_manager.load_lora(sft_ckpt, lora_alpha=1.0)
            logging.info(f"SFT LoRA fused into policy base from: {sft_ckpt}")

        # Freeze the fused base weights
        self.pipe.requires_grad_(False)

        # Add a new trainable PEFT LoRA on top of the fused base
        peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
        )
        denoising_model = self.pipe.denoising_model()
        self.policy_model = get_peft_model(denoising_model, peft_config)

        # Control gradients precisely: enable only lora_A / lora_B
        for param in self.policy_model.parameters():
            param.requires_grad_(False)
        for name, param in self.policy_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad_(True)

        self.pipe.dit = self.policy_model
        self.policy_model.train()

        trainable = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.policy_model.parameters())
        logging.info(f"Policy LoRA trainable params: {trainable:,} / {total:,}")
        assert trainable > 0, "Policy LoRA has no trainable parameters!"

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------- #
    #  Utilities
    # ---------------------------------------------------------------------- #
    def print_gpu_memory_usage(self, stage=""):
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"[{stage}] GPU Memory: Current={current_memory:.2f}GB, "
                         f"Max={max_memory:.2f}GB, Reserved={reserved_memory:.2f}GB")
            return current_memory, max_memory
        return 0, 0

    def prepare_model_inputs(self, batch):
        x_win = batch["x_win"].to(self.device)
        x_lose = batch["x_lose"].to(self.device)
        prompt_emb_win = batch["prompt_emb_win"]
        prompt_emb_lose = batch["prompt_emb_lose"]

        for key in prompt_emb_win:
            if isinstance(prompt_emb_win[key], torch.Tensor):
                prompt_emb_win[key] = prompt_emb_win[key].to(self.device)
                prompt_emb_lose[key] = prompt_emb_lose[key].to(self.device)

        image_emb_win = {}
        if "image_emb_win" in batch:
            image_emb_win = batch["image_emb_win"]
            for key in image_emb_win:
                if isinstance(image_emb_win[key], torch.Tensor):
                    image_emb_win[key] = image_emb_win[key].to(self.device)

        image_emb_lose = {}
        if "image_emb_lose" in batch:
            image_emb_lose = batch["image_emb_lose"]
            for key in image_emb_lose:
                if isinstance(image_emb_lose[key], torch.Tensor):
                    image_emb_lose[key] = image_emb_lose[key].to(self.device)

        return {
            "x_win": x_win,
            "x_lose": x_lose,
            "prompt_emb_win": prompt_emb_win,
            "prompt_emb_lose": prompt_emb_lose,
            "image_emb_win": image_emb_win,
            "image_emb_lose": image_emb_lose,
        }

    def forward_model(self, model, noisy_latent, timestep, prompt_emb, image_emb,
                      use_grad_ckpt=True):
        return model(
            noisy_latent,
            timestep=timestep,
            **prompt_emb,
            **image_emb,
            use_gradient_checkpointing=use_grad_ckpt,
        )

    # ---------------------------------------------------------------------- #
    #  Training step
    # ---------------------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        # Make sure both pipes are on the correct device
        if self.pipe.device != self.device:
            self.pipe.to(self.device)
        if self.ref_pipe.device != self.device:
            self.ref_pipe.to(self.device)

        m_win = batch["m_win"].mean().item()
        m_lose = batch["m_lose"].mean().item()
        inputs = self.prepare_model_inputs(batch)

        noise = torch.randn_like(inputs['x_win'])
        timestep_id = torch.randint(
            0, self.pipe.scheduler.num_train_timesteps,
            (self.config.data.batch_size,)
        )
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            device=self.device, dtype=self.pipe.torch_dtype
        )

        noisy_x_win = self.pipe.scheduler.add_noise(inputs['x_win'], noise, timestep)
        velocity_win = self.pipe.scheduler.training_target(inputs['x_win'], noise, timestep)
        noisy_x_lose = self.pipe.scheduler.add_noise(inputs['x_lose'], noise, timestep)
        velocity_lose = self.pipe.scheduler.training_target(inputs['x_lose'], noise, timestep)

        # ---------- Policy forward (with gradients) ----------
        self.policy_model.train()
        with torch.set_grad_enabled(True):
            velocity_win_pred = self.forward_model(
                self.policy_model, noisy_x_win, timestep,
                inputs["prompt_emb_win"], inputs["image_emb_win"]
            )
            velocity_lose_pred = self.forward_model(
                self.policy_model, noisy_x_lose, timestep,
                inputs["prompt_emb_lose"], inputs["image_emb_lose"]
            )

        # ---------- Reference forward (fused, frozen ref_pipe, no gradients) ----------
        if self.config.training.train_strategy == 'dpo':
            ref_dit = self.ref_pipe.denoising_model()
            ref_dit.eval()
            with torch.no_grad():
                velocity_ref_win_pred = self.forward_model(
                    ref_dit, noisy_x_win, timestep,
                    inputs["prompt_emb_win"], inputs["image_emb_win"],
                    use_grad_ckpt=False
                )
                velocity_ref_lose_pred = self.forward_model(
                    ref_dit, noisy_x_lose, timestep,
                    inputs["prompt_emb_lose"], inputs["image_emb_lose"],
                    use_grad_ckpt=False
                )
        else:
            velocity_ref_win_pred = None
            velocity_ref_lose_pred = None

        velocities = {
            "win": velocity_win_pred,
            "lose": velocity_lose_pred,
            "win_ref": velocity_ref_win_pred,
            "lose_ref": velocity_ref_lose_pred,
            "win_target": velocity_win,
            "lose_target": velocity_lose,
        }
        forward_inputs = {
            "noisy_x_win": noisy_x_win,
            "noisy_x_lose": noisy_x_lose,
            "timestep": timestep,
            "scheduler": self.pipe.scheduler,
            "prompt_emb_win": inputs["prompt_emb_win"],
            "prompt_emb_lose": inputs["prompt_emb_lose"],
            "image_emb_win": inputs["image_emb_win"],
            "image_emb_lose": inputs["image_emb_lose"],
        }

        loss_output: LossOutput = self.loss_strategy.calculate_loss(velocities, forward_inputs)
        loss = loss_output.loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)
        self.log("win_metric", m_win, prog_bar=False)
        self.log("lose_metric", m_lose, prog_bar=False)
        self.log("metric_gap", m_win - m_lose, prog_bar=False)

        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            max_norm=float('inf')
        )
        self.log("grad_norm", total_norm, prog_bar=True)

        for key, item in loss_output.metrics.items():
            self.log(key, item, prog_bar=True)

        return loss

    # ---------------------------------------------------------------------- #
    #  Optimizer
    # ---------------------------------------------------------------------- #
    def configure_optimizers(self):
        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found!"

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.config.training.learning_rate,
        )
        total_steps = self.config.training.max_steps
        warmup_steps = int(total_steps * 0.015)

        lr_scheduler = get_scheduler(
            "constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    # ---------------------------------------------------------------------- #
    #  Save checkpoint (policy LoRA only)
    # ---------------------------------------------------------------------- #
    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        lora_state_dict = get_peft_model_state_dict(self.policy_model)
        corrected_state_dict = {}
        for key, value in lora_state_dict.items():
            if key.startswith('base_model.model.'):
                new_key = key.replace('base_model.model.', '')
            elif key.startswith('base_model.'):
                new_key = key.replace('base_model.', '')
            else:
                new_key = key
            corrected_state_dict[new_key] = value
        checkpoint.update(corrected_state_dict)


# --------------------------------------------------------------------------- #
#  Dataset & entry point
# --------------------------------------------------------------------------- #
def setup_dataset(config: RewardTrainerConfig):
    dataset = DPOLatentDataset(
        metadata_path=config.data.metadata_path,
        metric_name=config.data.metric_name,
        metric_mode=config.data.metric_mode,
        min_gap=config.data.min_gap,
        metric_threshold=config.data.metric_threshold,
        filter_static=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.dataloader_num_workers,
    )
    return dataloader


def get_experiment_name(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


@hydra.main(config_path="config", config_name="train_hybrid", version_base=None)
def train(config: RewardTrainerConfig):
    import warnings
    warnings.filterwarnings("ignore", message=".*AccumulateGrad.*stream.*")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    print(f"Training configuration:\n{OmegaConf.to_yaml(config)}")
    dataloader = setup_dataset(config)
    model = FlowDPOTrainer(config)

    current_time = datetime.now()
    date_str = current_time.strftime("%Y_%d_%m-%H_%M_%S")
    output_dir = os.path.join(config.logging.output_path, config.data.metric_name, date_str)
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")

    logger = TensorBoardLogger(save_dir=tensorboard_dir, name="reward_lora")
    experiment_name = get_experiment_name(config.logging.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.logging.output_path}/{experiment_name}",
        save_top_k=config.logging.save_top_k,
        every_n_train_steps=config.logging.checkpoint_every_n_steps,
        filename="{epoch}-step={step}-{train_loss:.4f}",
        auto_insert_metric_name=False,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=config.training.precision,
        strategy=config.training.strategy,
        default_root_dir=output_dir,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
    )

    trainer.fit(model, dataloader)
    print(f"Training completed. Model saved to {output_dir}")


cs = ConfigStore.instance()
cs.store(name="reward_config", node=RewardTrainerConfig)

if __name__ == "__main__":
    train()