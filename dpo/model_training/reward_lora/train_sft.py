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
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from omegaconf import MISSING, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from torch.optim.lr_scheduler import LambdaLR

from dataset import DPOLatentDataset
from loss import create_loss_strategy, SFTLossStrategy, DPOLossStrategy, LossOutput


@dataclass
class LoraTrainConfig:
    rank: int = 64
    alpha: float = 128.0
    target_modules: List[str] = field(
        default_factory=lambda: ["q", "k", "v", "o", "ffn.0", "ffn.2"]
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
    inlier_regression_lambda: float = 0.0
    # ── Two-phase training ──────────────────────────────────────────────
    # Phase 1: SFT warm-up, trains the LoRA to predict flow velocities
    #          on winners before DPO preference signal kicks in.
    # Empirical references:
    #   - InstructVideo / VADER style DPO: 200-500 SFT warm-up steps
    #   - Diffusion-DPO paper: ~5% of total steps as SFT initialisation
    # 500 steps is a safe default; increase to 1000 for larger datasets.
    sft_warmup_steps: int = 500
    # LR for SFT phase. Typically same or slightly higher than DPO LR.
    sft_learning_rate: float = 1e-5


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
    """
    Two-phase trainer:
      Phase 1 (steps 0 … sft_warmup_steps-1):  SFT on winner samples.
              LR linearly warms up from 0 to sft_learning_rate.
      Phase 2 (steps sft_warmup_steps … max_steps): DPO with cosine decay.
              LR starts at learning_rate and decays to 0 following a cosine schedule.

    Why this schedule?
      - SFT warm-up aligns the LoRA weights with the flow target before the
        potentially noisy DPO preference signal is introduced, preventing
        early gradient explosions observed in cold-start DPO training.
      - Linear LR warm-up during SFT avoids large updates on randomly
        initialised LoRA parameters.
      - Cosine decay during DPO is the standard choice in RLHF/DPO
        literature (InstructGPT, Diffusion-DPO) and empirically outperforms
        constant or step-decay schedules on preference fine-tuning tasks.
    """

    def __init__(self, config: RewardTrainerConfig):
        super().__init__()
        self.config = config

        # Pre-build both loss strategies so we can switch cheaply at runtime.
        # auxiliary losses (e.g. static penalty) are only applied in DPO phase.
        aux_losses = []
        if config.training.static_penalty_lambda > 0:
            from loss import StaticPenaltyLoss
            aux_losses.append(StaticPenaltyLoss(weight=config.training.static_penalty_lambda))

        self.sft_loss_strategy = SFTLossStrategy(auxiliary_losses=[])
        self.dpo_loss_strategy = DPOLossStrategy(
            beta=config.training.beta,
            auxiliary_losses=aux_losses
        )

        # Load model
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(config.model.dit_path):
            model_manager.load_models([config.model.dit_path])
        else:
            dit_path = config.model.dit_path.split(",")
            model_manager.load_models([dit_path])

        # Initialize pipeline
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        self.pipe.requires_grad_(False)
        peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
        )
        denoising_model = self.pipe.denoising_model()
        self.peft_model = get_peft_model(denoising_model, peft_config)
        self.pipe.dit = self.peft_model
        self.peft_model.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── helpers ────────────────────────────────────────────────────────

    @property
    def in_sft_phase(self) -> bool:
        return self.global_step < self.config.training.sft_warmup_steps

    def print_gpu_memory_usage(self, stage=""):
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
            logging.info(f"[{stage}] GPU Memory: Current: {current_memory:.2f}GB, "
                         f"Max: {max_memory:.2f}GB, Reserved: {reserved_memory:.2f}GB")
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

    def forward_model(self, model, noisy_latent, timestep, prompt_emb, image_emb):
        return model(
            noisy_latent,
            timestep=timestep,
            **prompt_emb,
            **image_emb,
            use_gradient_checkpointing=True,
        )

    # ── core training step ─────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        if self.pipe.device != self.device:
            self.pipe.to(self.device)

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

        # ── Phase 1: SFT warm-up ───────────────────────────────────────
        if self.in_sft_phase:
            with torch.set_grad_enabled(True):
                velocity_win_pred = self.forward_model(
                    self.peft_model,
                    noisy_x_win,
                    timestep,
                    inputs["prompt_emb_win"],
                    inputs["image_emb_win"],
                )

            velocities = {
                "win": velocity_win_pred,
                "win_target": velocity_win,
                # DPO fields not used during SFT, but keep keys for safety
                "lose": None,
                "lose_target": velocity_lose,
                "win_ref": None,
                "lose_ref": None,
            }
            loss_output: LossOutput = self.sft_loss_strategy.calculate_loss(
                velocities,
                {   # SFT strategy doesn't use inputs dict, but pass for aux compat
                    "noisy_x_win": noisy_x_win,
                    "noisy_x_lose": noisy_x_lose,
                    "timestep": timestep,
                    "scheduler": self.pipe.scheduler,
                    **{k: inputs[k] for k in ("prompt_emb_win", "prompt_emb_lose",
                                               "image_emb_win", "image_emb_lose")},
                }
            )
            # Measure the gradient norm before returning the loss in training_step.
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.peft_model.parameters(), max_norm=float('inf')  # measure only, do not clip
            )
            self.log("grad_norm", total_norm, prog_bar=True)

            self.log("phase", 0.0, prog_bar=True)

        # ── Phase 2: DPO ───────────────────────────────────────────────
        else:
            with torch.set_grad_enabled(True):
                velocity_win_pred = self.forward_model(
                    self.peft_model,
                    noisy_x_win,
                    timestep,
                    inputs["prompt_emb_win"],
                    inputs["image_emb_win"],
                )
                velocity_lose_pred = self.forward_model(
                    self.peft_model,
                    noisy_x_lose,
                    timestep,
                    inputs["prompt_emb_lose"],
                    inputs["image_emb_lose"],
                )

            # Reference model forward (adapter disabled → base weights)
            self.peft_model.disable_adapter_layers()
            with torch.no_grad():
                velocity_ref_win_pred = self.forward_model(
                    self.peft_model,
                    noisy_x_win,
                    timestep,
                    inputs["prompt_emb_win"],
                    inputs["image_emb_win"],
                )
                velocity_ref_lose_pred = self.forward_model(
                    self.peft_model,
                    noisy_x_lose,
                    timestep,
                    inputs["prompt_emb_lose"],
                    inputs["image_emb_lose"],
                )
            self.peft_model.enable_adapter_layers()

            velocities = {
                "win": velocity_win_pred,
                "lose": velocity_lose_pred,
                "win_ref": velocity_ref_win_pred,
                "lose_ref": velocity_ref_lose_pred,
                "win_target": velocity_win,
                "lose_target": velocity_lose,
            }
            loss_output: LossOutput = self.dpo_loss_strategy.calculate_loss(
                velocities,
                {
                    "noisy_x_win": noisy_x_win,
                    "noisy_x_lose": noisy_x_lose,
                    "timestep": timestep,
                    "scheduler": self.pipe.scheduler,
                    **{k: inputs[k] for k in ("prompt_emb_win", "prompt_emb_lose",
                                               "image_emb_win", "image_emb_lose")},
                }
            )
            self.log("phase", 1.0, prog_bar=True)

        loss = loss_output.loss * self.pipe.scheduler.training_weight(timestep)

        self.log("train_loss", loss, prog_bar=True)
        self.log("win_metric", m_win, prog_bar=False)
        self.log("lose_metric", m_lose, prog_bar=False)
        self.log("metric_gap", m_win - m_lose, prog_bar=False)

        # Measure the gradient norm before returning the loss in training_step.
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.peft_model.parameters(), max_norm=float('inf')  # measure only, do not clip
        )
        self.log("grad_norm", total_norm, prog_bar=True)

        for key, item in loss_output.metrics.items():
            self.log(key, item, prog_bar=True)
        return loss

    # ── optimizer & LR schedule ────────────────────────────────────────

    def configure_optimizers(self):
        """
        Single optimizer, composite LR schedule:

          [0, W)        — linear warm-up from 0 → sft_learning_rate  (SFT phase)
          [W, W+100)    — linear transition from sft_lr → dpo_lr      (smooth handoff)
          [W+100, T)    — cosine decay from dpo_lr → 0                (DPO phase)

        where W = sft_warmup_steps, T = max_steps.

        The 100-step linear bridge avoids a discontinuous LR jump when the
        objective switches from SFT to DPO.
        """
        trainable_modules = filter(
            lambda p: p.requires_grad,
            self.pipe.denoising_model().parameters()
        )
        optimizer = torch.optim.Adam(
            trainable_modules,
            lr=self.config.training.sft_learning_rate,  # base LR for lambda scaling
        )

        W = self.config.training.sft_warmup_steps
        T = self.config.training.max_steps
        sft_lr = self.config.training.sft_learning_rate
        dpo_lr = self.config.training.learning_rate
        lr_ratio = dpo_lr / sft_lr if sft_lr > 0 else 1.0
        bridge = 100  # steps for SFT→DPO LR transition

        import math

        def lr_lambda(current_step: int) -> float:
            # ── Phase 1: linear warm-up (SFT) ──────────────────────────
            if current_step < W:
                return float(current_step) / max(1, W)

            # ── Bridge: smooth LR transition ────────────────────────────
            if current_step < W + bridge:
                t = (current_step - W) / bridge          # 0 → 1
                return 1.0 + t * (lr_ratio - 1.0)        # sft_lr → dpo_lr

            # ── Phase 2: cosine decay (DPO) ─────────────────────────────
            dpo_steps = T - (W + bridge)
            progress = (current_step - (W + bridge)) / max(1, dpo_steps)
            progress = min(progress, 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_ratio * cosine_factor               # dpo_lr → 0

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── checkpoint ────────────────────────────────────────────────────

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()

        lora_state_dict = get_peft_model_state_dict(self.peft_model)

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


# ── dataset & dataloader ───────────────────────────────────────────────

def setup_dataset(config: RewardTrainerConfig):
    dataset = DPOLatentDataset(
        metadata_path=config.data.metadata_path,
        metric_name=config.data.metric_name,
        metric_mode=config.data.metric_mode,
        min_gap=config.data.min_gap,
        metric_threshold=config.data.metric_threshold,
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


# ── entry point ───────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="train", version_base=None)
def train(config: RewardTrainerConfig):
    print(f"Training configuration:\n{OmegaConf.to_yaml(config)}")
    print(f"[Two-phase] SFT warm-up for {config.training.sft_warmup_steps} steps, "
          f"then DPO for remaining {config.training.max_steps - config.training.sft_warmup_steps} steps.")

    dataloader = setup_dataset(config)
    model = FlowDPOTrainer(config)

    current_time = datetime.now()
    date_str = current_time.strftime("%Y_%d_%m-%H_%M_%S")
    output_dir = os.path.join(
        config.logging.output_path, config.data.metric_name, date_str
    )
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
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, dataloader)
    print(f"Training completed. Model saved to {output_dir}")


# Register configuration with Hydra
cs = ConfigStore.instance()
cs.store(name="reward_config", node=RewardTrainerConfig)

if __name__ == "__main__":
    train()