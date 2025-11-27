"""
Training Agent for LLM fine-tuning with PyTorch Lightning.

This module implements a LightningModule that manages the training loop
for causal models (CausalLM) with support for:
- QLoRA and PEFT
- Mixed precision training
- Gradient accumulation
- Structured logging

Can be integrated with Agent Lightning for advanced orchestration.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class LLMTrainingAgent(pl.LightningModule):
    """
    LightningModule for LLM training with QLoRA.
    
    Manages the forward pass, loss calculation, optimization and logging.
    Compatible with Agent Lightning for advanced orchestration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float = 2.0e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: Optional[int] = None,
        lr_scheduler_type: str = "cosine",
        max_grad_norm: float = 0.3,
        fp16: bool = True,
        bf16: bool = False,
    ):
        """
        Initialize the Training Agent.
        
        Args:
            model: Model with PEFT/LoRA applied
            tokenizer: Tokenizer for the model
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            max_steps: Maximum number of steps (for scheduler)
            lr_scheduler_type: Scheduler type ("cosine", "linear", "constant")
            max_grad_norm: Gradient clipping
            fp16: Use FP16 mixed precision
            bf16: Use BF16 mixed precision (requires Ampere+ GPU)
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.bf16 = bf16
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        
        # Metrics for logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Model forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Model logits [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        For CausalLM, the model automatically calculates the loss when
        labels are passed. Labels are the same input_ids shifted by 1 position.
        Tokens with label -100 are ignored in loss calculation.
        
        Args:
            batch: Data batch with 'input_ids', 'attention_mask', 'labels'
            batch_idx: Batch index
            
        Returns:
            Scalar loss
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass with labels (model calculates loss internally)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # Pass labels for automatic loss calculation
        )
        
        loss = outputs.loss
        
        # Log loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # For DDP
        )
        
        # Calculate perplexity (exp(loss))
        perplexity = torch.exp(loss)
        self.log(
            "train/perplexity",
            perplexity,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        
        # Save output for epoch-level aggregation
        self.training_step_outputs.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Data batch with 'input_ids', 'attention_mask', 'labels'
            batch_idx: Batch index
            
        Returns:
            Scalar loss
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass in eval mode
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        loss = outputs.loss
        
        # Log validation loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        self.log(
            "val/perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        
        # Save output for aggregation
        self.validation_step_outputs.append(loss.detach())
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            logger.info(f"Epoch {self.current_epoch} - Average Training Loss: {avg_loss:.4f}")
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            logger.info(f"Epoch {self.current_epoch} - Average Validation Loss: {avg_loss:.4f}")
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler for Lightning
        """
        # Filter trainable parameters (only those with LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # AdamW optimizer
        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Calculate max_steps if not specified
        if self.max_steps is None:
            # Estimate based on trainer (will be updated by Lightning)
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.max_steps
        
        # Scheduler configuration
        if self.lr_scheduler_type == "cosine":
            # Cosine annealing with warmup
            if self.warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.warmup_steps,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max_steps - self.warmup_steps,
                    eta_min=self.learning_rate * 0.1,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_steps],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max_steps,
                    eta_min=self.learning_rate * 0.1,
                )
        elif self.lr_scheduler_type == "linear":
            # Linear decay with warmup
            if self.warmup_steps > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.warmup_steps,
                )
                linear_scheduler = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=max_steps - self.warmup_steps,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, linear_scheduler],
                    milestones=[self.warmup_steps],
                )
            else:
                scheduler = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=max_steps,
                )
        else:
            # Constant (no scheduler)
            scheduler = None
        
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Update every step, not every epoch
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Hook called before each optimizer step.
        Applies gradient clipping.
        """
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )
