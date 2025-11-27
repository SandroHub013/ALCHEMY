"""
Training Agent per fine-tuning LLM con PyTorch Lightning.

Questo modulo implementa un LightningModule che gestisce il training loop
per modelli causali (CausalLM) con supporto per:
- QLoRA e PEFT
- Mixed precision training
- Gradient accumulation
- Logging strutturato

Può essere integrato con Agent Lightning per orchestrazione avanzata.
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
    LightningModule per il training di LLM con QLoRA.
    
    Gestisce il forward pass, calcolo loss, ottimizzazione e logging.
    Compatibile con Agent Lightning per orchestrazione avanzata.
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
        Inizializza il Training Agent.
        
        Args:
            model: Modello con PEFT/LoRA applicato
            tokenizer: Tokenizer per il modello
            learning_rate: Learning rate iniziale
            weight_decay: Weight decay per ottimizzatore
            warmup_steps: Numero di step per warmup
            max_steps: Numero massimo di step (per scheduler)
            lr_scheduler_type: Tipo di scheduler ("cosine", "linear", "constant")
            max_grad_norm: Clipping dei gradienti
            fp16: Usa mixed precision FP16
            bf16: Usa mixed precision BF16 (richiede GPU Ampere+)
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
        
        # Salva hyperparameters per logging
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        
        # Metriche per logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modello.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits del modello [batch_size, seq_len, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Step di training.
        
        Per CausalLM, il modello calcola automaticamente la loss se vengono
        passati i labels. I labels sono gli stessi input_ids shiftati di 1 posizione.
        I token con label -100 vengono ignorati nel calcolo della loss.
        
        Args:
            batch: Batch di dati con 'input_ids', 'attention_mask', 'labels'
            batch_idx: Indice del batch
            
        Returns:
            Loss scalare
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass con labels (il modello calcola la loss internamente)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # Passa labels per calcolo loss automatico
        )
        
        loss = outputs.loss
        
        # Log della loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # Per DDP
        )
        
        # Calcola perplexity (exp(loss))
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
        
        # Salva output per aggregazione epoch-level
        self.training_step_outputs.append(loss.detach())
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Step di validazione.
        
        Args:
            batch: Batch di dati con 'input_ids', 'attention_mask', 'labels'
            batch_idx: Indice del batch
            
        Returns:
            Loss scalare
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass in modalità eval
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        loss = outputs.loss
        
        # Log della loss di validazione
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        # Calcola perplexity
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
        
        # Salva output per aggregazione
        self.validation_step_outputs.append(loss.detach())
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Chiamato alla fine di ogni epoch di training."""
        if self.training_step_outputs:
            avg_loss = torch.stack(self.training_step_outputs).mean()
            logger.info(f"Epoch {self.current_epoch} - Average Training Loss: {avg_loss:.4f}")
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Chiamato alla fine di ogni epoch di validazione."""
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            logger.info(f"Epoch {self.current_epoch} - Average Validation Loss: {avg_loss:.4f}")
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configura ottimizzatore e learning rate scheduler.
        
        Returns:
            Dizionario con ottimizzatore e scheduler per Lightning
        """
        # Filtra i parametri trainable (solo quelli con LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Ottimizzatore AdamW
        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Calcola max_steps se non specificato
        if self.max_steps is None:
            # Stima basata su trainer (sarà aggiornato da Lightning)
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.max_steps
        
        # Configurazione scheduler
        if self.lr_scheduler_type == "cosine":
            # Cosine annealing con warmup
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
            # Linear decay con warmup
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
            # Constant (nessun scheduler)
            scheduler = None
        
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Aggiorna ogni step, non ogni epoch
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Hook chiamato prima di ogni step dell'ottimizzatore.
        Applica gradient clipping.
        """
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

