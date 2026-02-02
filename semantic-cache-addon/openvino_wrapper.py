import os
import logging
import time
import shutil
from typing import List, Union, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer

logger = logging.getLogger("reranker.openvino")

class OpenVINOReranker:
    """
    Wrapper for OpenVINO Reranker models to mimic CrossEncoder API.
    Optimized for Intel Hardware (NUC iGPU).
    """
    def __init__(self, model_name: str, device: str = "GPU", cache_dir: str = "/share/semantic-cache/openvino"):
        self.model_name = model_name
        self.device = device.upper()
        self.cache_dir = os.path.join(cache_dir, model_name.replace("/", "_"))
        
        logger.info(f"Initializing OpenVINO Reranker: {model_name} on {self.device}")
        
        # Check if already exported
        if os.path.exists(self.cache_dir) and os.path.exists(os.path.join(self.cache_dir, "openvino_model.xml")):
            logger.info(f"Loading cached OpenVINO model from {self.cache_dir}...")
            from optimum.intel import OVModelForSequenceClassification
            self.model = OVModelForSequenceClassification.from_pretrained(
                self.cache_dir, 
                device=self.device,
                ov_config={"PERFORMANCE_HINT": "LATENCY"}
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.cache_dir)
        else:
            logger.info(f"Exporting model to OpenVINO (this may take a minute)...")
            from optimum.intel import OVModelForSequenceClassification
            
            # Export and Load
            self.model = OVModelForSequenceClassification.from_pretrained(
                model_name,
                export=True,
                use_cache=False,
                device=self.device
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save for next time
            logger.info(f"Saving OpenVINO model to {self.cache_dir}...")
            self.model.save_pretrained(self.cache_dir)
            self.tokenizer.save_pretrained(self.cache_dir)

        # Optimize for latency if on GPU
        if "GPU" in self.device:
            logger.info("Compiling model for GPU latency...")
            self.model.reshape(1, 512) # Typical max seq length
            self.model.compile()

        logger.info("OpenVINO Model Ready.")

    def predict(self, sentences: List[List[str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Predict matching scores for a list of (query, document) pairs.
        Returns: np.ndarray of scores (logits)
        """
        # Sentences is list of [query, doc]
        if len(sentences) == 0:
            return np.array([])

        all_scores = []
        
        # Simple batching
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Predict
            # OVModel forward args are slightly different but **inputs usually works
            # We assume model returns logits
            with torch.no_grad():
                results = self.model(**inputs)
                logits = results.logits
                if logits.shape[1] > 1:
                     # Some models output [non_entailment, entailment] or similar
                     # We usually want the last column for "relevance"
                     scores = logits[:, -1]
                else:
                     scores = logits.squeeze(-1)
                
                all_scores.extend(scores.cpu().numpy())
                
        return np.array(all_scores)
