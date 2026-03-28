"""
GAN Data Generator Module for TG-GAT DDoS Detection System.

This module implements Generative Adversarial Networks for generating
synthetic Zero-Day DDoS attack samples to enhance model robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from torch.utils.data import DataLoader, TensorDataset
import wandb

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    Generator network for synthetic DDoS attack generation.
    
    Takes random noise and generates realistic network traffic features
    that resemble Zero-Day DDoS attacks.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Generator.
        
        Args:
            config: Configuration dictionary containing GAN parameters
        """
        super(Generator, self).__init__()
        
        self.latent_dim = config['gan']['latent_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.output_dim = config['model']['node_dim'] + config['model']['edge_dim']
        self.num_layers = config['gan']['generator_layers']
        
        # Build the generator network
        layers = []
        in_dim = self.latent_dim
        
        for i in range(self.num_layers):
            out_dim = self.hidden_dim // (2 ** i)
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            in_dim = out_dim
        
        # Final layer
        layers.append(nn.Linear(in_dim, self.output_dim))
        layers.append(nn.Tanh())  # Normalize output to [-1, 1]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            z: Random noise tensor
            
        Returns:
            Generated features tensor
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs synthetic traffic.
    
    Takes network traffic features and outputs probability of being real.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Discriminator.
        
        Args:
            config: Configuration dictionary containing GAN parameters
        """
        super(Discriminator, self).__init__()
        
        self.input_dim = config['model']['node_dim'] + config['model']['edge_dim']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_layers = config['gan']['discriminator_layers']
        
        # Build the discriminator network
        layers = []
        in_dim = self.input_dim
        
        for i in range(self.num_layers):
            out_dim = self.hidden_dim // (2 ** i)
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            in_dim = out_dim
        
        # Final layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input features tensor
            
        Returns:
            Probability tensor
        """
        return self.model(x)


class GANGenerator:
    """
    GAN-based synthetic data generator for Zero-Day DDoS attacks.
    
    Trains a GAN to generate realistic attack samples that can be used
    to enhance the training dataset and improve model robustness.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GAN generator.
        
        Args:
            config: Configuration dictionary containing GAN parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['gan']['gan_lr'],
            betas=(config['gan']['beta1'], config['gan']['beta2'])
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['gan']['gan_lr'],
            betas=(config['gan']['beta1'], config['gan']['beta2'])
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training parameters
        self.n_critic = config['gan']['n_critic']
        self.latent_dim = config['gan']['latent_dim']
        
        # Training history
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'real_scores': [],
            'fake_scores': []
        }
    
    def train(self, real_data: torch.Tensor, epochs: int = 100, batch_size: int = 64):
        """
        Train the GAN on real DDoS attack data.
        
        Args:
            real_data: Real DDoS attack samples
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info(f"Training GAN on {len(real_data)} real samples")
        
        # Create dataset and dataloader
        dataset = TensorDataset(real_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_real_scores = 0.0
            epoch_fake_scores = 0.0
            num_batches = 0
            
            for batch_idx, (real_batch,) in enumerate(dataloader):
                batch_size = real_batch.size(0)
                real_batch = real_batch.to(self.device)
                
                # Labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                for _ in range(self.n_critic):
                    self.d_optimizer.zero_grad()
                    
                    # Real data
                    real_outputs = self.discriminator(real_batch)
                    d_loss_real = self.criterion(real_outputs, real_labels)
                    
                    # Fake data
                    noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                    fake_data = self.generator(noise)
                    fake_outputs = self.discriminator(fake_data.detach())
                    d_loss_fake = self.criterion(fake_outputs, fake_labels)
                    
                    # Total discriminator loss
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                # Generate fake data
                noise = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_outputs = self.discriminator(fake_data)
                
                # Generator loss (want discriminator to think fake is real)
                g_loss = self.criterion(fake_outputs, real_labels)
                g_loss.backward()
                self.g_optimizer.step()
                
                # Update metrics
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_real_scores += real_outputs.mean().item()
                epoch_fake_scores += fake_outputs.mean().item()
                num_batches += 1
            
            # Average metrics for epoch
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_real_score = epoch_real_scores / num_batches
            avg_fake_score = epoch_fake_scores / num_batches
            
            # Store history
            self.training_history['g_loss'].append(avg_g_loss)
            self.training_history['d_loss'].append(avg_d_loss)
            self.training_history['real_scores'].append(avg_real_score)
            self.training_history['fake_scores'].append(avg_fake_score)
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
                logger.info(f"Real Score: {avg_real_score:.4f}, Fake Score: {avg_fake_score:.4f}")
                
                # Log to W&B
                if wandb.run:
                    wandb.log({
                        'gan/g_loss': avg_g_loss,
                        'gan/d_loss': avg_d_loss,
                        'gan/real_score': avg_real_score,
                        'gan/fake_score': avg_fake_score,
                        'epoch': epoch
                    })
    
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate synthetic DDoS attack samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples tensor
        """
        self.generator.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            generated_samples = self.generator(noise)
        
        self.generator.train()
        
        logger.info(f"Generated {num_samples} synthetic samples")
        return generated_samples.cpu()
    
    def generate_diverse_attacks(self, num_samples_per_type: int = 100) -> Dict[str, torch.Tensor]:
        """
        Generate diverse types of Zero-Day attacks.
        
        Args:
            num_samples_per_type: Number of samples per attack type
            
        Returns:
            Dictionary of attack type to generated samples
        """
        attack_types = ['syn_flood', 'udp_flood', 'http_flood', 'slowloris', 'amplification']
        generated_attacks = {}
        
        for attack_type in attack_types:
            # Modify noise generation for specific attack characteristics
            samples = self._generate_attack_specific_samples(attack_type, num_samples_per_type)
            generated_attacks[attack_type] = samples
            
            logger.info(f"Generated {num_samples_per_type} samples for {attack_type}")
        
        return generated_attacks
    
    def _generate_attack_specific_samples(self, attack_type: str, num_samples: int) -> torch.Tensor:
        """
        Generate samples specific to an attack type.
        
        Args:
            attack_type: Type of attack to generate
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples tensor
        """
        self.generator.eval()
        
        with torch.no_grad():
            # Generate noise with specific characteristics for each attack type
            if attack_type == 'syn_flood':
                # SYN flood: high packet rate, low payload
                noise = self._generate_attack_noise(num_samples, attack_type)
            elif attack_type == 'udp_flood':
                # UDP flood: high volume, varied ports
                noise = self._generate_attack_noise(num_samples, attack_type)
            elif attack_type == 'http_flood':
                # HTTP flood: application layer, varied requests
                noise = self._generate_attack_noise(num_samples, attack_type)
            elif attack_type == 'slowloris':
                # Slowloris: slow, persistent connections
                noise = self._generate_attack_noise(num_samples, attack_type)
            elif attack_type == 'amplification':
                # Amplification: small request, large response
                noise = self._generate_attack_noise(num_samples, attack_type)
            else:
                noise = torch.randn(num_samples, self.latent_dim).to(self.device)
            
            generated_samples = self.generator(noise)
        
        self.generator.train()
        return generated_samples.cpu()
    
    def _generate_attack_noise(self, num_samples: int, attack_type: str) -> torch.Tensor:
        """
        Generate noise with characteristics specific to attack type.
        
        Args:
            num_samples: Number of samples
            attack_type: Type of attack
            
        Returns:
            Noise tensor
        """
        base_noise = torch.randn(num_samples, self.latent_dim).to(self.device)
        
        # Modify noise based on attack type
        if attack_type == 'syn_flood':
            # High frequency components for SYN flood
            base_noise[:, :self.latent_dim//4] *= 2.0
        elif attack_type == 'udp_flood':
            # Random port variations
            base_noise[:, self.latent_dim//4:self.latent_dim//2] *= 1.5
        elif attack_type == 'http_flood':
            # Application layer patterns
            base_noise[:, self.latent_dim//2:3*self.latent_dim//4] *= 1.2
        elif attack_type == 'slowloris':
            # Low frequency, persistent patterns
            base_noise[:, 3*self.latent_dim//4:] *= 0.5
        elif attack_type == 'amplification':
            # High amplification factor
            base_noise *= 1.8
        
        return base_noise
    
    def save_models(self, filepath: str):
        """
        Save generator and discriminator models.
        
        Args:
            filepath: Base path for saving models
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, filepath)
        
        logger.info(f"Saved GAN models to {filepath}")
    
    def load_models(self, filepath: str):
        """
        Load generator and discriminator models.
        
        Args:
            filepath: Path to load models from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Loaded GAN models from {filepath}")
    
    def evaluate_quality(self, real_data: torch.Tensor, generated_data: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the quality of generated samples.
        
        Args:
            real_data: Real DDoS samples
            generated_data: Generated samples
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Statistical similarity
        real_mean = real_data.mean(dim=0)
        real_std = real_data.std(dim=0)
        gen_mean = generated_data.mean(dim=0)
        gen_std = generated_data.std(dim=0)
        
        metrics['mean_difference'] = torch.mean(torch.abs(real_mean - gen_mean)).item()
        metrics['std_difference'] = torch.mean(torch.abs(real_std - gen_std)).item()
        
        # Discriminator score
        self.discriminator.eval()
        with torch.no_grad():
            real_scores = self.discriminator(real_data.to(self.device))
            fake_scores = self.discriminator(generated_data.to(self.device))
        
        metrics['real_score'] = real_scores.mean().item()
        metrics['fake_score'] = fake_scores.mean().item()
        metrics['discriminator_confusion'] = abs(real_scores.mean().item() - 0.5)
        
        self.discriminator.train()
        
        logger.info(f"Generated data quality metrics: {metrics}")
        return metrics
