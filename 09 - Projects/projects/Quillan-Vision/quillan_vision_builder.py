# Quillan Vision Builder - Council & Swarm Architecture
# Novel multi-agent visual perception system for Quillan-Ronin
# Part of the Quillan-Ronin cognitive architecture

import os
from typing import Optional, Dict
import torch
import torch.nn as nn
from quillan_council_vision import (
    QuillanVisionSystem,
    VisionCouncil,
    SwarmIntelligenceModule,
    QuillanVisionTracker,
    build_quillan_council_vision
)


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def build_quillan_council_predictor(
    input_dim: int = 3,
    hidden_dim: int = 256,
    num_classes: int = 80,
    num_particles: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    enable_tracking: bool = True
) -> Dict:
    """
    Build Quillan Council Vision System with optional tracking.
    
    This is the main entry point for the novel council/swarm architecture.
    
    Args:
        input_dim: Number of input channels (3 for RGB)
        hidden_dim: Hidden dimension for feature processing
        num_classes: Number of output classes
        num_particles: Number of particles for swarm optimization
        device: Device to place model on
        enable_tracking: Whether to enable temporal tracking
        
    Returns:
        Dictionary containing:
        - vision_system: The QuillanVisionSystem
        - tracker: QuillanVisionTracker (if enabled)
        - metadata: System configuration
        
    Example:
        >>> system = build_quillan_council_predictor()
        >>> results = system['vision_system'](images)
        >>> if system['tracker']:
        >>>     council_consensus = system['tracker'].get_council_consensus()
    """
    # Build the vision system
    vision_system = build_quillan_council_vision(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_particles=num_particles,
        device=device
    )
    
    # Create tracker if enabled
    tracker = None
    if enable_tracking:
        tracker = QuillanVisionTracker(vision_system)
    
    return {
        'vision_system': vision_system,
        'tracker': tracker,
        'metadata': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'num_particles': num_particles,
            'device': device,
            'council_members': ['edge_detector', 'motion_tracker', 'semantic_segmenter', 'depth_estimator'],
            'architecture': 'council_swarm'
        }
    }


def build_quillan_image_model(
    input_dim: int = 3,
    hidden_dim: int = 256,
    num_classes: int = 80,
    num_particles: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: Optional[str] = None,
):
    """
    Build Quillan Vision image model using council/swarm architecture.
    
    Args:
        input_dim: Number of input channels
        hidden_dim: Hidden dimension
        num_classes: Number of output classes
        num_particles: Swarm particle count
        device: Device to load model on
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        QuillanVisionSystem instance
    """
    system = build_quillan_council_vision(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_particles=num_particles,
        device=device
    )
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        system.load_state_dict(checkpoint, strict=False)
    
    return system


def build_quillan_video_model(
    input_dim: int = 3,
    hidden_dim: int = 256,
    num_classes: int = 80,
    num_particles: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: Optional[str] = None,
):
    """
    Build Quillan Vision video model with temporal tracking.
    
    Args:
        input_dim: Number of input channels
        hidden_dim: Hidden dimension
        num_classes: Number of output classes
        num_particles: Swarm particle count
        device: Device to load model on
        checkpoint_path: Optional path to model checkpoint
        
    Returns:
        QuillanVisionTracker instance
    """
    vision_system = build_quillan_council_vision(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_particles=num_particles,
        device=device
    )
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vision_system.load_state_dict(checkpoint, strict=False)
    
    tracker = QuillanVisionTracker(vision_system)
    
    return tracker


def build_quillan_predictor(
    version: str = "council_swarm",
    input_dim: int = 3,
    hidden_dim: int = 256,
    num_classes: int = 80,
    num_particles: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_path: Optional[str] = None,
    enable_tracking: bool = False,
    **kwargs
):
    """
    Build a Quillan Vision predictor.
    
    This is the unified entry point for all Quillan Vision architectures.
    
    Args:
        version: Architecture version - "council_swarm" for novel architecture
        input_dim: Number of input channels
        hidden_dim: Hidden dimension
        num_classes: Number of output classes
        num_particles: Swarm particle count
        device: Device to place model on
        checkpoint_path: Optional path to model checkpoint
        enable_tracking: Enable temporal tracking for video
        **kwargs: Additional arguments
        
    Returns:
        QuillanVisionSystem or QuillanVisionTracker depending on tracking
        
    Example:
        # Build council/swarm system:
        predictor = build_quillan_predictor(version="council_swarm")
        results = predictor(images)
        
        # Build with tracking:
        tracker = build_quillan_predictor(version="council_swarm", enable_tracking=True)
        tracker.track_frame(frame)
        consensus = tracker.get_council_consensus()
    """
    if version == "council_swarm":
        if enable_tracking:
            return build_quillan_video_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_particles=num_particles,
                device=device,
                checkpoint_path=checkpoint_path
            )
        else:
            return build_quillan_image_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_particles=num_particles,
                device=device,
                checkpoint_path=checkpoint_path
            )
    else:
        raise ValueError(f"Unknown version: {version!r}. Use 'council_swarm'.")
