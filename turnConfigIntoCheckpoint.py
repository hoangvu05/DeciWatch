import os
import torch
import yaml
from lib.models.deciwatch import DeciWatch
from lib.core.config import parse_args

def clean_path(path):
    """Handle special characters in SCC directory names"""
    return path.replace("['", "").replace("']", "")

def create_checkpoint():
    # 1. Configure paths - UPDATE THESE TO MATCH YOUR ENVIRONMENT
    base_dir = "/projectnb/cs585bp/students/anhvu/DeciWatch"
    results_dir = clean_path(os.path.join(base_dir, "results/25-04-2025_2-41-50_['jhmdb_simplepose']"))
    
    # Try different possible weight files
    possible_weight_files = [
        "model_best.pth",
        "checkpoint.pth.tar", 
        "latest.pth",
        "weights.pth"
    ]
    
    # Find first existing weights file
    weights_path = None
    for f in possible_weight_files:
        if os.path.exists(os.path.join(results_dir, f)):
            weights_path = os.path.join(results_dir, f)
            break
            
    if not weights_path:
        raise FileNotFoundError(f"No weight files found in {results_dir}")
    
    config_path = os.path.join(results_dir, "config.yaml")
    output_path = os.path.join(base_dir, "deciwatch_checkpoint.pth.tar")

    # 2. Load config - Handle YACS format properly
    cfg, _ = parse_args()
    
    # Manually load YAML and convert to dict for YACS
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert YAML dict to YACS config
    from yacs.config import CfgNode
    cfg = CfgNode(config_dict)
    
    # Set required fields that might be missing
    if not hasattr(cfg, 'MODEL'):
        cfg.MODEL = CfgNode()
    if not hasattr(cfg.MODEL, 'INPUT_DIM'):
        cfg.MODEL.INPUT_DIM = 30  # Default for JHMDB 2D (15 keypoints * 2)
    
    # 3. Initialize model
    model = DeciWatch(
        input_dimension=cfg.MODEL.INPUT_DIM,
        sample_interval=cfg.SAMPLE_INTERVAL,
        encoder_hidden_dim=cfg.MODEL.ENCODER_EMBEDDING_DIMENSION,
        decoder_hidden_dim=cfg.MODEL.DECODER_EMBEDDING_DIMENSION,
        dropout=cfg.MODEL.DROPOUT,
        nheads=cfg.MODEL.ENCODER_HEAD,
        enc_layers=cfg.MODEL.ENCODER_TRANSFORMER_BLOCK,
        dec_layers=cfg.MODEL.DECODER_TRANSFORMER_BLOCK,
        recovernet_interp_method=cfg.MODEL.DECODER_INTERP,
        recovernet_mode=cfg.MODEL.DECODER,
        pre_norm=cfg.TRAIN.PRE_NORM
    )

    # 4. Load weights - handle different formats
    weights = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in weights:
        model.load_state_dict(weights['state_dict'])
    else:
        model.load_state_dict(weights)

    # 5. Save proper checkpoint
    torch.save({
        'state_dict': model.state_dict(),
        'epoch': cfg.TRAIN.EPOCH if hasattr(cfg.TRAIN, 'EPOCH') else 10,
        'config': cfg.dump(),
        'optimizer': None
    }, output_path)

    print(f"Successfully created checkpoint at {output_path}")
    print("Verification:")
    print(torch.load(output_path, map_location='cpu').keys())

if __name__ == "__main__":
    create_checkpoint()