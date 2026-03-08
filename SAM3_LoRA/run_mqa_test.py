import os
import yaml
import torch
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
import sys

# Add sam3 dir to path to find mqa_evaluator module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sam3"))
from mqa_evaluator import evaluate_mqa_on_dataset

def main():
    config_path = "configs/disaster_fast_config.yaml"
    weights_path = "outputs/sam3_lora_fast/best_lora_weights.pt"
    
    if not os.path.exists(weights_path):
        weights_path = "outputs/sam3_lora_fast/last_lora_weights.pt"
    
    if not os.path.exists(weights_path):
        print("No weights found!")
        return

    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Building SAM3 model...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=True
    )
    
    print("Applying LoRA...")
    lora_cfg = config["lora"]
    lora_config = LoRAConfig(
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
        apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
        apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
        apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
        apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
    )
    model = apply_lora_to_model(model, lora_config)
    
    print(f"Loading weights from {weights_path}")
    load_lora_weights(model, weights_path)
    model.to(device)
    model.eval()
    
    print("\nRunning MQA Evaluation...")
    try:
        mqa_results = evaluate_mqa_on_dataset(
            model=model,
            device=device.type,
            scenarios_path="../test_data/100_images_test/common_samples/sample_RDC.json",
            images_dir="../test_data/100_images_test/test_images",
            threshold=0.25
        )
        
        print("\n" + "="*50)
        print(f"MQA Results:")
        print(f"Accuracy: {mqa_results['accuracy']:.2%}")
        print(f"MAE:      {mqa_results['mae']:.4f}")
        print("="*50)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
