"""Per-token loss diagnostic: PEFT checkpoint-1000 on original vs adaptive batch."""
import argparse, json, torch, torch.nn.functional as F
from scripts.experiment_partitioned_counting.train_adaptive_partition_sft import (
    AdaptivePartitionDataset, resolve_processor_source, load_bootstrap_model, force_default_cpu_device
)
from scripts.experiment_partitioned_counting.train_partitioned_double_scaffold_sft import (
    PartitionedFSC147Dataset, IGNORE_INDEX, compute_partitioned_ce_loss,
    embed_tokens, resolve_internvl_core, resolve_vision_tower, module_dtype, load_unilip_constants
)
from scripts.counting_grpo.train_hf_multi_image_count_sft import load_processor, HFMultiImageCollator

JSONL = '/data/amondal/UniCount/outputs/experiment_partitioned_counting/train/train_sft_partitioned_scaffold.jsonl'
ANN   = '/data/amondal/FSC147_hf/annotation_FSC147_384.json'
CKPT  = '/data/amondal/unicount_runs/partitioned_double_scaffold_sft_v1/checkpoint-1000'

args = argparse.Namespace(
    model_name_or_path=CKPT,
    base_model_name_or_path='/data/amondal/model_cache/UniLIP-3B',
    processor_name_or_path='/data/amondal/model_cache/UniLIP-3B',
    trust_remote_code=1,
    attn_implementation='eager',
    allow_attn_fallback=1,
    bf16=True,
)
args.processor_name_or_path = resolve_processor_source(args)
processor = load_processor(args)
tokenizer = processor.tokenizer

ds_orig = PartitionedFSC147Dataset(data_path=JSONL, annotation_json=ANN, processor=processor, max_seq_length=1024, image_size=448)
ds_adap = AdaptivePartitionDataset(jsonl_path=JSONL, processor=processor, annotation_json=ANN, max_seq_length=1024, image_size=448, strict_images=True)
collator = HFMultiImageCollator(processor)
batch_orig = {k: v.cuda() if torch.is_tensor(v) else v for k, v in collator([ds_orig[0]]).items()}
batch_adap = {k: v.cuda() if torch.is_tensor(v) else v for k, v in collator([ds_adap[0]]).items()}

print("\n[LOADING PEFT MODEL]")
with force_default_cpu_device():
    peft_model = load_bootstrap_model(args)
peft_model.config.use_cache = False
peft_model.eval()

with torch.no_grad():
    loss_orig = compute_partitioned_ce_loss(peft_model, batch_orig)
    loss_adap = compute_partitioned_ce_loss(peft_model, batch_adap)

print(f"\nPEFT loss ORIGINAL batch (8 supervised tokens):  {float(loss_orig):.4f}")
print(f"PEFT loss ADAPTIVE batch (11 supervised tokens): {float(loss_adap):.4f}")

def per_token_losses(model, batch):
    input_ids = batch['input_ids']
    labels = batch['labels']
    model_module = model.module if hasattr(model, 'module') else model
    language_model = model_module.get_model().language_model
    text_embeds = embed_tokens(language_model, input_ids)
    pixel_values = batch.get('pixel_values')
    if pixel_values is not None:
        internvl_core = resolve_internvl_core(model_module)
        vision_tower = resolve_vision_tower(model_module, internvl_core)
        vdtype = module_dtype(vision_tower)
        pixel_values = pixel_values.to(device=text_embeds.device, dtype=vdtype)
        image_token_id = load_unilip_constants()["UND_IMAGE_TOKEN_IDX"]
        image_token_mask = input_ids == image_token_id
        feature_layer = getattr(model_module.config, 'vision_feature_layer', None)
        feature_strategy = getattr(model_module.config, 'vision_feature_select_strategy', None)
        with torch.no_grad():
            vision_outputs = vision_tower(pixel_values=pixel_values, return_dict=True, output_hidden_states=(feature_layer != -1))
            vf = vision_outputs.last_hidden_state if feature_layer == -1 else vision_outputs.hidden_states[feature_layer]
            if feature_strategy == 'default': vf = vf[:, 1:, :]
            channels = vf.shape[1]; feature_size = int(channels**0.5); batch_size = vf.shape[0]
            vf = vf.reshape(batch_size, feature_size, feature_size, -1)
            vf = internvl_core.pixel_shuffle(vf, scale_factor=internvl_core.config.downsample_ratio)
            vf = vf.reshape(batch_size, -1, vf.shape[-1])
            image_embeds = internvl_core.multi_modal_projector(vf)
        flat_embeds = image_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype).flatten(0,1)
        text_embeds = text_embeds.clone()
        text_embeds[image_token_mask] = flat_embeds
    attn = batch['attention_mask']
    position_ids = torch.cumsum(attn.int(), dim=1) - 1
    position_ids[position_ids < 0] = 0
    with torch.no_grad():
        outputs = language_model(inputs_embeds=text_embeds, attention_mask=attn, position_ids=position_ids,
                                 output_hidden_states=False, return_dict=True, use_cache=False)
    logits = model_module.lm_head(outputs.last_hidden_state)
    shift_logits = logits[0, :-1, :]
    shift_labels = labels[0, 1:]
    nonign = shift_labels != IGNORE_INDEX
    supervised_ids = shift_labels[nonign]
    supervised_logits = shift_logits[nonign]
    probs = F.softmax(supervised_logits.float(), dim=-1)
    gold_probs = probs[torch.arange(len(supervised_ids)), supervised_ids]
    per_tok_loss = -gold_probs.log()
    decoded = [tokenizer.decode([t]) for t in supervised_ids.tolist()]
    return list(zip(decoded, supervised_ids.tolist(), per_tok_loss.tolist()))

print("\nPEFT per-token losses on ORIGINAL batch:")
for dec, tid, l in per_token_losses(peft_model, batch_orig):
    print(f"  tok={repr(dec):25s} id={tid:7d}  loss={l:.4f}")

print("\nPEFT per-token losses on ADAPTIVE batch:")
for dec, tid, l in per_token_losses(peft_model, batch_adap):
    print(f"  tok={repr(dec):25s} id={tid:7d}  loss={l:.4f}")
