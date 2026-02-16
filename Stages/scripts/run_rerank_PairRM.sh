# Relabeled offline preference dataset
python rerank_PairRM_preference.py \
    --data_path datasets/ultrafeedback_binarized \
    --blender_model_path pretrained_lms/PairRM \
    --output_path generate_outputs/iter_data/iter0/pairwise/xx.jsonl


# On-policy sampling preference dataset
python rerank_PairRM_preference.py \
    --data_path generate_outputs/iter_data/iter0/exploration/xx.jsonl \
    --blender_model_path pretrained_lms/PairRM \
    --output_path generate_outputs/iter_data/iter0/pairwise/xx.jsonl

# Boundary Measurement
python rerank_PairRM_listwise.py \
    --prev_preference_path datasets/ultrafeedback_binarized \
    --curr_sampled_path generate_outputs/iter_data/iter0/exploration/llama3-it-online.jsonl \
    --blender_model_path pretrained_lms/PairRM \
    --output_path generate_outputs/test_prefs/...

# Quality comparison between offline-original and offline-llama
python rerank_PairRM_pairwise.py \
    --prev_preference_path datasets/ultrafeedback_binarized \
    --curr_sampled_path generate_outputs/iter_data/iter0/exploration/llama3-it-online.jsonl \
    --blender_model_path pretrained_lms/PairRM \
    --output_path generate_outputs/iter_data/iter0/pairwise/...
