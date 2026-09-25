#!/usr/bin/env bash
set -euo pipefail

# Gradual ImageNet-C: each corruption follows severity 1-2-3-4-5-4-3-2-1
# with 1,000 images per segment. AdaInit remains continuous; baselines receive
# a source reset at known segment boundaries by default.

method="${1:-adainit}"
device="${2:-cuda:0}"
seed="${3:-2022}"
if (( $# >= 3 )); then shift 3; else shift "$#"; fi
extra_args=("$@")

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${ADAINIT_PYTHON:-python}"
data_root="${DATA_ROOT:-$repo_dir/data}"
num_cpus="${NUM_CPUS:-4}"
samples_per_domain="${SAMPLES_PER_DOMAIN:-1000}"
detail_samples="${DETAIL_SAMPLES:-100}"
run_tag="${RUN_TAG:-paper}"
dry_run="${DRY_RUN:-0}"
baseline_reset="${BOUNDARY_RESET:-true}"
severity_text="${GRADUAL_SEVERITIES:-1 2 3 4 5 4 3 2 1}"
only_corruption="${CORRUPTION:-all}"

methods=(adainit no_adaptation tent sar sar2 come adadem cotta nctta)
corruptions=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression)

contains() {
  local wanted="$1" item
  shift
  for item in "$@"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}

if [[ "$method" == "all" ]]; then
  selected_methods=("${methods[@]}")
elif contains "$method" "${methods[@]}"; then
  selected_methods=("$method")
else
  echo "Unknown method: $method" >&2
  echo "Choose one of: ${methods[*]} all" >&2
  exit 2
fi
if [[ "$only_corruption" != "all" ]]; then
  contains "$only_corruption" "${corruptions[@]}" || { echo "Unknown corruption: $only_corruption" >&2; exit 2; }
  corruptions=("$only_corruption")
fi

read -r -a severities <<< "$severity_text"
(( ${#severities[@]} > 0 )) || { echo "GRADUAL_SEVERITIES cannot be empty." >&2; exit 2; }
for severity in "${severities[@]}"; do
  [[ "$severity" =~ ^[1-5]$ ]] || { echo "Invalid severity: $severity" >&2; exit 2; }
done
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer." >&2; exit 2; }
[[ "$num_cpus" =~ ^[0-9]+$ ]] || { echo "NUM_CPUS must be non-negative." >&2; exit 2; }
[[ "$samples_per_domain" =~ ^[1-9][0-9]*$ ]] || { echo "SAMPLES_PER_DOMAIN must be positive." >&2; exit 2; }
(( samples_per_domain <= 50000 )) || { echo "SAMPLES_PER_DOMAIN cannot exceed 50,000." >&2; exit 2; }
[[ "$detail_samples" =~ ^[0-9]+$ ]] || { echo "DETAIL_SAMPLES must be non-negative." >&2; exit 2; }
[[ "$run_tag" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RUN_TAG." >&2; exit 2; }
[[ "$dry_run" == "0" || "$dry_run" == "1" ]] || { echo "DRY_RUN must be 0 or 1." >&2; exit 2; }
[[ "$baseline_reset" == "true" || "$baseline_reset" == "false" ]] || { echo "BOUNDARY_RESET must be true or false." >&2; exit 2; }
command -v "$python_bin" >/dev/null || { echo "Python executable not found: $python_bin" >&2; exit 1; }
if [[ "$dry_run" == "0" && ! -d "$data_root/ILSVRC/imagenet-c" ]]; then
  echo "ImageNet-C not found at $data_root/ILSVRC/imagenet-c" >&2
  exit 1
fi

stream=()
for corruption in "${corruptions[@]}"; do
  for severity in "${severities[@]}"; do
    stream+=("imagenet_c_deterministic-${corruption}-${severity}")
  done
done
data_names="$(IFS=';'; printf '%s' "${stream[*]}")"
sampling_ratio="$(awk -v n="$samples_per_domain" 'BEGIN { printf "%.12g", n / 50000.0 }')"
num_domains="${#stream[@]}"

adainit_args=(
  --adainit_backend nctta
  --adainit_candidate_mode full
  --adainit_initialization_selector sequential_view_evidence
  --adainit_detector_mode adaptive
  --adainit_signature_momentum 0.95
  --adainit_signature_window_size 1
  --adainit_detector_signature_source instantaneous
  --adainit_signature_distance standardized
  --adainit_score_momentum 0.95
  --adainit_drift_beta 3.0
  --adainit_drift_min_ratio 2.0
  --adainit_min_reference_samples 32
  --adainit_drift_confirmations 6
  --adainit_trigger_cooldown 32
  --adainit_cache_size 16
  --adainit_cache_admission periodic_health
  --adainit_cache_eviction knowledge_fingerprint
  --adainit_knowledge_fingerprint local_delta
  --adainit_fingerprint_window 32
  --adainit_fingerprint_magnitude_weight 0.05
  --adainit_cache_insert_interval 32
  --adainit_cache_on_detection false
  --adainit_cache_optimizer_state false
  --adainit_cache_health_mode infomax_update
  --adainit_cache_health_min_information 0.05
  --adainit_cache_health_max_concentration 0.90
  --adainit_cache_health_min_update_cosine -0.50
  --adainit_cache_health_min_fingerprint_norm 1e-8
  --adainit_cache_health_update_weight 0.10
  --adainit_history_retrieval knowledge_fingerprint
  --adainit_history_fingerprint_max_distance -1
  --adainit_history_fingerprint_max_ratio -1
  --adainit_max_history_candidates 4
  --adainit_history_min_age 64
  --adainit_history_match_ratio -1
  --adainit_history_max_distance -1
  --adainit_history_radius_multiplier -1
  --adainit_source_for_unseen_only false
  --adainit_source_min_current_entropy -1
  --adainit_history_max_entropy_increase -1
  --adainit_num_views 8
  --adainit_evidence_horizon 32
  --adainit_evidence_min_samples 32
  --adainit_evidence_max_samples 64
  --adainit_evidence_confidence_scale 1.0
  --adainit_evidence_timing prequential
  --adainit_sequential_score view_infomax
  --adainit_sequential_view_jsd_weight 1.0
  --adainit_sequential_context_weight 1.0
  --adainit_selection_margin 0
  --adainit_source_selection_margin 0.05
  --adainit_max_marginal_entropy_increase -1
  --adainit_candidate_eval_batch_size 32
)

cd "$repo_dir"
for current_method in "${selected_methods[@]}"; do
  method_args=(--fishers false)
  reset="$baseline_reset"
  checkpoint_every=0
  case "$current_method" in
    adainit)
      reset=false
      checkpoint_every="${CHECKPOINT_EVERY_DOMAINS:-9}"
      method_args+=(--optimizer SGD --nu 5.0 --eta 1.0 "${adainit_args[@]}")
      ;;
    nctta) method_args+=(--nu 5.0 --eta 1.0) ;;
    come) method_args+=(--K 1000) ;;
    adadem) method_args+=(--adadem_pi 0.1 --adadem_mode adadem) ;;
    cotta)
      method_args+=(
        --optimizer SGD
        --alpha_teacher "${COTTA_ALPHA_TEACHER:-0.999}"
        --threshold_cotta "${COTTA_THRESHOLD:-0.1}"
        --restore_prob "${COTTA_RESTORE_PROB:-0.001}"
        --aug_size "${COTTA_AUG_SIZE:-32}"
        --cotta_loss symmetric_ce
      )
      ;;
  esac

  job_name="imagenetc_gradual${num_domains}x${samples_per_domain}/${current_method}/${run_tag}_seed${seed}"
  command=(
    "$python_bin" run_exp.py
    --job_name "$job_name"
    --root_path "$repo_dir/logs"
    --data_path "$data_root"
    --base_data_name imagenet
    --src_data_name imagenet
    --data_names "$data_names"
    --model_name vit_base_patch16_224
    --model_adaptation_method "$current_method"
    --model_selection_method last_iterate
    --data_wise sample_wise
    --batch_size 1
    --domain_sampling_name uniform
    --domain_sampling_ratio "$sampling_ratio"
    --domain_replay_visits 1
    --domain_replay_samples_per_visit 0
    --reset_adaptation_on_domain_boundary "$reset"
    --lr "${LR:-3.125e-5}"
    --n_train_steps 1
    --episodic false
    --offline_pre_adapt false
    --stochastic_restore_model false
    --inter_domain HomogeneousNoMixture
    --intra_domain_shuffle true
    --record_preadapted_perf false
    --record_first_n_per_domain "$detail_samples"
    --checkpoint_every_domains "$checkpoint_every"
    --num_cpus "$num_cpus"
    --seed "$seed"
    --device "$device"
    "${method_args[@]}"
    "${extra_args[@]}"
  )
  echo "Gradual: method=$current_method reset=$reset domains=$num_domains seed=$seed device=$device"
  if [[ "$dry_run" == "1" ]]; then
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
  else
    "${command[@]}"
  fi
done
