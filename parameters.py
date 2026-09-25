# -*- coding: utf-8 -*-
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Output and runtime.
    parser.add_argument("--job_name", default="tta", type=str)
    parser.add_argument("--root_path", default="./logs", type=str)
    parser.add_argument("--data_path", default="path/to/your/dataset/root", type=str)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_cpus", default=2, type=int)
    parser.add_argument(
        "--checkpoint_every_domains",
        default=5,
        help="save a recovery checkpoint after every N completed test domains; 0 disables it",
        type=int,
    )
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help=(
            "path to a CTTA recovery .pt file or its run directory containing "
            "latest_checkpoint.json"
        ),
        type=str,
    )

    # define the task & model & adaptation & selection method.
    parser.add_argument(
        "--model_name",
        default="vit_base_patch16_224",
        choices=["vit_base_patch16_224"],
        type=str,
    )
    parser.add_argument(
        "--model_adaptation_method",
        default="tent",
        choices=[
            "no_adaptation",
            "tent",
            "sar",
            "cotta",
            "nctta",
            "come",
            "adadem",
            "sar2",
            "adainit",
        ],
        type=str,
    )
    parser.add_argument(
        "--model_selection_method",
        default="last_iterate",
        choices=["last_iterate"],
        type=str,
    )
    parser.add_argument("--task", default="classification", type=str)

    # define the test scenario.
    parser.add_argument("--test_scenario", default=None, type=str)
    parser.add_argument(
        "--base_data_name",
        default="imagenet",
        choices=["imagenet"],
        type=str,
    )
    parser.add_argument("--src_data_name", default="imagenet", choices=["imagenet"], type=str)
    parser.add_argument(
        "--data_names", default="imagenet_c_deterministic-gaussian_noise-5", type=str
    )
    parser.add_argument(
        "--data_wise",
        default="batch_wise",
        choices=["batch_wise", "sample_wise"],
        type=str,
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument(
        "--update_on_domain_partial_batch",
        default=False,
        help=(
            "whether to adapt on the final incomplete batch of each contiguous "
            "domain; false evaluates that batch without updating"
        ),
        type=str2bool,
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=["SGD", "Adam", "AdamW"],
        help=(
            "optional optimizer override; when omitted, use the adaptation "
            "method's registered default"
        ),
        type=str,
    )
    parser.add_argument("--n_train_steps", default=1, type=int)
    parser.add_argument("--offline_pre_adapt", default=False, type=str2bool)
    parser.add_argument("--episodic", default=False, type=str2bool)
    parser.add_argument("--intra_domain_shuffle", default=True, type=str2bool)
    parser.add_argument(
        "--inter_domain",
        default="HomogeneousNoMixture",
        choices=["HomogeneousNoMixture"],
        type=str,
    )
    # Test domain
    parser.add_argument("--domain_sampling_name", default="uniform", type=str)
    parser.add_argument("--domain_sampling_ratio", default=1.0, type=float)
    parser.add_argument(
        "--domain_replay_visits",
        default=1,
        help=(
            "number of round-major visits used to replay each sampled test domain; "
            "1 preserves the ordinary test stream"
        ),
        type=int,
    )
    parser.add_argument(
        "--domain_replay_samples_per_visit",
        default=0,
        help=(
            "samples assigned to each domain visit when domain_replay_visits > 1; "
            "the sampled domain must contain exactly visits times this value"
        ),
        type=int,
    )
    parser.add_argument(
        "--reset_adaptation_on_domain_boundary",
        default=False,
        help=(
            "reset the adaptation model, optimizer, and method-local state at every "
            "known contiguous domain boundary"
        ),
        type=str2bool,
    )
    # Method-independent online adaptation controls.
    parser.add_argument("--stochastic_restore_model", default=False, type=str2bool)
    parser.add_argument("--restore_prob", default=0.01, type=float)
    parser.add_argument(
        "--alpha_teacher",
        default=None,
        help="EMA coefficient for CoTTA's weight-averaged teacher.",
        type=float,
    )
    parser.add_argument(
        "--threshold_cotta",
        default=None,
        help=(
            "source-anchor confidence threshold below which CoTTA uses its "
            "augmentation-averaged teacher prediction"
        ),
        type=float,
    )
    parser.add_argument(
        "--cotta_loss",
        default=None,
        choices=["teacher_ce", "symmetric_ce"],
        help="official CoTTA distillation loss variant for the target benchmark.",
        type=str,
    )
    parser.add_argument("--fishers", default=False, type=str2bool)
    parser.add_argument(
        "--fisher_size",
        default=5000,
        type=int,
        help="number of samples to compute fisher information matrix.",
    )
    parser.add_argument(
        "--fisher_alpha",
        type=float,
        default=1.5,
        help="the trade-off between entropy and regularization loss",
    )
    # method-wise hparams
    parser.add_argument(
        "--aug_size",
        default=32,
        help="number of per-image augmentation operations in memo and ttt",
        type=int,
    )
    # AdaInit: initialization controller, backend, and ablation switches.
    parser.add_argument(
        "--adainit_backend",
        default=None,
        choices=["tent", "come", "sar", "adadem", "nctta"],
        help="TTA update rule wrapped by AdaInit (default: nctta).",
        type=str,
    )
    parser.add_argument(
        "--adadem_pi",
        default=None,
        help="AdaDEM class-marginal exponential update rate.",
        type=float,
    )
    parser.add_argument(
        "--adadem_mode",
        default=None,
        choices=["adadem", "adadem-norm", "adadem-mec"],
        help="AdaDEM loss variant (default: adadem).",
        type=str,
    )
    parser.add_argument(
        "--adainit_signature_momentum",
        default=None,
        help="alpha in the momentum domain signature update.",
        type=float,
    )
    parser.add_argument(
        "--adainit_signature_window_size",
        default=None,
        help="recent samples pooled into the detection/retrieval signature.",
        type=int,
    )
    parser.add_argument(
        "--adainit_require_full_signature_window",
        default=None,
        help=(
            "delay drift-detector initialization until the rolling signature "
            "window is full; enables a strict buffer-level detector"
        ),
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_signature_distance",
        default=None,
        choices=["wasserstein", "standardized", "zoa_symmetric_kl"],
        help="distance used consistently by detection and history retrieval.",
        type=str,
    )
    parser.add_argument(
        "--adainit_detector_distance",
        default=None,
        choices=["same", "wasserstein", "standardized", "zoa_symmetric_kl"],
        help=(
            "distance used only by drift detection; same reuses the cache/retrieval "
            "signature distance"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_detector_signature_source",
        default=None,
        choices=["window", "instantaneous"],
        help=(
            "signature presented to the drift detector: the rolling retrieval "
            "window or the current sample's spatial feature statistics"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_score_momentum",
        default=None,
        help="momentum for segment-local EW drift-score statistics.",
        type=float,
    )
    parser.add_argument(
        "--adainit_drift_beta",
        default=None,
        help="number of EW standard deviations used by the drift threshold.",
        type=float,
    )
    parser.add_argument(
        "--adainit_fixed_drift_threshold",
        default=None,
        help="fixed distribution-distance threshold used by the ZOA-style detector.",
        type=float,
    )
    parser.add_argument(
        "--adainit_drift_min_ratio",
        default=None,
        help="required multiplicative exceedance over the adaptive threshold.",
        type=float,
    )
    parser.add_argument("--adainit_eps", default=None, type=float)
    parser.add_argument(
        "--adainit_min_reference_samples",
        default=None,
        help="accepted scores required before adaptive triggering is enabled.",
        type=int,
    )
    parser.add_argument(
        "--adainit_drift_confirmations",
        default=None,
        help=(
            "consecutive threshold exceedances required to promote the provisional "
            "MDS; recovery selection is proposed at the first exceedance"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_detector_mode",
        default=None,
        choices=["adaptive", "fixed", "disabled", "periodic", "hybrid"],
        help="detector ablation: proposed, never trigger, or fixed-period trigger.",
        type=str,
    )
    parser.add_argument(
        "--adainit_periodic_interval",
        default=None,
        help="number of samples between triggers in periodic mode.",
        type=int,
    )
    parser.add_argument(
        "--adainit_periodic_include_source",
        default=None,
        help=(
            "allow source as a periodic-probe candidate when the detector "
            "score also satisfies adainit_source_min_drift_ratio"
        ),
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_trigger_cooldown",
        default=None,
        help="minimum quiet period after a selection event.",
        type=int,
    )
    parser.add_argument("--adainit_cache_size", default=None, type=int)
    parser.add_argument(
        "--adainit_cache_eviction",
        default=None,
        choices=["signature", "parameter_cosine", "knowledge_fingerprint"],
        help=(
            "bounded-cache eviction rule: signature diversity or ZOA-style "
            "adaptation-vector/knowledge-fingerprint diversity"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_cache_admission",
        default=None,
        choices=["mds_stable", "periodic_health"],
        help=(
            "cache-write controller: the legacy MDS-stable schedule or an "
            "MDS-independent periodic schedule with an unlabeled health gate"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_knowledge_fingerprint",
        default=None,
        choices=["disabled", "source_delta", "local_delta"],
        help=(
            "adaptable-parameter response stored with each cache state; "
            "local_delta uses a fixed causal singleton window"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_fingerprint_window",
        default=None,
        help="singleton updates represented by one local knowledge fingerprint.",
        type=int,
    )
    parser.add_argument(
        "--adainit_fingerprint_magnitude_weight",
        default=None,
        help="weight of layer-wise log update-magnitude distance.",
        type=float,
    )
    parser.add_argument(
        "--adainit_cache_insert_interval",
        default=None,
        help="periodically insert one lightweight adaptable-parameter anchor every N samples.",
        type=int,
    )
    parser.add_argument(
        "--adainit_cache_signature_source",
        default=None,
        choices=["window", "mds"],
        help="pair anchors with the recent window or the slow domain prototype.",
        type=str,
    )
    parser.add_argument(
        "--adainit_cache_min_segment_samples",
        default=None,
        help="minimum stable-segment age before a periodic anchor is inserted.",
        type=int,
    )
    parser.add_argument(
        "--adainit_cache_on_detection",
        default=None,
        help="save the pre-shift terminal trajectory as a cache anchor.",
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_cache_optimizer_state",
        default=None,
        help="cache and resume the backend optimizer together with model state.",
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_cache_health_mode",
        default=None,
        choices=["disabled", "infomax_update"],
        help=(
            "label-free cache admission health check based on recent prediction "
            "diversity and split-window update consistency"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_cache_health_min_information",
        default=None,
        help="minimum recent-sample predictive mutual information for cache admission.",
        type=float,
    )
    parser.add_argument(
        "--adainit_cache_health_max_concentration",
        default=None,
        help="maximum probability mass assigned to one class across the cache window.",
        type=float,
    )
    parser.add_argument(
        "--adainit_cache_health_min_update_cosine",
        default=None,
        help="minimum cosine consistency between the two half-window updates.",
        type=float,
    )
    parser.add_argument(
        "--adainit_cache_health_min_fingerprint_norm",
        default=None,
        help=(
            "minimum total norm of the stored knowledge fingerprint; use a "
            "small positive value to reject source-duplicate anchors"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_cache_health_update_weight",
        default=None,
        help="weight of nonnegative update consistency in the stored health score.",
        type=float,
    )
    parser.add_argument(
        "--adainit_history_retrieval",
        default=None,
        choices=["signature", "knowledge_fingerprint"],
        help="cache preselection key used before hard query-conditioned scoring.",
        type=str,
    )
    parser.add_argument(
        "--adainit_history_fingerprint_max_distance",
        default=None,
        help="maximum knowledge-fingerprint distance; negative disables the gate.",
        type=float,
    )
    parser.add_argument(
        "--adainit_history_fingerprint_max_ratio",
        default=None,
        help="maximum nearest/median fingerprint-distance ratio; negative disables it.",
        type=float,
    )
    parser.add_argument(
        "--adainit_max_history_candidates",
        default=None,
        help=(
            "signature-ranked cache states evaluated by the deployable hard "
            "selector; 0 evaluates the full cache"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_history_min_age",
        default=None,
        help=(
            "minimum samples elapsed since cache insertion before a state may "
            "serve as a hard rollback initialization; 0 disables the age gate"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_history_match_ratio",
        default=None,
        help=(
            "maximum nearest/median signature-distance ratio for a familiar-domain "
            "match; negative disables the gate"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_history_max_distance",
        default=None,
        help="maximum absolute standardized distance for a familiar domain.",
        type=float,
    )
    parser.add_argument(
        "--adainit_history_radius_multiplier",
        default=None,
        help="maximum query-distance/local-radius ratio; negative disables it.",
        type=float,
    )
    parser.add_argument(
        "--adainit_source_for_unseen_only",
        default=None,
        help="exclude source when a familiar historical domain is matched.",
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_source_min_drift_ratio",
        default=None,
        help="minimum drift-score/threshold ratio before source is considered.",
        type=float,
    )
    parser.add_argument(
        "--adainit_source_min_current_entropy",
        default=None,
        help="minimum current post-update marginal entropy before source may win.",
        type=float,
    )
    parser.add_argument(
        "--adainit_history_score_bonus",
        default=None,
        help="signature-prior bonus applied to a matched history recovery score.",
        type=float,
    )
    parser.add_argument(
        "--adainit_history_min_drift_ratio",
        default=None,
        help=(
            "minimum detector score/threshold ratio required before a cached "
            "history may be selected; 0 disables this hard feasibility gate"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_context_diversity_weight",
        default=None,
        help=(
            "weight of recent-sample prediction diversity subtracted from the "
            "transformed-view recovery score; 0 disables it"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_history_max_entropy_increase",
        default=None,
        help="maximum history-vs-current post-update marginal entropy increase.",
        type=float,
    )
    parser.add_argument(
        "--adainit_periodic_selection_margin",
        default=None,
        help="minimum recovery-score gain for a periodic hard history switch.",
        type=float,
    )
    parser.add_argument(
        "--adainit_periodic_history_max_entropy_increase",
        default=None,
        help="entropy guard for periodic hard history switches.",
        type=float,
    )
    parser.add_argument(
        "--adainit_prefer_matched_history",
        default=None,
        help="directly reuse a confidently matched historical trajectory.",
        type=str2bool,
    )
    parser.add_argument("--adainit_num_views", default=None, type=int)
    parser.add_argument(
        "--adainit_selection_window",
        default=None,
        help="recent observations averaged in counterfactual recovery scoring.",
        type=int,
    )
    parser.add_argument(
        "--adainit_counterfactual_steps", default=None, type=int
    )
    parser.add_argument(
        "--adainit_counterfactual_batch_source",
        default=None,
        choices=["current", "buffer"],
        help=(
            "samples used by each temporary candidate update; buffer replays the "
            "causal selection window only inside discarded counterfactual branches"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_candidate_eval_batch_size",
        default=None,
        help=(
            "micro-batch size for candidate-view inference; 0 evaluates all views "
            "at once"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_audit_candidate_accuracy",
        default=None,
        help=(
            "diagnostic only: log each candidate's labeled view-ensemble accuracy; "
            "labels never enter detection, scoring, or selection"
        ),
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_initialization_selector",
        default=None,
        choices=[
            "surrogate",
            "causal_horizon_entropy",
            "sequential_view_evidence",
            "oracle_future_accuracy",
        ],
        help=(
            "initialization selector; sequential_view_evidence accumulates causal "
            "singleton evidence, applies a confidence-bound safety gate after every "
            "sample, and hard-selects one complete branch; causal_horizon_entropy "
            "is the fixed-horizon ablation; oracle_future_accuracy is a label-leaking "
            "diagnostic upper bound and is invalid as a deployable TTA result"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_evidence_horizon",
        default=None,
        help=(
            "maximum rolling causal-evidence window retained per independently "
            "evolved hard initialization branch"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_evidence_min_samples",
        default=None,
        help=(
            "minimum causal singleton observations before the sequential "
            "confidence-bound selector may commit a non-current initialization"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_evidence_max_samples",
        default=None,
        help=(
            "maximum duration of one recovery proposal; if no candidate clears "
            "the safety gate, AdaInit keeps the uninterrupted current trajectory"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_evidence_confidence_scale",
        default=None,
        help=(
            "standard-error multiplier in the lower confidence bound on each "
            "candidate's marginal-entropy advantage over current"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_evidence_warmup",
        default=None,
        help=(
            "initial causal branch samples used for adaptation but excluded from "
            "the hard-selection evidence score"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_evidence_timing",
        default=None,
        choices=["post_update", "prequential"],
        help=(
            "score each branch after updating on the same singleton (paper-style) "
            "or before updating it using only prior causal observations"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_sequential_score",
        default=None,
        choices=["entropy", "view_infomax"],
        help=(
            "rolling hard-selection score; view_infomax combines per-sample "
            "entropy, View JSD, and cross-sample predictive diversity"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_sequential_view_jsd_weight",
        default=None,
        help="View JSD weight in the view_infomax sequential score.",
        type=float,
    )
    parser.add_argument(
        "--adainit_sequential_context_weight",
        default=None,
        help="cross-sample marginal-entropy reward in the view_infomax score.",
        type=float,
    )
    parser.add_argument(
        "--adainit_oracle_horizon",
        default=None,
        help="future labeled samples replayed per candidate by the diagnostic oracle.",
        type=int,
    )
    parser.add_argument(
        "--adainit_oracle_max_history_candidates",
        default=None,
        help=(
            "nearest cache entries replayed by the oracle; 0 evaluates the full cache"
        ),
        type=int,
    )
    parser.add_argument(
        "--adainit_oracle_include_source",
        default=None,
        help="include the source initialization in the future-accuracy oracle.",
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_oracle_min_accuracy_gain",
        default=None,
        help="minimum Acc@H gain over current required for an oracle switch.",
        type=float,
    )
    parser.add_argument(
        "--adainit_oracle_log_surrogates",
        default=None,
        help=(
            "diagnostic only: align every candidate's causal entropy/view proxy "
            "with its future-label Oracle utility"
        ),
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_oracle_log_future_view_surrogates",
        default=None,
        help=(
            "diagnostic only: record each hard Oracle branch's transformed-view "
            "statistics over its causal future replay; never affects selection"
        ),
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_native_anchor",
        default=None,
        help=(
            "maintain an uninterrupted native-backend parameter trajectory as "
            "one additional hard initialization candidate"
        ),
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_native_anchor_selection_margin",
        default=None,
        help=(
            "minimum recovery-score gain required to hard-switch back to the "
            "uninterrupted native-backend initialization"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_selection_margin",
        default=None,
        help=(
            "minimum marginal-entropy improvement required to replace the "
            "current state; 0 recovers the paper argmin rule"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_source_selection_margin",
        default=None,
        help="minimum recovery-score improvement required to switch to source.",
        type=float,
    )
    parser.add_argument(
        "--adainit_reset_current_optimizer",
        default=None,
        help=(
            "whether selecting the current model also starts a fresh backend "
            "optimizer; true matches the draft equation, while false preserves "
            "the uninterrupted optimizer trajectory"
        ),
        type=str2bool,
    )
    parser.add_argument(
        "--adainit_max_marginal_entropy_increase",
        default=None,
        help=(
            "maximum allowed post-update marginal-entropy increase when a "
            "non-current candidate wins another surrogate; a negative value "
            "disables this guard"
        ),
        type=float,
    )
    parser.add_argument(
        "--adainit_recovery_score",
        default=None,
        choices=["post_entropy", "entropy_delta", "view_jsd", "backend_loss"],
        help=(
            "counterfactual surrogate: entropy after the temporary update "
            "(paper rule), its change from before the update, transformed-view "
            "JSD, or the wrapped backend's unlabeled adaptation loss"
        ),
        type=str,
    )
    parser.add_argument(
        "--adainit_candidate_mode",
        default=None,
        choices=[
            "full",
            "current_only",
            "source_only",
            "nearest_history",
            "counterfactual_history_only",
        ],
        help="candidate/selection ablation; full is the proposed method.",
        type=str,
    )
    parser.add_argument(
        "--nu",
        default=None,
        help="method-specific nu coefficient; uses the algorithm default when omitted",
        type=float,
    )
    parser.add_argument(
        "--eta",
        default=None,
        help="method-specific eta coefficient; uses the algorithm default when omitted",
        type=float,
    )
    parser.add_argument(
        "--sar_margin_e0",
        default=None,
        help="entropy reliability threshold used by SAR/SAR2",
        type=float,
    )
    parser.add_argument(
        "--thre_ent",
        default=None,
        help="entropy filter threshold used by NCTTA",
        type=float,
    )
    parser.add_argument(
        "--margin_ent",
        default=None,
        help="entropy reweighting margin used by NCTTA",
        type=float,
    )
    parser.add_argument(
        "--top_k",
        default=None,
        help="number of top classes retained by NCTTA's mixup target",
        type=int,
    )
    parser.add_argument(
        "--K",
        default=None,
        help="number of dataset classes used by COME",
        type=int,
    )
    # metrics
    parser.add_argument(
        "--record_preadapted_perf",
        default=False,
        help="record performance on the local batch prior to implementing test-time adaptation.",
        type=str2bool,
    )
    parser.add_argument(
        "--record_first_n_per_domain",
        default=1000,
        help=(
            "for sample-wise tests, record the first N samples of each domain; "
            "for batch-wise tests, any positive value records every batch once; "
            "use 0 to disable detailed records"
        ),
        type=int,
    )
    # misc
    parser.add_argument(
        "--grad_checkpoint",
        default=False,
        help="Trade computation for gpu space.",
        type=str2bool,
    )
    parser.add_argument("--debug", default=False, help="Display logs.", type=str2bool)

    # parse conf.
    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
