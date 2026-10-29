export type Device = "cpu" | "cuda" | "directml";
export type Page = "studio" | "library" | "training" | "models" | "storage";
export interface Metric {
  learning_rate?: number;
  next_learning_rate?: number;
  learning_rate_reduced?: boolean;
  best_epoch?: number;
  context_probability?: number;
  train_groups?: Record<string, number>;
  validation_groups?: Record<string, number>;
  free_validation?: RolloutSuite;
  performance?: {
    training_seconds: number;
    examples_per_second: number;
    precision: string;
    peak_allocated_mib: number | null;
    peak_reserved_mib: number | null;
  };
  onset_precision?: number;
  onset_recall?: number;
  onset_f1?: number;
  motion_mae?: number;
  object_type_accuracy?: number;
  rhythm_accuracy?: number;
  rhythm_top3_accuracy?: number;
  loss_terms?: Record<string, number>;
  epoch: number;
  train_loss: number;
  validation_loss: number | null;
}
export interface RolloutSuite {
  songs: number;
  attempted: number;
  sections: number;
  max_objects_per_section: number;
  max_objects_full_song: number | null;
  means: Record<string, number>;
  failures: { map_id: string; reason: string }[];
  epoch?: number;
  split?: string;
}
export interface TrainingSummary {
  best_epoch: number;
  best_loss: number | null;
  last_epoch: number;
  next_learning_rate: number;
  stale_epochs: number;
  patience: number;
  stop_reason: string | null;
  training_songs: number;
  validation_songs: number;
  test_songs: number;
}
export interface Job {
  cleanup_warning?: string;
  training_summary?: TrainingSummary;
  warnings?: string[];
  id: string;
  kind: string;
  name: string;
  status: string;
  progress: number;
  message: string;
  metrics: Metric[];
  result?: {
    output_id?: string;
    imported?: number;
    skipped_other_modes?: number;
    error_count?: number;
    errors?: string[];
    calculated?: number;
    failed?: number;
  };
}
export interface Dataset {
  id: string;
  name: string;
  count: number;
  songs: number;
}
export interface StyleProfile {
  maps: number;
  stars_min?: number | null;
  stars_max?: number | null;
  /** Median values in UI units, keyed like the style controls. */
  style: Record<string, number>;
}
export interface Model {
  model_config?: { style_version?: number; width?: number; [key: string]: unknown };
  style_version?: number;
  mapper_vocab?: string[];
  mapper_profiles?: Record<string, StyleProfile>;
  star_profiles?: Record<string, StyleProfile>;
  rated_maps?: number;
  warm_start?: { model_id: string; name?: string; copied_tensors: number; new_tensors: number };
  free_generation_suite?: RolloutSuite;
  training_summary?: TrainingSummary;
  free_generation?: {
    objects: number;
    reference_onset_f1: number;
    heavily_repaired_fraction: number;
    temporal_overlaps: number;
  };
  dataset_id?: string;
  continuable?: boolean;
  training_status?: string;
  parameters?: number;
  test_songs?: number;
  best_epoch?: number;
  architecture: string;
  warnings?: string[];
  id: string;
  name: string;
  maps: number;
  training_songs: number;
  validation_songs: number;
  best_loss: number | null;
  created_at: string;
  metrics: Metric[];
}
export interface Output {
  id: string;
  name: string;
  objects?: number;
  bpm?: number;
  stars?: number | null;
  aim?: number | null;
  speed?: number | null;
  target_stars?: number | null;
  created_at: string;
}
export interface BaseModel {
  id: string;
  name: string;
  architecture?: string;
  current?: boolean;
  style_version?: number;
}
export interface State {
  deletion_blocked: boolean;
  datasets: Dataset[];
  models: Model[];
  /** Every model with weights, including older architectures, as a training start point. */
  base_models?: BaseModel[];
  jobs: Job[];
  outputs: Output[];
  map_count: number;
  device: Device;
  reader_ready: boolean;
  difficulty_ready?: boolean;
  default_path: string;
}
export interface MapItem {
  star_rating: number | null;
  id: string;
  title: string;
  artist: string;
  creator: string;
  version: string;
  objects: number;
  duration: number;
  density: number;
  source: string;
  difficulty: Record<string, number>;
}
export interface Hardware {
  cpu: boolean;
  cuda: boolean;
  directml: boolean;
  name: string;
}
export interface HitObject {
  curve?: number[][] | null;
  repeats?: number;
  new_combo?: boolean;
  x: number;
  y: number;
  time: number;
  slider: boolean;
  duration: number;
  end_x: number;
  end_y: number;
}
export interface PreviewSound {
  time: number;
  end?: number;
  sample: "hitnormal" | "slidertick" | "sliderslide";
  volume: number;
}
export interface Preview {
  sound_events?: PreviewSound[];
  cs?: number;
  ar?: number;
  quality?: {
    timing_source: string;
    timing_sections: number;
    aim_speed_p95: number;
    heavily_repaired_fraction: number;
    temporal_overlaps: number;
    rhythm?: string;
    difficulty?: Record<string, number> | null;
    calibration?: { round: number; stars: number | null }[];
    patterns?: Record<string, number>;
  };
  objects: HitObject[];
  duration: number;
  bpm: number;
  offset: number;
  waveform: number[];
}
