"""Five-update FM data diagnostic, covering the exact 320 suspect examples."""
from hydra.core.config_store import ConfigStore

NAME = 'h3_teacher_fm_spike320_bs64_5iter'

def experiment_node():
    return {
        'job': {'name': NAME},
        'model': {'config': {'h3': {
            'method': 'h3_distill',
            'data': {'manifest': '/mnt/data/butong/codebase_h3/data/h3_pcm_spike_window_151_160_320_v1/manifest.jsonl',
                     'dmd_manifest': '', 'height': 320, 'width': 512, 'num_frames': None,
                     'frame_buckets': [], 'size_schedule': [], 'resolution_schedule': [],
                     'batch_size': 1, 'shuffle': False, 'drop_last': False},
            'dmd': {'enabled': False, 'data_mode': 'data', 'rf_loss_weight': 1.,
                    'dmd_loss_weight': 0., 'video_loss_weight': 1., 'audio_loss_weight': .2,
                    'loss_reduction': 'mean', 'rollout_solver': 'euler'},
            'pcm': {'enabled': False, 'warmup_steps': 0},
            'schedule': {'video_shift': 6., 'audio_shift': 3.,
                         'train_video_shift': 6., 'train_audio_shift': 3.,
                         'train_q_min': 0., 'train_q_max': 1.,
                         'inference_q': None, 'num_inference_steps': 50},
            'training': {'grad_clip_enabled': True, 'max_grad_norm': 1.},
        }}},
        'optimizer': {'lr': 2e-6, 'betas': [0., .999], 'weight_decay': .01},
        # Peak PCM LR, no warmup: this test should not hide spikes behind tiny updates.
        'scheduler': {'warm_up_steps': [0], 'f_start': [1.]},
        'trainer': {'max_iter': 5, 'grad_accum_iter': 2, 'logging_iter': 1,
                    'callbacks': {
                        'sample_diagnostics': {'_target_': 'rcm.callbacks.h3_diagnostics.H3SampleDiagnostics'},
                        'fixed_prompt': {'every_n': 5, 'run_at_start': True, 'num_prompts': 4},
                    }},
        'checkpoint': {'save_iter': 5, 'save_latest_iter': 0, 'save_on_exit': True},
    }

def register_experiments():
    ConfigStore.instance().store(group='experiment', name=NAME, package='_global_', node=experiment_node())
