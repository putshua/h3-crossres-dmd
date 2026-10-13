"""Opt-in per-rank sample losses for short data investigations."""
import json
from pathlib import Path
from imaginaire.utils.callback import Callback
from imaginaire.utils import distributed

class H3SampleDiagnostics(Callback):
    def on_train_start(self, model, iteration=0):
        root=Path(self.config.job.path_local)/'sample_diagnostics'
        root.mkdir(exist_ok=True,parents=True)
        self.path=root/f'rank_{distributed.get_rank():05d}.jsonl'
        self.iteration=None

    def on_training_step_batch_end(self, model, data, output_batch, loss, iteration=0):
        if self.iteration!=iteration:
            self.iteration=iteration
            self.microbatch=0
            self.previous={}
        # H3 trainer returns a running sum divided by the accumulation count.
        # Undo that aggregation to retain the original sample's raw objective.
        accumulation=model.gradient_accumulation_steps(iteration)
        current={k:float(v.detach().cpu()) for k,v in output_batch.items()}
        raw={k:(v-self.previous.get(k,0.))*accumulation for k,v in current.items()}
        self.previous=current
        row={'iteration':iteration+1,'microbatch':self.microbatch,
             'rank':distributed.get_rank(),'sample_ids':data['sample_ids'],
             'num_frames':17*((data['video_latents'].shape[2]-2)//5)+5,**raw}
        with self.path.open('a') as f:f.write(json.dumps(row)+'\n')
        self.microbatch+=1
