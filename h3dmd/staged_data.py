"""Iteration-addressable real-AV / prompt-only source switching, including resume."""
from itertools import islice
from torch.utils.data import Dataset, Sampler
from h3dmd.data import CachedH3Dataset, AlternatingFrameSampler


class StagedH3Dataset(Dataset):
    def __init__(self, config):
        d = config.data
        self.pcm = CachedH3Dataset(d.manifest, d.cache_root,
                                  height=d.height, width=d.width, num_frames=d.num_frames)
        self.dmd = CachedH3Dataset(d.dmd_manifest,
                                  height=d.height, width=d.width, num_frames=d.num_frames)
        if any(e.get("shape_only") for e in self.pcm.entries):
            raise ValueError("PCM warmup requires real AV; shape-only input is forbidden")
        if not all(e.get("shape_only") for e in self.dmd.entries):
            raise ValueError("separate DMD source must be explicit prompt-only data")
        self.entries = self.pcm.entries + self.dmd.entries
        self.pcm[0]
        self.dmd[0]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.pcm[index] if index < len(self.pcm) else self.dmd[index - len(self.pcm)]


class StagedH3Sampler(Sampler):
    def __init__(self, dataset, config, *, rank=0, world_size=1):
        self.dataset, self.config = dataset, config
        self.rank, self.world_size, self.start = rank, world_size, 0

    def set_epoch(self, iteration):
        # The inherited trainer supplies the resumed global iteration here.
        self.start = int(iteration)

    def __len__(self):
        c = self.config
        pcm_updates = max(0, min(c.pcm.warmup_steps, c.training.max_steps) - self.start)
        dmd_updates = max(0, c.training.max_steps - max(self.start, c.pcm.warmup_steps))
        return (pcm_updates * c.staged.pcm_gradient_accumulation_steps
                + dmd_updates * c.training.gradient_accumulation_steps)

    def _indices(self, dataset, schedule, *, seed, accumulation=1):
        schedule = [frame for frame in schedule for _ in range(accumulation)]
        epoch = 0
        while True:
            sampler = AlternatingFrameSampler(dataset, schedule, rank=self.rank,
                world_size=self.world_size, seed=seed, shuffle=self.config.data.shuffle)
            sampler.set_epoch(epoch)
            yield from sampler
            epoch += 1

    def __iter__(self):
        c = self.config
        boundary = c.pcm.warmup_steps
        if self.start < boundary:
            # Alternate each PCM update; no reason to repeat one duration six times.
            accum = c.staged.pcm_gradient_accumulation_steps
            stream = self._indices(self.dataset.pcm, [124, 243, 362],
                                   seed=c.training.seed, accumulation=accum)
            yield from islice(stream, self.start * accum,
                              min(boundary, c.training.max_steps) * accum)
        if c.training.max_steps > max(self.start, boundary):
            # Fake warmup alternates durations; then each fake:G block owns one
            # duration. Both indices restart at their own phase boundary.
            if c.training.gradient_accumulation_steps != 1:
                raise ValueError("staged DMD currently requires global accumulation=1")
            fake_end = boundary + c.staged.fake_warmup_steps
            first = max(self.start, boundary)
            stream = self._indices(self.dataset.dmd, [124, 243, 362], seed=c.training.seed + 1)
            for i in islice(stream, first - boundary, max(0, min(fake_end, c.training.max_steps) - boundary)):
                yield len(self.dataset.pcm) + i
            first = max(self.start, fake_end)
            schedule = [f for f in (124, 243, 362)
                        for _ in range(c.dmd.student_update_frequency)]
            stream = self._indices(self.dataset.dmd, schedule, seed=c.training.seed + 2)
            for i in islice(stream, max(0, first - fake_end), max(0, c.training.max_steps - fake_end)):
                yield len(self.dataset.pcm) + i
