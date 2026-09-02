"""scheduler 单位契约与 world-size 不变性回归测试。

历史 bug：8 卡校准 run 把 warmup 乘了 num_processes、total horizon 没乘，
导致 LR 在约 total/8 个 global update 就跌到 1e-8 floor。
这些测试锁定「每个成功的 global optimizer update 看到的 LR 与 world size 无关」。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from f5_tts.model.trainer import (
    LR_FLOOR_FACTOR,
    Trainer,
    build_scheduler,
    resolve_scheduler_contract,
    resolve_split_batches,
)

LEARNING_RATE = 1e-5
FLOOR_FACTOR = LR_FLOOR_FACTOR


class SchedulerHarness:
    """模拟 AcceleratedScheduler：每个成功 optimizer step 推进 multiplier 个 inner step。

    参考 accelerate/scheduler.py step()：split_batches=False 时循环 num_processes 次，
    且 optimizer.step_was_skipped 为真时直接 return（不推进 LR）。
    """

    def __init__(self, *, global_warmup, global_total, num_processes, split_batches=False):
        self.contract = resolve_scheduler_contract(
            global_warmup_updates=global_warmup,
            global_total_updates=global_total,
            num_processes=num_processes,
            split_batches=split_batches,
        )
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.optimizer = torch.optim.AdamW([self.param], lr=LEARNING_RATE)
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_updates=self.contract["warmup_updates"],
            total_updates=self.contract["total_updates"],
        )
        self.global_update = 0

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def optimizer_step(self, *, skipped=False):
        """走一次 global optimizer update；skipped 模拟 AMP overflow 跳步。"""
        if skipped:
            return  # LR 与训练身份都不推进
        for _ in range(self.contract["scheduler_step_multiplier"]):
            self.scheduler.step()
        self.global_update += 1

    def trace(self, until, *, anchors=None):
        anchors = set(anchors or range(until + 1))
        seen = {0: self.lr} if 0 in anchors else {}
        while self.global_update < until:
            self.optimizer_step()
            if self.global_update in anchors:
                seen[self.global_update] = self.lr
        return seen


class SchedulerContractTests(unittest.TestCase):
    def test_multiplier_follows_split_batches(self):
        without = resolve_scheduler_contract(
            global_warmup_updates=100, global_total_updates=5000, num_processes=8, split_batches=False
        )
        self.assertEqual(without["scheduler_step_multiplier"], 8)
        self.assertEqual(without["warmup_updates"], 800)
        self.assertEqual(without["total_updates"], 40000)

        with_split = resolve_scheduler_contract(
            global_warmup_updates=100, global_total_updates=5000, num_processes=8, split_batches=True
        )
        self.assertEqual(with_split["scheduler_step_multiplier"], 1)
        self.assertEqual(with_split["warmup_updates"], 100)
        self.assertEqual(with_split["total_updates"], 5000)

    def test_global_semantics_preserved_for_forensics(self):
        contract = resolve_scheduler_contract(
            global_warmup_updates=100, global_total_updates=5000, num_processes=8, split_batches=False
        )
        self.assertEqual(contract["global_warmup_updates"], 100)
        self.assertEqual(contract["global_total_updates"], 5000)
        self.assertEqual(contract["num_processes"], 8)
        self.assertFalse(contract["split_batches"])

    def test_warmup_clamped_in_global_units(self):
        # 短 smoke：warmup 请求超过 horizon 时必须在 global 单位下 clamp，
        # 否则 warmup/total 单位不一致会得到负的 decay total_iters。
        contract = resolve_scheduler_contract(
            global_warmup_updates=500, global_total_updates=120, num_processes=8, split_batches=False
        )
        self.assertEqual(contract["global_warmup_updates"], 120)
        self.assertEqual(contract["global_warmup_updates_requested"], 500)
        self.assertEqual(contract["warmup_updates"], 960)
        self.assertEqual(contract["total_updates"], 960)
        self.assertLessEqual(contract["warmup_updates"], contract["total_updates"])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resolve_scheduler_contract(
                global_warmup_updates=100, global_total_updates=5000, num_processes=0, split_batches=False
            )
        with self.assertRaises(ValueError):
            resolve_scheduler_contract(
                global_warmup_updates=-1, global_total_updates=5000, num_processes=8, split_batches=False
            )
        with self.assertRaises(ValueError):
            resolve_scheduler_contract(
                global_warmup_updates=100, global_total_updates=-1, num_processes=8, split_batches=False
            )


class WorldSizeInvarianceTests(unittest.TestCase):
    ANCHORS = (0, 1, 50, 99, 100, 101, 500, 625, 1000, 2500, 4999, 5000)

    def trace_for(self, num_processes, *, split_batches=False):
        harness = SchedulerHarness(
            global_warmup=100, global_total=5000, num_processes=num_processes, split_batches=split_batches
        )
        return harness.trace(max(self.ANCHORS), anchors=self.ANCHORS)

    def test_one_and_eight_gpu_traces_match(self):
        single = self.trace_for(1)
        eight = self.trace_for(8)
        for update in self.ANCHORS:
            with self.subTest(global_update=update):
                self.assertAlmostEqual(single[update], eight[update], delta=1e-16)

    def test_split_batches_matches_single_process(self):
        single = self.trace_for(1)
        split = self.trace_for(8, split_batches=True)
        for update in self.ANCHORS:
            with self.subTest(global_update=update):
                self.assertAlmostEqual(single[update], split[update], delta=1e-16)

    def test_warmup_then_decay_shape(self):
        trace = self.trace_for(8)
        self.assertLess(trace[1], trace[50])
        self.assertLess(trace[50], trace[100])
        self.assertAlmostEqual(trace[100], LEARNING_RATE, delta=1e-12)
        self.assertGreater(trace[100], trace[101])
        self.assertGreater(trace[1000], trace[2500])
        self.assertGreater(trace[2500], trace[4999])

    def test_floor_not_reached_before_horizon(self):
        # 这是历史 bug 的直接门禁：8 卡下 LR 不得在 ~625 步就到 floor。
        trace = self.trace_for(8)
        floor = LEARNING_RATE * FLOOR_FACTOR
        self.assertGreater(trace[625], 1e-6)
        self.assertGreater(trace[1000], 1e-6)
        self.assertGreater(trace[4999], floor)
        self.assertAlmostEqual(trace[5000], floor, delta=1e-15)

    def test_regression_reproduces_historical_bug(self):
        # 只乘 warmup 不乘 total 时，8 卡在 global 500 的 LR 正是坏 run
        # checkpoint 里记录的 2.38095245714e-06，且 1000 步后已在 floor。
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=LEARNING_RATE)
        scheduler = build_scheduler(optimizer, warmup_updates=100 * 8, total_updates=5000)
        lrs = {}
        for update in range(1, 1001):
            for _ in range(8):
                scheduler.step()
            lrs[update] = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(lrs[500], 2.38095245714e-06, delta=1e-15)
        self.assertLess(lrs[1000], 1e-12)


class GradientAccumulationAndSkipTests(unittest.TestCase):
    ANCHORS = (0, 1, 100, 101, 500, 1000)

    def test_gradient_accumulation_does_not_change_global_trace(self):
        # scheduler 只在成功的 global optimizer update 上推进，
        # 因此 micro-batch 累积倍数不该改变 global 轨迹。
        baseline = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=8).trace(
            max(self.ANCHORS), anchors=self.ANCHORS
        )

        accumulated = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=8)
        seen = {0: accumulated.lr}
        while accumulated.global_update < max(self.ANCHORS):
            accumulated.optimizer_step()  # 4 个 micro-batch 才触发一次，这里等价于一次
            if accumulated.global_update in self.ANCHORS:
                seen[accumulated.global_update] = accumulated.lr

        for update in self.ANCHORS:
            with self.subTest(global_update=update):
                self.assertAlmostEqual(baseline[update], seen[update], delta=1e-16)

    def test_skipped_optimizer_step_advances_nothing(self):
        harness = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=8)
        for _ in range(50):
            harness.optimizer_step()
        lr_before, update_before = harness.lr, harness.global_update

        for _ in range(5):
            harness.optimizer_step(skipped=True)

        self.assertEqual(harness.global_update, update_before)
        self.assertAlmostEqual(harness.lr, lr_before, delta=1e-18)

    def test_skipped_steps_do_not_desynchronise_lr_from_update(self):
        clean = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=8)
        clean.trace(120)

        noisy = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=8)
        while noisy.global_update < 120:
            noisy.optimizer_step(skipped=True)
            noisy.optimizer_step()

        self.assertEqual(clean.global_update, noisy.global_update)
        self.assertAlmostEqual(clean.lr, noisy.lr, delta=1e-18)


class SchedulerResumeTests(unittest.TestCase):
    def resume_matches_uninterrupted(self, num_processes):
        uninterrupted = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=num_processes)
        uninterrupted.trace(200)

        first = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=num_processes)
        first.trace(120)
        state = {
            "optimizer": first.optimizer.state_dict(),
            "scheduler": first.scheduler.state_dict(),
            "update": first.global_update,
        }

        resumed = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=num_processes)
        resumed.optimizer.load_state_dict(state["optimizer"])
        resumed.scheduler.load_state_dict(state["scheduler"])
        resumed.global_update = state["update"]
        self.assertAlmostEqual(resumed.lr, first.lr, delta=1e-18)

        resumed.trace(200)
        self.assertEqual(resumed.global_update, uninterrupted.global_update)
        self.assertAlmostEqual(resumed.lr, uninterrupted.lr, delta=1e-16)

    def test_single_gpu_resume(self):
        self.resume_matches_uninterrupted(1)

    def test_eight_gpu_resume(self):
        self.resume_matches_uninterrupted(8)

    def test_resume_lr_matches_across_world_sizes(self):
        single = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=1)
        single.trace(120)
        eight = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=8)
        eight.trace(120)
        self.assertAlmostEqual(single.lr, eight.lr, delta=1e-16)


class ShortHorizonSmokeTests(unittest.TestCase):
    def test_smoke_horizon_keeps_planned_schedule(self):
        # smoke 只跑 120 步，但 scheduler horizon 仍按 5000 规划，
        # 因此 120 步时 LR 必须还在正常区间，而不是已经到 floor。
        harness = SchedulerHarness(global_warmup=100, global_total=5000, num_processes=8)
        trace = harness.trace(120, anchors=(1, 50, 100, 101, 120))
        self.assertLess(trace[1], trace[100])
        self.assertGreater(trace[100], trace[120])
        self.assertGreater(trace[120], 1e-6)

    def test_warmup_only_horizon(self):
        # horizon <= warmup 时只建 warmup scheduler，末端应到达 base LR。
        harness = SchedulerHarness(global_warmup=500, global_total=120, num_processes=8)
        self.assertEqual(harness.contract["warmup_updates"], harness.contract["total_updates"])
        harness.trace(120)
        self.assertAlmostEqual(harness.lr, LEARNING_RATE, delta=1e-12)


class StopAfterUpdatesTests(unittest.TestCase):
    """smoke 的 stop_after_updates 必须只提前停步，绝不缩短 scheduler horizon。

    如果它参与了 horizon 计算，120 步 smoke 验证的就是一条压缩过的
    schedule，而不是将要真跑的 5000 步 schedule——那样 smoke 就失去意义。
    """

    class FakeTrainer:
        """只带 predicate 所需字段的替身，避免构造完整 Trainer。"""

        def __init__(self, *, epochs=1, grad_accumulation_steps=1, max_updates=None, stop_after_updates=None):
            self.epochs = epochs
            self.grad_accumulation_steps = grad_accumulation_steps
            self.max_updates = max_updates
            self.stop_after_updates = stop_after_updates

        # 必须沿用原方法名：_should_stop_training 内部会调用 self._reached_max_updates。
        _training_update_horizon = Trainer._training_update_horizon
        _reached_max_updates = Trainer._reached_max_updates
        _should_stop_training = Trainer._should_stop_training

    def test_stop_after_updates_does_not_shrink_horizon(self):
        smoke = self.FakeTrainer(max_updates=5000, stop_after_updates=120)
        full = self.FakeTrainer(max_updates=5000)
        # 两者 horizon 必须一致，scheduler 才会规划同一条曲线。
        self.assertEqual(smoke._training_update_horizon(40000), full._training_update_horizon(40000))
        self.assertEqual(smoke._training_update_horizon(40000), 5000)

    def test_smoke_contract_matches_full_run_contract(self):
        smoke = self.FakeTrainer(max_updates=5000, stop_after_updates=120)
        contract = resolve_scheduler_contract(
            global_warmup_updates=100,
            global_total_updates=smoke._training_update_horizon(40000),
            num_processes=8,
            split_batches=False,
        )
        self.assertEqual(contract["warmup_updates"], 800)
        self.assertEqual(contract["total_updates"], 40000)

    def test_stop_after_updates_stops_the_loop(self):
        smoke = self.FakeTrainer(max_updates=5000, stop_after_updates=120)
        self.assertFalse(smoke._should_stop_training(119))
        self.assertTrue(smoke._should_stop_training(120))
        self.assertTrue(smoke._should_stop_training(121))
        # max_updates 尚未到达，所以停下来的原因只能是 stop_after_updates。
        self.assertFalse(smoke._reached_max_updates(120))

    def test_max_updates_still_stops_without_stop_after(self):
        full = self.FakeTrainer(max_updates=5000)
        self.assertFalse(full._should_stop_training(4999))
        self.assertTrue(full._should_stop_training(5000))

    def test_no_limits_never_stops_early(self):
        unlimited = self.FakeTrainer()
        self.assertFalse(unlimited._should_stop_training(0))
        self.assertFalse(unlimited._should_stop_training(10**6))


class SplitBatchesResolutionTests(unittest.TestCase):
    """split_batches 决定 step multiplier，读不到时必须报错而不是静默默认 False。

    静默默认在当前配置下"碰巧"正确，但那正是历史 bug 的形状：
    一个没人察觉的错误单位，安静地毁掉整条 LR 曲线。
    """

    class Config:
        def __init__(self, split_batches):
            self.split_batches = split_batches

    def test_prefers_dataloader_config(self):
        accelerator = types.SimpleNamespace(dataloader_config=self.Config(True), split_batches=False)
        self.assertTrue(resolve_split_batches(accelerator))

    def test_falls_back_to_accelerator_attribute(self):
        accelerator = types.SimpleNamespace(split_batches=True)
        self.assertTrue(resolve_split_batches(accelerator))

    def test_raises_when_neither_source_exists(self):
        with self.assertRaises(AttributeError):
            resolve_split_batches(types.SimpleNamespace())

    def test_coerces_to_bool(self):
        accelerator = types.SimpleNamespace(dataloader_config=self.Config(0))
        self.assertIs(resolve_split_batches(accelerator), False)


class LrTraceTests(unittest.TestCase):
    """The LR trace is the instrument whose absence let this bug run 5000 updates.

    It must stay off by default (so it cannot perturb a real run's artifacts) and,
    when on, be a rank-0 factual record that can be compared across world sizes.
    """

    class FakeTrainer:
        def __init__(self, *, lr_trace_path=None, is_main=True, lr=1e-5):
            self.lr_trace_path = lr_trace_path
            self.is_main = is_main
            self.scheduler = types.SimpleNamespace(get_last_lr=lambda: [lr])

        _write_lr_trace = Trainer._write_lr_trace
        _record_lr = Trainer._record_lr

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="lr-trace-test-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def read(self, path):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]

    def test_disabled_by_default_writes_nothing(self):
        trainer = self.FakeTrainer()
        trainer._write_lr_trace({"record": "scheduler_contract"})
        trainer._record_lr(1)
        self.assertEqual(sorted(self.root.iterdir()), [])

    def test_records_updates_in_order(self):
        path = self.root / "lr_trace.jsonl"
        trainer = self.FakeTrainer(lr_trace_path=str(path))
        for update in (1, 2, 3):
            trainer._record_lr(update)
        records = self.read(path)
        self.assertEqual([r["global_update"] for r in records], [1, 2, 3])
        self.assertTrue(all(r["lr"] == 1e-5 for r in records))

    def test_non_main_rank_writes_nothing(self):
        # Eight ranks appending to one file would interleave into nonsense.
        path = self.root / "lr_trace.jsonl"
        self.FakeTrainer(lr_trace_path=str(path), is_main=False)._record_lr(1)
        self.assertFalse(path.exists())

    def test_creates_parent_directory(self):
        # The first record is written while the scheduler is built, before any
        # checkpoint save has created the run directory.
        path = self.root / "not_yet" / "lr_trace.jsonl"
        self.FakeTrainer(lr_trace_path=str(path))._record_lr(1)
        self.assertTrue(path.is_file())

    def test_contract_record_is_written_first(self):
        path = self.root / "lr_trace.jsonl"
        trainer = self.FakeTrainer(lr_trace_path=str(path))
        contract = resolve_scheduler_contract(
            global_warmup_updates=100, global_total_updates=5000, num_processes=8, split_batches=False
        )
        trainer._write_lr_trace({"record": "scheduler_contract", **contract})
        trainer._record_lr(1)
        records = self.read(path)
        self.assertEqual(records[0]["record"], "scheduler_contract")
        self.assertEqual(records[0]["warmup_updates"], 800)
        self.assertEqual(records[0]["total_updates"], 40000)
        self.assertEqual(records[1]["global_update"], 1)

    def test_survives_being_reopened(self):
        # Written per record so an interrupted run still leaves a readable trace.
        path = self.root / "lr_trace.jsonl"
        self.FakeTrainer(lr_trace_path=str(path))._record_lr(1)
        self.FakeTrainer(lr_trace_path=str(path))._record_lr(2)
        self.assertEqual([r["global_update"] for r in self.read(path)], [1, 2])


class SmokeDiagnosticPowerTests(unittest.TestCase):
    """A world-size smoke is only meaningful if it runs past warmup.

    Under the historical bug, warmup was scaled correctly and only the decay
    horizon was wrong, so 1-GPU and 8-GPU traces are *identical* for the whole
    warmup phase. A smoke of <= num_warmup_updates would therefore have passed
    against the broken code. These tests pin that reasoning so a future change
    that shortens the smoke below the warmup boundary fails loudly instead of
    silently turning the check into theatre.
    """

    WARMUP = 100
    TOTAL = 5000

    def buggy_eight_gpu_trace(self, updates):
        # warmup scaled by world size, horizon left in global units -- the bug.
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=LEARNING_RATE)
        scheduler = build_scheduler(optimizer, warmup_updates=self.WARMUP * 8, total_updates=self.TOTAL)
        trace = {}
        for update in range(1, updates + 1):
            for _ in range(8):
                scheduler.step()
            trace[update] = optimizer.param_groups[0]["lr"]
        return trace

    def reference_trace(self, updates):
        harness = SchedulerHarness(global_warmup=self.WARMUP, global_total=self.TOTAL, num_processes=1)
        return harness.trace(updates, anchors=range(1, updates + 1))

    def test_warmup_phase_cannot_reveal_the_bug(self):
        buggy = self.buggy_eight_gpu_trace(self.WARMUP)
        reference = self.reference_trace(self.WARMUP)
        for update in range(1, self.WARMUP + 1):
            with self.subTest(global_update=update):
                self.assertAlmostEqual(buggy[update], reference[update], delta=1e-15)

    def test_bug_becomes_visible_on_the_first_decay_update(self):
        buggy = self.buggy_eight_gpu_trace(self.WARMUP + 1)
        reference = self.reference_trace(self.WARMUP + 1)
        self.assertGreater(abs(buggy[self.WARMUP + 1] - reference[self.WARMUP + 1]), 1e-15)

    def test_smoke_length_of_120_detects_the_bug(self):
        # 120 is the configured smoke length; it must sit past the warmup
        # boundary by a real margin, not by luck.
        buggy = self.buggy_eight_gpu_trace(120)
        reference = self.reference_trace(120)
        relative_gap = abs(buggy[120] - reference[120]) / reference[120]
        self.assertGreater(relative_gap, 0.01, "smoke at 120 updates must show a clear divergence")
        self.assertGreater(120, self.WARMUP, "smoke must run past warmup to have any diagnostic power")


if __name__ == "__main__":
    unittest.main()
