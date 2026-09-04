# vendor/

第三方代码的原样副本，**请勿手工修改** —— 改动会在下次同步上游时丢失。

## kotoba_whisper.py

- 来源：HuggingFace `kotoba-tech/kotoba-whisper-v1.1`，commit `cf813454956c24499abaf549f4fbbdb89e496903`
- 许可：Apache-2.0（随原仓库）
- 用途：日文 ASR（Visual-novel-whisper 是它在 Galgame 语料上的微调）的自定义
  `KotobaWhisperPipeline` 实现，由 [asr_backends.py](../asr_backends.py) 加载。

### 为什么要放本地副本

Visual-novel-whisper 的 `config.json` 里写的是

```json
"custom_pipelines": {"kotoba-whisper": {"impl": "kotoba-tech/kotoba-whisper-v1.1--kotoba_whisper.KotobaWhisperPipeline"}}
```

`--` 前面是**远程仓库名**，所以即便模型权重在本地，`trust_remote_code=True` 仍会去
HuggingFace 取这份 pipeline 代码。而两个 `.bat` 都把 `HF_HOME` 指向仓库内
`.huggingface/`，与用户默认缓存 `~/.cache/huggingface` 对不上，于是加载时必然联网、
断网即失败。放一份本地副本后 `asr_backends` 直接 `pipeline_class=` 传类，离线可用。

### 运行依赖

这份代码硬性 `import stable_whisper` 与 `punctuators`，缺任一个直接 ImportError，
关掉 `stable_ts` / `punctuator` 开关也绕不过去：

```bash
pip install punctuators
pip install stable-ts --no-deps     # 必须 --no-deps
```

`stable-ts` 依赖的 `openai-whisper==20231117` 其 setup.py 仍在用已被新版 setuptools
移除的 `pkg_resources`，不加 `--no-deps` 构建必然失败。我们只取文本，`stable_whisper`
仅用于时间戳精修，跳过它的依赖不影响使用。
