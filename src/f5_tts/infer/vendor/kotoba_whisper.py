from typing import Union, Optional, Dict, List, Any
import requests

import torch
import numpy as np

from transformers.pipelines.audio_utils import ffmpeg_read
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline, chunk_iter
from transformers.utils import is_torchaudio_available
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from stable_whisper import WhisperResult
from punctuators.models import PunctCapSegModelONNX


class Punctuator:

    ja_punctuations = ["!", "?", "、", "。"]

    def __init__(self, model: str = "pcs_47lang"):
        self.punctuation_model = PunctCapSegModelONNX.from_pretrained(model)

    def punctuate(self, pipeline_chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        def validate_punctuation(raw: str, punctuated: str):
            if 'unk' in punctuated.lower() or any(p in raw for p in self.ja_punctuations):
                return raw
            if punctuated.count("。") > 1:
                ind = punctuated.rfind("。")
                punctuated = punctuated.replace("。", "")
                punctuated = punctuated[:ind] + "。" + punctuated[ind:]
            return punctuated

        text_edit = self.punctuation_model.infer([c['text'] for c in pipeline_chunk])
        return [
            {
                'timestamp': c['timestamp'],
                'text': validate_punctuation(c['text'], "".join(e))
            } for c, e in zip(pipeline_chunk, text_edit)
        ]


def _fix_timestamp(sample_rate: int, result: List[Dict[str, Any]], audio: np.ndarray) -> WhisperResult or None:

    def replace_none_ts(parts):
        total_dur = round(audio.shape[-1] / sample_rate, 3)
        _medium_dur = _ts_nonzero_mask = None

        def ts_nonzero_mask() -> np.ndarray:
            nonlocal _ts_nonzero_mask
            if _ts_nonzero_mask is None:
                _ts_nonzero_mask = np.array([(p['end'] or p['start']) is not None for p in parts])
            return _ts_nonzero_mask

        def medium_dur() -> float:
            nonlocal _medium_dur
            if _medium_dur is None:
                nonzero_dus = [p['end'] - p['start'] for p in parts if None not in (p['end'], p['start'])]
                nonzero_durs = np.array(nonzero_dus)
                _medium_dur = np.median(nonzero_durs) * 2 if len(nonzero_durs) else 2.0
            return _medium_dur

        def _curr_max_end(start: float, next_idx: float) -> float:
            max_end = total_dur
            if next_idx != len(parts):
                mask = np.flatnonzero(ts_nonzero_mask()[next_idx:])
                if len(mask):
                    _part = parts[mask[0]+next_idx]
                    max_end = _part['start'] or _part['end']

            new_end = round(start + medium_dur(), 3)
            if new_end > max_end:
                return max_end
            return new_end

        for i, part in enumerate(parts, 1):
            if part['start'] is None:
                is_first = i == 1
                if is_first:
                    new_start = round((part['end'] or 0) - medium_dur(), 3)
                    part['start'] = max(new_start, 0.0)
                else:
                    part['start'] = parts[i - 2]['end']
            if part['end'] is None:
                no_next_start = i == len(parts) or parts[i]['start'] is None
                part['end'] = _curr_max_end(part['start'], i) if no_next_start else parts[i]['start']

    words = [dict(start=word['timestamp'][0], end=word['timestamp'][1], word=word['text']) for word in result]
    replace_none_ts(words)
    return WhisperResult([words], force_order=True, check_sorted=True)


def fix_timestamp(pipeline_output: List[Dict[str, Any]], audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
    result = _fix_timestamp(sample_rate=sample_rate, audio=audio, result=pipeline_output)
    result.adjust_by_silence(
        audio,
        q_levels=20,
        k_size=5,
        sample_rate=sample_rate,
        min_word_dur=None,
        word_level=True,
        verbose=True,
        nonspeech_error=0.1,
        use_word_position=True
    )
    if result.has_words:
        result.regroup(True)
    return [{"timestamp": [s.start, s.end], "text": s.text} for s in result.segments]


class KotobaWhisperPipeline(AutomaticSpeechRecognitionPipeline):

    def __init__(self,
                 model: "PreTrainedModel",
                 feature_extractor: Union["SequenceFeatureExtractor", str] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 device: Union[int, "torch.device"] = None,
                 torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
                 punctuator: bool = True,
                 stable_ts: bool = False,
                 **kwargs):
        self.type = "seq2seq_whisper"
        self.stable_ts = stable_ts
        if punctuator:
            self.punctuator = Punctuator()
        else:
            self.punctuator = None
        super().__init__(
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs
        )

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)

        stride = None
        extra = {}
        if isinstance(inputs, dict):
            stride = inputs.pop("stride", None)
            # Accepting `"array"` which is the key defined in `datasets` for
            # better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to AutomaticSpeechRecognitionPipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            extra = inputs
            inputs = _inputs
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    raise ImportError(
                        "torchaudio is required to resample audio samples in AutomaticSpeechRecognitionPipeline. "
                        "The torchaudio package can be installed through: `pip install torchaudio`."
                    )

                inputs = F.resample(
                    torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate
                ).numpy()
                ratio = self.feature_extractor.sampling_rate / in_sampling_rate
            else:
                ratio = 1
            if stride is not None:
                if stride[0] + stride[1] > inputs.shape[0]:
                    raise ValueError("Stride is too large for input")

                # Stride needs to get the chunk length here, it's going to get
                # swallowed by the `feature_extractor` later, and then batching
                # can add extra data in the inputs, so we need to keep track
                # of the original length in the stride so we can cut properly.
                stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))
        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
        if len(inputs.shape) != 1:
            raise ValueError("We expect a single channel audio input for AutomaticSpeechRecognitionPipeline")

        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6

            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]

            # XXX: Carefuly, this variable will not exist in `seq2seq` setting.
            # Currently chunking is not possible at this level for `seq2seq` so
            # it's ok.
            align_to = getattr(self.model.config, "inputs_to_logits_ratio", 1)
            chunk_len = int(round(chunk_length_s * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_left = int(round(stride_length_s[0] * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_right = int(round(stride_length_s[1] * self.feature_extractor.sampling_rate / align_to) * align_to)

            if chunk_len < stride_left + stride_right:
                raise ValueError("Chunk length must be superior to stride length")

            for item in chunk_iter(
                    inputs, self.feature_extractor, chunk_len, stride_left, stride_right, self.torch_dtype
            ):
                item["audio_array"] = inputs
                yield item
        else:
            if inputs.shape[0] > self.feature_extractor.n_samples:
                processed = self.feature_extractor(
                    inputs,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    truncation=False,
                    padding="longest",
                    return_tensors="pt",
                )
            else:
                processed = self.feature_extractor(
                    inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
                )

            if self.torch_dtype is not None:
                processed = processed.to(dtype=self.torch_dtype)
            if stride is not None:
                processed["stride"] = stride
            yield {"is_last": True, "audio_array": inputs, **processed, **extra}

    def _forward(self, model_inputs, return_timestamps=False, **generate_kwargs):
        attention_mask = model_inputs.pop("attention_mask", None)
        stride = model_inputs.pop("stride", None)
        is_last = model_inputs.pop("is_last")
        audio_array = model_inputs.pop("audio_array")
        encoder = self.model.get_encoder()
        # Consume values so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if type(return_timestamps) is not bool:
            raise ValueError("return_timestamps should be bool")
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # custom processing for Whisper timestamps and word-level timestamps
        generate_kwargs["return_timestamps"] = True
        if inputs.shape[-1] > self.feature_extractor.nb_max_frames:
            generate_kwargs["input_features"] = inputs
        else:
            generate_kwargs["encoder_outputs"] = encoder(inputs, attention_mask=attention_mask)

        tokens = self.model.generate(attention_mask=attention_mask, **generate_kwargs)
        # whisper longform generation stores timestamps in "segments"
        out = {"tokens": tokens}
        if self.type == "seq2seq_whisper":
            if stride is not None:
                out["stride"] = stride

        # Leftover
        extra = model_inputs
        return {"is_last": is_last, "audio_array": audio_array, **out, **extra}

    def postprocess(self,
                    model_outputs,
                    decoder_kwargs: Optional[Dict] = None,
                    return_timestamps=None,
                    return_language=None):
        assert len(model_outputs) > 0
        for model_output in model_outputs:
            audio_array = model_output.pop("audio_array")[0]
        outputs = super().postprocess(
            model_outputs=model_outputs,
            decoder_kwargs=decoder_kwargs,
            return_timestamps=True,
            return_language=return_language
        )
        if self.stable_ts:
            outputs["chunks"] = fix_timestamp(
                pipeline_output=outputs["chunks"], audio=audio_array, sample_rate=self.feature_extractor.sampling_rate
            )
        if self.punctuator:
            outputs["chunks"] = self.punctuator.punctuate(outputs["chunks"])
        outputs["text"] = "".join([c["text"] for c in outputs["chunks"]])
        if not return_timestamps:
            outputs.pop("chunks")
        return outputs

