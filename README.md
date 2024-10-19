
# F5 TTS for Swift (WIP)

Implementation of [F5-TTS](https://arxiv.org/abs/2410.06885) in Swift, using the [MLX Swift]([https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx-swift)) framework.

You can listen to a [sample here](https://s3.amazonaws.com/lucasnewman.datasets/f5tts/sample.wav) that was generated in ~11 seconds on an M3 Max MacBook Pro.

See the [Python repository](https://github.com/lucasnewman/f5-tts-mlx) for additional details on the model architecture.
This repository is based on the original Pytorch implementation available [here](https://github.com/SWivid/F5-TTS).


## Installation

The `F5TTS` Swift package can be built and run from Xcode or SwiftPM.

A pretrained model is available [on Huggingface](https://hf.co/lucasnewman/f5-tts-mlx).


## Usage

```swift
import Vocos
import F5TTS

let f5tts = try await F5TTS.fromPretrained(repoId: "lucasnewman/f5-tts-mlx")
let vocos = try await Vocos.fromPretrained(repoId: "lucasnewman/vocos-mel-24khz-mlx") // if decoding to audio output

let inputAudio = MLXArray(...)

let (outputAudio, _) = f5tts.sample(
    cond: inputAudio,
    text: ["This is the caption for the reference audio and generation text."],
    duration: ...,
    vocoder: vocos.decode) { progress in
        print("Progress: \(Int(progress * 100))%")
    }
```

## Appreciation

[Yushen Chen](https://github.com/SWivid) for the original Pytorch implementation of F5 TTS and pretrained model.

[Phil Wang](https://github.com/lucidrains) for the E2 TTS implementation that this model is based on.

## Citations

```bibtex
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```

```bibtex
@inproceedings{Eskimez2024E2TE,
    title   = {E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS},
    author  = {Sefik Emre Eskimez and Xiaofei Wang and Manthan Thakker and Canrun Li and Chung-Hsien Tsai and Zhen Xiao and Hemin Yang and Zirun Zhu and Min Tang and Xu Tan and Yanqing Liu and Sheng Zhao and Naoyuki Kanda},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270738197}
}
```

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
