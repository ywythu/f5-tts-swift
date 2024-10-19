import Foundation
import Hub
import MLX
import MLXNN
import MLXRandom
import Vocos

// MARK: - F5TTS

public class F5TTS: Module {
    enum F5TTSError: Error {
        case unableToLoadModel
        case unableToLoadReferenceAudio
        case unableToDetermineDuration
    }

    public let melSpec: MelSpec
    public let transformer: DiT

    let dim: Int
    let numChannels: Int
    let vocabCharMap: [String: Int]

    init(
        transformer: DiT,
        melSpec: MelSpec,
        vocabCharMap: [String: Int]
    ) {
        self.melSpec = melSpec
        self.numChannels = self.melSpec.nMels
        self.transformer = transformer
        self.dim = transformer.dim
        self.vocabCharMap = vocabCharMap

        super.init()
    }

    private func odeint(fun: (Float, MLXArray) -> MLXArray, y0: MLXArray, t: MLXArray) -> MLXArray {
        var ys = [y0]
        var yCurrent = y0

        for i in 0..<(t.shape[0] - 1) {
            let tCurrent = t[i].item(Float.self)
            let dt = t[i + 1].item(Float.self) - tCurrent

            let k1 = fun(tCurrent, yCurrent)
            let mid = yCurrent + 0.5 * dt * k1

            let k2 = fun(tCurrent + 0.5 * dt, mid)
            let yNext = yCurrent + dt * k2

            ys.append(yNext)
            yCurrent = yNext
        }

        return MLX.stacked(ys, axis: 0)
    }

    private func sample(
        cond: MLXArray,
        text: [String],
        duration: Any,
        lens: MLXArray? = nil,
        steps: Int = 32,
        cfgStrength: Double = 2.0,
        swayCoef: Double? = -1.0,
        seed: Int? = nil,
        maxDuration: Int = 4096,
        vocoder: ((MLXArray) -> MLXArray)? = nil,
        noRefAudio: Bool = false,
        editMask: MLXArray? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) -> (MLXArray, MLXArray) {
        MLX.eval(self.parameters())

        var cond = cond

        // raw wave

        if cond.ndim == 2 {
            cond = cond.reshaped([cond.shape[1]])
            cond = self.melSpec(x: cond)
        }

        let batch = cond.shape[0]
        let condSeqLen = cond.shape[1]
        var lens = lens ?? MLX.full([batch], values: condSeqLen, type: Int.self)

        // text

        let inputText = listStrToIdx(text, vocabCharMap: vocabCharMap)
        let textLens = (inputText .!= -1).sum(axis: -1)
        lens = MLX.maximum(textLens, lens)

        var condMask = lensToMask(t: lens)
        if let editMask = editMask {
            condMask = condMask & editMask
        }

        // duration

        var duration = (duration as? Int).map { MLX.full([batch], values: $0, type: Int.self) } ?? duration as! MLXArray
        duration = MLX.clip(MLX.maximum(lens + 1, duration), min: 0, max: maxDuration)
        let maxDur = duration.max().item(Int.self)

        cond = MLX.padded(cond, widths: [.init((0, 0)), .init((0, maxDur - condSeqLen)), .init((0, 0))])
        condMask = MLX.padded(condMask, widths: [.init((0, 0)), .init((0, maxDur - condMask.shape[1]))], value: MLXArray(false))
        condMask = condMask.expandedDimensions(axis: -1)
        let stepCond = MLX.where(condMask, cond, MLX.zeros(like: cond))

        let mask: MLXArray? = (batch > 1) ? lensToMask(t: duration) : nil

        if noRefAudio {
            cond = MLX.zeros(like: cond)
        }

        // neural ode

        let fn: (Float, MLXArray) -> MLXArray = { t, x in
            let pred = self.transformer(
                x: x,
                cond: stepCond,
                text: inputText,
                time: MLXArray(t),
                dropAudioCond: false,
                dropText: false,
                mask: mask
            )

            guard cfgStrength > 1e-5 else { return pred }

            let nullPred = self.transformer(
                x: x,
                cond: stepCond,
                text: inputText,
                time: MLXArray(t),
                dropAudioCond: true,
                dropText: true,
                mask: mask
            )

            progressHandler?(Double(t))

            return pred + (pred - nullPred) * cfgStrength
        }

        // noise input

        var y0: [MLXArray] = []
        for dur in duration {
            if let seed = seed {
                MLXRandom.seed(UInt64(seed))
            }
            let noise = MLXRandom.normal([dur.item(Int.self), self.numChannels])
            y0.append(noise)
        }
        let y0Padded = padSequence(y0, paddingValue: 0.0)

        var t = MLXArray.linspace(Float32(0.0), Float32(1.0), count: steps)

        if let coef = swayCoef {
            t = t + coef * (MLX.cos(MLXArray(.pi) / 2 * t) - 1 + t)
        }

        let trajectory = self.odeint(fun: fn, y0: y0Padded, t: t)
        let sampled = trajectory[-1]
        var out = MLX.where(condMask, cond, sampled)

        if let vocoder = vocoder {
            out = vocoder(out)
        }

        out.eval()

        return (out, trajectory)
    }

    public func generate(
        text: String,
        referenceAudioURL: URL? = nil,
        referenceAudioText: String? = nil,
        duration: TimeInterval? = nil,
        cfg: Double = 2.0,
        sway: Double = -1.0,
        speed: Double = 1.0,
        seed: Int? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> MLXArray {
        print("Loading Vocos model...")
        let vocos = try await Vocos.fromPretrained(repoId: "lucasnewman/vocos-mel-24khz-mlx")

        // load the reference audio + text

        var audio: MLXArray
        let referenceText: String

        if let referenceAudioURL {
            audio = try F5TTS.loadAudioArray(url: referenceAudioURL)
            referenceText = referenceAudioText ?? ""
        } else {
            let refAudioAndCaption = try F5TTS.referenceAudio()
            (audio, referenceText) = refAudioAndCaption
        }

        let refAudioDuration = Double(audio.shape[0]) / Double(F5TTS.sampleRate)
        print("Using reference audio with duration: \(refAudioDuration)")

        // use a heuristic to determine the duration if not provided

        var generatedDuration = duration
        if generatedDuration == nil {
            generatedDuration = F5TTS.estimatedDuration(refAudio: audio, refText: referenceText, text: text)
        }

        guard let generatedDuration else {
            throw F5TTSError.unableToDetermineDuration
        }
        print("Using generated duration: \(generatedDuration)")

        // generate the audio

        let normalizedAudio = F5TTS.normalizeAudio(audio: audio)

        let processedText = referenceText + " " + text
        let frameDuration = Int((refAudioDuration + generatedDuration) * F5TTS.framesPerSecond)
        print("Generating \(generatedDuration) seconds (\(frameDuration) total frames) of audio...")

        let (outputAudio, _) = self.sample(
            cond: normalizedAudio.expandedDimensions(axis: 0),
            text: [processedText],
            duration: frameDuration,
            steps: 32,
            cfgStrength: cfg,
            swayCoef: sway,
            seed: seed,
            vocoder: vocos.decode
        ) { progress in
            print("Generation progress: \(progress)")
        }

        let generatedAudio = outputAudio[audio.shape[0]...]
        return generatedAudio
    }
}

// MARK: - Pretrained Models

public extension F5TTS {
    static func fromPretrained(repoId: String, downloadProgress: ((Progress) -> Void)? = nil) async throws -> F5TTS {
        let modelDirectoryURL = try await Hub.snapshot(from: repoId, matching: ["*.safetensors", "*.txt"]) { progress in
            downloadProgress?(progress)
        }
        return try self.fromPretrained(modelDirectoryURL: modelDirectoryURL)
    }

    static func fromPretrained(modelDirectoryURL: URL) throws -> F5TTS {
        let modelURL = modelDirectoryURL.appendingPathComponent("model.safetensors")
        let modelWeights = try loadArrays(url: modelURL)

        // mel spec

        guard let filterbankURL = Bundle.module.url(forResource: "mel_filters", withExtension: "npy") else {
            throw F5TTSError.unableToLoadModel
        }
        let filterbank = try MLX.loadArray(url: filterbankURL)

        // vocab

        let vocabURL = modelDirectoryURL.appendingPathComponent("vocab.txt")
        guard let vocabString = try String(data: Data(contentsOf: vocabURL), encoding: .utf8) else {
            throw F5TTSError.unableToLoadModel
        }

        let vocabEntries = vocabString.split(separator: "\n").map { String($0) }
        let vocab = Dictionary(uniqueKeysWithValues: zip(vocabEntries, vocabEntries.indices))

        // model

        let dit = DiT(
            dim: 1024,
            depth: 22,
            heads: 16,
            ffMult: 2,
            textNumEmbeds: vocab.count,
            textDim: 512,
            convLayers: 4
        )
        let f5tts = F5TTS(transformer: dit, melSpec: MelSpec(filterbank: filterbank), vocabCharMap: vocab)

        // load weights

        var weights = [String: MLXArray]()
        for (key, value) in modelWeights {
            weights[key] = value
        }
        try f5tts.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])

        return f5tts
    }
}

// MARK: - Utilities

public extension F5TTS {
    static var sampleRate: Int = 24000
    static var hopLength: Int = 256
    static var framesPerSecond: Double = .init(sampleRate) / Double(hopLength)

    static func loadAudioArray(url: URL) throws -> MLXArray {
        return try AudioUtilities.loadAudioFile(url: url)
    }

    static func referenceAudio() throws -> (MLXArray, String) {
        guard let url = Bundle.module.url(forResource: "test_en_1_ref_short", withExtension: "wav") else {
            throw F5TTSError.unableToLoadReferenceAudio
        }

        return try (
            self.loadAudioArray(url: url),
            "Some call me nature, others call me mother nature."
        )
    }

    static func normalizeAudio(audio: MLXArray, targetRMS: Double = 0.1) -> MLXArray {
        let rms = Double(audio.square().mean().sqrt().item(Float.self))
        if rms < targetRMS {
            return audio * targetRMS / rms
        }
        return audio
    }

    static func estimatedDuration(refAudio: MLXArray, refText: String, text: String, speed: Double = 1.0) -> TimeInterval {
        let refDurationInFrames = refAudio.shape[0] / self.hopLength
        let pausePunctuation = "。，、；：？！"
        let refTextLength = refText.utf8.count + 3 * pausePunctuation.utf8.count
        let genTextLength = text.utf8.count + 3 * pausePunctuation.utf8.count

        let refAudioToTextRatio = Double(refDurationInFrames) / Double(refTextLength)
        let textLength = Double(genTextLength) / speed
        let estimatedDurationInFrames = Int(refAudioToTextRatio * textLength)

        let estimatedDuration = TimeInterval(estimatedDurationInFrames) / Self.framesPerSecond
        print("Using duration of \(estimatedDuration) seconds (\(estimatedDurationInFrames) frames) for generated speech.")

        return estimatedDuration
    }
}

// MLX utilities

func lensToMask(t: MLXArray, length: Int? = nil) -> MLXArray {
    let maxLength = length ?? t.max(keepDims: false).item(Int.self)
    let seq = MLXArray(0..<maxLength)
    let expandedSeq = seq.expandedDimensions(axis: 0)
    let expandedT = t.expandedDimensions(axis: 1)
    return MLX.less(expandedSeq, expandedT)
}

func padToLength(_ t: MLXArray, length: Int, value: Float? = nil) -> MLXArray {
    let ndim = t.ndim

    guard let seqLen = t.shape.last, length > seqLen else {
        return t[0..., .ellipsis]
    }

    let paddingValue = MLXArray(value ?? 0.0)

    let padded: MLXArray
    switch ndim {
    case 1:
        padded = MLX.padded(t, widths: [.init((0, length - seqLen))], value: paddingValue)
    case 2:
        padded = MLX.padded(t, widths: [.init((0, 0)), .init((0, length - seqLen))], value: paddingValue)
    case 3:
        padded = MLX.padded(t, widths: [.init((0, 0)), .init((0, length - seqLen)), .init((0, 0))], value: paddingValue)
    default:
        fatalError("Unsupported padding dims: \(ndim)")
    }

    return padded[0..., .ellipsis]
}

func padSequence(_ t: [MLXArray], paddingValue: Float = 0) -> MLXArray {
    let maxLen = t.map { $0.shape.last ?? 0 }.max() ?? 0
    let t = MLX.stacked(t, axis: 0)
    return padToLength(t, length: maxLen, value: paddingValue)
}

func listStrToIdx(_ text: [String], vocabCharMap: [String: Int], paddingValue: Int = -1) -> MLXArray {
    let listIdxTensors = text.map { str in str.map { char in vocabCharMap[String(char), default: 0] }}
    let mlxArrays = listIdxTensors.map { MLXArray($0) }
    let paddedText = padSequence(mlxArrays, paddingValue: Float(paddingValue))
    return paddedText.asType(.int32)
}
