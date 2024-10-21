import Foundation
import MLX
import MLXNN

class DurationInputEmbedding: Module {
    let proj: Linear
    let conv_pos_embed: ConvPositionEmbedding

    init(melDim: Int, textDim: Int, outDim: Int) {
        self.proj = Linear(melDim + textDim, outDim)
        self.conv_pos_embed = ConvPositionEmbedding(dim: outDim)
        super.init()
    }

    func callAsFunction(
        cond: MLXArray,
        textEmbed: MLXArray
    ) -> MLXArray {
        var output = proj(MLX.concatenated([cond, textEmbed], axis: -1))
        output = conv_pos_embed(output) + output
        return output
    }
}

public class DurationTransformer: Module {
    let dim: Int
    let time_embed: TimestepEmbedding
    let text_embed: TextEmbedding
    let input_embed: DurationInputEmbedding
    let rotary_embed: RotaryEmbedding
    let transformer_blocks: [DiTBlock]
    let norm_out: RMSNorm
    let depth: Int

    init(
        dim: Int,
        depth: Int = 8,
        heads: Int = 8,
        dimHead: Int = 64,
        dropout: Float = 0.0,
        ffMult: Int = 4,
        melDim: Int = 100,
        textNumEmbeds: Int = 256,
        textDim: Int? = nil,
        convLayers: Int = 0
    ) {
        self.dim = dim
        let actualTextDim = textDim ?? melDim
        self.time_embed = TimestepEmbedding(dim: dim)
        self.text_embed = TextEmbedding(textNumEmbeds: textNumEmbeds, textDim: actualTextDim, convLayers: convLayers)
        self.input_embed = DurationInputEmbedding(melDim: melDim, textDim: actualTextDim, outDim: dim)
        self.rotary_embed = RotaryEmbedding(dim: dimHead)
        self.depth = depth

        self.transformer_blocks = (0 ..< depth).map { _ in
            DiTBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: ffMult, dropout: dropout)
        }

        self.norm_out = RMSNorm(dimensions: dim)

        super.init()
    }

    func callAsFunction(
        cond: MLXArray,
        text: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batchSize = cond.shape[0]
        let seqLen = cond.shape[1]

        let t = time_embed(MLX.ones([batchSize], type: Float32.self))
        let textEmbed = text_embed(text, seqLen: seqLen)
        var x = input_embed(cond: cond, textEmbed: textEmbed)

        let rope = rotary_embed.forwardFromSeqLen(seqLen)

        for block in transformer_blocks {
            x = block(x, t: t, mask: mask, rope: rope)
        }

        return norm_out(x)
    }
}

public class DurationPredictor: Module {
    enum DurationPredictorError: Error {
        case unableToLoadModel
        case unableToLoadReferenceAudio
        case unableToDetermineDuration
    }

    public let melSpec: MelSpec
    public let transformer: DurationTransformer

    let dim: Int
    let numChannels: Int
    let vocabCharMap: [String: Int]
    let to_pred: Sequential

    init(
        transformer: DurationTransformer,
        melSpec: MelSpec,
        vocabCharMap: [String: Int]
    ) {
        self.melSpec = melSpec
        self.numChannels = self.melSpec.nMels
        self.transformer = transformer
        self.dim = transformer.dim
        self.vocabCharMap = vocabCharMap

        self.to_pred = Sequential(layers: [
            Linear(dim, 1, bias: false), Softplus()
        ])

        super.init()
    }

    func callAsFunction(_ cond: MLXArray, text: [String]) -> MLXArray {
        var cond = cond

        // raw wave

        if cond.ndim == 2 {
            cond = cond.reshaped([cond.shape[1]])
            cond = melSpec(x: cond)
        }

        let batch = cond.shape[0]
        let condSeqLen = cond.shape[1]
        var lens = MLX.full([batch], values: condSeqLen, type: Int.self)

        // text

        let inputText = listStrToIdx(text, vocabCharMap: vocabCharMap)
        let textLens = (inputText .!= -1).sum(axis: -1)
        lens = MLX.maximum(textLens, lens)

        var output = transformer(cond: cond, text: inputText)
        output = to_pred(output).mean()
        output.eval()

        return output
    }
}
