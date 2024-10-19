import ArgumentParser
import MLX
import F5TTS
import Foundation
import Vocos

@main
struct GenerateAudio: AsyncParsableCommand {
    @Argument(help: "Text to generate speech from")
    var text: String
    
    @Option(name: .long, help: "Duration of the generated audio in seconds")
    var duration: Double?
    
    @Option(name: .long, help: "Path to the reference audio file")
    var refAudioPath: String?
    
    @Option(name: .long, help: "Text spoken in the reference audio")
    var refAudioText: String?
    
    @Option(name: .long, help: "Model name to use")
    var model: String = "lucasnewman/f5-tts-mlx"
    
    @Option(name: .long, help: "Output path for the generated audio")
    var outputPath: String = "output.wav"
    
    @Option(name: .long, help: "Strength of classifier free guidance")
    var cfg: Float = 2.0
    
    @Option(name: .long, help: "Coefficient for sway sampling")
    var sway: Float = -1.0
    
    @Option(name: .long, help: "Speed factor for the duration heuristic")
    var speed: Float = 1.0
    
    @Option(name: .long, help: "Seed for noise generation")
    var seed: Int?
    
    func run() async throws {
        let sampleRate = 24_000
        let hopLength = 256
        let framesPerSec = Double(sampleRate) / Double(hopLength)
        let targetRMS: Float = 0.1
        
        let f5tts = try await F5TTS.fromPretrained(repoId: model)
        let vocos = try await Vocos.fromPretrained(repoId: "lucasnewman/vocos-mel-24khz-mlx")
        
        var audio: MLXArray
        let referenceText: String
        
        if let refPath = refAudioPath {
            audio = try AudioUtilities.loadAudioFile(url: URL(filePath: refPath))
            referenceText = refAudioText ?? "Some call me nature, others call me mother nature."
        } else if let refURL = Bundle.main.url(forResource: "test_en_1_ref_short", withExtension: "wav") {
            audio = try AudioUtilities.loadAudioFile(url: refURL)
            referenceText = "Some call me nature, others call me mother nature."
        } else {
            fatalError("No reference audio file specified.")
        }
        
        let rms = audio.square().mean().sqrt().item(Float.self)
        if rms < targetRMS {
            audio = audio * targetRMS / rms
        }
        
        // use a heuristic to determine the duration if not provided
        let refAudioDuration = Double(audio.shape[0]) / framesPerSec
        var generatedDuration = duration
        
        if generatedDuration == nil {
            let refAudioLength = audio.shape[0] / hopLength
            let pausePunctuation = "。，、；：？！"
            let refTextLength = referenceText.utf8.count + 3 * pausePunctuation.utf8.count
            let genTextLength = text.utf8.count + 3 * pausePunctuation.utf8.count
            
            let durationInFrames = refAudioLength + Int((Double(refAudioLength) / Double(refTextLength)) * (Double(genTextLength) / Double(speed)))
            let estimatedDuration = Double(durationInFrames - refAudioLength) / framesPerSec
            
            print("Using duration of \(estimatedDuration) seconds for generated speech.")
            generatedDuration = estimatedDuration
        }
        
        guard let generatedDuration else {
            fatalError("Unable to determine duration.")
        }
        
        let processedText = referenceText + " " + text
        let frameDuration = Int((refAudioDuration + generatedDuration) * framesPerSec)
        print("Generating \(frameDuration) frames of audio...")
        
        let startTime = Date()
        
        let (outputAudio, _) = f5tts.sample(
            cond: audio.expandedDimensions(axis: 0),
            text: [processedText],
            duration: frameDuration,
            steps: 32,
            cfgStrength: cfg,
            swayCoef: sway,
            seed: seed,
            vocoder: vocos.decode
        )
        
        let generatedAudio = outputAudio[audio.shape[0]...]
        
        let elapsedTime = Date().timeIntervalSince(startTime)
        print("Generated \(Double(generatedAudio.count) / Double(sampleRate)) seconds of audio in \(elapsedTime) seconds.")
        
        try AudioUtilities.saveAudioFile(url: URL(filePath: outputPath), samples: generatedAudio)
    }
}
