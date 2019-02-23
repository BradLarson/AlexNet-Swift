import TensorFlow
import Foundation

//let arguments = CommandLine.arguments
//guard (arguments.count > 2) else { fatalError("Usage: AlexNet [image directory] [weights directory]") }
//let imageDirectory = arguments[1]
//let weightsDirectory = arguments[2]
let imageDirectory = "images"
let weightsDirectory = "weights"

let batchSize: Int32 = 60
let classCount = 2

let fakeImageBatch = Tensor<Float>(zeros: [batchSize, 227, 227, 3])
//let fakeLabelBatch = Tensor<Int32>(oneHotAtIndices: Tensor<Int32>([0]), depth: classCount)
let fakeLabelBatch = Tensor<Int32>(zeros: [batchSize])

print("Fake labels: \(fakeLabelBatch)")

let weightsDirectoryURL = URL(fileURLWithPath:weightsDirectory)
let learningPhaseIndicator = LearningPhaseIndicator()
var alexNet = try! AlexNet(classCount: classCount, learningPhaseIndicator: learningPhaseIndicator, weightDirectory: weightsDirectoryURL)

let imageDirectoryURL = URL(fileURLWithPath:imageDirectory)

//alexNet.loadInitialWeights(from:weightsDirectoryURL)

let optimizer = SGD<AlexNet, Float>(learningRate: 0.001, momentum: 0.9)

print("Start of training process")

let startTime = NSDate()
for batchNumber in 0..<60 {
    var currentLoss = Tensor<Float>(zeros: [1])
    let gradients = gradient(at: alexNet) { model -> Tensor<Float> in
        currentLoss = loss(model: model, images: fakeImageBatch, labels: fakeLabelBatch)
        return currentLoss
    }
    optimizer.update(&alexNet.allDifferentiableVariables, along: gradients)
    print("Completed batch \(batchNumber), loss: \(currentLoss)")
}
let endTime = -startTime.timeIntervalSinceNow

print("End of training process, took \(endTime)s")
