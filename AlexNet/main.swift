import TensorFlow
import Foundation

//let arguments = CommandLine.arguments
//guard (arguments.count > 2) else { fatalError("Usage: AlexNet [image directory] [weights directory]") }
//let imageDirectory = arguments[1]
//let weightsDirectory = arguments[2]
let imageDirectory = "images"
let weightsDirectory = "weights"

//let batchSize: Int32 = 60
//let classCount = 2
//let fakeImageBatch = Tensor<Float>(zeros: [batchSize, 227, 227, 3])
//let fakeLabelBatch = Tensor<Int32>(zeros: [batchSize])

// Load image dataset

let trainingImageDirectoryURL = URL(fileURLWithPath:"\(imageDirectory)/train")
let trainingImageDataset = try! ImageDataset(imageDirectory: trainingImageDirectoryURL, imageSize: (227, 227), batchSize: 60, byteOrdering: .bgr, pixelMeanToSubtract: 0.0)
let classCount = trainingImageDataset.classes
let batchSize = trainingImageDataset.imageData.shape[0]
let validationImageDirectoryURL = URL(fileURLWithPath:"\(imageDirectory)/val")
let validationImageDataset = try! ImageDataset(imageDirectory: validationImageDirectoryURL, imageSize: (227, 227), batchSize: 60, byteOrdering: .bgr, pixelMeanToSubtract: 0.0)
print("Dataset classes: \(trainingImageDataset.classes), labels: \(trainingImageDataset.labels)")

// Initialize network
let weightsDirectoryURL = URL(fileURLWithPath:weightsDirectory)
var alexNet = try! AlexNet(classCount: classCount, weightDirectory: weightsDirectoryURL)

let dumpTensorImages = false
if dumpTensorImages {
    alexNet.debugOutput(for: "images/val/dog/pexels-photo-1851471.jpeg")
}

// Train
//let optimizer = SGD<AlexNet, Float>(learningRate: 0.001, momentum: 0.9, decay: 0.00001)
let optimizer = SGD(for: alexNet, learningRate: 0.002, momentum: 0.9)
let validationInterval = 10

print("Start of training process")
print("Epoch, loss, accuracy(train), accuracy(val)")

Context.local.learningPhase = .training

let startTime = NSDate()
for epochNumber in 0..<100 {
    let shuffledDataset = trainingImageDataset.combinedDataset.shuffled(sampleCount: 60, randomSeed: Int64(epochNumber)).batched(60)
    for datasetItem in shuffledDataset {
        let (currentLoss, gradients) = valueWithGradient(at: alexNet) { model -> Tensor<Float> in
            return loss(model: model, images: datasetItem.first, labels: datasetItem.second)
        }

        optimizer.update(&alexNet.allDifferentiableVariables, along: gradients)

        let currentTrainingAccuracy = accuracy(model: alexNet, images: trainingImageDataset.imageData, labels: trainingImageDataset.imageLabels)
        
        
        if ((epochNumber % validationInterval) == 0) {
            let currentValidationAccuracy = accuracy(model: alexNet, images: validationImageDataset.imageData, labels: validationImageDataset.imageLabels)
            print("\(epochNumber), \(currentLoss), \(currentTrainingAccuracy), \(currentValidationAccuracy)")
        } else {
            print("\(epochNumber), \(currentLoss), \(currentTrainingAccuracy)")
        }
    }
}
let endTime = -startTime.timeIntervalSinceNow

print("End of training process, took \(endTime)s")
