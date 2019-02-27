import TensorFlow
import Foundation

// Assumes 227x227 input size
public struct AlexNet: Layer {
    var conv1: Conv2D<Float> // Note: declared as var because of learnable parameters
    let norm1: LRN
    @noDerivative let pool1: MaxPool2D<Float>
    var conv2: Conv2D<Float>
    let norm2: LRN
    @noDerivative let pool2: MaxPool2D<Float>
    var conv3: Conv2D<Float>
    var conv4: Conv2D<Float>
    var conv5: Conv2D<Float>
    @noDerivative let pool5: MaxPool2D<Float>
    var fc6: Dense<Float>
    let drop6: Dropout<Float>
    var fc7: Dense<Float>
    let drop7: Dropout<Float>
    var fc8: Dense<Float>
    
//    @noDerivative let fullyConnectedWidth = 4096
    @noDerivative let fullyConnectedWidth = 256
//    @noDerivative let fullyConnectedWidth = 16

    public init(classCount: Int, learningPhaseIndicator: LearningPhaseIndicator, weightDirectory: URL? = nil) throws {
        if let directory = weightDirectory {
            // Load pretrained convolutional weights
  
            // Why can't I use this initializer?
//            self.conv1 = Conv2D<Float>(filter: conv1Weights, bias: conv1Bias, strides: (Int32(4), Int32(4)), padding: .valid)
            self.conv1 = Conv2D(filterShape: (11, 11, 3, 96), strides: (4, 4), padding: .valid)
            let conv1Weights = try loadWeights(from: "conv1.weights", directory: directory, filterShape: (11, 11, 3, 96))
            let conv1Bias = try loadBiases(from: "conv1.biases", directory: directory)
            self.conv1.filter = conv1Weights
            self.conv1.bias = conv1Bias
            
            self.conv2 = Conv2D(filterShape: (5, 5, 96, 256), strides: (1, 1), padding: .same)
            let conv2Weights = try loadWeights(from: "conv2.weights", directory: directory, filterShape: (5, 5, 96, 256))
            let conv2Bias = try loadBiases(from: "conv2.biases", directory: directory)
            self.conv2.filter = conv2Weights
            self.conv2.bias = conv2Bias

            self.conv3 = Conv2D(filterShape: (3, 3, 256, 384), strides: (1, 1), padding: .same)
            let conv3Weights = try loadWeights(from: "conv3.weights", directory: directory, filterShape: (3, 3, 256, 384))
            let conv3Bias = try loadBiases(from: "conv3.biases", directory: directory)
            self.conv3.filter = conv3Weights
            self.conv3.bias = conv3Bias

            self.conv4 = Conv2D(filterShape: (3, 3, 384, 384), strides: (1, 1), padding: .same)
            let conv4Weights = try loadWeights(from: "conv4.weights", directory: directory, filterShape: (3, 3, 384, 384))
            let conv4Bias = try loadBiases(from: "conv4.biases", directory: directory)
            self.conv4.filter = conv4Weights
            self.conv4.bias = conv4Bias

            self.conv5 = Conv2D(filterShape: (3, 3, 384, 256), strides: (1, 1), padding: .same)
            let conv5Weights = try loadWeights(from: "conv5.weights", directory: directory, filterShape: (3, 3, 384, 256))
            let conv5Bias = try loadBiases(from: "conv5.biases", directory: directory)
            self.conv5.filter = conv5Weights
            self.conv5.bias = conv5Bias
        } else {
            // Random initialization
            self.conv1 = Conv2D(filterShape: (11, 11, 3, 96), strides: (4, 4), padding: .valid)
            self.conv2 = Conv2D(filterShape: (5, 5, 96, 256), strides: (1, 1), padding: .same)
            self.conv3 = Conv2D(filterShape: (3, 3, 256, 384), strides: (1, 1), padding: .same)
            self.conv4 = Conv2D(filterShape: (3, 3, 384, 384), strides: (1, 1), padding: .same)
            self.conv5 = Conv2D(filterShape: (3, 3, 384, 256), strides: (1, 1), padding: .same)
        }
        
        self.norm1 = LRN(depthRadius: 5, bias: 1.0, alpha: 0.0001, beta: 0.75)
        self.pool1 = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .valid)
        self.norm2 = LRN(depthRadius: 5, bias: 1.0, alpha: 0.0001, beta: 0.75)
        self.pool2 = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .valid)
        self.pool5 = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .valid)

        self.fc6 = Dense(inputSize: 9216, outputSize: fullyConnectedWidth, activation: relu) // 6 * 6 * 256 on input
        self.drop6 = Dropout<Float>(probability: 0.5, learningPhaseIndicator: learningPhaseIndicator)
        self.fc7 = Dense(inputSize: fullyConnectedWidth, outputSize: fullyConnectedWidth, activation: relu)
        self.drop7 = Dropout<Float>(probability: 0.5, learningPhaseIndicator: learningPhaseIndicator)
        self.fc8 = Dense(inputSize: fullyConnectedWidth, outputSize: classCount, activation: { $0 } )
    }
    
    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>) -> Tensor<Float> {
        let conv1Result = relu(conv1.applied(to: input))
        let norm1Result = norm1.applied(to: conv1Result)
        let pool1Result = pool1.applied(to: norm1Result)
        let conv2Result = relu(conv2.applied(to: pool1Result))
        let norm2Result = norm2.applied(to: conv2Result)
        let pool2Result = pool2.applied(to: norm2Result)
        let conv3Result = relu(conv3.applied(to: pool2Result))
        let conv4Result = relu(conv4.applied(to: conv3Result))
        let conv5Result = relu(conv5.applied(to: conv4Result))
        let pool5Result = pool5.applied(to: conv5Result)
        let reshapedIntermediate = pool5Result.reshaped(toShape: Tensor<Int32>([pool5Result.shape[Int32(0)], 9216]))
        let fc6Result = fc6.applied(to: reshapedIntermediate)
        let drop6Result = drop6.applied(to: fc6Result)
        let fc7Result = fc7.applied(to: drop6Result)
        let drop7Result = drop7.applied(to: fc7Result)
        let fc8Result = fc8.applied(to: drop7Result)

        return fc8Result
    }
}

@differentiable(wrt: model)
func loss(model: AlexNet, images: Tensor<Float>, labels: Tensor<Int32>) -> Tensor<Float> {
    let logits = model.applied(to: images)
    let oneHotLabels = Tensor<Float>(oneHotAtIndices: labels, depth: logits.shape[1])
    let crossEntropyLoss = softmaxCrossEntropy(logits: logits, labels: oneHotLabels)
    return crossEntropyLoss
}

func accuracy(model: AlexNet, images: Tensor<Float>, labels: Tensor<Int32>) -> Float {
    let results = softmax(model.applied(to: images))
    let predictedLabels = results.argmax(squeezingAxis: 1)
    let labelComparison = predictedLabels .== labels
    let matches = labelComparison.scalars.reduce(0.0){ accumulator, value in
        if value { return accumulator + 1} else { return accumulator }
    }
    
    return 100.0 * Float(matches) / Float(labels.shape[0])
}
