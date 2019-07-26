import TensorFlow
import Foundation

// Assumes 227x227 input size
public struct AlexNet: Layer {
    var conv1: Conv2D<Float> // Note: declared as var because of learnable parameters
    @noDerivative let norm1: LRN
    @noDerivative let pool1: MaxPool2D<Float>
    var conv2: Conv2D<Float>
    @noDerivative let norm2: LRN
    @noDerivative let pool2: MaxPool2D<Float>
    var conv3: Conv2D<Float>
    var conv4: Conv2D<Float>
    var conv5: Conv2D<Float>
    @noDerivative let pool5: MaxPool2D<Float>
    var fc6: Dense<Float>
    @noDerivative let drop6: Dropout<Float>
    var fc7: Dense<Float>
    @noDerivative let drop7: Dropout<Float>
    var fc8: Dense<Float>
    
//    @noDerivative let fullyConnectedWidth = 4096
    @noDerivative let fullyConnectedWidth = 256

    public init(classCount: Int, weightDirectory: URL? = nil) throws {
        if let directory = weightDirectory {
            // Load pretrained convolutional weights
  
            let conv1Weights = try loadWeights(from: "conv1.weights", directory: directory, filterShape: (11, 11, 3, 96))
            let conv1Bias = try loadBiases(from: "conv1.biases", directory: directory)
            self.conv1 = Conv2D<Float>(filter: conv1Weights, bias: conv1Bias, activation: relu, strides: (4, 4), padding: .valid)
            
            let conv2Weights = try loadWeights(from: "conv2.weights", directory: directory, filterShape: (5, 5, 96, 256))
            let conv2Bias = try loadBiases(from: "conv2.biases", directory: directory)
            self.conv2 = Conv2D<Float>(filter: conv2Weights, bias: conv2Bias, activation: relu, strides: (1, 1), padding: .same)

            let conv3Weights = try loadWeights(from: "conv3.weights", directory: directory, filterShape: (3, 3, 256, 384))
            let conv3Bias = try loadBiases(from: "conv3.biases", directory: directory)
            self.conv3 = Conv2D<Float>(filter: conv3Weights, bias: conv3Bias, activation: relu, strides: (1, 1), padding: .same)

            let conv4Weights = try loadWeights(from: "conv4.weights", directory: directory, filterShape: (3, 3, 384, 384))
            let conv4Bias = try loadBiases(from: "conv4.biases", directory: directory)
            self.conv4 = Conv2D<Float>(filter: conv4Weights, bias: conv4Bias, activation: relu, strides: (1, 1), padding: .same)

            let conv5Weights = try loadWeights(from: "conv5.weights", directory: directory, filterShape: (3, 3, 384, 256))
            let conv5Bias = try loadBiases(from: "conv5.biases", directory: directory)
            self.conv5 = Conv2D<Float>(filter: conv5Weights, bias: conv5Bias, activation: relu, strides: (1, 1), padding: .same)
        } else {
            // Random initialization
            self.conv1 = Conv2D(filterShape: (11, 11, 3, 96), strides: (4, 4), padding: .valid, activation: relu)
            self.conv2 = Conv2D(filterShape: (5, 5, 96, 256), strides: (1, 1), padding: .same, activation: relu)
            self.conv3 = Conv2D(filterShape: (3, 3, 256, 384), strides: (1, 1), padding: .same, activation: relu)
            self.conv4 = Conv2D(filterShape: (3, 3, 384, 384), strides: (1, 1), padding: .same, activation: relu)
            self.conv5 = Conv2D(filterShape: (3, 3, 384, 256), strides: (1, 1), padding: .same, activation: relu)
        }
        
        self.norm1 = LRN(depthRadius: 5, bias: 1.0, alpha: 0.0001, beta: 0.75)
        self.pool1 = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .valid)
        self.norm2 = LRN(depthRadius: 5, bias: 1.0, alpha: 0.0001, beta: 0.75)
        self.pool2 = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .valid)
        self.pool5 = MaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .valid)

        // TODO: Find a faster way to initialize these
        // The Gaussian distributions here are crucial to fast convergence and high generalization accuracy
        let fc6Bias = Tensor<Float>(repeating: 0.1, shape: TensorShape(fullyConnectedWidth))
        let fc6Weight = Tensor<Float>(randomNormal: TensorShape(9216, fullyConnectedWidth), mean: Tensor<Float>(0.0), standardDeviation: Tensor<Float>(0.005))
        self.fc6 = Dense(weight: fc6Weight, bias: fc6Bias, activation: relu) // 6 * 6 * 256 on input
        self.drop6 = Dropout<Float>(probability: 0.5)
        
        let fc7Bias = Tensor<Float>(repeating: 0.1, shape: TensorShape(fullyConnectedWidth))
        let fc7Weight = Tensor<Float>(randomNormal: TensorShape(fullyConnectedWidth, fullyConnectedWidth), mean: Tensor<Float>(0.0), standardDeviation: Tensor<Float>(0.005))
        self.fc7 = Dense(weight: fc7Weight, bias: fc7Bias, activation: relu)
        self.drop7 = Dropout<Float>(probability: 0.5)
        
        let fc8Bias = Tensor<Float>(repeating: 0.0, shape: TensorShape(classCount))
        let fc8Weight = Tensor<Float>(randomNormal: TensorShape(fullyConnectedWidth, classCount), mean: Tensor<Float>(0.0), standardDeviation: Tensor<Float>(0.01))
        self.fc8 = Dense(weight: fc8Weight, bias: fc8Bias, activation: { $0 })
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let conv1Result = conv1(input)
        let norm1Result = norm1(conv1Result)
        let pool1Result = pool1(norm1Result)
        let conv2Result = conv2(pool1Result)
        let norm2Result = norm2(conv2Result)
        let pool2Result = pool2(norm2Result)
        let conv3Result = conv3(pool2Result)
        let conv4Result = conv4(conv3Result)
        let conv5Result = conv5(conv4Result)
        let pool5Result = pool5(conv5Result)
        let reshapedIntermediate = pool5Result.reshaped(toShape: Tensor<Int32>([Int32(pool5Result.shape[0]), 9216]))
        let fc6Result = fc6(reshapedIntermediate)
        let drop6Result = drop6(fc6Result)
        let fc7Result = fc7(drop6Result)
        let drop7Result = drop7(fc7Result)
        let fc8Result = fc8(drop7Result)

        return fc8Result
    }
}

@differentiable(wrt: model)
func loss(model: AlexNet, images: Tensor<Float>, labels: Tensor<Int32>) -> Tensor<Float> {
    let logits = model(images)
    let crossEntropyLoss = softmaxCrossEntropy(logits: logits, labels: labels)
    return crossEntropyLoss
}

func accuracy(model: AlexNet, images: Tensor<Float>, labels: Tensor<Int32>) -> Float {
    Context.local.learningPhase = .inference

    let results = softmax(model(images))
    let predictedLabels = results.argmax(squeezingAxis: 1)
    let labelComparison = predictedLabels .== labels
    let matches = labelComparison.scalars.reduce(0.0){ accumulator, value in
        if value { return accumulator + 1} else { return accumulator }
    }
    
    return 100.0 * Float(matches) / Float(labels.shape[0])
}
