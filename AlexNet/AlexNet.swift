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
    @noDerivative var rng: ARC4RandomNumberGenerator = ARC4RandomNumberGenerator(seed: 120910)
    
//    @noDerivative let fullyConnectedWidth = 4096
    @noDerivative let fullyConnectedWidth = 256
//    @noDerivative let fullyConnectedWidth = 16

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
        let fc6Bias = Tensor<Float>(shape: TensorShape(Int32(fullyConnectedWidth)), repeating: 0.1)
        let fc6Weight = Tensor<Float>(randomNormal: TensorShape(Int32(9216), Int32(fullyConnectedWidth)), mean: 0.0, stddev: 0.005, generator: &rng)
        self.fc6 = Dense(weight: fc6Weight, bias: fc6Bias, activation: relu) // 6 * 6 * 256 on input
        self.drop6 = Dropout<Float>(probability: 0.5)
        
        let fc7Bias = Tensor<Float>(shape: TensorShape(Int32(fullyConnectedWidth)), repeating: 0.1)
        let fc7Weight = Tensor<Float>(randomNormal: TensorShape(Int32(fullyConnectedWidth), Int32(fullyConnectedWidth)), mean: 0.0, stddev: 0.005, generator: &rng)
        self.fc7 = Dense(weight: fc7Weight, bias: fc7Bias, activation: relu)
        self.drop7 = Dropout<Float>(probability: 0.5)
        
        let fc8Bias = Tensor<Float>(shape: TensorShape(Int32(classCount)), repeating: 0.0)
        let fc8Weight = Tensor<Float>(randomNormal: TensorShape(Int32(fullyConnectedWidth), Int32(classCount)), mean: 0.0, stddev: 0.01, generator: &rng)
        self.fc8 = Dense(weight: fc8Weight, bias: fc8Bias, activation: { $0 })
    }

    
    
    @differentiable(wrt: (self, input))
    public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
        let conv1Result = conv1.applied(to: input, in: context)
        let norm1Result = norm1.applied(to: conv1Result, in: context)
        let pool1Result = pool1.applied(to: norm1Result, in: context)
        let conv2Result = conv2.applied(to: pool1Result, in: context)
        let norm2Result = norm2.applied(to: conv2Result, in: context)
        let pool2Result = pool2.applied(to: norm2Result, in: context)
        let conv3Result = conv3.applied(to: pool2Result, in: context)
        let conv4Result = conv4.applied(to: conv3Result, in: context)
        let conv5Result = conv5.applied(to: conv4Result, in: context)
        let pool5Result = pool5.applied(to: conv5Result, in: context)
        let reshapedIntermediate = pool5Result.reshaped(toShape: Tensor<Int32>([pool5Result.shape[Int32(0)], 9216]))
        let fc6Result = fc6.applied(to: reshapedIntermediate, in: context)
        let drop6Result = drop6.applied(to: fc6Result, in: context)
        let fc7Result = fc7.applied(to: drop6Result, in: context)
        let drop7Result = drop7.applied(to: fc7Result, in: context)
        let fc8Result = fc8.applied(to: drop7Result, in: context)

        return fc8Result
    }
}

@differentiable(wrt: model)
func loss(model: AlexNet, in context: Context, images: Tensor<Float>, labels: Tensor<Int32>) -> Tensor<Float> {
    let logits = model.applied(to: images, in: context)
    let crossEntropyLoss = softmaxCrossEntropy(logits: logits, labels: labels)
    return crossEntropyLoss
}

func accuracy(model: AlexNet, images: Tensor<Float>, labels: Tensor<Int32>) -> Float {
    let inferenceContext = Context(learningPhase: .inference)

    let results = softmax(model.applied(to: images, in: inferenceContext))
    let predictedLabels = results.argmax(squeezingAxis: 1)
    let labelComparison = predictedLabels .== labels
    let matches = labelComparison.scalars.reduce(0.0){ accumulator, value in
        if value { return accumulator + 1} else { return accumulator }
    }
    
    return 100.0 * Float(matches) / Float(labels.shape[0])
}
