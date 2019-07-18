import TensorFlow

struct LRN: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    @noDerivative let depthRadius: Int64
    @noDerivative let bias: Double
    @noDerivative let alpha: Double
    @noDerivative let beta: Double

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return localResponseNorm(input, depthRadius: depthRadius, bias: bias, alpha: alpha, beta: beta)
    }
}


// Per Dan Zheng's guidance in the gist here: https://gist.github.com/dan-zheng/b4c772f00c90a514eec61db265139c3f
@differentiable(wrt: (input), vjp: vjpLocalResponseNorm)
func localResponseNorm<T : TensorFlowFloatingPoint>(_ input: Tensor<T>, depthRadius: Int64 = 5, bias: Double = 1, alpha: Double = 0.0001, beta: Double = 0.75) -> Tensor<T> {
    return Raw.lRN(input, depthRadius: depthRadius, bias: bias, alpha: alpha, beta: beta)
}

func vjpLocalResponseNorm<T : TensorFlowFloatingPoint>(_ input: Tensor<T>, depthRadius: Int64 = 5, bias: Double = 1, alpha: Double = 0.0001, beta: Double = 0.75) -> (Tensor<T>, (Tensor<T>) -> (Tensor<T>)) {
    let value = localResponseNorm(input, depthRadius: depthRadius, bias: bias, alpha: alpha, beta: beta)
    return (value, { v in
        Raw.lRNGrad(inputGrads: v, inputImage: input, outputImage: value, depthRadius: depthRadius, bias: bias, alpha: alpha, beta: beta)
    })
}
