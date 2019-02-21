import TensorFlow

struct LRN: Layer {
    typealias Input = Tensor<Float>
    
    typealias Output = Tensor<Float>
    
    @noDerivative let depthRadius: Int64
    @noDerivative let bias: Double
    @noDerivative let alpha: Double
    @noDerivative let beta: Double

    @differentiable(wrt:(self, input))
    func applied(to input: Tensor<Float>) -> Tensor<Float> {
        return input
//        return Raw.lRN(input, depthRadius: depthRadius, bias: bias, alpha: alpha, beta: beta)
    }
}
