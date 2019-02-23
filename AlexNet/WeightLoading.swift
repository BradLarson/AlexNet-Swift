import Foundation
import TensorFlow

extension Data {
    init<T>(fromArray values: [T]) {
        var values = values
        self.init(buffer: UnsafeBufferPointer(start: &values, count: values.count))
    }
    
    func toArray<T>(type: T.Type) -> [T] {
        return self.withUnsafeBytes {
            [T](UnsafeBufferPointer(start: $0, count: self.count/MemoryLayout<T>.stride))
        }
    }
}

// Caffe weight ordering: [c_o][c_i][h][w]
// TensorFlow weight ordering: [h][w][c_i][c_o]

func loadWeights(from weightsFile: String, directory: URL, filterShape: (Int, Int, Int, Int)) throws -> Tensor<Float> {
    // TODO: A more gentle failure if the file is missing
    let weightsFileURL = directory.appendingPathComponent(weightsFile).absoluteURL
    let inputData = try Data(contentsOf: weightsFileURL)
    let weightValues = inputData.toArray(type:Float.self)
    print("Loading \(weightValues.count) weights from file: \(weightsFile)")

    guard (weightValues.count == (filterShape.0 * filterShape.1 * filterShape.2 * filterShape.3)) else {
        print("Weight mismatch, expected: \(filterShape.0 * filterShape.1 * filterShape.2 * filterShape.3), read: \(weightValues.count)")
        return Tensor<Float>(zeros: [Int32(filterShape.0), Int32(filterShape.1), Int32(filterShape.2), Int32(filterShape.3)])
    }
    
    var reorderedWeights = [Float](repeating: 0.0 , count: weightValues.count)
    
    let (kernelHeight, kernelWidth, inputChannels, outputChannels) = filterShape

    let outputChannelWidth = /*groupedInputChannels * */ kernelWidth * kernelHeight

    var reorderedWeightIndex = 0
    for heightInKernel in 0..<kernelHeight {
        for widthInKernel in 0..<kernelWidth {
            for inputChannel in 0..<inputChannels {
                for outputChannel in 0..<outputChannels {
                    let calculatedIndex = outputChannel * outputChannelWidth + inputChannel * kernelWidth * kernelHeight + heightInKernel * kernelWidth + widthInKernel
                    if (calculatedIndex >= weightValues.count) || (reorderedWeightIndex >= weightValues.count) {
                        print("Out of bounds calculatedIndex: \(calculatedIndex), reorderedWeightIndex: \(reorderedWeightIndex)")
                        continue
                    }
                    
                    reorderedWeights[reorderedWeightIndex] = weightValues[calculatedIndex]
                    reorderedWeightIndex += 1
                }
            }
        }
    }
    
    return Tensor<Float>(shape: [Int32(filterShape.0), Int32(filterShape.1), Int32(filterShape.2), Int32(filterShape.3)], scalars: reorderedWeights)
}

func loadBiases(from biasFile: String, directory: URL) throws -> Tensor<Float> {
    let biasFileURL = directory.appendingPathComponent(biasFile).absoluteURL
    let inputData = try Data(contentsOf: biasFileURL)
    return Tensor<Float>(inputData.toArray(type:Float.self))
}
