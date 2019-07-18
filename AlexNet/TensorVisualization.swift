import TensorFlow
import Foundation
#if os(OSX)
import Cocoa

func dataProviderReleaseCallback(_ context:UnsafeMutableRawPointer?, data:UnsafeRawPointer, size:Int) {
    data.deallocate()
}
#endif

extension Tensor where Scalar == Float {
    public func saveOutputMosaicImageToDisk(prefix:String, spaceBetweenSlices:Int = 1) {
        // TODO: Rework this using TensorFlow Raw functions
#if os(OSX)
        print("Saving layer output: \(prefix)")

        let (mosaicImageData, imageWidth, imageHeight) = mosaicImageOfOutput(spaceBetweenLayers:spaceBetweenSlices)
        let imageByteSize = imageWidth * imageHeight * 4

        guard let dataProvider = CGDataProvider(dataInfo:nil, data:mosaicImageData, size:imageByteSize, releaseData:dataProviderReleaseCallback) else {fatalError("Could not allocate a CGDataProvider")}
        let defaultRGBColorSpace = CGColorSpaceCreateDeviceRGB()
        let mosaicImage = CGImage(width:Int(imageWidth), height:Int(imageHeight), bitsPerComponent:8, bitsPerPixel:32, bytesPerRow:4 * Int(imageWidth), space:defaultRGBColorSpace, bitmapInfo:CGBitmapInfo() /*| CGImageAlphaInfo.Last*/, provider:dataProvider, decode:nil, shouldInterpolate:false, intent:.defaultIntent)!
        
        let fileURL = URL(fileURLWithPath:"").appendingPathComponent("\(prefix)_output.png").absoluteURL
        let bitmapRepresentation = NSBitmapImageRep(cgImage:mosaicImage)
        let imageData = bitmapRepresentation.representation(using: .png, properties: [NSBitmapImageRep.PropertyKey(rawValue: ""):""])!
        try! imageData.write(to:fileURL)
#else
#endif
    }

    public func mosaicImageOfOutput(spaceBetweenLayers:Int = 1) -> (UnsafeMutablePointer<UInt8>, Int, Int) {
        guard (self.shape.rank == 4) else { fatalError("Tried to write out non-rank-4 tensor, rank:\(self.shape.rank)") }
        let width = Int(self.shape[2])
        let height = Int(self.shape[1])
        let numLayers = Int(self.shape[3])
//        let sliceByteSize = width * height * 4
        
        let numberOfLayersAcross = Int(round(Double.sqrt(Double(numLayers))))
        let numberOfLayersDown = Int(ceil(Double(numLayers) / Double(numberOfLayersAcross)))
        
        let imageWidth = width * numberOfLayersAcross + (numberOfLayersAcross * spaceBetweenLayers)
        let imageHeight = height * numberOfLayersDown + (numberOfLayersDown * spaceBetweenLayers)
        let imageByteSize = imageWidth * imageHeight * 4
        let imageData = UnsafeMutablePointer<UInt8>.allocate(capacity:imageByteSize)
        
        let minValue = self.min().scalarized()
        let maxValue = self.max().scalarized()
        let meanValue = self.mean().scalarized()
        
        print("Layer min:\(minValue), max:\(maxValue), mean:\(meanValue)")

        let totalBytes = imageWidth * imageHeight
        // Initialize with black
        for currentPixelIndex in 0..<totalBytes {
            imageData[currentPixelIndex * 4] = 0
            imageData[(currentPixelIndex * 4) + 1] = 0
            imageData[(currentPixelIndex * 4) + 2] = 0
            imageData[(currentPixelIndex * 4) + 3] = 255
        }

        let floatBuffer = self[0].scalars
        
        for currentLayerY in 0..<height {
            for currentLayerX in 0..<width {
                for layerIndex in 0..<numLayers {
                    let layerLocationInY = Int(floor(Double(layerIndex) / Double(numberOfLayersAcross)))
                    let layerLocationInX = layerIndex - (layerLocationInY * numberOfLayersAcross)
                    let startingLayerOutputByteLocation = (layerLocationInY * imageWidth * (height + spaceBetweenLayers) + layerLocationInX * (width + spaceBetweenLayers)) * 4
                    let firstOutputByteLocation = startingLayerOutputByteLocation + (currentLayerY * imageWidth + currentLayerX) * 4
                
//                let floatValue = self[Int32(0)][Int32(currentLayerY)][Int32(currentLayerX)][Int32(layerIndex)].scalarized()
                    let floatValue = floatBuffer[currentLayerY * width * numLayers + currentLayerX * numLayers + layerIndex]
                    let (redChannel, blueChannel, greenChannel) = visualizationColor(from: floatValue, scale: .heatmap, minValue: minValue, maxValue: maxValue)

                    imageData[firstOutputByteLocation] = redChannel
                    imageData[firstOutputByteLocation + 1] = blueChannel
                    imageData[firstOutputByteLocation + 2] = greenChannel
                    imageData[firstOutputByteLocation + 3] = 255
                }
            }
        }
        
        return (imageData, imageWidth, imageHeight)
    }
}


enum LayerVisualizationColorScale {
    case greyscale
    case heatmap
}

func visualizationColor(from value:Float, scale:LayerVisualizationColorScale, minValue:Float = 0.0, maxValue:Float = 1.0) -> (red:UInt8, green:UInt8, blue:UInt8) {
    let dv = maxValue - minValue
    guard value.isFinite else { return (red:255, green:0, blue:0) }
    guard (maxValue > minValue) else { return (red:255, green:255, blue:255) }
    
    switch scale {
    case .greyscale:
        let scaledValue = (value - minValue) / dv
        let grey = UInt8(max(min(round(scaledValue * 255), 255), 0))
        return (red:grey, green:grey, blue:grey)
    case .heatmap:
        // Drawn from Amro's answer on Stack Overflow here: http://stackoverflow.com/a/7811134/19679
        
        let redValue:Float
        let greenValue:Float
        let blueValue:Float
        
        if (value < (minValue + 0.2 * dv)) {
            redValue = 0.0
            greenValue = 0.0
            blueValue = 0.5 + 2.5 * (value - minValue) / dv
        } else if (value < (minValue + 0.4 * dv)) {
            redValue = 0.0
            greenValue = 5 * (value - minValue - 0.2 * dv) / dv
            blueValue = 1.0
        } else if (value < (minValue + 0.6 * dv)) {
            redValue = 0
            blueValue = 1.0 + 5 * (minValue + 0.4 * dv - value) / dv
            greenValue = 1.0
        } else if (value < (minValue + 0.8 * dv)) {
            redValue = 5 * (value - minValue - 0.6 * dv) / dv
            blueValue = 0
            greenValue = 1.0
        } else {
            greenValue = 1 + 5 * (minValue + 0.8 * dv - value) / dv
            blueValue = 0
            redValue = 1.0
        }
        
        return (red:UInt8(max(min(round(redValue * 255), 255), 0)), green:UInt8(max(min(round(greenValue * 255), 255), 0)), blue:UInt8(max(min(round(blueValue * 255), 255), 0)))
    }
}

public extension AlexNet {
    func debugOutput(for testFile: String) {
        Context.local.learningPhase = .inference

        let testImageURL = URL(fileURLWithPath: testFile)
        let imageFloats = loadImageUsingTensorFlow(from: testImageURL, size: (227, 227), byteOrdering: .bgr, pixelMeanToSubtract: 0.0)!
        let input = Tensor<Float>(shape:[1, 227, 227, 3], scalars: imageFloats)

        input.saveOutputMosaicImageToDisk(prefix: "input")
        let conv1Result = conv1(input)
        conv1Result.saveOutputMosaicImageToDisk(prefix: "conv1")
        let norm1Result = norm1(conv1Result)
        norm1Result.saveOutputMosaicImageToDisk(prefix: "norm1")
        let pool1Result = pool1(norm1Result)
        pool1Result.saveOutputMosaicImageToDisk(prefix: "pool1")
        let conv2Result = conv2(pool1Result)
        conv2Result.saveOutputMosaicImageToDisk(prefix: "conv2")
        let norm2Result = norm2(conv2Result)
        norm2Result.saveOutputMosaicImageToDisk(prefix: "norm2")
        let pool2Result = pool2(norm2Result)
        pool2Result.saveOutputMosaicImageToDisk(prefix: "pool2")
        let conv3Result = conv3(pool2Result)
        conv3Result.saveOutputMosaicImageToDisk(prefix: "conv3")
        let conv4Result = conv4(conv3Result)
        conv4Result.saveOutputMosaicImageToDisk(prefix: "conv4")
        let conv5Result = conv5(conv4Result)
        conv5Result.saveOutputMosaicImageToDisk(prefix: "conv5")
        let pool5Result = pool5(conv5Result)
        pool5Result.saveOutputMosaicImageToDisk(prefix: "pool5")
    }
}
