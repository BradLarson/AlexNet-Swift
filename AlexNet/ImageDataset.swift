import Foundation
import TensorFlow

#if os(OSX)
import Cocoa
#endif

struct ImageDataset {
    let classes:Int
    let batchSize:Int
    let labels:[String]
//    let combinedImageData:TensorPair<Tensor<Float>,Tensor<Int32>>
    let imageData:Tensor<Float>
    let imageLabels:Tensor<Int32>
    let combinedDataset:Dataset<Zip2TensorGroup<Tensor<Float>, Tensor<Int32>>>
    var shuffledAndBatchedDataset: Dataset<Zip2TensorGroup<Tensor<Float>, Tensor<Int32>>> {
        get {
            return combinedDataset.shuffled(sampleCount:imageLabels.shape[0], randomSeed: 0).batched(self.batchSize)
        }
    }
    
    enum ByteOrdering {
        case bgr
        case rgb
    }
    
    init(imageDirectory: URL, imageSize: (Int, Int), batchSize: Int, byteOrdering: ByteOrdering = .rgb, pixelMeanToSubtract: Float = 0.0) throws {
        let dirContents = try FileManager.default.contentsOfDirectory(at:imageDirectory, includingPropertiesForKeys: [.isDirectoryKey], options:[.skipsHiddenFiles])
        
        self.batchSize = batchSize
        var newImageData:[Float] = []
        var newLabels:[String] = []
        var newImageLabels:[Int32] = []
        var currentLabel:Int32 = 0
        for directoryURL in dirContents {

            newLabels.append(directoryURL.lastPathComponent)
            
            // resourceValues(forKeys:) has not yet been implemented on Linux
//            let resourceValues = try directoryURL.resourceValues(forKeys: [.isDirectoryKey])
//            guard resourceValues.isDirectory! else { continue }
            let subdirContents = try FileManager.default.contentsOfDirectory(at:directoryURL, includingPropertiesForKeys: [.isDirectoryKey], options:[.skipsHiddenFiles])
            for fileURL in subdirContents {
                if let imageFloats = loadImageUsingTensorFlow(from: fileURL, size: imageSize, byteOrdering: byteOrdering, pixelMeanToSubtract: pixelMeanToSubtract) {
//                if let imageFloats = loadImageUsingCoreGraphics(from: fileURL, size: imageSize, byteOrdering: byteOrdering, pixelMeanToSubtract: pixelMeanToSubtract) {
                    newImageData.append(contentsOf: imageFloats)
                    newImageLabels.append(currentLabel)
                }
            }
            currentLabel += 1
        }
        
        self.classes = newLabels.count
        self.imageData = Tensor<Float>(shape:[newImageLabels.count, imageSize.0, imageSize.1, 3], scalars: newImageData)
        self.imageLabels = Tensor<Int32>(newImageLabels)
        self.labels = newLabels
        
        let testImageDataset = Dataset(elements: self.imageData)
        let testImageLabels = Dataset(elements: self.imageLabels)
        self.combinedDataset = zip(testImageDataset, testImageLabels)
    }
}

func loadImageUsingTensorFlow(from fileURL: URL, size: (Int, Int), byteOrdering: ImageDataset.ByteOrdering, pixelMeanToSubtract: Float) -> [Float]? {
    let loadedFile = Raw.readFile(filename: StringTensor(fileURL.absoluteString))
    let loadedJpeg = Raw.decodeJpeg(contents: loadedFile, channels: 3, dctMethod: "")
    let resizedImage = Raw.resizeBilinear(images: Tensor<UInt8>([loadedJpeg]), size: Tensor<Int32>([Int32(size.0), Int32(size.1)])) - pixelMeanToSubtract
    if (byteOrdering == .bgr) {
        let reversedChannelImage = Raw.reverse(resizedImage, dims: Tensor<Bool>([false, false, false, true]))
        return reversedChannelImage.scalars
    } else {
        return resizedImage.scalars
    }
}

#if os(OSX)

func loadImageUsingCoreGraphics(from fileURL: URL, size: (Int, Int), byteOrdering: ImageDataset.ByteOrdering, pixelMeanToSubtract: Float) -> [Float]? {
    let totalImageSize = size.0 * size.1
    var imageFloats = [Float](repeating: 0.0, count: size.0 * size.1 * 3)
    
    guard let currentImage = NSImage(contentsOf: fileURL) else { return nil }
    
    let rawImageData = UnsafeMutablePointer<UInt8>.allocate(capacity:size.0 * size.1 * 4)
    
    let genericRGBColorspace = CGColorSpaceCreateDeviceRGB()
    
    let imageContext = CGContext(data:rawImageData, width:size.0, height:size.1, bitsPerComponent:8, bytesPerRow:size.0 * 4, space:genericRGBColorspace,  bitmapInfo:CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    imageContext?.draw(currentImage.cgImage(forProposedRect:nil, context:nil, hints:nil)!, in:CGRect(x:0.0, y:0.0, width:CGFloat(size.0), height:CGFloat(size.1)))
    for currentPixelIndex in 0..<totalImageSize {
        if (byteOrdering == .bgr) {
            imageFloats[(currentPixelIndex * 3) + 2] = Float(rawImageData[currentPixelIndex * 4]) - pixelMeanToSubtract
            imageFloats[(currentPixelIndex * 3) + 1] = Float(rawImageData[(currentPixelIndex * 4) + 1]) - pixelMeanToSubtract
            imageFloats[(currentPixelIndex * 3)] = Float(rawImageData[(currentPixelIndex * 4) + 2]) - pixelMeanToSubtract
        } else {
            imageFloats[(currentPixelIndex * 3)] = Float(rawImageData[currentPixelIndex * 4]) - pixelMeanToSubtract
            imageFloats[(currentPixelIndex * 3) + 1] = Float(rawImageData[(currentPixelIndex * 4) + 1]) - pixelMeanToSubtract
            imageFloats[(currentPixelIndex * 3) + 2] = Float(rawImageData[(currentPixelIndex * 4) + 2]) - pixelMeanToSubtract
        }
    }
    
    rawImageData.deallocate()
    
    return imageFloats
}

// Drawn from Atika's answer on Stack Overflow here: https://stackoverflow.com/a/42915296/19679
extension NSImage {
    func resized(to newSize: NSSize) -> NSImage? {
        if let bitmapRep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(newSize.width), pixelsHigh: Int(newSize.height), bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .calibratedRGB, bytesPerRow: 0, bitsPerPixel: 0) {
            bitmapRep.size = newSize
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmapRep)
            draw(in: NSRect(x: 0, y: 0, width: newSize.width, height: newSize.height), from: .zero, operation: .copy, fraction: 1.0)
            NSGraphicsContext.restoreGraphicsState()
            
            let resizedImage = NSImage(size: newSize)
            resizedImage.addRepresentation(bitmapRep)
            return resizedImage
        }
        
        return nil
    }
}
#endif
