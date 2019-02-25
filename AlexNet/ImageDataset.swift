import Foundation
import TensorFlow

#if os(OSX)
import Cocoa
#endif
//let imageDataset = Dataset(elements: <#T##_#>)

struct ImageDataset {
    let classes:Int
    let labels:[String]
    let imageData:Tensor<Float>
    let imageLabels:Tensor<Int32>
    
    enum ByteOrdering {
        case bgr
        case rgb
    }
    
    init(imageDirectory: URL, imageSize: (Int, Int), byteOrdering: ByteOrdering = .rgb) throws {
        let dirContents = try FileManager.default.contentsOfDirectory(at:imageDirectory, includingPropertiesForKeys: [.isDirectoryKey], options:[.skipsHiddenFiles])
        
        var newImageData:[Float] = []
        var newLabels:[String] = []
        var newImageLabels:[Int32] = []
        var currentLabel:Int32 = 0
        for directoryURL in dirContents {
            newLabels.append(directoryURL.lastPathComponent)
            print("Adding image category: \(directoryURL.lastPathComponent)")
            
            let resourceValues = try directoryURL.resourceValues(forKeys: [.isDirectoryKey])
            guard resourceValues.isDirectory! else { continue }
            let subdirContents = try FileManager.default.contentsOfDirectory(at:directoryURL, includingPropertiesForKeys: [.isDirectoryKey], options:[.skipsHiddenFiles])
            for fileURL in subdirContents {
                let totalImageSize = imageSize.0 * imageSize.1
                var imageFloats = [Float](repeating: 0.0, count: imageSize.0 * imageSize.1 * 3)
                // Note: using 0..255 dynamic range to match Caffe training conditions

#if os(OSX)
                guard let currentImage = NSImage(contentsOf: fileURL) else { continue }

                let targetImageSize = CGSize(width:imageSize.0, height:imageSize.1)
                let rawImageData = UnsafeMutablePointer<CChar>.allocate(capacity:imageSize.0 * imageSize.1 * 4)
                
                let genericRGBColorspace = CGColorSpaceCreateDeviceRGB()
                
                let imageContext = CGContext(data:rawImageData, width:imageSize.0, height:imageSize.1, bitsPerComponent:8, bytesPerRow:imageSize.0 * 4, space:genericRGBColorspace,  bitmapInfo:CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
                //        CGContextSetBlendMode(imageContext, kCGBlendModeCopy); // From Technical Q&A QA1708: http://developer.apple.com/library/ios/#qa/qa1708/_index.html
                imageContext?.draw(currentImage.cgImage(forProposedRect:nil, context:nil, hints:nil)!, in:CGRect(x:0.0, y:0.0, width:CGFloat(imageSize.0), height:CGFloat(imageSize.1)))
                for currentPixelIndex in 0..<totalImageSize {
                    if (byteOrdering == .bgr) {
                        imageFloats[(currentPixelIndex * 3) + 2] = Float(rawImageData[currentPixelIndex * 4])
                        imageFloats[(currentPixelIndex * 3) + 1] = Float(rawImageData[(currentPixelIndex * 4) + 1])
                        imageFloats[(currentPixelIndex * 3)] = Float(rawImageData[(currentPixelIndex * 4) + 2])
                    } else {
                        imageFloats[(currentPixelIndex * 3)] = Float(rawImageData[currentPixelIndex * 4])
                        imageFloats[(currentPixelIndex * 3) + 1] = Float(rawImageData[(currentPixelIndex * 4) + 1])
                        imageFloats[(currentPixelIndex * 3) + 2] = Float(rawImageData[(currentPixelIndex * 4) + 2])
                    }
                }
                
                rawImageData.deallocate()
#else
                // Resize image
                // Pull into BGR colorspace?
#endif

                
                newImageData.append(contentsOf: imageFloats)
                newImageLabels.append(currentLabel)
            }
            currentLabel += 1
        }
        
        self.classes = newLabels.count
        self.imageData = Tensor<Float>(shape:[Int32(newImageLabels.count), Int32(imageSize.0), Int32(imageSize.1), 3], scalars: newImageData)
        self.imageLabels = Tensor<Int32>(newImageLabels)
        self.labels = newLabels
    }
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
