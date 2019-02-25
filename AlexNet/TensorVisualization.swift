import TensorFlow
import Foundation
#if os(OSX)
import Cocoa

func dataProviderReleaseCallback(_ context:UnsafeMutableRawPointer?, data:UnsafeRawPointer, size:Int) {
    data.deallocate()
}
#endif

extension Tensor {
    public func saveOutputMosaicImageToDisk(prefix:String, spaceBetweenSlices:Int = 1) {
#if os(OSX)
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
        let sliceByteSize = width * height * 4
        
        let numberOfLayersAcross = Int(round(sqrt(Double(numLayers))))
        let numberOfLayersDown = Int(ceil(Double(numLayers) / Double(numberOfLayersAcross)))
        
        let imageWidth = width * numberOfLayersAcross + (numberOfLayersAcross * spaceBetweenLayers)
        let imageHeight = height * numberOfLayersDown + (numberOfLayersDown * spaceBetweenLayers)
        let imageByteSize = imageWidth * imageHeight * 4
        let imageData = UnsafeMutablePointer<UInt8>.allocate(capacity:imageByteSize)
        
        let totalBytes = imageWidth * imageHeight
        // Initialize with black
        for currentPixelIndex in 0..<totalBytes {
            imageData[currentPixelIndex * 4] = 0
            imageData[(currentPixelIndex * 4) + 1] = 0
            imageData[(currentPixelIndex * 4) + 2] = 0
            imageData[(currentPixelIndex * 4) + 3] = 255
        }
        
        // TODO: Pull in my image coloration code
        
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
