# AlexNet in Swift #

This is an implementation of the classic <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks">AlexNet</a> convolutional neural network architecture in <a href="https://github.com/tensorflow/swift">Swift for TensorFlow</a>.

It currently builds and runs on both the Mac (via an Xcode project) and on Ubuntu 18.04 (via the `swift build` command in the main directory). The application has been updated to be current with the Swift for TensorFlow API as of the April 18, 2019 snapshot.

On the Mac, you may need to open up your Xcode's File | Project Settings... menu option and switch the Build System to Legacy Build System to get this to compile.

On Ubuntu, running `swift build` will deposit the binary in the .build directory, so to run this against the weights and images that ship with the project, you'll want to run the following from the base project directory:

    .build/x86_64-unknown-linux/debug/AlexNet