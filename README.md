<img align="right" alt="GIF" src="https://github.com/AbidHasanPiash/Hand-Pose-Estimation/blob/main/output_gif.gif" width="300px">

# Hand-Pose-Estimation
Hand pose estimation is the task of finding the joints of the hand from an image or set of video frames.

I used an existing method called Multiview Bootstrapping to find hand key-points. Then I added two features, one is drawing skeletons of hand and the other is the classification of all fingers. 

# Requirment
- OpenCV 4.5.1
- Numpy 1.20.1

- ### prototxt
The structure of the neural network is stored as a `.prototxt` file. 
- ### caffemodel
The weights of the layers of the neural network are stored as a `.caffemodel` file.

`.prototext` file is already added in the `Hand` directory. You just need to download the `.caffemodel` file from the [link](http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel) and put it on `Hand` directory.

# Features
- [x] Hand key-points detection
- [x] Estimating skeletons 
- [x] Classification of fingers
- [ ] Left and right hand detection
- [ ] Multiple hand detection

### Insperations
The base model of this project is comes from [here](https://github.com/xar47x/pose)
