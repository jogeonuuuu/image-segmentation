# ADE20K-model


https://github.com/rstrudel/segmenter/blob/master/segm/data/config/ade20k.yml

std::vector<cv::Vec3b> colors(150);
    colors[12]=cv::Vec3b(255,0,255);
    colors[33]=colors[15]=colors[56]=colors[64]=cv::Vec3b(255,255,0);
    colors[19]=colors[30]=colors[31]=colors[75]=colors[101]=cv::Vec3b(0,255,255); // chair..
    colors[5]=cv::Vec3b(128,128,128);
    colors[0]=cv::Vec3b(255,0,0);
    colors[3]=cv::Vec3b(255,255,255);
    colors[10]=cv::Vec3b(0,255,0);
    colors[14]=cv::Vec3b(128,128,0);
    colors[41]=cv::Vec3b(0,0,255);
    colors[50]=cv::Vec3b(0,128,128);
    colors[74]=colors[143]=cv::Vec3b(128,0,128);
    colors[139]=cv::Vec3b(0,200,0); // fan

    cv::Mat colorMask(inputH, inputW, CV_8UC3, cv::Scalar(0,0,0));

    for (int row = 0; row < segmentationMask.rows; row++) {
        for (int col = 0; col < segmentationMask.cols; col++) {
            colorMask.at<cv::Vec3b>(row, col) = colors[segmentationMask.at<uchar>(row, col)];
        }
    }
