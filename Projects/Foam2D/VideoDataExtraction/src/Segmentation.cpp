#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

#include "../include/Segmentation.h"

static void doSomething(cv::Mat &image) {
//    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
//    cv::ximgproc::anisotropicDiffusion(image, image, 0.11, 0.01, 100);
//    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

//    cv::erode(image, image, cv::Mat::ones(3, 3, CV_8U));
//    cv::medianBlur(image, image, 3);
////    cv::medianBlur(image, image, 3);
////    cv::blur(image, image, cv::Size_<int>(3, 3));
////    cv::blur(image, image, cv::Size_<int>(3, 3));
////    cv::blur(image, image, cv::Size_<int>(3, 3));
//    auto clahe = cv::createCLAHE(2.0, cv::Size(16, 16));
//    clahe->apply(image, image);
//    cv::equalizeHist(image, image);

//    cv::Mat blur;
//    cv::blur(image, blur, cv::Size_<int>(10, 10));

    cv::Mat imagecopy;
    image.copyTo(imagecopy);
    cv::bilateralFilter(imagecopy, image, 11, 200, 200);
////    imshow("Blurred Image", blur);
//
//    cv::divide(image, blur, image, 255);
//    imshow("Division by blurred image", image);

//    cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
//    cv::ximgproc::anisotropicDiffusion(image, image, 0.4, 0.01, 100);
//    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
}

cv::Mat imageMatchSegmentation(const cv::Mat &src, cv::Mat &dst, cv::Mat &markers, std::vector<cv::Vec3b> &colors) {
//    imshow("Source Image", src);

    cv::Mat uncroppedimage;
    cv::cvtColor(src, uncroppedimage, cv::COLOR_BGR2GRAY);

    cv::Mat bgnd;
    uncroppedimage.copyTo(bgnd);
    cv::floodFill(bgnd, cv::Point(0, 0), 0);
    std::vector<std::vector<cv::Point> > backgroundcontour;
    cv::findContours(bgnd, backgroundcontour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect rect = cv::boundingRect(backgroundcontour[0]);
    cv::Mat image = uncroppedimage(rect);
    cv::copyMakeBorder(image, image, 100, 100, 100, 100, cv::BORDER_CONSTANT, cv::Scalar::all(0));
//    cv::floodFill(image, cv::Point(0, 0), 0);

    cv::blur(image, image, cv::Size_<int>(5, 5));
    auto clahe = cv::createCLAHE(10.0, cv::Size(8, 8));
    clahe->apply(image, image);
//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    clahe->apply(image, image);
//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    clahe->apply(image, image);
//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    clahe->apply(image, image);
//    cv::equalizeHist(image, image);
//    imshow("clahe", image);

    cv::blur(image, image, cv::Size_<int>(5, 5));

    cv::Mat blur;
    cv::blur(image, blur, cv::Size_<int>(10, 10));
//    imshow("Blurred Image", blur);

    cv::divide(image, blur, image, 255);
//    imshow("Division by blurred image", image);

//    clahe->apply(image, image);
//    cv::equalizeHist(image, image);
//    imshow("clahe2", image);

//    clahe->apply(image, image);

//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    imshow("Blurred Image1", image);
//
////    auto clahe = cv::createCLAHE(400.0, cv::Size(16, 16));
////    clahe->apply(image, image);
////    imshow("clahe", image);
//
    doSomething(image);
//    imshow("Blurred Image4", image);

    auto clahe2 = cv::createCLAHE(4.0, cv::Size(64, 64));
    clahe2->apply(image, image);
//    imshow("Clahe2", image);

    doSomething(image);
//    imshow("Blurred Image5", image);
    doSomething(image);
//    imshow("Blurred Image6", image);
//    doSomething(image);
//    imshow("Blurred Image7", image);
//    doSomething(image);
//    imshow("Blurred Image8", image);

    cv::bitwise_not(image, image);

//    {
//        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
//        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
//        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255);
//        std::cout << blob.type() << " " << blob.size << std::endl;
//        std::cout << image.type() << " " << image.size << std::endl;
//        auto net = cv::dnn::readNetFromCaffe("../../../../Projects/Foam2D/VideoDataExtraction/hed/deploy.prototxt",
//                                             "../../../../Projects/Foam2D/VideoDataExtraction/hed/hed_pretrained_bsds.caffemodel");
//        net.setInput(blob);
//        cv::Mat hed = net.forward();
//        std::cout << hed.type() << " " << hed.size << std::endl;
//        int sz[] = {hed.size[2], hed.size[3]};
//        cv::Mat hed2(2, sz, hed.type(), hed.ptr<double>(0));
//        std::cout << hed2.type() << " " << hed2.size << std::endl;
//
//        imshow("Hed", hed2);
//        hed2.convertTo(hed2, CV_8UC1, 255);
//        cv::equalizeHist(hed2, hed2);
//        imshow("Hed hist", hed2);
//
//        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
//    }

//    threshold(image, image, 5, 255, cv::THRESH_BINARY);
    cv::adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 201, -1);
    imshow("Bin", image);

    cv::Mat skel;
    cv::ximgproc::thinning(image, skel);
//    cv::bitwise_not(skel, skel);
//    cv::cvtColor(skel, skel, cv::COLOR_GRAY2BGR);
    imshow("Skeleton", skel);

//    std::vector<cv::Vec4f> lines;
//    cv::HoughLinesP(skel, lines, 2, CV_PI / 180 * 2, 30);
//    std::cout << "num lines " << lines.size() << std::endl;
//
//    for (size_t i = 0; i < lines.size(); i++) {
//        cv::Vec4f l = lines[i];
//        std::cout << l[0] << " " << l[1] << " " << l[2] << " " << l[3] << std::endl;
//        line(image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(125), 1, cv::LINE_AA);
//    }
//    imshow("Hough", image);

//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    imshow("Blurred Image2", image);
//
//    cv::ximgproc::RidgeDetectionFilter::create()->getRidgeFilteredImage(image, image);
//    imshow("Ridges", image);

//    cv::blur(image, image, cv::Size_<int>(3, 3));
//    cv::blur(image, image, cv::Size_<int>(3, 3));
//    imshow("Blurred Image3", image);
//

//    erodeHorz(image);
//    erodeVert(image);
//    erodeIso(image);
//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    cv::blur(image, image, cv::Size_<int>(5, 5));
//    doSomething(image);
//    imshow("Blurred Image4", image);
//    doSomething(image);
//    imshow("Blurred Image5", image);
//    doSomething(image);
//    imshow("Blurred Image6", image);
//    doSomething(image);
//    imshow("Blurred Image7", image);
//    doSomething(image);
//    imshow("Blurred Image8", image);
//
//    cv::equalizeHist(image, image);
//    imshow("Hist", image);

//    cv::ximgproc::RidgeDetectionFilter::create()->getRidgeFilteredImage(image, image);
//    imshow("Ridges", image);

//    cv::Mat skel;
//    cv::bitwise_not(image, image);
//    threshold(image, image, 5, 255, cv::THRESH_BINARY);
//    imshow("Bin", image);
//
//    cv::ximgproc::thinning(image, skel);
//    cv::bitwise_not(skel, skel);
//    cv::cvtColor(skel, skel, cv::COLOR_GRAY2BGR);
//    imshow("Skeleton0", skel);

//    cv::Canny(image, image, 200, 600, 5, true);
//    imshow("Canny", image);
////    threshold(image, image, 180, 255, cv::THRESH_BINARY);
////    imshow("Bin", image);
//
    cv::Mat kernel1 = cv::Mat::ones(5, 5, CV_8U);
    cv::erode(image, image, kernel1);
    cv::erode(image, image, kernel1);
    cv::dilate(image, image, kernel1);
    cv::dilate(image, image, kernel1);
//    imshow("Open", image);
    cv::dilate(image, image, kernel1);
    cv::dilate(image, image, kernel1);
    cv::dilate(image, image, kernel1);
//    imshow("Dilate", image);
    cv::dilate(image, image, kernel1);
    cv::dilate(image, image, kernel1);
    cv::dilate(image, image, kernel1);
    cv::erode(image, image, kernel1);
    cv::erode(image, image, kernel1);
    cv::erode(image, image, kernel1);
//    imshow("Close", image);
//
////    cv::Mat skel;
//    cv::ximgproc::thinning(image, skel);
//    cv::bitwise_not(skel, skel);
    cv::cvtColor(skel, skel, cv::COLOR_GRAY2BGR);
//    imshow("Skeleton", skel);

    cv::bitwise_not(image, image);
//    imshow("Peaks", image);
    cv::floodFill(image, {0, 0}, 0);
//    imshow("Flood", image);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Create the marker image for the watershed algorithm
    markers = cv::Mat::zeros(image.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i + 1)), -1);
    }
    markers.at<int>(1, 1) = contours.size() + 1; // Mark background

    // Perform the watershed algorithm
    cv::watershed(skel, markers);

//    cv::Mat mark;
//    markers.convertTo(mark, CV_8U);
//    bitwise_not(mark, mark);
//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark

    // Generate random colors
    colors.clear();
    for (size_t i = 0; i < contours.size(); i++) {
        int b = cv::theRNG().uniform(0, 256);
        int g = cv::theRNG().uniform(0, 256);
        int r = cv::theRNG().uniform(0, 256);
        colors.push_back(cv::Vec3b((uchar) b, (uchar) g, (uchar) r));
    }

    // Create the result image
    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<cv::Vec3b>(i + rect.y - 100, j + rect.x - 100) = colors[index - 1];
            }
        }
    }

//     Visualize the final image
//    imshow("Final Result", dst);

//    cv::waitKey();
    return dst;
}

using namespace std;
using namespace cv;

cv::Mat imageMatchSegmentationTutorial(cv::Mat src) {
    // Show the source image
    imshow("Source Image", src);

    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
//    Mat mask;
//    inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
//    src.setTo(Scalar(0, 0, 0), mask);
    // Show output image
    imshow("Black Background Image", src);
    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3, 3) <<
                                    1, 1, 1,
            1, -8, 1,
            1, 1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow("New Sharped Image", imgResult);
    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Binary Image", bw);
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++) {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }
    // Draw the background marker
    circle(markers, Point(5, 5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    imshow("Markers", markers8u);
    // Perform the watershed algorithm
    watershed(imgResult, markers);
    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
    }
    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contours.size())) {
                dst.at<Vec3b>(i, j) = colors[index - 1];
            }
        }
    }
    // Visualize the final image
    imshow("Final Result", dst);
    waitKey();

    return dst;
}
