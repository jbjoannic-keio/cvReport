#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>

// GENERAL

cv::Mat convertToGrayscale(cv::Mat img)
{
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    return imgGray;
}

cv::Mat readImage(std::string path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    return img;
}

void saveImage(std::string path, cv::Mat img)
{
    cv::imwrite(path, img);
}

// Create a vector of all the images path in a folder
std::vector<std::string> findAllFiles(std::string path)
{
    std::vector<std::string> imagePaths;
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".JPG")
        {
            imagePaths.push_back(entry.path());
        }
    }
    return imagePaths;
}

// Read all the images from a folder
std::vector<cv::Mat> readImagesFromPath(std::string imgPath)
{
    std::vector<cv::Mat> imgList;
    std::vector<std::string> imgNames = findAllFiles(imgPath);
    std::sort(imgNames.begin(), imgNames.end());
    for (size_t i = 0; i < imgNames.size(); i++)
    {
        imgList.push_back(readImage(imgNames[i]));
        std::cout << "Reading image: " << imgNames[i] << std::endl;
    }
    return imgList;
}

// NORMALIZATION AND ALIGNMENT
cv::Mat normalizeF(cv::Mat x)
{
    // Find max and min values of x
    double minVal, maxVal;
    cv::minMaxLoc(x, &minVal, &maxVal);
    cv::Mat normalized = (x - minVal) / (maxVal - minVal);
    return normalized;
}

// Align images using SIFT features
// Is not used in the final version because it is not robust enough to the defocus blur
std::vector<cv::Mat> alignImages(const std::vector<cv::Mat> &images)
{
    std::vector<cv::Mat> alignedImages;

    if (images.empty())
    {
        return alignedImages;
    }
    int mid = images.size() / 2;

    for (size_t i = 0; i < images.size(); i++)
    {
        if (i == mid)
        {
            alignedImages.push_back(images[i]);
            continue;
        }
        cv::Mat referenceImage = images[mid];
        cv::Mat targetImage = images[i];

        cv::Mat grayReference, grayTarget;
        cv::cvtColor(referenceImage, grayReference, cv::COLOR_BGR2GRAY);
        cv::cvtColor(targetImage, grayTarget, cv::COLOR_BGR2GRAY);

        // Detect keypoints and extract descriptors
        std::vector<cv::KeyPoint> keypointsReference, keypointsTarget;
        cv::Mat descriptorsReference, descriptorsTarget;

        // SIFT detector / descriptor
        cv::Ptr<cv::Feature2D> SIFT = cv::SIFT::create();
        SIFT->detectAndCompute(grayReference, cv::noArray(), keypointsReference, descriptorsReference);
        SIFT->detectAndCompute(grayTarget, cv::noArray(), keypointsTarget, descriptorsTarget);
        std::cout << "Keypoints: image " << i << "   " << keypointsReference.size() << std::endl;

        // Match keypoints
        std::vector<cv::DMatch> matches;
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
        matcher->match(descriptorsReference, descriptorsTarget, matches);
        std::cout << "matches: image " << i << "   " << matches.size() << std::endl;

        std::vector<cv::DMatch> goodMatches;
        double thresholdDist = 50;
        for (size_t j = 0; j < matches.size(); j++)
        {
            if (matches[j].distance < thresholdDist)
            {
                goodMatches.push_back(matches[j]);
            }
        }

        // Extract matched keypoints
        std::vector<cv::Point2f> referencePoints, targetPoints;
        for (size_t j = 0; j < goodMatches.size(); j++)
        {
            referencePoints.push_back(keypointsReference[goodMatches[j].queryIdx].pt);
            targetPoints.push_back(keypointsTarget[goodMatches[j].trainIdx].pt);
        }
        std::cout << "good: image " << i << "   " << referencePoints.size() << std::endl;

        // Find homography and warp the target image to align it with the reference image
        if (referencePoints.size() < 4 || targetPoints.size() < 4)
        {
            alignedImages.push_back(targetImage);
            continue;
        }
        cv::Mat homography = cv::findHomography(targetPoints, referencePoints, cv::RANSAC);
        cv::Mat alignedImage;
        std::cout << " M " << homography << std::endl;
        if (homography.empty())
        {
            alignedImage = targetImage;
        }
        else
        {
            cv::warpPerspective(targetImage, alignedImage, homography, referenceImage.size());
        }
        alignedImages.push_back(alignedImage);
    }

    return alignedImages;
}

// Create a range of lens disances, assuming that the lens is linearly over the photos
std::vector<float> lensLengthsC(float distanceMin, float distanceMax, float focalLength, int photoNb)
{
    // 1/f = 1/s + 1/o
    // We have omin and omax, f, and so smin and smax. We assess that s is linearly distributed.
    float smin = 1 / (1 / focalLength - 1 / distanceMin);
    float smax = 1 / (1 / focalLength - 1 / distanceMax);

    std::vector<float> lensLengths = std::vector<float>(photoNb);
    for (int i = 0; i < photoNb; i++)
    {
        lensLengths[i] = smin + (smax - smin) * i / (photoNb - 1);
    }
    return lensLengths;
}

// FOCUS STACKING
cv::Mat focusStack(cv::Mat alignedImg, int gaussianSize, int laplacianSize)
{
    cv::Mat imGray = convertToGrayscale(alignedImg);
    cv::Mat gaussianImg;
    cv::GaussianBlur(imGray, gaussianImg, cv::Size(gaussianSize, gaussianSize), 0);
    cv::Mat laplacianImg;
    cv::Laplacian(gaussianImg, laplacianImg, CV_64F, laplacianSize);
    return laplacianImg;
}

// Focus measure calculation (double summation of the laplacian)
std::vector<cv::Mat> focusMeasureCal(std::vector<cv::Mat> costVolume, int kernelSize = 9)
{
    std::vector<cv::Mat> focusMeasure = std::vector<cv::Mat>(costVolume.size());

    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_64F);

    for (int i = 0; i < costVolume.size(); i++)
    {
        cv::Mat focusImg = costVolume[i];
        focusMeasure[i] = cv::Mat::zeros(focusImg.rows, focusImg.cols, CV_64F);
        for (int j = 0; j < focusImg.rows; j++)
        {
            for (int k = 0; k < focusImg.cols; k++)
            {
                focusMeasure[i].at<double>(j, k) = focusImg.at<double>(j, k) * focusImg.at<double>(j, k);
            }
        }
        cv::filter2D(focusMeasure[i], focusMeasure[i], -1, kernel);
    }
    return focusMeasure;
}

// Argmax of the focus measure
cv::Mat argMaxF(std::vector<cv::Mat> focusMeasure)
{
    cv::Mat argMaxImg = cv::Mat::zeros(focusMeasure[0].rows, focusMeasure[0].cols, CV_64F);
    for (size_t i = 0; i < focusMeasure.size(); i++)
    {
        for (int j = 0; j < focusMeasure[i].rows; j++)
        {
            for (int k = 0; k < focusMeasure[i].cols; k++)
            {
                if (focusMeasure[i].at<double>(j, k) > focusMeasure[argMaxImg.at<double>(j, k)].at<double>(j, k))
                {
                    argMaxImg.at<double>(j, k) = i;
                }
            }
        }
    }
    return argMaxImg;
}

// returns in the order the three indexes of the three maximum values of the focus Measure
std::vector<int> argMaxThree(std::vector<cv::Mat> focusMeasure, int j, int k)
{
    std::vector<int> argMax;
    for (size_t i = 0; i < focusMeasure.size(); i++)
    {
        if (argMax.size() == 0)
        {
            argMax.push_back(i);
        }
        else if (argMax.size() == 1)
        {
            if (focusMeasure[i].at<double>(j, k) > focusMeasure[argMax[0]].at<double>(j, k))
            {
                argMax.push_back(argMax[0]);
                argMax[0] = i;
            }
            else
            {
                argMax.push_back(i);
            }
        }
        else if (argMax.size() == 2)
        {
            if (focusMeasure[i].at<double>(j, k) > focusMeasure[argMax[0]].at<double>(j, k))
            {
                argMax.push_back(argMax[1]);
                argMax[1] = argMax[0];
                argMax[0] = i;
            }
            else if (focusMeasure[i].at<double>(j, k) > focusMeasure[argMax[1]].at<double>(j, k))
            {
                argMax.push_back(argMax[1]);
                argMax[1] = i;
            }
            else
            {
                argMax.push_back(i);
            }
        }
        else
        {
            if (focusMeasure[i].at<double>(j, k) > focusMeasure[argMax[0]].at<double>(j, k))
            {
                argMax[2] = argMax[1];
                argMax[1] = argMax[0];
                argMax[0] = i;
            }
            else if (focusMeasure[i].at<double>(j, k) > focusMeasure[argMax[1]].at<double>(j, k))
            {
                argMax[2] = argMax[1];
                argMax[1] = i;
            }
            else if (focusMeasure[i].at<double>(j, k) > focusMeasure[argMax[2]].at<double>(j, k))
            {
                argMax[2] = i;
            }
        }
    }
    return argMax;
}

// Interpolation of the argmax, returns the depth map
cv::Mat argMaxFInterpolation(std::vector<cv::Mat> focusMeasure, std::vector<float> lensLengths, float focalLength)
{
    cv::Mat depthMap = cv::Mat::zeros(focusMeasure[0].rows, focusMeasure[0].cols, CV_64F);

    for (int j = 0; j < focusMeasure[0].rows; j++)
    {
        for (int k = 0; k < focusMeasure[0].cols; k++)
        {
            std::vector<int> argMax = argMaxThree(focusMeasure, j, k);
            std::sort(argMax.begin(), argMax.end());
            double MS1 = focusMeasure[argMax[0]].at<double>(j, k);
            double MS2 = focusMeasure[argMax[1]].at<double>(j, k);
            double MS3 = focusMeasure[argMax[2]].at<double>(j, k);
            double S1 = lensLengths[argMax[0]];
            double S2 = lensLengths[argMax[1]];
            double S3 = lensLengths[argMax[2]];
            double _S1 = ((log(MS2) - log(MS3)) * (S2 * S2 - S1 * S1)) / (2 * (S3 - S2) * ((log(MS2) - log(MS1)) + (log(MS2) - log(MS3))));
            double _S2 = ((log(MS2) - log(MS1)) * (S2 * S2 - S3 * S3)) / (2 * (S3 - S2) * ((log(MS2) - log(MS1)) + (log(MS2) - log(MS3))));
            double _S = _S1 - _S2;

            double o = (_S * focalLength) / (_S - focalLength);
            depthMap.at<double>(j, k) = std::min(std::max(o, 0.0), 400.0);
        }
    }
    return depthMap;
}

// create the non interpolated depth map and a all in focus image
std::pair<cv::Mat, cv::Mat> allInFocus(std::vector<cv::Mat> imgList, std::vector<cv::Mat> costVolume, int kernelSize, int gaussianSize)
{
    cv::Mat allInFocusImg = cv::Mat::zeros(imgList[0].rows, imgList[0].cols, CV_8UC3);
    int height, width;
    height = imgList[0].rows;
    width = imgList[0].cols;

    std::vector<cv::Mat> focusMeasure = focusMeasureCal(costVolume, kernelSize);
    cv::Mat argMax = argMaxF(focusMeasure);

    cv::Mat normalize = 255 - (normalizeF(argMax) * 255);
    cv::Mat depthMap;
    cv::GaussianBlur(normalize, depthMap, cv::Size(gaussianSize, gaussianSize), 0);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = argMax.at<double>(i, j);
            allInFocusImg.at<cv::Vec3b>(i, j) = imgList[index].at<cv::Vec3b>(i, j);
        }
    }

    return std::make_pair(depthMap, allInFocusImg);
}

// create the interpolated depth map
cv::Mat allInFocusInterpolation(std::vector<cv::Mat> imgList, std::vector<cv::Mat> costVolume, std::vector<float> lensLengths, float focalLength, int kernelSize, int gaussianSize)
{
    std::vector<cv::Mat> focusMeasure = focusMeasureCal(costVolume, kernelSize);
    cv::Mat argMaxInterpolation = argMaxFInterpolation(focusMeasure, lensLengths, focalLength);

    cv::Mat normalize = 255 - (normalizeF(argMaxInterpolation) * 255);
    cv::Mat depthMap;

    cv::GaussianBlur(normalize, depthMap, cv::Size(gaussianSize, gaussianSize), 0);
    return depthMap;
}

int main(int argc, char **argv)
{
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path parentPath = currentPath.parent_path();
    std::string imgPath = parentPath.string() + "/images/Final/1";
    std::string resultPath = parentPath.string() + "/images/Final/1/results";

    std::vector<cv::Mat> imgList = readImagesFromPath(imgPath);
    // imgList = alignImages(imgList);   // Not used in the final version
    std::vector<cv::Mat> stackedFocusImgs;

    std::vector<float> lensLengths = lensLengthsC(60, 1500, 26, imgList.size());

    for (size_t i = 0; i < imgList.size(); i++)
    {
        std::string focusPath = "focus_" + std::to_string(i) + ".jpg";
        cv::Mat laplacianImg = focusStack(imgList[i], 9, 9);
        stackedFocusImgs.push_back(laplacianImg);

        saveImage(resultPath + "/" + focusPath, laplacianImg);
    }

    std::vector<cv::Mat> costVolume = stackedFocusImgs;
    std::pair<cv::Mat, cv::Mat> allInFocusP = allInFocus(imgList, costVolume, 128, 9);

    cv::Mat depthMapInterpolation = allInFocusInterpolation(imgList, costVolume, lensLengths, 26, 128, 9);
    saveImage(resultPath + "/depthMap.jpg", allInFocusP.first);
    saveImage(resultPath + "/allInFocus.jpg", allInFocusP.second);
    saveImage(resultPath + "/depthMapInterpolation.jpg", depthMapInterpolation);
}