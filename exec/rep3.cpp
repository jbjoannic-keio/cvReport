#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "opencv2/opencv.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[])
{

    std::string folderPath = "./images/3/Overlap2"; // Path to the folder with images of the same panorama

    std::vector<std::string> imagePaths;

    // Create viewer
    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image", 800, 600);
    cv::namedWindow("Warped translation", cv::WINDOW_NORMAL);
    cv::resizeWindow("Warped translation", 800, 600);
    cv::namedWindow("Warped similarity", cv::WINDOW_NORMAL);
    cv::resizeWindow("Warped similarity", 800, 600);

    // Read all images from folder
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
        {
            imagePaths.push_back(entry.path());
        }
    }

    // create vectors to store images
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> imagesWithFeatures;                // images with features drawn on them (for visualization)
    std::vector<std::vector<cv::KeyPoint>> imagesKeypoints; // keypoints for each image
    std::vector<cv::Mat> imagesDescriptors;                 // descriptors for each image

    // create AKAZE detector, descriptor extractor. Other are possible
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();

    // detect keypoints and compute descriptors for each image
    for (int i = 0; i < imagePaths.size(); i++)
    {
        cv::Mat img = cv::imread(imagePaths[i]);

        cv::Mat imgWithFeatures = img.clone();

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        akaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

        imagesKeypoints.push_back(keypoints);
        imagesDescriptors.push_back(descriptors);

        // draw keypoints on image
        for (int j = 0; j < keypoints.size(); j++)
        {
            cv::circle(imgWithFeatures, keypoints[j].pt, 5, cv::Scalar(0, 255, 0), -1);
        }

        // draw image number on image, should be resize on low quality images
        cv::rectangle(imgWithFeatures, cv::Point(0, 0), cv::Point(400, 400), cv::Scalar(0, 0, 0), -1);
        cv::putText(imgWithFeatures, std::to_string(i), cv::Point(100, 300), cv::FONT_HERSHEY_SIMPLEX, 10, cv::Scalar(255, 255, 255), 5);

        images.push_back(img);
        imagesWithFeatures.push_back(imgWithFeatures);

        cv::imshow("Image", imgWithFeatures);

        std::string res_path = folderPath + "/results/" + "features_" + std::to_string(i) + ".png";
        cv::imwrite(res_path, imgWithFeatures);
        cv::waitKey(10);
    }

    // create brute force matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // create table for matches, given the fact that matches are computed just one time, we store them in a partcular vector
    // for example for 3 input images :
    // std::vector<std::vector<cv::KeyPoint>> matched1_all = {matched1_01, matched1_02, matched1_12} so nn_mathes_table = {0, 1, 2,    0, 0, 3,    0, 0, 0}
    cv::Mat nn_matches_table = cv::Mat::zeros(images.size(), images.size(), CV_32S);
    cv::Mat nn_matches_number = cv::Mat::zeros(images.size(), images.size(), CV_32S);

    std::vector<std::vector<std::vector<cv::DMatch>>> nn_matches_all;

    std::vector<std::vector<cv::KeyPoint>> matched1_all, matched2_all, inliers1_all, inliers2_all;

    std::vector<std::vector<cv::DMatch>> good_matches_all;

    for (int i = 0; i < images.size(); i++)
    {
        for (int j = i + 1; j < images.size(); j++)
        {
            std::cout << "Matching images " << i << " and " << j << std::endl;

            std::vector<std::vector<cv::DMatch>> nn_matches;
            matcher.knnMatch(imagesDescriptors[i], imagesDescriptors[j], nn_matches, 2);
            nn_matches_all.push_back(nn_matches);
            nn_matches_table.at<int>(i, j) = nn_matches_all.size();
            nn_matches_table.at<int>(j, i) = nn_matches_all.size();

            std::vector<cv::KeyPoint> matched1, matched2, inliers1, inliers2;
            std::vector<cv::DMatch> good_matches;

            for (size_t k = 0; k < nn_matches.size(); k++)
            {
                cv::DMatch first = nn_matches[k][0];
                float dist1 = nn_matches[k][0].distance;
                float dist2 = nn_matches[k][1].distance;

                matched1.push_back(imagesKeypoints[i][first.queryIdx]);
                matched2.push_back(imagesKeypoints[j][first.trainIdx]);

                if (dist1 < 0.5f * dist2)
                {
                    int new_i = static_cast<int>(inliers1.size());
                    inliers1.push_back(matched1[k]);
                    inliers2.push_back(matched2[k]);
                    good_matches.push_back(cv::DMatch(new_i, new_i, 0));
                }
            }
            matched1_all.push_back(matched1);
            matched2_all.push_back(matched2);
            inliers1_all.push_back(inliers1);
            inliers2_all.push_back(inliers2);

            good_matches_all.push_back(good_matches);
        }
    }

    // draw matches for each pair of images
    for (int i = 0; i < nn_matches_table.cols; i++)
    {
        for (int j = i + 1; j < nn_matches_table.rows; j++)
        {
            if (nn_matches_table.at<int>(i, j) != 0)
            {
                std::cout << "Image " << i << " and " << j << " have " << good_matches_all[nn_matches_table.at<int>(i, j) - 1].size() << " good matches" << std::endl;

                // store number of matches in a matrix
                nn_matches_number.at<int>(i, j) = good_matches_all[nn_matches_table.at<int>(i, j) - 1].size();
                nn_matches_number.at<int>(j, i) = nn_matches_number.at<int>(i, j);

                cv::Mat res;

                cv::drawMatches(imagesWithFeatures[i], inliers1_all[nn_matches_table.at<int>(i, j) - 1], imagesWithFeatures[j], inliers2_all[nn_matches_table.at<int>(i, j) - 1], good_matches_all[nn_matches_table.at<int>(i, j) - 1], res);
                cv::imshow("Image", res);

                std::string res_path = folderPath + "/results/" + "matches_" + std::to_string(i) + "_" + std::to_string(j) + ".png";
                cv::imwrite(res_path, res);
                cv::waitKey(10);
            }
        }
    }

    // vector of parameters of transformation for each images
    std::vector<Eigen::Matrix<float, 3, 3>> translations(images.size()); // tx, ty
    std::vector<Eigen::Matrix<float, 3, 3>> similarities(images.size()); // tx, ty, a, b

    // we count nb of matches for each image to select the first image to add to the panorama
    // vector to track which images have been processed
    std::vector<int> processed_images(images.size(), 0);
    std::vector<int> processed_images_order; // for the priority add method
    int max = 0;
    int max_i = 0;
    for (int i = 0; i < nn_matches_number.cols; i++)
    {
        int nb = 0;

        for (int j = 0; j < nn_matches_number.rows; j++)
        {
            nb += nn_matches_number.at<int>(i, j);
        }
        if (nb > max)
        {
            max = nb;
            max_i = i;
        }
        std::cout << "Image " << i << " has " << nb << " matches" << std::endl;
    }
    std::cout << "Image " << max_i << " has the most matches - " << max << std::endl;
    processed_images[max_i] = 1;
    processed_images_order.push_back(max_i);

    // max_i is the central image
    translations[max_i] = Eigen::Matrix<float, 3, 3>::Identity();
    similarities[max_i] = Eigen::Matrix<float, 3, 3>::Identity();

    // we add one by one remaining images and compute translation and similarity parameters
    for (int i = 1; i < imagePaths.size(); i++)
    {
        int max = 0;
        int max_j = 0;
        int max_k = 0;

        // we select the image with the most matches with the already processed images
        for (int j = 0; j < processed_images.size(); j++)
        {
            if (processed_images[j] == 0)
            {
                for (int k = 0; k < processed_images.size(); k++)
                {
                    if (processed_images[k] == 1)
                    {
                        if (nn_matches_number.at<int>(j, k) > max)
                        {
                            max = nn_matches_number.at<int>(j, k);
                            max_j = j;
                            max_k = k;
                        }
                    }
                }
            }
        }
        std::cout << "Image " << max_j << " has the most matches with " << max_k << " with nb_matches = " << max << std::endl;

        int nb_match = nn_matches_number.at<int>(max_j, max_k);

        // Translation computations
        Eigen::Matrix2f A_translation = Eigen::Matrix2f::Zero();
        Eigen::Matrix<float, 2, 1> b_translation = Eigen::Matrix<float, 2, 1>::Zero();
        Eigen::Matrix<float, 2, 1> translation_parameters = Eigen::Matrix<float, 2, 1>::Zero();
        for (int l = 0; l < nb_match; l++)
        {
            float x1, y1, x2, y2;

            // we select the right coordinates depending on the order of the images since we matched image int the increasing order of their index
            if (max_k < max_j)
            {
                x1 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y1 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
                x2 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y2 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
            }
            else
            {
                x1 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y1 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
                x2 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y2 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
            }
            A_translation(0, 0) += 1;
            A_translation(1, 1) += 1;

            b_translation(0) += x2 - x1;
            b_translation(1) += y2 - y1;
        }

        translation_parameters = A_translation.inverse() * b_translation; // we need to add the translations of the previous images
        translations[max_j] = Eigen::Matrix<float, 3, 3>::Identity();
        translations[max_j](0, 2) = translation_parameters(0);
        translations[max_j](1, 2) = translation_parameters(1);
        translations[max_j] = translations[max_j] * translations[max_k];

        // Similarity computations
        Eigen::Matrix<float, 4, 4> A_similarity = Eigen::Matrix<float, 4, 4>::Zero();
        Eigen::Matrix<float, 4, 1> b_similarity = Eigen::Matrix<float, 4, 1>::Zero();
        Eigen::Matrix<float, 4, 1> similarity_parameters = Eigen::Matrix<float, 4, 1>::Zero();

        for (int l = 0; l < nb_match; l++)
        {
            float x1, y1, x2, y2;

            // we select the right coordinates depending on the order of the images since we matched image int the increasing order of their index
            if (max_k < max_j)
            {
                x1 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y1 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
                x2 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y2 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
            }
            else
            {
                x1 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y1 = inliers2_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
                x2 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.x;
                y2 = inliers1_all[nn_matches_table.at<int>(max_j, max_k) - 1][l].pt.y;
            }

            // top left
            A_similarity(0, 0) += 1;
            A_similarity(1, 1) += 1;
            // bottom left
            A_similarity(2, 0) += x1;
            A_similarity(2, 1) += y1;
            A_similarity(3, 0) += -y1;
            A_similarity(3, 1) += x1;
            // top right
            A_similarity(0, 2) += x1;
            A_similarity(0, 3) += -y1;
            A_similarity(1, 2) += y1;
            A_similarity(1, 3) += x1;
            // bottom right
            A_similarity(2, 2) += x1 * x1 + y1 * y1;
            A_similarity(3, 3) += x1 * x1 + y1 * y1;

            b_similarity(0) += x2 - x1;
            b_similarity(1) += y2 - y1;
            b_similarity(2) += x1 * (x2 - x1) - y1 * (y2 - y1);
            b_similarity(3) += y1 * (x2 - x1) + x1 * (y2 - y1);
        }
        std::cout << "A_similarity" << std::endl;
        for (int m = 0; m < 4; m++)
        {
            for (int n = 0; n < 4; n++)
            {
                std::cout << A_similarity(m, n) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "b_similarity" << std::endl;
        for (int m = 0; m < 4; m++)
        {
            std::cout << b_similarity(m) << " ";
        }
        std::cout << std::endl;

        processed_images[max_j] = 1;
        processed_images_order.push_back(max_j);

        similarity_parameters = A_similarity.inverse() * b_similarity;
        similarities[max_j] = Eigen::Matrix<float, 3, 3>::Identity();
        similarities[max_j](0, 2) = similarity_parameters(0);
        similarities[max_j](1, 2) = similarity_parameters(1);
        similarities[max_j](0, 0) = 1 + similarity_parameters(2);
        similarities[max_j](1, 1) = 1 + similarity_parameters(2);
        similarities[max_j](0, 1) = -similarity_parameters(3);
        similarities[max_j](1, 0) = similarity_parameters(3);

        similarities[max_j] = similarities[max_j] * similarities[max_k]; // we need to add the similarities of the previous images
    }

    std::cout << "translations" << std::endl;
    for (int i = 0; i < translations.size(); i++)
    {
        std::cout << "image " << i << std::endl;
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                std::cout << translations[i](j, k) << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "similarities" << std::endl;
    for (int i = 0; i < similarities.size(); i++)
    {
        std::cout << "image " << i << std::endl;
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                std::cout << similarities[i](j, k) << " ";
            }
            std::cout << std::endl;
        }
    }

    int min_x_translation = 0;
    int min_y_translation = 0;
    int max_x_translation = 0;
    int max_y_translation = 0;
    for (int i = 0; i < images.size(); i++)
    {
        int length = images[i].cols;
        int height = images[i].rows;
        Eigen::Matrix<float, 3, 4> corners;
        corners << 0, length, length, 0,
            0, 0, height, height,
            1, 1, 1, 1;
        Eigen::Matrix<float, 3, 4> warped_corners = translations[i].inverse() * corners;
        int warped_x_min = warped_corners.row(0).minCoeff();
        int warped_x_max = warped_corners.row(0).maxCoeff();
        int warped_y_min = warped_corners.row(1).minCoeff();
        int warped_y_max = warped_corners.row(1).maxCoeff();
        if (warped_x_min < min_x_translation)
        {
            min_x_translation = warped_x_min;
        }
        if (warped_y_min < min_y_translation)
        {
            min_y_translation = warped_y_min;
        }
        if (warped_x_max > max_x_translation)
        {
            max_x_translation = warped_x_max;
        }
        if (warped_y_max > max_y_translation)
        {
            max_y_translation = warped_y_max;
        }
        std::cout << "image " << i << std::endl;
        std::cout << "warped_x_min " << warped_x_min << std::endl;
        std::cout << "warped_x_max " << warped_x_max << std::endl;
        std::cout << "warped_y_min " << warped_y_min << std::endl;
        std::cout << "warped_y_max " << warped_y_max << std::endl;
    }
    std::cout << "min_x_translation " << min_x_translation << std::endl;
    std::cout << "min_y_translation " << min_y_translation << std::endl;
    std::cout << "max_x_translation " << max_x_translation << std::endl;
    std::cout << "max_y_translation " << max_y_translation << std::endl;

    // we need to add the translations of the previous images
    for (int i = 0; i < translations.size(); i++)
    {
        translations[i](0, 2) += min_x_translation;
        translations[i](1, 2) += min_y_translation;
    }

    int min_x_similarity = 0;
    int min_y_similarity = 0;
    int max_x_similarity = 0;
    int max_y_similarity = 0;
    for (int i = 0; i < images.size(); i++)
    {
        int length = images[i].cols;
        int height = images[i].rows;
        Eigen::Matrix<float, 3, 4> corners;
        corners << 0, length, length, 0,
            0, 0, height, height,
            1, 1, 1, 1;
        Eigen::Matrix<float, 3, 4> warped_corners = similarities[i].inverse() * corners;
        int warped_x_min = warped_corners.row(0).minCoeff();
        int warped_x_max = warped_corners.row(0).maxCoeff();
        int warped_y_min = warped_corners.row(1).minCoeff();
        int warped_y_max = warped_corners.row(1).maxCoeff();
        if (warped_x_min < min_x_similarity)
        {
            min_x_similarity = warped_x_min;
        }
        if (warped_y_min < min_y_similarity)
        {
            min_y_similarity = warped_y_min;
        }
        if (warped_x_max > max_x_similarity)
        {
            max_x_similarity = warped_x_max;
        }
        if (warped_y_max > max_y_similarity)
        {
            max_y_similarity = warped_y_max;
        }
        std::cout << "image " << i << std::endl;
        std::cout << "warped_x_min " << warped_x_min << std::endl;
        std::cout << "warped_x_max " << warped_x_max << std::endl;
        std::cout << "warped_y_min " << warped_y_min << std::endl;
        std::cout << "warped_y_max " << warped_y_max << std::endl;
    }
    std::cout << "min_x_similarity " << min_x_similarity << std::endl;
    std::cout << "min_y_similarity " << min_y_similarity << std::endl;
    std::cout << "max_x_similarity " << max_x_similarity << std::endl;
    std::cout << "max_y_similarity " << max_y_similarity << std::endl;

    // Add the minimum translation to the similarity matrices
    for (int i = 0; i < similarities.size(); i++)
    {
        similarities[i](0, 2) += min_x_similarity;
        similarities[i](1, 2) += min_y_similarity;
    }

    std::vector<cv::Mat> warped_images_translation(images.size());
    std::vector<cv::Mat> warped_images_similarity(images.size());
    for (int i = 0; i < translations.size(); i++)
    {

        int length = images[i].cols;
        int height = images[i].rows;

        // Convert into opencv matrices

        cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                translation.at<double>(j, k) = translations[i](j, k);
            }
        }
        cv::Mat similarity = cv::Mat::eye(3, 3, CV_64F);
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                similarity.at<double>(j, k) = similarities[i](j, k);
            }
        }

        cv::warpPerspective(images[i], warped_images_translation[i], translation, cv::Size(max_x_translation - min_x_translation, max_y_translation - min_y_translation), cv::WARP_INVERSE_MAP);
        cv::warpPerspective(images[i], warped_images_similarity[i], similarity, cv::Size(max_x_similarity - min_x_similarity, max_y_similarity - min_y_similarity), cv::WARP_INVERSE_MAP);

        cv::imwrite(folderPath + "/results/" + "warped_translation_" + std::to_string(i) + ".png", warped_images_translation[i]);
        cv::imwrite(folderPath + "/results/" + "warped_similarity_" + std::to_string(i) + ".png", warped_images_similarity[i]);

        cv::imshow("Image", images[i]);
        cv::imshow("Warped translation", warped_images_translation[i]);
        cv::imshow("Warped similarity", warped_images_similarity[i]);
        cv::waitKey(10);
    }

    cv::Mat fused_image_translation_mean = cv::Mat(max_y_translation - min_y_translation, max_x_translation - min_x_translation, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat fused_image_translation_priority = warped_images_translation[processed_images_order[0]];

    for (int i = 0; i < warped_images_translation.size(); i++)
    {
        fused_image_translation_mean += warped_images_translation[i] / warped_images_translation.size();
    }
    for (int i = 1; i < processed_images_order.size(); i++)
    {
        cv::Mat mask;
        cv::compare(fused_image_translation_priority, cv::Scalar(0, 0, 0), mask, cv::CMP_EQ);
        warped_images_translation[processed_images_order[i]].copyTo(fused_image_translation_priority, mask);
    }

    cv::imwrite(folderPath + "/results/" + "fused_image_translation_mean.png", fused_image_translation_mean);
    cv::imshow("Image", fused_image_translation_mean);
    cv::imwrite(folderPath + "/results/" + "fused_image_translation_priority.png", fused_image_translation_priority);
    cv::imshow("Image", fused_image_translation_priority);
    cv::waitKey(10);

    // USE OPENCV STITCHER
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    cv::Mat stitchedImage;
    cv::Stitcher::Status status = stitcher->stitch(images, stitchedImage);
    if (status != cv::Stitcher::OK)
    {
        std::cout << "Can't stitch images, error code = " << int(status) << std::endl;
        return EXIT_FAILURE;
    }
    cv::imwrite(folderPath + "/results/" + "stitchedImage.png", stitchedImage);
    cv::imshow("Image", stitchedImage);
    cv::waitKey(10);
}