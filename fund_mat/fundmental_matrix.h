#include <iostream>

#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <utility>

class fund_mat
{
public:
    fund_mat(const std::vector<cv::KeyPoint> &img_kpt, 
    	     const std::vector<cv::KeyPoint> &query_kpt, 
    	     const std::vector< cv::DMatch > &match_result,
    	     float sigma, int iterations);

    void FindFundamental(std::vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21);

    bool calc_rt(std::vector<bool> &vbMatchesInliers, cv::Mat &F21,
                 cv::Mat &R21, cv::Mat &t21,float minParallax, int minTriangulated);

private:

    void Normalize(const std::vector<cv::KeyPoint> &vKeys, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1,const std::vector<cv::Point2f> &vP2);

    bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

    int CheckRT(const cv::Mat &R, const cv::Mat &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2,
                          const std::vector<cv::DMatch> &vMatches12, std::vector<bool> &vbMatchesInliers,
                          const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);

    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);



    // Keypoints from Current Frame (Frame 1)
    std::vector<cv::KeyPoint> mvKeys1;

    // Keypoints from Reference Frame (Frame 2)
    std::vector<cv::KeyPoint> mvKeys2;

    // Current Matches from Current to Reference
    // queryidx is current
    // trainidx is reference
    std::vector< cv::DMatch > mvMatches12;

    // Ransac max iterations
    int mMaxIterations;

    // Calibration
    cv::Mat camera_calibration = (cv::Mat_<float>(3, 3) <<611.4509887695312, 0.0, 433.2039794921875, 0.0, 611.4857177734375, 249.4730224609375, 0.0, 0.0, 1.0);

    // Standard Deviation and Variance
    float mSigma, mSigma2;

   
    // Ransac sets
    std::vector< std::vector<size_t> > mvSets;   
    
 };