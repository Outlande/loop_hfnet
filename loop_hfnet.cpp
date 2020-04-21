#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <hfnet_msgs/Hfnet.h>
#include <geometry_msgs/Point32.h>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv/cv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fundmental_matrix.h>

#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <queue>

int getFileRows(std::string fileName){
	std::ifstream fileStream;

	std::string tmp;
	int count = 0;// 行数计数器

	fileStream.open(fileName);//ios::in 表示以只读的方式读取文件

	if(fileStream.fail())//文件打开失败:返回0
		return 0;

	else//文件存在
	{
		while(getline(fileStream,tmp,'\n'))//读取一行
		{
			if (tmp.size() > 0 )
				count++;
		}

		fileStream.close();

		return count;
	}
}

class Ros_solver
{
public:
    Ros_solver();

    void Start();

private:
    struct RGBD_hfnet 
    {
		std_msgs::Header header;
		cv::Mat color;
		cv::Mat depth;
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat local_desc;
		cv::Mat global_desc;

		Eigen::Quaterniond quaternion;
		Eigen::Vector3d translation;
    };

    struct glb_dis_index
    {
	    float distance;
	    RGBD_hfnet query;
    };

    //queue for save data
    std::queue<RGBD_hfnet> image_queue;

    std::vector<RGBD_hfnet> frame_queue;

    //load data function 
    void img_load(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD);
    void kpt_load(const hfnet_msgs::Hfnet::ConstPtr& msg);

    void trans_calc(const std::vector< std::vector<cv::DMatch> > &match_results, 
                    const RGBD_hfnet &image_work, 
                    const std::vector<RGBD_hfnet> &match_frames);

    // topic name 
    std::string image_topic;
    std::string depth_topic;
    std::string kpt_topic;
    std::string query_topic;

    //subscrible and publisher
    message_filters::Subscriber<sensor_msgs::Image> image_msg, depth_msg;
    ros::Subscriber hfnet_msg;

    ros::Publisher loop_result;

    //color and depth Aligned use message_filters
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_policy;
    message_filters::Synchronizer<sync_policy> sum_query_img;

    //compare distance
    static bool compare_glb(glb_dis_index a, glb_dis_index b);
    
};


bool Ros_solver::compare_glb(glb_dis_index a, glb_dis_index b)
{
	return a.distance < b.distance; //升序排列
}


Ros_solver::Ros_solver() : sum_query_img(sync_policy(10))
{
	ros::NodeHandle input;

	// image_topic = "/cam0/image_raw";
	image_topic = "/d400/color/image_raw";
	depth_topic = "/d400/depth/image_raw";
	kpt_topic = "/features";

	//chatter_pub = input.advertise<sensor_msgs::Imu>("chatter", 1000);

	image_msg.subscribe(input, image_topic, 1000);
	depth_msg.subscribe(input, depth_topic, 1000);
	sum_query_img.connectInput(image_msg, depth_msg);
	sum_query_img.registerCallback(boost::bind(&Ros_solver::img_load, this, _1, _2));
	hfnet_msg = input.subscribe(kpt_topic, 1000, &Ros_solver::kpt_load, this);

}


void Ros_solver::img_load(const sensor_msgs::ImageConstPtr& msgRGB, const sensor_msgs::ImageConstPtr& msgD)
{
	cv_bridge::CvImageConstPtr cv_ptrRGB;
	try
	{
		cv_ptrRGB = cv_bridge::toCvCopy(msgRGB,sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	cv_bridge::CvImageConstPtr cv_ptrD;
	try
	{
		cv_ptrD = cv_bridge::toCvCopy(msgD, sensor_msgs::image_encodings::TYPE_16UC1);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	RGBD_hfnet data;
	data.header = msgRGB->header;
	data.color = cv_ptrRGB->image;
	data.depth = cv_ptrD->image;
	image_queue.push(data);
}



//load detector and descripitor
void Ros_solver::kpt_load(const hfnet_msgs::Hfnet::ConstPtr& msg)
{
	// find the image with exact match of stamp
	RGBD_hfnet image;
	bool found = false;
	int num = 0;
  	while ((image_queue.size() < 2)||(num > 500))
  	{
    	usleep(200);
    	num++;
  	}
  	while (!image_queue.empty())
  	{
	    image = image_queue.front();
	    image_queue.pop();
	    if (image.header.stamp == msg->header.stamp)
	    {
        	found = true;
        	break;
	    }
    else if(image.header.stamp < msg->header.stamp)
  		ROS_WARN("There is not features for image at %lf", msg->header.stamp.toSec());
    else
  		break;
  	}
	if(!found)
	{
		ROS_WARN("Could not find matched image for feature at %lf", msg->header.stamp.toSec());
		return;
	}

	// Copy the ros image message to cv::Mat.
	for (auto p: msg->local_points)
	{
		cv::KeyPoint kp;
		kp.pt.x = p.x;
		kp.pt.y = p.y;
		kp.octave = 0;
		image.keypoints.push_back(kp);
	}
	image.local_desc.create(image.keypoints.size(), 256, CV_32F);
	int n_rows = image.local_desc.rows;
	int n_cols = image.local_desc.cols;
	for(int j = 0; j<n_rows; j++)
	{
		uchar* data = image.local_desc.ptr<uchar>(j);
		auto data_ = msg->local_desc[j];
		for (int i = 0; i<n_cols; i++)
			data[i] = data_.data[i];
	}
	image.global_desc = cv::Mat(msg->global_desc.data, CV_32F);

	// invoke the worker
	frame_queue.push_back(image);
    
}

void Ros_solver::trans_calc(const std::vector< std::vector<cv::DMatch> > &match_results, 
                            const RGBD_hfnet &image_work, 
                            const std::vector<RGBD_hfnet> &match_frames)
{
	if (match_results.size() != match_frames.size())
	{
		std::cout<<"size of matches is not the same"<<std::endl;
		return;
	}
	std::vector< std::vector<bool> > Inliers_points;
	std::vector<float> scores;
	std::vector<cv::Mat> fund_matrixs;

	std::vector<cv::Mat> R_mats;
	std::vector<cv::Mat> T_mats;
	std::vector<bool> relocal_result;
	
	for(int idx=0; idx<match_results.size(); idx++)
	{	
		std::vector<bool> Inliers_point;
		float score;
		cv::Mat fund_matrix;

		fund_mat fundmental_calc(image_work.keypoints, match_frames[idx].keypoints, match_results[idx], 1, 200);

		fundmental_calc.FindFundamental(Inliers_point, score, fund_matrix);

		Inliers_points.push_back(Inliers_point);
		scores.push_back(score);
		fund_matrixs.push_back(fund_matrix);

		cv::Mat R_mat;
		cv::Mat T_mat;

		relocal_result.push_back(fundmental_calc.calc_rt(Inliers_point, fund_matrix, R_mat, T_mat, 1.0, 50));

		R_mats.push_back(R_mat);
		T_mats.push_back(T_mat);
	}

	for(int idx=0; idx<scores.size(); idx++)
	{	
		if (relocal_result[idx])
		{
			cv::Mat img_ori = image_work.color;
			cv::Mat img_que = match_frames[idx].color;

			cv::Mat image_combine(img_ori.rows,img_ori.cols+img_que.cols+1, img_ori.type());

			img_ori.colRange(0, img_ori.cols).copyTo(image_combine.colRange(0, img_ori.cols));
			img_que.colRange(0, img_que.cols).copyTo(image_combine.colRange(img_ori.cols+1, image_combine.cols));

			int inlier_num=0;

			for(int j=0; j<Inliers_points[idx].size(); j++)
			{
				if (Inliers_points[idx][j])
				{
					cv::Point ori_pt(image_work.keypoints[j].pt.x, image_work.keypoints[j].pt.y);
					cv::Point que_pt(img_ori.cols+1+match_frames[idx].keypoints[j].pt.x, match_frames[idx].keypoints[j].pt.y);

					cv::circle(image_combine, ori_pt, 1, cv::Scalar(0, 0, 255));
					cv::circle(image_combine, que_pt, 1, cv::Scalar(0, 0, 255));
					
					cv::line(image_combine, ori_pt, que_pt, cv::Scalar(255, 0, 0), 1);
					inlier_num++;
				}
			}

			//匹配图位姿欧拉矩阵
		    Eigen::Isometry3d match_matrix = Eigen::Isometry3d::Identity();
		    match_matrix.rotate(match_frames[idx].quaternion.toRotationMatrix());
		    match_matrix.pretranslate(match_frames[idx].translation);
			
			//转换欧拉矩阵
			Eigen::Matrix3d R_eigen;
			Eigen::Vector3d T_eigen;

			cv::cv2eigen(R_mats[idx], R_eigen);
			cv::cv2eigen(T_mats[idx], T_eigen);
			
		    Eigen::Isometry3d trans_matrix = Eigen::Isometry3d::Identity();
		    trans_matrix.rotate(R_eigen);
		    trans_matrix.pretranslate(T_eigen);
		    
		    //位姿结果
		    Eigen::Isometry3d image_matrix = match_matrix*trans_matrix.inverse();

		    Eigen::Quaterniond res_R(image_matrix.rotation());
		    res_R.normalize();
		    Eigen::Vector3d res_T=image_matrix.translation();

			std::cout<<"idx "<<idx<<std::endl;
			std::cout<<"time_stamp "<<match_frames[idx].header.stamp<<std::endl;
			std::cout<<"fund_mat: "<<std::endl<<fund_matrixs[idx]<<std::endl;
			std::cout<<"inlier_num "<<inlier_num<<std::endl;
			std::cout<<"score "<<scores[idx]<<std::endl;
			std::cout<<"mean score "<<scores[idx]/inlier_num<<std::endl;
			std::cout<<"R_mat "<<std::endl<<res_R.coeffs().transpose()<<std::endl;
			std::cout<<"T_mat "<<std::endl<<res_T.transpose()<<std::endl;
			//std::cout<<"relocal result "<<relocal_result[idx]<<std::endl;
			//1560025113.03674531 
			//23.555883407592773 21.034465789794922 0.0 
			//0.0 0.0 -0.14400844017438966 0.9895764594807919

			std::cout<<std::endl;
			
			//cv::imshow("results", image_combine);
			//cv::waitKey(0);
		}
		
		
	}

}


void Ros_solver::Start()
{
	while (ros::ok())
	{
		std::cout<<"start"<<std::endl;


	    /*导入图片序列*/
	    //========================================================================================================
	    //========================================================================================================

	    std::vector<cv::String> img_names, des_names,kpt_names,glb_names; // notice here that we are using the Opencv's embedded "String" class
	    cv::String img_folder = "/media/outsider/8f6e8ca1-387d-4c79-9354-fbbcf6ea8d9a/openloris-scene-png-version/OLORIS_Cafe/cafe1/color/"; // again we are using the Opencv's embedded "String" class
	    cv::String des_folder = "/media/outsider/8f6e8ca1-387d-4c79-9354-fbbcf6ea8d9a/intel/cafe-1/des/*.txt"; // again we are using the Opencv's embedded "String" class
	    cv::String kpt_folder = "/media/outsider/8f6e8ca1-387d-4c79-9354-fbbcf6ea8d9a/intel/cafe-1/kpt/*.txt";
	    cv::String glb_folder = "/media/outsider/8f6e8ca1-387d-4c79-9354-fbbcf6ea8d9a/intel/cafe-1/glb/*.txt";

	    cv::glob(img_folder, img_names);
	    cv::glob(des_folder, des_names); 
	    cv::glob(kpt_folder, kpt_names); 
	    cv::glob(glb_folder, glb_names);

	    if ((img_names.size() == des_names.size())&&(des_names.size() == kpt_names.size())&&(kpt_names.size() == glb_names.size()))
	    	std::cout<<"query_size is right"<<std::endl;
	    else
	    {
			std::cout<<"query_size is wrong"<<std::endl;
			return;
	    }
	      

	    //for (int i=0; i<img_names.size(); i++)
	    for (int i=0; i<200; i++)
	    {
			RGBD_hfnet load_queue;
			load_queue.color  = cv::imread(img_names.at(i));

			load_queue.header.stamp.sec = std::stoi(img_names.at(i).substr(105,10));
			load_queue.header.stamp.nsec = std::stoi(img_names.at(i).substr(116,6))*1e3;

			//load results for hfnet
			int row;
			row = getFileRows(des_names.at(i));

			cv::Mat des = cv::Mat::zeros(row, 256, CV_32F);
			cv::Mat glb = cv::Mat::zeros(4096, 1, CV_32F);

			load_queue.keypoints.reserve(row);

			std::ifstream fileStream;
			fileStream.open(des_names.at(i));
			std::string a;
	    	for (int i=0;i<row;i++)
	    	{
	        	for (int j=0;j<256;j++)
	        	{
					fileStream >> a;
					des.at<float>(i,j) = std::stof(a); 
	        	}
	      	}
	      	fileStream.close();

			fileStream.open(kpt_names.at(i));

			for (int i=0;i<row;i++)
			{
				cv::KeyPoint kp;
				fileStream >> a;
				kp.pt.x = std::stof(a);
				fileStream >> a;
				kp.pt.y = std::stof(a);
				kp.octave = 0;
				load_queue.keypoints.push_back(kp);
			}
	      
			fileStream.close();

			fileStream.open(glb_names.at(i));

    		for (int i=0;i<4096;i++)
    		{
        		fileStream >> a;
        		glb.at<float>(0,i) = std::stof(a);
      		}

      		fileStream.close();

			load_queue.local_desc = des;
			load_queue.global_desc = glb;
			std::cout<<"load image "<<i<<std::endl;
			frame_queue.push_back(load_queue);
		}
		//========================================================================================================
		//========================================================================================================

		/*导入groundtruth*/
		//暂时采用最近的值作为图像的gt
		//========================================================================================================
		//========================================================================================================
		{
			std::string gt_file = "/media/outsider/8f6e8ca1-387d-4c79-9354-fbbcf6ea8d9a/intel/groundtruth/per-sequence/cafe1-1/groundtruth.txt";

			std::ifstream fileStream;
			fileStream.open(gt_file);
			std::string a;
			std_msgs::Header time_load;
			int i=0;
			
			while(i<frame_queue.size())
			{
				fileStream >> a;
				time_load.stamp.sec = std::stoi(a.substr(0,10));
				time_load.stamp.nsec = std::stoi(a.substr(11,10))*10;

				if ((frame_queue.at(i).header.stamp.sec<time_load.stamp.sec)||
				  ((frame_queue.at(i).header.stamp.sec==time_load.stamp.sec)&&(frame_queue.at(i).header.stamp.nsec<time_load.stamp.nsec)))
				{
					fileStream >> a;
					frame_queue.at(i).translation[0]=std::stof(a);
					fileStream >> a;
					frame_queue.at(i).translation[1]=std::stof(a);
					fileStream >> a;
					frame_queue.at(i).translation[2]=std::stof(a);

					fileStream >> a;
					frame_queue.at(i).quaternion.x()=std::stof(a);
					fileStream >> a;
					frame_queue.at(i).quaternion.y()=std::stof(a);
					fileStream >> a;
					frame_queue.at(i).quaternion.z()=std::stof(a);
					fileStream >> a;
					frame_queue.at(i).quaternion.w()=std::stof(a);

					i++;
		    	}
		    	else
				{
					for (int j=0;j<7;j++)
						fileStream >> a;
				}
		  }
		}

	    //========================================================================================================
	    //========================================================================================================

	    /*导入索引图片*/
	    //========================================================================================================
	    //========================================================================================================

		std::string img_root = "/media/outsider/8f6e8ca1-387d-4c79-9354-fbbcf6ea8d9a/openloris-scene-png-version/OLORIS_Cafe/cafe2/color/";
		std::string kpt_root = "/media/outsider/8f6e8ca1-387d-4c79-9354-fbbcf6ea8d9a/intel/cafe-2/";
		std::string time = "1560025113.032917";

		RGBD_hfnet image_work;
		image_work.color  = cv::imread(img_root+time+".png");

		image_work.header.stamp.sec = std::stoi(time.substr(0,10));
		image_work.header.stamp.nsec = std::stoi(time.substr(11,6))*1e3;


		//load results for hfnet
		int row;
		row = getFileRows(kpt_root+"des/"+time+".txt");

		cv::Mat des = cv::Mat::zeros(row, 256, CV_32F);
		cv::Mat glb = cv::Mat::zeros(4096, 1, CV_32F);
		image_work.keypoints.reserve(row);

		std::ifstream fileStream;
		fileStream.open(kpt_root+"des/"+time+".txt");
		std::string a;
		for (int i=0;i<row;i++)
		{
			for (int j=0;j<256;j++)
			{
				fileStream >> a;
				des.at<float>(i,j) = std::stof(a); 
			}
		}

		fileStream.close();

		fileStream.open(kpt_root+"kpt/"+time+".txt");
		
		for (int i=0;i<row;i++)
		{
			cv::KeyPoint kp;
			fileStream >> a;			
			kp.pt.x = std::stof(a);
			fileStream >> a;
			kp.pt.y = std::stof(a);
			kp.octave = 0;
			image_work.keypoints.push_back(kp);
				
		}

		fileStream.close();

		fileStream.open(kpt_root+"glb/"+time+".txt");

		for (int i=0;i<4096;i++)
		{
			fileStream >> a;
			glb.at<float>(0,i) = std::stof(a);
		}
		image_work.local_desc = des;
		image_work.global_desc = glb;

		fileStream.close();
    	
	
    	//========================================================================================================
    	//========================================================================================================

		std::vector<glb_dis_index> glb_distance;

		cv::Mat distance = cv::Mat::zeros(1, 1, CV_32F);

		for (auto p:frame_queue)
		{
			//calculate global descriptor distance
			glb_dis_index now_query;
			cv::reduce((image_work.global_desc-p.global_desc).mul((image_work.global_desc-p.global_desc)), distance, 0, CV_REDUCE_SUM);
			now_query.distance = distance.at<float>(0,0);
			now_query.query = p;
			glb_distance.push_back(now_query);
		}
		//sort by global distance
		std::sort(glb_distance.begin(), glb_distance.end(), compare_glb);
		//select top 20 image
		while(glb_distance.size()>15)
			glb_distance.pop_back();
		
		// for code in hfnet use norml2 and knnmatch 2 to match 
		cv::BFMatcher match(cv::NORM_L2);

		std::vector< std::vector<cv::DMatch> > match_results;

		std::vector<RGBD_hfnet> match_frames;

		std::vector< std::vector<cv::DMatch> >::iterator ite; 

		float thresh = 0.90;

		for (auto p:glb_distance)
		{
			std::vector< std::vector<cv::DMatch> > knnMatches;
			match.knnMatch(image_work.local_desc, p.query.local_desc, knnMatches, 2);
			std::vector<cv::DMatch> match_result;

			for(ite=knnMatches.begin(); ite!=knnMatches.end(); ite++)
			{
				//hfnet 源码中的方法，thresh 为0.90
				if (((*ite).front().distance/(*ite).back().distance) < thresh)
				  //匹配上的结果
				  match_result.push_back((*ite).front());
			}

			match_results.push_back(match_result);
			match_frames.push_back(p.query);
		}

		trans_calc(match_results, image_work, match_frames);

		frame_queue.clear();
  	}
}


int main(int argc, char **argv)
{
	// ros start
	ros::init(argc, argv, "loop_closure");

	Ros_solver ros_solver;
	ros::AsyncSpinner spinner(0);
	spinner.start();

	ros_solver.Start();

	spinner.stop();
	std::cout<<"finished"<<std::endl;

	return 0;
}