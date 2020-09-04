#pragma once
#pragma warning(disable : 4996)

#pragma   push_macro("min")  
#pragma   push_macro("max")  
#undef   min  
#undef   max
#include <pcl/io/pcd_io.h>  //pcl的pcd格式文件的输入输出头文件
#include <pcl/point_cloud.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#pragma   pop_macro("min")  
#pragma   pop_macro("max")

#include <opencv2/opencv.hpp>

#include <iostream>


using PointT = pcl::PointXYZI;

struct CalibCfg {
	std::string calibdataPath, calibdataFilename, pcdFilePath, mode, estimateMethod;
	bool coordinateDifference;

	cv::Mat capture_pos, grab_pos, robot_pos, camera_pos;
};

struct CalibData {
	bool coordinateDifference;
	cv::Mat transMatrix, translate;
};

class Calibration {
public:
	CalibCfg cfg;
	void readCfg(const std::string& file_path);
	void printCfg();
	CalibData calibrate();

private:


};

struct SphereParam {
	// PassThrough
	float PassThroughXMin = -670.0;
	float PassThroughXMax = 480.0;
	float PassThroughYMin = -550.0;
	float PassThroughYMax = 560.0;
	float PassThroughZMin = 600.0;
	float PassThroughZMax = 1500.0;
	float PassThroughIMin = 0;
	float PassThroughIMax = 91500.0;
	float BGLeafSize = 1.0;
	// StatisticalOutlierRemoval
	int SORMeanK = 50;
	float SORStddevMulThresh = 0.8;
	// Plane segmentation
	int PLANormalKSearch = 50;
	float PLANormalDistanceWeight = 0.1;
	int PLAMaxIterations = 100;
	float PLADistanceThreshold = 30;
	// Sphere segmentation
	int SPHNormalKSearch = 50;
	float SPHNormalDistanceWeight = 0.1;
	int SPHMaxIterations = 100;
	float SPHDistanceThreshold = 20;
	float SPHMinRadius = 100;
	float SPHMaxRadius = 300;
};

class SphereDetector {
public:
	SphereDetector();
	~SphereDetector() {};
public:

	void readParam(const std::string& file_path);
	void printParam();
	void setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

	pcl::PointCloud<PointT>::Ptr getPassThroughCloud();
	pcl::PointCloud<PointT>::Ptr getFilteredCloud();
	pcl::PointCloud<PointT>::Ptr getPlaneCloud();
	pcl::PointCloud<PointT>::Ptr getNoplaneCloud();
	pcl::PointCloud<PointT>::Ptr getInlierCloud();
	Eigen::VectorXf getModelCoefficients();
private:

	SphereParam sp;

	Eigen::VectorXf coefficients;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_pass,
		cloud_filtered, cloud_subplane, cloud_noplane, cloud_s,
		cloud_background_;

	bool PassThroughDone;

};


class Visualizer {
public:
	Visualizer(const std::string& win_name);
	Visualizer(const std::string& win_name,
		const int& rows, const int& cols);
	~Visualizer() {};
public:
	//template <typename PointT>
	//pcl::PointCloud<PointT>::Ptr cloud;
	//template <typename PointCloudT>
	void setCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int viewport);
	void setSphere(Eigen::VectorXf coef,
		float r,float g,float b);
private:
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1, cloud2,
		cloud3, cloud4, cloud5, cloud6;
	std::map<int, pcl::PointCloud<pcl::PointXYZI>::Ptr> clouds;
	bool setSphereFlag;
	std::vector<float> sphereCoef;

};



pcl::PointCloud<pcl::PointXYZI>::Ptr  createSphere(
	std::vector<float> origin, float radius, int point_num);



