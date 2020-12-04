// PCL_calibrationSphere.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#pragma warning(disable : 4996)

#include "sphereDetection.h"
#pragma   push_macro("min")  
#pragma   push_macro("max")  
#undef   min  
#undef   max
#include <pcl/console/parse.h>
#pragma   pop_macro("min")  
#pragma   pop_macro("max")

#include <future>
#include <iostream>
//#include <cmath>
//#include <algorithm>


  // 生成试验数据：12张标准球点云图
void generate12() {
  std::vector <std::vector<float> > origins = {
    {0,0,1200},
    {100,0,1150},
    {100,-100,1100},
    {0,-100,1050},
    {-100,-100,1000},
    {-100,0,950},
    {-100,100,900},
    {0,100,850},
    {100,100,800},
    {100,0,750},
    {100,-100,700},
    {0,-100,650},
  };
  for (int i = 0; i < 12; i++) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = createSphere(origins[i], 100, 10000);
    std::string save_name = "0/" + std::to_string(i) + ".pcd";
    pcl::io::savePCDFile(save_name, *cloud);
  }
}

int main(int argc, char** argv)
{
  //generate12();
	//pcl::PointCloud<PointT>::Ptr cloud1 = createSphere({ 0,0,0 }, 100, 10000);
	//pcl::io::savePCDFile("1.pcd", *cloud1);
	//pcl::PointCloud<PointT>::Ptr cloud2 = createSphere({ 100,100,100 }, 100, 10000);
	//pcl::io::savePCDFile("2.pcd", *cloud2);
	//pcl::PointCloud<PointT>::Ptr cloud3 = createSphere({ -100,-100,-100 }, 100, 10000);
	//pcl::io::savePCDFile("3.pcd", *cloud3);

		// pcl::console::parse
	std::string config_path;
	bool path_specified = pcl::console::find_switch(argc, argv, "-config_path");
	if (path_specified)
		pcl::console::parse(argc, argv, "-config_path", config_path);
	//std::cout << "config_path : " << config_path << std::endl;
  // 读取配置文件
  Calibration clb;
  clb.readCfg(config_path);
  clb.printCfg();

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PCDReader reader;
  reader.read<pcl::PointXYZI>(clb.cfg.pcdFilePath+"0.pcd", *cloud);//3层3个.

  SphereDetector sdet;
  sdet.readParam("sphere_param.json");
  sdet.printParam();

  Visualizer vis("3D Viewer",3,2);
  



  clb.cfg.camera_pos = cv::Mat::zeros(clb.cfg.robot_pos.size(),CV_32FC1);
  std::cout << "clb.cfg.camera_pos=\n" << clb.cfg.camera_pos << std::endl;

  std::string ch;
  for (int i = 0; i < clb.cfg.robot_pos.rows; i++) {
    if (ch == "q") {
      break;
    }
    std::cout << "Image "<< i << std::endl;
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    reader.read<pcl::PointXYZI>(clb.cfg.pcdFilePath + std::to_string(i)+".pcd", *cloud);
    
    vis.setCloud(cloud, 1);
    sdet.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered = sdet.getFilteredCloud();
    vis.setCloud(cloud_filtered, 2);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane = sdet.getPlaneCloud();
    vis.setCloud(cloud_plane, 3);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_noplane = sdet.getNoplaneCloud();
    vis.setCloud(cloud_noplane, 4);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_s = sdet.getInlierCloud();
    vis.setCloud(cloud_s, 5);
    vis.setCloud(cloud, 6);
    Eigen::VectorXf coef = sdet.getModelCoefficients();
    std::cout << "coef = \n" << coef << std::endl;
    vis.setSphere(coef, 0.2, 0.6, 0.2);
    clb.cfg.camera_pos.at<float>(i, 0) = coef[0];
    clb.cfg.camera_pos.at<float>(i, 1) = coef[1];
    clb.cfg.camera_pos.at<float>(i, 2) = coef[2];

    std::getline(std::cin, ch);
  }
  std::cout << "clb.cfg.camera_pos=\n" << clb.cfg.camera_pos << std::endl;
  cv::FileStorage fs;
  std::string camera_pos_fn = "camera_pos.json";
  fs.open(camera_pos_fn, cv::FileStorage::WRITE);
  fs << "camera_pos" << clb.cfg.camera_pos;
  fs.release();
  
  auto c_data = clb.calibrate();













}

