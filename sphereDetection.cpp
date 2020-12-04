#include "sphereDetection.h"


#pragma   push_macro("min")  
#pragma   push_macro("max")  
#undef   min  
#undef   max
#include <pcl/sample_consensus/ransac.h>          // 采样一致性
#include <pcl/sample_consensus/sac_model_sphere.h>// 球模型
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transformation_from_correspondences.h>

#pragma   pop_macro("min")  
#pragma   pop_macro("max")
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <future>

std::mutex mtx;
std::condition_variable conv;
//pcl::visualization::PCLVisualizer viewer("3D viewer");


void Calibration::readCfg(const std::string& file_path) {
	cv::FileStorage fs;
	fs.open(file_path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cout << "can't open file " << file_path << std::endl;
	}
	else {
		fs["calibdataPath"] >> this->cfg.calibdataPath;
		fs["calibdataFilename"] >> this->cfg.calibdataFilename;
		fs["pcdFilePath"] >> this->cfg.pcdFilePath;
		fs["mode"] >> this->cfg.mode;
		fs["coordinateDifference"] >> this->cfg.coordinateDifference;
		fs["estimateMethod"] >> this->cfg.estimateMethod;
		fs["capture_pos"] >> this->cfg.capture_pos;
		fs["grab_pos"] >> this->cfg.grab_pos;
		fs["robot_pos"] >> this->cfg.robot_pos;
	}
	fs.release();
};
void Calibration::printCfg() {
	std::cout << "The Calibration Config : " << std::endl;
	std::cout << "calibdataPath = " <<
		this->cfg.calibdataPath << std::endl;
	std::cout << "calibdataFilename = " <<
		this->cfg.calibdataFilename << std::endl;
	std::cout << "pcdFilePath = " <<
		this->cfg.pcdFilePath << std::endl;


	std::cout << "mode = " <<
		this->cfg.mode << std::endl;
	std::cout << "coordinateDifference = " <<
		this->cfg.coordinateDifference << std::endl;
	std::cout << "estimateMethod = " <<
		this->cfg.estimateMethod << std::endl;
	std::cout << "capture_pos = " <<
		this->cfg.capture_pos << std::endl;
	std::cout << "grab_pos = " <<
		this->cfg.grab_pos << std::endl;
	std::cout << "robot_pos = " <<
		this->cfg.robot_pos << std::endl;
	std::cout << "camera_pos = " <<
		this->cfg.camera_pos << std::endl;
}

CalibData Calibration::calibrate() {
	//pcl::TransformationFromCorrespondences tfc;
	//tfc.add

	// 计算转换矩阵
	CalibData cdata;
	cv::Mat homo_camera, homo_robot, homo_camera_inv, robot_negate, to_points;

	if (cfg.mode == "EyeInHand")
		// EyeInHand模式下机器人运动方向于视觉中方向相反。
		to_points = -(this->cfg.robot_pos);
	else if (cfg.mode == "EyeToHand")
		to_points = this->cfg.robot_pos;
	else {
		std::cout << "Mode error: EyeInHand or EyeToHand. " << std::endl;
		return cdata;
	}
	std::cout << "to_points =\n"<<to_points << std::endl;
	if (this->cfg.coordinateDifference) {
		this->cfg.camera_pos.col(-1) = -this->cfg.camera_pos.col(-1);
	}
	cdata.coordinateDifference = this->cfg.coordinateDifference;

	cv::vconcat(this->cfg.camera_pos.t(), cv::Mat::ones(1, this->cfg.camera_pos.rows, CV_32FC1), homo_camera);
	std::cout << "homo_camera= \n" << homo_camera << std::endl;
	cv::vconcat(to_points.t(), cv::Mat::ones(1, this->cfg.robot_pos.rows, CV_32FC1), homo_robot);
	std::cout << "homo_robot = \n" << homo_robot << std::endl;
	if (this->cfg.estimateMethod == "SVD") {
		// 使用伪逆矩阵
		cv::invert(homo_camera, homo_camera_inv, cv::DECOMP_SVD);  // 伪逆矩阵
		cdata.transMatrix = homo_robot * homo_camera_inv;
		std::cout << "transMatrix=\n" << cdata.transMatrix << std::endl;
	}
	else if (this->cfg.estimateMethod == "RANSAC") {
		//***************************使用cv::estimateAffine3D*******************************
		cv::Mat trans_m, trans_m32, trans_m_homo, inliers;
		cv::estimateAffine3D(this->cfg.camera_pos, to_points, trans_m, inliers);
		std::cout << "trans_m = \n" << trans_m << std::endl;
		std::cout << "inliers=\n" << inliers << std::endl;
		trans_m.convertTo(trans_m32, CV_32FC1);
		cv::Mat m = (cv::Mat_<float>(1, 4) << 0, 0, 0, 1);
		cv::vconcat(trans_m32, m, cdata.transMatrix);
		std::cout << "transMatrix=\n" << cdata.transMatrix << std::endl;
		//***************************使用cv::estimateAffine3D end*******************************
	}
	else if (this->cfg.estimateMethod == "FromCorr") {
		//*****************使用pcl::TransformationFromCorrespondences**************************
		pcl::TransformationFromCorrespondences transFromCorr;

		for (int i = 0; i < to_points.rows; i++) {
			Eigen::Vector3f from(
				cfg.camera_pos.at<float>(i,0),
				cfg.camera_pos.at<float>(i, 1), 
				cfg.camera_pos.at<float>(i, 2));
			Eigen::Vector3f to(
				to_points.at<float>(i, 0),
				to_points.at<float>(i, 1),
				to_points.at<float>(i, 2));
			transFromCorr.add(from, to, 1.0);
		}
		 auto m = transFromCorr.getTransformation().matrix();
		 cv::eigen2cv(m, cdata.transMatrix);
		 std::cout << "transMatrix=\n" << cdata.transMatrix << std::endl;
		//***************使用pcl::TransformationFromCorrespondences end************************
	}
	else {
		std::cout << "Unkwnown estimateMethod" << std::endl;
	}

	// 求translate
	// (此处不求translate,改为在TranslateCorrection项目中动态较准)
	//std::cout << "capture_pos = \n" << this->cfg.capture_pos << std::endl;
	//std::cout << "grab_pos = \n" << this->cfg.grab_pos << std::endl;
	//cdata.translate = this->cfg.grab_pos - this->cfg.capture_pos;
	//std::cout << "translate = \n" << cdata.translate << std::endl;
	cdata.translate = (cv::Mat_<float>(1, 3) << 0, 0, 0);

	// 写入标定结果文件
	cv::FileStorage fs;
	std::string save_path = cfg.calibdataPath + cfg.calibdataFilename;
	fs.open(save_path, cv::FileStorage::WRITE);
	fs << "transform_matrix" << cdata.transMatrix << "translate" << cdata.translate<<
		"coordinateDifference"<<cfg.coordinateDifference;
	fs.release();
	// 计算重投影点及重投影误差
	std::cout << "cdata.transMatrix.type() = " <<
		cdata.transMatrix.type() << std::endl;
	std::cout << "homo_camera.type() = " <<
		homo_camera.type() << std::endl;
	cv::Mat robot_again = cdata.transMatrix * homo_camera;
	std::cout << "robot_transform =\n" << robot_again << std::endl;
	cv::Mat err = robot_again - homo_robot;
	std::cout << "err =\n" << err << std::endl;
	return cdata;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr  createSphere(
	std::vector<float> origin, float radius, int point_num) {
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
	cv::RNG rng((unsigned)time(nullptr));
	for (int i = 0; i < point_num; i++) {
		pcl::PointXYZI pt_xyzi;
		pt_xyzi.x = rng.uniform(origin[0] - radius, origin[0] + radius);
		pt_xyzi.y = rng.uniform(origin[1] - radius, origin[1] + radius);
		pt_xyzi.z = -(std::sqrt(-std::pow(pt_xyzi.x - origin[0], 2) -
			std::pow(pt_xyzi.y - origin[1], 2) + std::pow(radius,2))) + origin[2];
		//std::cout << "x = " << pt_xyzi.x << " y = " << pt_xyzi.y <<
		//	" z = "<<pt_xyzi.z<<std::endl;
		pt_xyzi.intensity = pt_xyzi.z;
		if (!isnan(pt_xyzi.z)) {
			cloud->push_back(pt_xyzi);
		}
	}
	return cloud;
}

SphereDetector::SphereDetector() {

	cloud_pass.reset(new pcl::PointCloud<PointT>());
	cloud_filtered.reset(new pcl::PointCloud<PointT>());
	cloud_subplane.reset(new pcl::PointCloud<pcl::PointXYZI>());
	cloud_noplane.reset(new pcl::PointCloud<PointT>());
	cloud_s.reset(new pcl::PointCloud<PointT>());	
	cloud_background_.reset(new pcl::PointCloud<PointT>());
	this->coefficients.resize(4);
	this->PassThroughDone = false;

	pcl::PCDReader reader;
	reader.read<PointT>("D:/projects/PCL_calibrationSphere/PCL_calibrationSphere/0.pcd", *cloud_background_);
};

void SphereDetector::readParam(const std::string& file_path) {
	cv::FileStorage fs;
	fs.open(file_path, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		std::cout << "can't open file " << file_path << std::endl;
	}
	else {
		fs["PassThroghXMin"] >> this->sp.PassThroughXMin;
		fs["PassThroghXMax"] >> this->sp.PassThroughXMax;
		fs["PassThroghYMin"] >> this->sp.PassThroughYMin;
		fs["PassThroghYMax"] >> this->sp.PassThroughYMax;
		fs["PassThroghZMin"] >> this->sp.PassThroughZMin;
		fs["PassThroghZMax"] >> this->sp.PassThroughZMax;
		fs["PassThroghIMin"] >> this->sp.PassThroughIMin;
		fs["PassThroghIMax"] >> this->sp.PassThroughIMax;

		fs["BGLeafSize"] >> this->sp.BGLeafSize;

		fs["SORMeanK"] >> this->sp.SORMeanK;
		fs["SORStddevMulThresh"] >> this->sp.SORStddevMulThresh;

		fs["PLANormalKSearch"] >> this->sp.PLANormalKSearch;
		fs["PLANormalDistanceWeight"] >> this->sp.PLANormalDistanceWeight;
		fs["PLAMaxIterations"] >> this->sp.PLAMaxIterations;
		fs["PLADistanceThreshold"] >> this->sp.PLADistanceThreshold;

		fs["SPHNormalKSearch"] >> this->sp.SPHNormalKSearch;
		fs["SPHNormalDistanceWeight"] >> this->sp.SPHNormalDistanceWeight;
		fs["SPHMaxIterations"] >> this->sp.SPHMaxIterations;
		fs["SPHDistanceThreshold"] >> this->sp.SPHDistanceThreshold;
		fs["SPHMinRadius"] >> this->sp.SPHMinRadius;
		fs["SPHMaxRadius"] >> this->sp.SPHMaxRadius;
	}
	fs.release();
};

void SphereDetector::printParam() {
	std::cout << "SphereDetector Params: " << std::endl;
	std::cout << "PassThroghXMin = " <<
		this->sp.PassThroughXMin << std::endl;
	std::cout << "PassThroghXMax = " <<
		this->sp.PassThroughXMax << std::endl;
	std::cout << "PassThroghYMin = " <<
		this->sp.PassThroughYMin << std::endl;
	std::cout << "PassThroghYMax = " <<
		this->sp.PassThroughYMax << std::endl;
	std::cout << "PassThroghZMin = " <<
		this->sp.PassThroughZMin << std::endl;
	std::cout << "PassThroghZMax = " <<
		this->sp.PassThroughZMax << std::endl;
	std::cout << "PassThroghIMin = " <<
		this->sp.PassThroughIMin << std::endl;
	std::cout << "PassThroghIMax = " <<
		this->sp.PassThroughIMax << std::endl;
	std::cout << "BGLeafSize = " <<
		this->sp.BGLeafSize << std::endl;

	std::cout << "SORMeanK = " <<
		this->sp.SORMeanK << std::endl;
	std::cout << "SORStddevMulThresh = " <<
		this->sp.SORStddevMulThresh << std::endl;

	std::cout << "PLANormalKSearch = " <<
		this->sp.PLANormalKSearch << std::endl;
	std::cout << "PLANormalDistanceWeight = " <<
		this->sp.PLANormalDistanceWeight << std::endl;
	std::cout << "PLAMaxIterations = " <<
		this->sp.PLAMaxIterations << std::endl;
	std::cout << "PLADistanceThreshold = " <<
		this->sp.PLADistanceThreshold << std::endl;

	std::cout << "SPHNormalKSearch = " <<
		this->sp.SPHNormalKSearch << std::endl;
	std::cout << "SPHNormalDistanceWeight = " <<
		this->sp.SPHNormalDistanceWeight << std::endl;
	std::cout << "SPHMaxIterations = " <<
		this->sp.SPHMaxIterations << std::endl;
	std::cout << "SPHDistanceThreshold = " <<
		this->sp.SPHDistanceThreshold << std::endl;
	std::cout << "SPHMinRadius = " <<
		this->sp.SPHMinRadius << std::endl;
	std::cout << "SPHMaxRadius = " <<
		this->sp.SPHMaxRadius << std::endl;
}

void SphereDetector::setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
	//this->PassThroughDone = true;
	std::cout << "cloud->points.size() = " <<
		cloud->points.size() << std::endl;
	//// Pass Though
	pcl::ConditionAnd<PointT>::Ptr range_cond(new
		pcl::ConditionAnd<PointT>());
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::GT, this->sp.PassThroughXMin)));
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("x", pcl::ComparisonOps::LT, this->sp.PassThroughXMax)));
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::GT, this->sp.PassThroughYMin)));
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("y", pcl::ComparisonOps::LT, this->sp.PassThroughYMax)));
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::GT, this->sp.PassThroughZMin)));
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("z", pcl::ComparisonOps::LT, this->sp.PassThroughZMax)));
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("intensity", pcl::ComparisonOps::GT,
			this->sp.PassThroughIMin)));
	range_cond->addComparison(pcl::FieldComparison<PointT>::ConstPtr(new
		pcl::FieldComparison<PointT>("intensity", pcl::ComparisonOps::LT,
			this->sp.PassThroughIMax)));

	pcl::ConditionalRemoval<PointT> condrem; // 条件直通滤波器
	condrem.setCondition(range_cond);
	condrem.setInputCloud(cloud);
	//condrem.setKeepOrganized(true);
	condrem.filter(*this->cloud_pass);
	std::cout << "cloud_pass->points.size() = " <<
		cloud_pass->points.size() << std::endl;
	//// ***********背景减除********
	//pcl::PointIndices::Ptr ind_foreground(new pcl::PointIndices),
	//	ind_background(new pcl::PointIndices);
	//pcl::KdTreeFLANN<PointT> kdtree_onbackground;
	//kdtree_onbackground.setInputCloud(this->cloud_background_);
	//for (int i = 0; i < this->cloud_pass->points.size(); i++) {
	//	std::vector<int> pointIdxNKNSearch(1);
	//	std::vector<float> pointNKNSquaredDistance(1);
	//	kdtree_onbackground.nearestKSearch(this->cloud_pass->points[i], 1,
	//		pointIdxNKNSearch, pointNKNSquaredDistance);
	//	if (pointNKNSquaredDistance[0]<sp.BGLeafSize) {
	//		ind_background->indices.push_back(i);
	//	}
	//	else {
	//		ind_foreground->indices.push_back(i);
	//	}
	//}
	//pcl::io::savePCDFile("cloud_pass.pcd", *this->cloud_pass);
	//pcl::ExtractIndices<PointT> extract_foreg;
	//extract_foreg.setInputCloud(this->cloud_pass);
	//extract_foreg.setNegative(false);
	//extract_foreg.setIndices(ind_foreground);
	//pcl::PointCloud<PointT>::Ptr foreground(new pcl::PointCloud<PointT>()),
	//	background(new pcl::PointCloud<PointT>());
	//extract_foreg.filter(*foreground);
	//extract_foreg.setIndices(ind_background);
	//extract_foreg.filter(*background);
	//pcl::io::savePCDFile("foreground.pcd", *foreground);
	//pcl::io::savePCDFile("background.pcd", *background);
	// ***********背景减除end********
	// 滤波
		// sor
	pcl::StatisticalOutlierRemoval<PointT> sor;// StatisticalOutlierRemoval滤波器 
	sor.setInputCloud(cloud_pass);
	sor.setMeanK(sp.SORMeanK);
	sor.setStddevMulThresh(sp.SORStddevMulThresh);
	sor.filter(*this->cloud_filtered);
	std::cout << "cloud_filtered->points.size() = " <<
		cloud_filtered->points.size() << std::endl;

	// SAC球体分割
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal>seg_n;
	pcl::SACSegmentation<PointT> seg;

	pcl::ExtractIndices<PointT> extract;
	pcl::ExtractIndices<pcl::Normal> extract_normals;
	pcl::PointIndices::Ptr inliers_sphere(new pcl::PointIndices),
		inliers_plane(new pcl::PointIndices);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_plane_normals(new pcl::PointCloud<pcl::Normal>),
		cloud_noplane_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::ModelCoefficients::Ptr coef_plane(new pcl::ModelCoefficients),
		coef_sphere(new pcl::ModelCoefficients);

	ne.setSearchMethod(tree);

	ne.setKSearch(sp.PLANormalKSearch);


	seg_n.setOptimizeCoefficients(true);
	seg_n.setMethodType(pcl::SAC_RANSAC);
	seg_n.setNormalDistanceWeight(sp.PLANormalDistanceWeight);
	seg_n.setModelType(pcl::SACMODEL_NORMAL_PLANE);
	seg_n.setMaxIterations(sp.PLAMaxIterations);
	seg_n.setDistanceThreshold(sp.PLADistanceThreshold);

	// 排除平面
	pcl::PointCloud<PointT>::Ptr cloud_sor4planar(new pcl::PointCloud<PointT>());

	cloud_subplane.reset(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::copyPointCloud(*cloud_filtered, *cloud_sor4planar); //复制

	int i = 0, nr_points = (int)cloud_filtered->points.size();
	//若剩余点大于阈值则继续找平面
	while (cloud_sor4planar->points.size() > 10000)
	{
		// Segment the largest planar component from the remaining cloud
		ne.setInputCloud(cloud_sor4planar);
		ne.compute(*cloud_plane_normals);


		seg_n.setInputCloud(cloud_sor4planar);
		seg_n.setInputNormals(cloud_plane_normals);
		seg_n.segment(*inliers_plane, *coef_plane);
		if (inliers_plane->indices.size() == 0) {
			std::cout << "Could not estimate a planar model for the given dataset." <<
				std::endl;
			break;
		}
		extract.setInputCloud(cloud_sor4planar);
		extract.setNegative(false);
		extract.setIndices(inliers_plane);
		pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
		extract.filter(*cloud_plane);
		std::cout << "cloud_plane->points.size() = " <<
			cloud_plane->points.size() << std::endl;

		for (pcl::PointCloud<PointT>::iterator it = cloud_plane->points.begin();
			it < cloud_plane->points.end(); ++it) {
			pcl::PointXYZI p_xyzi;
			p_xyzi.x = it->x;
			p_xyzi.y = it->y;
			p_xyzi.z = it->z;
			p_xyzi.intensity = i;
			//cloud_subplane->points.push_back(p_xyzl);
			cloud_subplane->push_back(p_xyzi);
		}
		// Extract the planar inliers from the input cloud
		extract.setNegative(true);
		extract.filter(*cloud_sor4planar);

		std::cout << "cloud_sor4planar->points.size() = " <<
			cloud_sor4planar->points.size() << std::endl;
			// 得到平面后对sub_plane进行二次滤波
		// 1使用sor 
		 //this->sor.setInputCloud(cloud_plane);
		 //this->sor.filter(*cloud_plane_filtered);
		// 2使用radius_outlier_removal
		//this->ror.setInputCloud(cloud_plane);
		//this->ror.filter(*cloud_plane_filtered);
		i++;
	}
	pcl::copyPointCloud(*cloud_sor4planar, *cloud_noplane); //复制
	std::cout << "cloud_noplane->points.size() = " <<
		cloud_noplane->points.size() << std::endl;
	ne.setKSearch(sp.SPHNormalKSearch);
	ne.setInputCloud(cloud_noplane);
	ne.compute(*cloud_noplane_normals);
	seg_n.setModelType(pcl::SACMODEL_NORMAL_SPHERE);

	seg_n.setNormalDistanceWeight(sp.SPHNormalDistanceWeight);
	seg_n.setMaxIterations(sp.SPHMaxIterations);
	seg_n.setDistanceThreshold(sp.SPHDistanceThreshold);
	seg_n.setRadiusLimits(sp.SPHMinRadius, sp.SPHMaxRadius);
	seg_n.setInputCloud(cloud_noplane);
	seg_n.setInputNormals(cloud_noplane_normals);

	seg_n.segment(*inliers_sphere, *coef_sphere);
	std::cout << "inliers_sphere->indices.size() = " <<
		inliers_sphere->indices.size() << std::endl;
	extract.setInputCloud(cloud_noplane);
	extract.setIndices(inliers_sphere);
	extract.setNegative(false);
	extract.filter(*this->cloud_s);


	if (inliers_sphere->indices.size()) {
		for (int i = 0; i < 4; i++) 
			coefficients[i] = coef_sphere->values[i];
	}
	else {
		for (int i = 0; i < 4; i++)
			coefficients[i] = -1;
	}

	
		

	////// created RandomSampleConsensus object and compute the appropriated model
	//std::vector<int> inliers;
	//pcl::SampleConsensusModelSphere<pcl::PointXYZI>::Ptr
	//model_s(new pcl::SampleConsensusModelSphere<pcl::PointXYZI>(cloud_noplane));
	//pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(model_s);
	//ransac.setDistanceThreshold(10);
	//
	//ransac.computeModel();
	//ransac.getInliers(inliers);
	////this->coefficients.resize(4);
	//ransac.getModelCoefficients(this->coefficients);
	////ransac
	//std::cout << "cloud_pass.size() = " << cloud_pass->points.size() << std::endl;
	//std::cout << "inliers.size() = "<<inliers.size() << std::endl;
	//pcl::copyPointCloud(*cloud_noplane, inliers, *this->cloud_s);




}

pcl::PointCloud<PointT>::Ptr SphereDetector::getPassThroughCloud() {
	return this->cloud_pass;
}

pcl::PointCloud<PointT>::Ptr SphereDetector::getFilteredCloud() {
	return this->cloud_filtered;
}

pcl::PointCloud<PointT>::Ptr SphereDetector::getPlaneCloud() {
	return this->cloud_subplane;
}

pcl::PointCloud<PointT>::Ptr SphereDetector::getNoplaneCloud() {
	return this->cloud_noplane;
}

pcl::PointCloud<PointT>::Ptr SphereDetector::getInlierCloud() {
	return this->cloud_s;
}

Eigen::VectorXf SphereDetector::getModelCoefficients() {
	return this->coefficients;
}






Visualizer::Visualizer(const std::string& win_name) {

}

Visualizer::Visualizer(const std::string& win_name,
	const int& rows, const int& cols) {
	this->cloud1.reset(new pcl::PointCloud<PointT>());
	this->cloud2.reset(new pcl::PointCloud<PointT>());
	this->cloud3.reset(new pcl::PointCloud<PointT>());
	this->cloud4.reset(new pcl::PointCloud<PointT>());
	this->cloud5.reset(new pcl::PointCloud<PointT>());
	this->cloud6.reset(new pcl::PointCloud<PointT>());
	this->clouds = {
		{ 1, cloud1 },
		{ 2, cloud2 },
		{ 3, cloud3 },
		{ 4, cloud4 },
		{ 5, cloud5 },
		{ 6, cloud6 }
	};
	this->setSphereFlag = false;
	std::thread viewer_t([&] {
	if (rows * cols > 6) {
		pcl::console::print_error("ERROR : Too much ViewPorts!\n");
	}

	float view_width = 1 / cols;
	float view_height = 1 / rows;
	pcl::visualization::PCLVisualizer viewer("3D viewer");
	int v1(0), v2(0), v3(0), v4(0), v5(0), v6(0);
	//for (int i = 0; i < rows; i++) {
	//	for (int j = 0; j < cols; j++) {
	//		//this->viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1); // (Xmin, Ymin, Xmax, Ymax)
	//		pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
	//		clouds.emplace_back(cloud);

	//	}
	//}
	//std::cout << "clouds.size()"<<clouds.size() << std::endl;
	//viewPort1 original
	viewer.createViewPort(0.0, 0.5, 0.33, 1.0, v1); // (Xmin, Ymin, Xmax, Ymax)
	viewer.setBackgroundColor(0.05, 0.05, 0.05, v1); // Setting background to a dark grey
	viewer.addText("Original", 10, 10, "v1 text", v1);
	viewer.addCoordinateSystem(1000, "original_cloud_coor", v1);
	viewer.addPointCloud<PointT>(clouds[1], "cloud1", v1);
	//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
	//	1, "original_cloud");
	//viewPort2 sloud_sor
	viewer.createViewPort(0.33, 0.5, 0.66, 1.0, v2);
	viewer.setBackgroundColor(0.3, 0.3, 0.3, v2); // Setting background to a dark grey
	viewer.addText("PassThrough", 10, 10, "v2 text", v2);
	viewer.addCoordinateSystem(1000, "pass_through_coor", v2);
	viewer.addPointCloud<PointT>(clouds[2], "cloud2", v2);
	//viewPort3 sloud_sor
	viewer.createViewPort(0.66, 0.5, 1.0, 1.0, v3);
	viewer.setBackgroundColor(0.05, 0.05, 0.05, v3); // Setting background to a dark grey
	viewer.addText("Planes", 10, 10, "v3 text", v3);
	viewer.addCoordinateSystem(1000, "planes_coor", v3);
	viewer.addPointCloud<PointT>(clouds[3], "cloud3", v3);
	//viewPort4 sloud_sor
	viewer.createViewPort(0.0, 0.0, 0.33, 0.5, v4);
	viewer.setBackgroundColor(0.3, 0.3, 0.3, v4); // Setting background to a dark grey
	viewer.addText("Noplane", 10, 10, "v4 text", v4);
	viewer.addCoordinateSystem(1000, "noplane_coor", v4);
	viewer.addPointCloud<PointT>(clouds[4], "cloud4", v4);
		//viewPort5 sloud_sor
	viewer.createViewPort(0.33, 0.0, 0.66, 0.5, v5);
	viewer.setBackgroundColor(0.05, 0.05, 0.05, v5); // Setting background to a dark grey
	viewer.addText("Inliers", 10, 10, "v5 text", v5);
	viewer.addCoordinateSystem(1000, "inliers_coor", v5);
	viewer.addPointCloud<PointT>(clouds[5], "cloud5", v5);
		//viewPort6 sloud_sor
	viewer.createViewPort(0.66, 0.0, 1.0, 0.5, v6);
	viewer.setBackgroundColor(0.3, 0.3, 0.3, v6); // Setting background to a dark grey
	viewer.addText("SphereModel", 10, 10, "v6 text", v6);
	viewer.addCoordinateSystem(1000, "sphere_coor", v6);
	viewer.addPointCloud<PointT>(clouds[6], "cloud6", v6);


	while (!viewer.wasStopped())
	{
		viewer.updatePointCloud<PointT>(cloud1, "cloud1");
		viewer.updatePointCloud<PointT>(cloud2, "cloud2");
		viewer.updatePointCloud<PointT>(cloud3, "cloud3");
		//viewer.updatePointCloud<PointT>(cloud4, "cloud4");
		viewer.removePointCloud("cloud4", v4);
		viewer.addPointCloud<PointT>(clouds[4], "cloud4", v4);
		//viewer.updatePointCloud<PointT>(cloud5, "cloud5");
		viewer.removePointCloud("cloud5", v5);
		viewer.addPointCloud<PointT>(clouds[5], "cloud5", v5);
		viewer.updatePointCloud<PointT>(cloud6, "cloud6");
		if (this->setSphereFlag) {
			if (sphereCoef[3] < 0) {
				viewer.removeShape("sphere", v6);
				viewer.addSphere(pcl::PointXYZ(0,0,0), 0, sphereCoef[4],
					sphereCoef[5], sphereCoef[6], "sphere", v6);
			}
			else {
				pcl::PointXYZ center;
				center.x = sphereCoef[0];
				center.y = sphereCoef[1];
				center.z = sphereCoef[2];
				viewer.removeShape("sphere", v6);
				viewer.addSphere(center, sphereCoef[3], sphereCoef[4],
					sphereCoef[5], sphereCoef[6], "sphere", v6);
			}
			this->setSphereFlag = false;
		}
		viewer.spinOnce(100);
	}
	});
	viewer_t.detach();
}

void Visualizer::setCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int viewport){
	pcl::copyPointCloud(*cloud, *(clouds[viewport])); //复制
}

void Visualizer::setSphere(Eigen::VectorXf coef,
	float r, float g, float b) {
	this->sphereCoef = { coef[0], coef[1], coef[2], coef[3], r, g, b };
	this->setSphereFlag = true;
}
