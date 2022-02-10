#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>             // 法线
#include <pcl/features/fpfh_omp.h>
#include <pcl/visualization/pcl_visualizer.h>   // 可视化
#include <pcl/visualization/pcl_plotter.h>
#include <Eigen/Core>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/time.h>
#include <pcl/common/distances.h>
#include <pcl/console/parse.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/extract_indices.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/visualization/pcl_visualizer.h> 
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/surface/mls.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/correspondence_estimation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <direct.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <io.h>
#include <string>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <windows.h>
#include <iostream>
#include <chrono>

using namespace std;

boost::mutex cloud_mutex;
pcl::visualization::PCLPlotter plotter;
//pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
pcl::FPFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>());

// structure used to pass arguments to the callback function
struct callback_args {
	pcl::PointCloud<pcl::PointXYZ>::Ptr clicked_points_3d;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

// callback function
void pp_callback(const pcl::visualization::PointPickingEvent& event, void* args)
{
	plotter.clearPlots();
	struct callback_args* data = (struct callback_args *)args;
	if (event.getPointIndex() == -1)
		return;
	pcl::PointXYZ current_point;
	event.getPoint(current_point.x, current_point.y, current_point.z);
	data->clicked_points_3d->points.clear();
	data->clicked_points_3d->points.push_back(current_point);

	// Draw clicked points in red:
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(data->clicked_points_3d, 255, 0, 0);
	data->viewerPtr->removePointCloud("clicked_points");
	data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
	data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
	std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;

	int num = event.getPointIndex();
	plotter.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs, "fpfh", num);
	plotter.plot();
}

int main(int argc, char *argv[]) {
	bool display = true;
	bool downSampling = false;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile<pcl::PointXYZ>("C:\\Users\\Ren\\Desktop\\點雲匹配測試檔\\B_Scan_1.ply", *cloud);
	std::cout << "Cloud size: " << cloud->points.size() << std::endl;


	//  Normal estimation
	auto t1 = chrono::steady_clock::now();
	pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
	pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);
	ne.setKSearch(10);
	//    ne.setRadiusSearch(0.03);
	ne.compute(*normals);
	auto t2 = chrono::steady_clock::now();
	auto dt = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
	cout << "Time cost of Normal estimation: " << dt << endl;

	// fpfh or fpfh_omp
	fpfh.setInputCloud(cloud);
	fpfh.setInputNormals(normals);
	fpfh.setSearchMethod(tree);
	fpfh.setRadiusSearch(10);
	fpfh.compute(*fpfhs);
	t1 = chrono::steady_clock::now();
	dt = chrono::duration_cast<chrono::duration<double> >(t1 - t2).count();
	cout << "Time cost of FPFH estimation: " << dt << endl;

	pcl::FPFHSignature33 descriptor;
	for (int i = 0; i<10; ++i) {
		int index = i + rand() % cloud->points.size();
		descriptor = fpfhs->points[index];
		std::cout << " -- fpfh for point " << index << ":\n" << descriptor << std::endl;
	}

	if (display) {
		plotter.addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs, "fpfh", 100);

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
		viewer->addPointCloudNormals<pcl::PointXYZ, pcl::PointNormal>(cloud, normals, 10, 0.05, "normals");  // display every 1 points, and the scale of the arrow is 10
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();

		// Add point picking callback to viewer:
		struct callback_args cb_args;
		pcl::PointCloud<pcl::PointXYZ>::Ptr clicked_points_3d(new pcl::PointCloud<pcl::PointXYZ>);
		cb_args.clicked_points_3d = clicked_points_3d;
		cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);
		viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);
		std::cout << "Shift + click on three floor points, then press 'Q'..." << std::endl;

		//        viewer->spin();
		//        cloud_mutex.unlock();

		while (!viewer->wasStopped()) {
			viewer->spinOnce(100); // Spin until 'Q' is pressed
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}

	return 0;
}