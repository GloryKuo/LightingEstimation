#include "LightingEstimation_marker.h"
using namespace cv;
using namespace std;

LightingEstimation_marker::LightingEstimation_marker(){
	double stopPixelVal = 10.0;
	opt = new nlopt::opt(nlopt::LN_COBYLA, 5);
	opt->set_stopval(stopPixelVal);
}

LightingEstimation_marker::~LightingEstimation_marker(){
	delete opt;
}

void LightingEstimation_marker::setHomoMatrix(Mat H){
	_homoMatrix = H.clone();
}

void LightingEstimation_marker::setInputImg(Mat img){
	_img = img.clone();
}

bool LightingEstimation_marker::estimate(){
	return estimate(_img, _homoMatrix);
}

bool LightingEstimation_marker::estimate(cv::Mat img, cv::Mat homography){
	Mat shading = makeShading(img);
	//imshow("shading", shading);
	float normal[3] = {0, 0, 1};
	vector<Point2f> imgPts, worldPts_;
	vector<Point3f> worldPts;
	imgPts.reserve(shading.rows*shading.cols);
	worldPts_.reserve(shading.rows*shading.cols);
	worldPts.reserve(shading.rows*shading.cols);

	for(int i=0;i<shading.rows;i++)
		for(int j=0;j<shading.cols;j++){
			imgPts.push_back(Point2f(i, j));
		}
	perspectiveTransform(imgPts, worldPts_, homography);
		
	for(int i=0;i<worldPts_.size();i++)
		worldPts.push_back(Point3f(worldPts_[i].x, worldPts_[i].y, 0));

	return true;
}

double LightingEstimation_marker::objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data){
}

double LightingEstimation_marker::constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data){
	return 0.0;
}
