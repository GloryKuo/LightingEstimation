#include "LightingEstimation_marker.h"
#include "GradientFilter.h"
/* _LE_DEBUG mode is slower */
//#define _LE_DEBUG

using namespace cv;
using namespace std;

LightingEstimation_marker::LightingEstimation_marker(){
	set_initGuess = false;

	double stopPixelVal = 10.0;
	double stopMaxtime = 0.3;
	opt = new nlopt::opt(nlopt::LN_COBYLA, 5);
	opt->set_stopval(stopPixelVal);
	opt->set_maxtime(stopMaxtime);
	opt->set_ftol_rel(0.01);
	vector<double> lb(5), ub(5);
	lb[0] = 0.0;
	lb[1] = 0.0;
	lb[2] = -500.0;
	lb[3] = -500.0;
	lb[4] = 0.0;

	ub[0] = 0.5;
	ub[1] = 1.0;
	ub[2] = 500.0;
	ub[3] = 500.0;
	ub[4] = 300.0;
	opt->set_lower_bounds(lb);
	opt->set_upper_bounds(ub);

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

double LightingEstimation_marker::estimate(){
	return estimate(_img, _homoMatrix);
}

double LightingEstimation_marker::estimate(cv::Mat img, cv::Mat homography){

	/////* setup */////////////////////////////////////////////////////
	Mat shading = makeShading(img);
	//imshow("shading", shading);
	double normal[3] = {0.0, 0.0, 1.0};           //Assume only simple plane, all normal=0.
	vector<Point2f> imgPts, worldPts_;
	vector<Point3f> worldPts;
	imgPts.reserve(shading.rows*shading.cols);
	worldPts_.reserve(shading.rows*shading.cols);
	worldPts.reserve(shading.rows*shading.cols);

	/////* convert image coordinate to world coordinate *//////////////////////////

	for(int i=0;i<shading.rows;i++)
		for(int j=0;j<shading.cols;j++){
			imgPts.push_back(Point2f((float)j, (float)i));
		}
	perspectiveTransform(imgPts, worldPts_, homography);
		
	for(int i=0;i<worldPts_.size();i++)
		worldPts.push_back(Point3f(worldPts_[i].x, worldPts_[i].y, 0));   //Assume only simple plane, let all z=0.

	/////* optmization */////////////////////////////////////////////////////
	setInitGuess();
	objfunc_data data;
	data._intensity = shading;
	for(int j=0;j<3;j++)
		data._normal[j] = normal[j];
	data._pts_world = worldPts;
	data._pts_img = imgPts;

	double cost = 0.0;
	vector<double> x;
	x.push_back(intensity_ambient);
	x.push_back(intensity_diffuse);
	x.push_back(light_position.x);
	x.push_back(light_position.y);
	x.push_back(light_position.z);

	opt->set_min_objective(objFunc, &data);
	nlopt::result result = opt->optimize(x, cost);
	cout<<"finish "<<result<<endl;
	intensity_ambient = x[0];
	intensity_diffuse = x[1];
	light_position = Point3f((float)x[2], (float)x[3], (float)x[4]);

	return cost;
}

void LightingEstimation_marker::setInitGuess()
{
	if(!set_initGuess)
		setInitGuess(0.1, 0.5, 0.0f, 0.0f, 0.0f);
	else{
		cout<<"Use prior guess."<<endl;
	}
}

void LightingEstimation_marker::setInitGuess(double ambient, double diffuse, float x, float y, float z)
{
	set_initGuess = true;
	intensity_ambient = ambient;
	intensity_diffuse = diffuse;
	light_position = Point3f(x, y, z);        //³æ¦ìmm
}

double LightingEstimation_marker::objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data){
	objfunc_data *data =  reinterpret_cast<objfunc_data*>(objFunc_data);
	Mat I = data->_intensity;
	Mat n(3, 1, CV_64FC1, &(data->_normal));	
	Mat l(3, 1, CV_64FC1);
	double sumCost = 0.0;

	for(int i=0;i<data->_pts_world.size();i++){	
		double *lp = l.ptr<double>(0);
		lp[0] = x[2] - data->_pts_world[i].x;
		lp[1] = x[3] - data->_pts_world[i].y;
		lp[2] = x[4] - data->_pts_world[i].z;	
		double Ip = I.at<double>((int)data->_pts_img[i].y, (int)data->_pts_img[i].x);
		double cost = Ip - x[0] - x[1]*(n.dot(l)/norm(l, NORM_L2));
		sumCost += cost*cost;
		/*system("cls");
		cout<<"progress: "<<i+1<<"/"<<data->_pts_world.size()<<endl;
		cout<<"cost = "<<cost*cost<<endl;*/
	}	
#ifdef _LE_DEBUG
	static int itrCnt = 0;
	system("cls");
	cout<<"iteration "<<++itrCnt<<":"<<endl;
	cout<<"x = ["<<x[0]<<" "<<x[1]<<" "<<x[2]<<" "<<x[3]<<" "<<x[4]<<"]"<<endl;
	cout<<"sumCost = "<<sumCost<<endl;
#endif
	return sumCost;
}

double LightingEstimation_marker::constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data)
{
	return 0.0;
}

void LightingEstimation_marker::outputData(double output[5])
{
	output[0] = intensity_ambient;
	output[1] = intensity_diffuse;
	output[2] = light_position.x;
	output[3] = light_position.y;
	output[4] = light_position.z;
}

Mat LightingEstimation_marker::makeShading(Mat src)
{
	if(src.channels()!=1)
		cvtColor(src, src, CV_BGR2GRAY);
	GradientFilter gf;
	gf.init(src);
	return gf.optimize();
}

Mat& LightingEstimation_marker::getShading()
{
	Mat *output = new Mat();
	_shading.convertTo(*output, CV_8UC1, 255);
	return *output;
}
