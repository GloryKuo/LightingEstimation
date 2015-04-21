#include "LightingEstimation_marker.h"
#include "GradientFilter.h"

using namespace cv;
using namespace std;

LightingEstimation_marker::LightingEstimation_marker()
{
	set_initGuess = false;
	_normal[0] = 0.0;      //normal initial guess n=(0, 0, 1)
	_normal[1] = 0.0;
	_normal[2] = 1.0;
	
	const double stopPixelVal = 0.0001;
	const double stopMaxtime = 120;    //seconds
	const int para_dimention = 8;     //number of paramter to optimize
	opt = new nlopt::opt(nlopt::LN_COBYLA, para_dimention);
	opt->set_stopval(stopPixelVal);
	opt->set_maxtime(stopMaxtime);
	opt->set_ftol_rel(0.001);
	vector<double> lb(8), ub(8);
	lb[0] = 0.0;
	lb[1] = 0.5;
	lb[2] = -1.0;
	lb[3] = -1.0;
	lb[4] = 0.0;
	lb[5] = -1000.0;
	lb[6] = -1000.0;
	lb[7] = 0.0;

	ub[0] = 0.5;
	ub[1] = 1.0;
	ub[2] = 1.0;
	ub[3] = 1.0;
	ub[4] = 1.0;
	ub[5] = 1000.0;
	ub[6] = 1000.0;
	ub[7] = 500.0;
	opt->set_lower_bounds(lb);
	opt->set_upper_bounds(ub);

}

LightingEstimation_marker::~LightingEstimation_marker()
{
	delete opt;
}

void LightingEstimation_marker::setHomoMatrix(Mat H)
{
	_homoMatrix = H.clone();
}

void LightingEstimation_marker::setInputImg(Mat img)
{
	_img = img.clone();
	//_img = img;
}

double LightingEstimation_marker::estimate()
{
	return estimate(_img, _homoMatrix);
}

double LightingEstimation_marker::estimate(cv::Mat img, double imgpts[4][2], double objpts[4][2])
{
	cv::Mat H;
	LE_marker::computeHomgraphy(imgpts, objpts, H);
	return estimate(img, H);
}

double LightingEstimation_marker::estimate(cv::Mat img, cv::Mat homography)
{

	/////* setup */////////////////////////////////////////////////////
	Mat shading = makeShading(img);
	//imshow("shading", shading);
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
		
	for(int i=0;i<worldPts_.size();i++){
		worldPts.push_back(Point3f(worldPts_[i].x, worldPts_[i].y, 0));   //Assume only simple plane, let all z=0.
	}

	/////* optmization */////////////////////////////////////////////////////
	setInitGuess();
	objfunc_data data;
	data._intensity = shading;
	data._pts_world = worldPts;
	data._pts_img = imgPts;

	double cost = 0.0;
	vector<double> x;
	x.push_back(_ambient);
	x.push_back(_diffuse);
	x.push_back(_normal[0]);
	x.push_back(_normal[1]);
	x.push_back(_normal[2]);
	x.push_back(_lightPos.x);
	x.push_back(_lightPos.y);
	x.push_back(_lightPos.z);

	opt->set_min_objective(objFunc, &data);
	nlopt::result result = opt->optimize(x, cost);
#ifdef _LE_DEBUG
	cout<<"finish "<<result<<endl;
#endif
	_ambient = x[0];
	_diffuse = x[1];

	Mat n(3, 1, CV_64FC1);
	for(int i=0;i<3;i++)
		n.at<double>(i) = x[i+2];
	double nn = norm(n, NORM_L2);
	n /= nn;      // normalize to unit vector
	for(int i=0;i<3;i++)
		_normal[i] = n.at<double>(i);
	_lightPos = Point3f((float)x[5], (float)x[6], (float)x[7]);

	return cost;
}

void LightingEstimation_marker::setInitGuess()
{
	if(!set_initGuess)
		setInitGuess(0.4, 0.6, 0.0f, 0.0f, 0.0f);
	else{
#ifdef _LE_DEBUG
		cout<<"Use prior guess."<<endl;
#endif
	}
}

void LightingEstimation_marker::setInitGuess(double ambient, double diffuse, float x, float y, float z)
{
	set_initGuess = true;
	_ambient = ambient;
	_diffuse = diffuse;
	_lightPos = Point3f(x, y, z);        //³æ¦ìmm
}

double LightingEstimation_marker::objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data)
{
	/* x := {ambient, diffuse, nx, ny, nz, lx, ly, lz} */

	objfunc_data *data =  reinterpret_cast<objfunc_data*>(objFunc_data);
	Mat I = data->_intensity;
	Mat n(3, 1, CV_64FC1);	
	Mat l(3, 1, CV_64FC1);
	double sumCost = 0.0;

	for(int i=0;i<3;i++)
		n.at<double>(i) = x[i+2];
	double nn = norm(n, NORM_L2);
	for(int i=0;i<3;i++)
		n /= nn;      // normalize to unit vector

// TODO: OpenCL
	for(int i=0;i<data->_pts_world.size();i++){	
		double Zp = (x[2]*data->_pts_world[i].x + x[3]*data->_pts_world[i].y)/(-x[4]);

		double *lp = l.ptr<double>(0);
		lp[0] = x[5] - data->_pts_world[i].x;
		lp[1] = x[6] - data->_pts_world[i].y;
		lp[2] = x[7] - Zp;	
		double nl = norm(l, NORM_L2);
		l /= nl;

		double Ip = I.at<double>((int)data->_pts_img[i].y, (int)data->_pts_img[i].x);
		double cost = Ip - x[0] - x[1]*(n.dot(l));
		sumCost += cost*cost;
		/*system("cls");
		cout<<"progress: "<<i+1<<"/"<<data->_pts_world.size()<<endl;
		cout<<"cost = "<<cost*cost<<endl;*/
	}	
#ifdef _LE_DEBUG
	static int itrCnt = 0;
	system("cls");
	cout<<"iteration "<<++itrCnt<<":"<<endl;
	cout<<"x = [";
	for(int i=0;i<7;i++)
		cout<<x[i]<<" ";
	cout<<x[7]<<"]"<<endl;
	cout<<"cost = "<<sumCost/data->_pts_world.size()<<endl;
#endif
	return sumCost/data->_pts_world.size();
}

double LightingEstimation_marker::constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data)
{
	return 0.0;
}

void LightingEstimation_marker::outputData(double output[8])
{
	output[0] = _ambient;
	output[1] = _diffuse;
	output[2] = _normal[0];
	output[3] = _normal[1];
	output[4] = _normal[2];
	output[5] = _lightPos.x;
	output[6] = _lightPos.y;
	output[7] = _lightPos.z;
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

void LightingEstimation_marker::computeHomgraphy(double imgpts[4][2], double objpts[4][2], cv::Mat& outputH)
{
	std::vector<Point2f> srcPts;
	std::vector<Point2f> desPts;
	for(int i=0;i<4;i++){
		srcPts.push_back(Point2f((float)imgpts[i][0], (float)imgpts[i][1]));
		desPts.push_back(Point2f((float)objpts[i][0], (float)objpts[i][1]));
	}
	outputH = getPerspectiveTransform(srcPts, desPts);    //find homography
}