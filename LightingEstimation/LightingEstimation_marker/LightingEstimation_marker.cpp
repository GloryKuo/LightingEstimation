#include "LightingEstimation_marker.h"
#include "GradientFilter.h"

using namespace cv;
using namespace std;

LightingEstimation_marker::LightingEstimation_marker()
{
	set_initGuess = false;

	const double stopPixelVal = 0.0001;
	const double stopMaxtime = 30;    //seconds
	const int para_dimention = 8;     //number of paramter to optimize
	opt = new nlopt::opt(nlopt::LN_COBYLA, para_dimention);
	opt->set_stopval(stopPixelVal);
	opt->set_maxtime(stopMaxtime);
	opt->set_ftol_rel(0.000001);
	vector<double> lb(8), ub(8);
	lb[0] = 0.0;
	lb[1] = 0.5;
	lb[2] = -1.0;
	lb[3] = -1.0;
	lb[4] = 0.0;
	lb[5] = -100.0;
	lb[6] = -100.0;
	lb[7] = 0.0;

	ub[0] = 0.5;
	ub[1] = 1.0;
	ub[2] = 1.0;
	ub[3] = 1.0;
	ub[4] = 1.0;
	ub[5] = 100.0;
	ub[6] = 100.0;
	ub[7] = 50.0;
	opt->set_lower_bounds(lb);
	opt->set_upper_bounds(ub);

}

LightingEstimation_marker::~LightingEstimation_marker()
{
	delete opt;
}

void LightingEstimation_marker::setHomoMatrix(std::vector<cv::Mat> Hs)
{
	for(int i=0;i<Hs.size();i++)
		_homoMatrix.push_back(Hs[i]);
}

void LightingEstimation_marker::setInputImg(Mat img)
{
	_img = img.clone();
	//_img = img;
	_label = Mat::zeros(Size(_img.cols, _img.rows), CV_32FC1);
}

void LightingEstimation_marker::setLabel(Mat label)
{
	unsigned short int nLabels = 0;
	std::vector<int> colors;
	bool newL = true;
	for(int i=0;i<label.rows;i++)
		for(int j=0;j<label.cols; j++){
			newL = true;
			for(int c=0;c<colors.size();c++){
				if(label.at<uchar>(i,j) == colors[c]){
					_label.at<float>(i,j) = c+1;
					newL = false;
					break;
				}
			}
			if(newL){
				nLabels++;
				colors.push_back(label.at<uchar>(i,j));
				_label.at<float>(i,j) = (float)nLabels;
			}
		}
}

double LightingEstimation_marker::estimate()
{
	return estimate(_img, _homoMatrix, _label);
}


double LightingEstimation_marker::estimate(cv::Mat img, std::vector<cv::Mat> homography, cv::Mat label)
{

	/////* setup */////////////////////////////////////////////////////
	Mat shading = makeShading(img);
	//imshow("shading", shading);

	vector<vector<Point2f>> imgPts, worldPts_;
	vector<vector<Point3f>> worldPts;

	/*imgPts.reserve(shading.rows*shading.cols);
	worldPts_.reserve(shading.rows*shading.cols);
	worldPts.reserve(shading.rows*shading.cols);*/

	/////* convert image coordinate to world coordinate *//////////////////////////

	for(int s=0;s<homography.size();s++){
		vector<Point2f> imgPts_tmp, worldPts_tmp;
		imgPts_tmp.reserve(shading.rows*shading.cols);
		worldPts_tmp.reserve(shading.rows*shading.cols);

		for(int i=0;i<shading.rows;i++)
			for(int j=0;j<shading.cols;j++){
				if(label.at<float>(i,j) == s)
					imgPts_tmp.push_back(Point2f((float)j, (float)i));
			}
		perspectiveTransform(imgPts_tmp, worldPts_tmp, homography[s]);
		imgPts.push_back(imgPts_tmp);
		worldPts_.push_back(worldPts_tmp);
	}
		
	vector<Point3f> worldPts3D_tmp;
	for(int i=0;i<worldPts_[0].size();i++)
		worldPts3D_tmp.push_back(Point3f(worldPts_[0].at(i).x, worldPts_[0].at(i).y, 0));   //the first plane, z=0.
	worldPts.push_back(worldPts3D_tmp);

	if(worldPts_.size()>=2){
		for(int i=0;i<worldPts_[1].size();i++)
			worldPts3D_tmp.push_back(Point3f(worldPts_[1][i].x, 40, worldPts_[1][i].y));   //the second plane, let y=40.
		worldPts.push_back(worldPts3D_tmp);
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

	for(int s=0;s<homography.size();s++)
		for(int i=0;i<3;i++)
			x.push_back(_normal[s][i]);
	/*x.push_back(_normal[0]);
	x.push_back(_normal[1]);
	x.push_back(_normal[2]);*/

	x.push_back(_lightPos.x);
	x.push_back(_lightPos.y);
	x.push_back(_lightPos.z);

	opt->set_min_objective(objFunc, &data);
	nlopt::result result = opt->optimize(x, cost);
#ifdef _LE_DEBUG
	std::cout<<"finish "<<result<<endl;
#endif
	_ambient = x[0];
	_diffuse = x[1];

	Mat n(3, 1, CV_64FC1);
	for(int s=0;s<homography.size();s++){
		for(int i=0;i<3;i++)
			n.at<double>(i) = x[i+2+s*3];
		double nn = norm(n, NORM_L2);
		n /= nn;      // normalize to unit vector
		for(int i=0;i<3;i++)
			_normal[s][i] = n.at<double>(i);
	}
	_lightPos = Point3d(x[5], x[6], x[7]);

	return cost;
}

void LightingEstimation_marker::setInitGuess()
{
	if(!set_initGuess)
		setInitGuess(0.4, 0.6, 0.0, 0.0, 0.0);
	else{
#ifdef _LE_DEBUG
		std::cout<<"Use prior guess."<<endl;
#endif
	}
}

void LightingEstimation_marker::setInitGuess(double ambient, double diffuse,
											 double x, double y, double z)
{
	set_initGuess = true;
	_ambient = ambient;
	_diffuse = diffuse;

	_normal[0][0] = 0.0;
	_normal[0][1] = 0.0;      //地板平面
	_normal[0][2] = 1.0;

	_normal[1][0] = 0.0;
	_normal[1][1] = -1.0;     //右側平面
	_normal[1][2] = 0.0;

	_normal[2][0] = 1.0;
	_normal[2][1] = 0.0;      //左側平面
	_normal[2][2] = 0.0;

	_lightPos = Point3d(x, y, z);        //單位mm
}

double LightingEstimation_marker::objFunc(const std::vector<double> &x, std::vector<double> &grad, void* objFunc_data)
{
	/* x := {ambient, diffuse, n1x, n1y, n1z, (n2x, ..., n3z), lx, ly, lz} */

	objfunc_data *data =  reinterpret_cast<objfunc_data*>(objFunc_data);
	Mat I = data->_intensity;
	Mat n(3, 1, CV_64FC1);	
	Mat l(3, 1, CV_64FC1);
	double sumCost = 0.0;

	for(int s=0;s<data->_pts_world.size();s++){
		for(int i=0;i<3;i++)
			n.at<double>(i) = x[i+2+s*3];
		double nn = norm(n, NORM_L2);
		for(int i=0;i<3;i++)
			n /= nn;      // normalize to unit vector

// TODO: OpenCL
		for(int i=0;i<data->_pts_world[s].size();i++){	
			double Zp = (x[2]*data->_pts_world[s][i].x + x[3]*data->_pts_world[s][i].y)/(-x[4]);

			double *lp = l.ptr<double>(0);
			lp[0] = x[5] - data->_pts_world[s][i].x;
			lp[1] = x[6] - data->_pts_world[s][i].y;
			lp[2] = x[7] - Zp;	
			double nl = norm(l, NORM_L2);
			l /= nl;

			double Ip = I.at<double>((int)data->_pts_img[s][i].y, (int)data->_pts_img[s][i].x);
			double cost = Ip - x[0] - x[1]*(n.dot(l));
			sumCost += cost*cost;
			/*system("cls");
			cout<<"progress: "<<i+1<<"/"<<data->_pts_world[s].size()<<endl;
			cout<<"cost = "<<cost*cost<<endl;*/
		}	
	}
	double averageCost = sumCost/(data->_intensity.rows*data->_intensity.cols);
#ifdef _LE_DEBUG
	static int itrCnt = 0;
	system("cls");
	cout<<"iteration "<<++itrCnt<<":"<<endl;
	cout<<"x = [";
	for(int i=0;i<x.size()-1;i++)
		cout<<x[i]<<" ";
	cout<<x[x.size()-1]<<"]"<<endl;
	cout<<"cost = "<<averageCost<<endl;
#endif
	return averageCost;
}

double LightingEstimation_marker::constraint(const std::vector<double> &x, std::vector<double> &grad, void* cons_data)
{
	return 0.0;
}

void LightingEstimation_marker::outputData(std::vector<double> &output)
{
	output.push_back(_ambient);
	output.push_back(_diffuse);
	for(int s=0;s<_homoMatrix.size();s++)
		for(int i=0;i<3;i++)
			output.push_back(_normal[s][i]);
	/*output[2] = _normal[0];
	output[3] = _normal[1];
	output[4] = _normal[2];*/
	output.push_back(_lightPos.x);
	output.push_back(_lightPos.y);
	output.push_back(_lightPos.z);
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