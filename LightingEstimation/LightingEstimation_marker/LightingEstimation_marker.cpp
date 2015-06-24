#include "LightingEstimation_marker.h"
#include "GradientFilter.h"

using namespace cv;
using namespace std;

LightingEstimation_marker::LightingEstimation_marker()
{
	set_initGuess = false;
}

LightingEstimation_marker::~LightingEstimation_marker()
{
	delete opt;
}

void LightingEstimation_marker::init(int numMarkers, double markerWidth)
{
	_halfMarkerWidth = markerWidth/2.0;

	const double stopPixelVal = 0.000001;
	const double stopMaxtime = 60;    //seconds
	const int para_dimention = 5+(numMarkers*3);     //number of paramter to optimize
	opt = new nlopt::opt(nlopt::LN_COBYLA, para_dimention);
	opt->set_stopval(stopPixelVal);
	opt->set_maxtime(stopMaxtime);
	opt->set_ftol_rel(0.000001);
	vector<double> lb(para_dimention), ub(para_dimention);
	lb[0] = 0.0;              //ambient
	ub[0] = 0.5;
	lb[1] = 0.2;              //diffuse
	ub[1] = 0.7;

	for(int i=0;i<numMarkers;i++){
		lb[i*3+2] = (i==3)? 0.0 : -1.0;      //normal_x
		ub[i*3+2] = 1.0;

		lb[i*3+3] = (i==2)? 0.0 : -1.0;      //normal_y
		ub[i*3+3] = 1.0;

		lb[i*3+4] = (i==1)? 0.0 : -1.0;      //normal_z
		ub[i*3+4] = 1.0;
	}
	
	lb[2+numMarkers*3] = -markerWidth*5.0;        //light_pos_x
	ub[2+numMarkers*3] = markerWidth*5.0;
	lb[2+numMarkers*3+1] = -markerWidth*5.0;        //light_pos_y
	ub[2+numMarkers*3+1] = markerWidth*5.0;
	lb[2+numMarkers*3+2] = 0.0;           //light_pos_z
	ub[2+numMarkers*3+2] = markerWidth*6.0;
	opt->set_lower_bounds(lb);
	opt->set_upper_bounds(ub);
	
	/*vector<double> init_step(para_dimention);
	init_step[0] = 0.1;
	init_step[1] = 0.1;
	for(int i=0;i<numMarkers*3;i++){
		init_step[2+i] = 0.05;
	}
	init_step[2+numMarkers*3] = 10.0;
	init_step[2+numMarkers*3+1] = 10.0;
	init_step[2+numMarkers*3+2] = 10.0;
	opt->set_initial_step(init_step);*/
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
	_label = Mat::ones(Size(_img.cols, _img.rows), CV_32FC1);
}

void LightingEstimation_marker::setLabel(Mat label)
{
	unsigned short int nLabels = 0;
	std::vector<int> colors;
	bool newL = true;
	for(int i=label.rows-1;i>=0;i--)
		for(int j=label.cols-1;j>=0;j--){
			newL = true;
			for(int c=0;c<colors.size();c++){
				if(label.at<uchar>(i,j) == colors[c]){
					_label.at<float>(i,j) = (float)(c+1);
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
	imshow("shading",shading);
	waitKey(10);
	
	/////* convert image coordinate to world coordinate *//////////////////////////
	
	vector<vector<Point2f>> imgPts_2d, markerPts_2d;
	vector<vector<Point3f>> worldPts_3d;

	/* transform image coordinate to marker coordinate respectively */
	computeCorrespondRelative(Size(shading.cols, shading.rows), homography, label, imgPts_2d, markerPts_2d);

		
	vector<Point3f> worldPts3D_tmp;
	worldPts3D_tmp.reserve(markerPts_2d[0].size());
	/* marker1 coordinate to world coordinate */
	for(int i=0;i<markerPts_2d[0].size();i++)
		worldPts3D_tmp.push_back(Point3f(markerPts_2d[0].at(i).x, markerPts_2d[0].at(i).y, 0));   //the first plane, z=0.
	worldPts_3d.push_back(worldPts3D_tmp);

	worldPts3D_tmp.clear();
	/* marker2 coordinate to world coordinate */
	if(markerPts_2d.size()>=2){
		vector<Point3f> markerPts_3d;
		markerPts_3d.reserve(markerPts_2d[1].size());
		for(int i=0;i<markerPts_2d[1].size();i++)
			markerPts_3d.push_back(Point3f(markerPts_2d[1].at(i).x, markerPts_2d[1].at(i).y, 0));

		Mat markerPts_3d_mat(3, (int)markerPts_3d.size(), CV_32FC1);
		for(int i=0;i<markerPts_3d.size();i++){
			markerPts_3d_mat.at<float>(0,i) = markerPts_3d[i].x;
			markerPts_3d_mat.at<float>(1,i) = markerPts_3d[i].y;
			markerPts_3d_mat.at<float>(2,i) = markerPts_3d[i].z;
		}
		
		
		/* transform marker2 coordinate to marker1 coordinate */
		/*double srcPts[4][2] = {{-_halfMarkerWidth, _halfMarkerWidth},
								{_halfMarkerWidth, _halfMarkerWidth},
								{_halfMarkerWidth, -_halfMarkerWidth},
								{-_halfMarkerWidth, -_halfMarkerWidth}};
		double desPts[4][2] = {{-_halfMarkerWidth, _halfMarkerWidth},
								{_halfMarkerWidth, _halfMarkerWidth},
								{_halfMarkerWidth, 0},
								{-_halfMarkerWidth, 0}};
		Mat homography_2_1;
		computeHomgraphy(srcPts, desPts, homography_2_1);cout<<homography_2_1<<endl;

		vector<Point2f> worldPts_2d;
		perspectiveTransform(markerPts_2d[1], worldPts_2d, homography_2_1);*/
		float mat[3][3] = {{1,0,0},{0,0,-1},{0,1,0}};
		Mat rorateMat(3, 3, CV_32FC1, mat);
		Mat markerPts_3d_mat_r = rorateMat * markerPts_3d_mat;

		for(int i=0;i<markerPts_3d_mat.cols;i++){
			markerPts_3d_mat_r.at<float>(1,i) += (float)_halfMarkerWidth;
			markerPts_3d_mat_r.at<float>(2,i) += (float)_halfMarkerWidth;
			/*if(i%10000==0) 
				cout<<imgPts_2d[1].at(i)<<" ==> "<<markerPts_3d_mat.col(i)<<" ==> "<<markerPts_3d_mat_r.col(i)<<endl;*/
			worldPts3D_tmp.push_back(Point3f(markerPts_3d_mat_r.at<float>(0,i), 
				markerPts_3d_mat_r.at<float>(1,i), markerPts_3d_mat_r.at<float>(2,i)));   //the second plane, let y=40.
		}
		worldPts_3d.push_back(worldPts3D_tmp);
	}

	/////* optmization */////////////////////////////////////////////////////
	setInitGuess();
	objfunc_data data;
	data._intensity = shading;
	data._pts_world = worldPts_3d;
	data._pts_img = imgPts_2d;
	data._marker_halfLen  = _halfMarkerWidth;

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

	cons_data *cdata = new cons_data[homography.size()];
	for(int i=0;i<homography.size();i++){
		cdata[i].index_marker = i;
		opt->add_equality_constraint(constraint, &(cdata[i]), 0);
	}

	nlopt::result result = opt->optimize(x, cost);
#ifdef _LE_DEBUG
	std::cout<<"finish "<<result<<endl;
#endif
	delete [] cdata;

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
	_lightPos = Point3d(x[2+homography.size()*3], x[2+homography.size()*3+1], x[2+homography.size()*3+2]);

	return cost;
}

void LightingEstimation_marker::computeCorrespondRelative(cv::Size imgSize, std::vector<cv::Mat> homography, cv::Mat label,
		std::vector<std::vector<Point2f>> &imgPts_2d, std::vector<std::vector<Point2f>> &worldPts_2d)
{
	for(int s=0;s<homography.size();s++){
		vector<Point2f> imgPts_tmp, worldPts_tmp;
		imgPts_tmp.reserve(imgSize.width*imgSize.height);
		worldPts_tmp.reserve(imgSize.width*imgSize.height);

		for(int i=0;i<imgSize.height;i++)
			for(int j=0;j<imgSize.width;j++){
				if(label.at<float>(i,j) == s+1)
					imgPts_tmp.push_back(Point2f((float)j, (float)i));
			}
		perspectiveTransform(imgPts_tmp, worldPts_tmp, homography[s]);
		imgPts_2d.push_back(imgPts_tmp);
		worldPts_2d.push_back(worldPts_tmp);
	}
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
	int numMarker = data->_pts_world.size();

	for(int s=0;s<data->_pts_world.size();s++){
		for(int i=0;i<3;i++)
			n.at<double>(i) = x[i+2+s*3];
		double nn = norm(n, NORM_L2);
		n /= nn;      // normalize to unit vector

		double plane_offset = data->_marker_halfLen;       //使平面平移至marker邊緣
// TODO: OpenCL

		for(int i=0;i<data->_pts_world[s].size();i++){
			/* 投影某3D點到平面上，計算light direction */
			double offset;
			if(s==0)
				offset = (n.at<double>(0)*data->_pts_world[s].at(i).x + n.at<double>(1)*data->_pts_world[s].at(i).y)/(-(n.at<double>(2)));
			else if(s==1)
				offset = (n.at<double>(0)*data->_pts_world[s].at(i).x + n.at<double>(2)*data->_pts_world[s].at(i).z)+plane_offset/(-(n.at<double>(1)));
			else if(s==2)
				offset = (n.at<double>(1)*data->_pts_world[s].at(i).y + n.at<double>(2)*data->_pts_world[s].at(i).z)+plane_offset/(-(n.at<double>(0)));
			else{
				std::cout<<"invalid \"_pts_world.size()\"."<<endl;
				return 0.0;
			}

			double *lp = l.ptr<double>(0);
			lp[0] = x[2+numMarker*3] - ((s==2)? offset : data->_pts_world[s].at(i).x);
			lp[1] = x[2+numMarker*3+1] - ((s==1)? offset : data->_pts_world[s].at(i).y);
			lp[2] = x[2+numMarker*3+2] - ((s==0)? offset : data->_pts_world[s].at(i).z);	
			double nl = norm(l, NORM_L2);
			l /= nl;

			/* 計算cost */
			double Ip = I.at<double>((int)data->_pts_img[s].at(i).y, (int)data->_pts_img[s].at(i).x);
			double cost = Ip - x[0] - x[1]*(n.dot(l));
			sumCost += cost*cost;
			/*system("cls");
			cout<<"progress: "<<i+1<<"/"<<data->_pts_world[s].size()<<endl;
			cout<<"cost = "<<cost*cost<<endl;*/
		}	
	}
	double averageCost = sqrt(sumCost)/(data->_intensity.rows*data->_intensity.cols);
#ifdef _LE_DEBUG
	static int itrCnt = 0;
	std::system("cls");
	std::cout<<"iteration "<<++itrCnt<<":"<<endl;
	std::cout<<"x = [";
	for(int i=0;i<x.size()-1;i++)
		std::cout<<x[i]<<" ";
	std::cout<<x[x.size()-1]<<"]"<<endl;
	std::cout<<"cost = "<<averageCost<<endl;
#endif
	return averageCost;
}

double LightingEstimation_marker::constraint(const std::vector<double> &x, std::vector<double> &grad, void* data)
{
	cons_data *cdata =  reinterpret_cast<cons_data*>(data);
	Mat n(3, 1, CV_64FC1);
	for(int i=0;i<3;i++)
		n.at<double>(i) = x[2 + (cdata->index_marker)*3 + i];
	double cost = norm(n, NORM_L2) - 1;
	cout<<"Constraint cost:\t"<<cost<<endl;
	return cost*cost;
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