#include "LightingEstimation.h"
#include "GradientFilter.h"

using namespace cv;
using namespace std;

void LightingEstimation::Line::recordValidPts(int imgWidth, int imgHeight){
	/* 紀錄在圖片中的所有點 */
	double t_step = MIN(1/abs(_coef[0]), 1/abs(_coef[1]));        //最小步長
	double t = 0.0;
	Point2i p = _prior;

	while (checkValidPt( p, Size2i(imgWidth, imgHeight) )){
		pts.push_back(p);
		t += t_step;
		p.x = _prior.x + static_cast<int>(ceil(_coef[0]*t) );
		p.y = _prior.y + static_cast<int>(ceil(_coef[1]*t) );		
	}
	reverse(pts.begin(), pts.end());
	t = 0.0;
	t_step *= -1;
	p = _prior;

	while (checkValidPt( p, Size2i(imgWidth, imgHeight) )){
		if(t!=0.0)
			pts.push_back(p);
		t += t_step;
		p.x = _prior.x + static_cast<int>(ceil(_coef[0]*t) );
		p.y = _prior.y + static_cast<int>(ceil(_coef[1]*t) );
	}
}

bool LightingEstimation::Line::checkValidPt(cv::Point2i p, cv::Size2i imgSize){
	return p.x>=0 && p.y>=0 && p.x<=imgSize.width && p.y<=imgSize.height;
}

LightingEstimation::Line& LightingEstimation::Line::clone()
{
	Line *l = new Line();
	for(int i=0;i<pts.size();i++){
		l->pts.push_back(pts[i]);
	}
	l->_coef[0] = _coef[0];
	l->_coef[1] = _coef[1];
	return *l;
}

void LightingEstimation::Line::printLine()
{
	cout<<"=========================="<<endl;
	cout<<"x = "<<_prior.x<<" + "<<_coef[0]<<"t;"<<endl;
	cout<<"y = "<<_prior.y<<" + "<<_coef[1]<<"t;"<<endl;

	if(pts.size() ==0){
		cout<<"this Line have not recordPt()!"<<endl;
		return;
	}

	for(int i=0;i<pts.size();i++){
		cout<<"("<<pts[i].x<<", "<<pts[i].y<<")  ";
	}
	cout<<"\n=========================="<<endl;
}

LightingEstimation::LightingEstimation(void)
{
}


LightingEstimation::~LightingEstimation(void)
{
}

Mat LightingEstimation::makeShading(Mat src)
{
	GradientFilter gf;
	gf.init(src);
	return gf.optimize();
}

LightingEstimation::Line LightingEstimation::detectBiSymmetry(Mat shading, Point2i prior)
{
	vector<LightingEstimation::Line> lines;
	vector<double> dist(lines.size());
	generateHypotheses(lines, prior, shading.cols, shading.rows);

	for(int i=0;i<lines.size();i++){
		int numValidPt = 0;
		double d_sum = 0.0;
		for(int j=0;j<lines[i].pts.size();j++){
			/* 找出經過p的垂直線 */
			Point2i p = lines[i].pts[j];
			double coef[2]= {-1*lines[i]._coef[1], lines[i]._coef[0]};    // 方向向量(a,b) -> (-b,a)
			double t_step = MIN(1/coef[0], 1/coef[1]);        //最小步長

			Point2i z1 = p, z2 = p;
			Size imgSize(shading.cols, shading.rows);
			double d=10000000.0;
			double t=0.0;
			while(Line::checkValidPt(z1, imgSize) && Line::checkValidPt(z2, imgSize)){
				z1 = Point2i(p.x+coef[0]*t, p.y+coef[1]*t);
				z2 = Point2i(p.x+coef[0]*(-1*t), p.y+coef[1]*(-1*t));
				if(shading.at<uchar>(z1) == shading.at<uchar>(z2)){
					d = (z1.x-z2.x)*(z1.x-z2.x) + (z1.y-z2.y)*(z1.y-z2.y);
					numValidPt++;
					break;
				}
				t += t_step;
			}
			d_sum += d;
		}
		dist[i] = d_sum / numValidPt;
	}
	int *minIdx = NULL;
	double *minVal = NULL;
	cv::minMaxIdx(dist, minVal, NULL, minIdx);

	Line symmetryAxis = lines[*minIdx].clone();
	return symmetryAxis;
}

void LightingEstimation::generateHypotheses( vector<LightingEstimation::Line> &lines, Point2i prior, int imgWidth, int imgHeight )
{
	/* 點斜式 y-y0 = m(x-x0) */
	double m;      //直線斜率
	const double theta_step = 0.5;    //夾角間隔
	for(double theta=(-90.0)+theta_step; theta<90.0; theta+=theta_step){
		m = tan(theta*3.1415926/180);
		LightingEstimation::Line l;
		l._coef[0] = 1;
		l._coef[1] = m;
		l._prior = prior;
		lines.push_back(l);
	}
	for(int i=0;i<lines.size();i++){
		lines[i].recordValidPts(imgWidth, imgHeight);
	}
}

void LightingEstimation::drawLines(cv::Mat src, cv::Mat &des, std::vector<LightingEstimation::Line> lines)
{
	des = Mat(src.rows, src.cols, CV_8UC3);
	Mat img[3] = {src, src, src};
	merge(img, 3, des);

	for(int i=0;i<lines.size();i++){
		for(int j=0;j<lines[i].pts.size();j++)
			line(des, lines[i].pts[j], lines[i].pts[j], Scalar(0, 0, 200));
	}
}