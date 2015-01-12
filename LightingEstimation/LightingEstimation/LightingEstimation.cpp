#include "LightingEstimation.h"
#include "GradientFilter.h"

using namespace cv;
using namespace std;

void LightingEstimation::Line::recordPt(cv::Point2i prior, double coef[2], int imgWidth, int imgHeight){  
	_prior = prior;
	_coef[0] = coef[0];
	_coef[1] = coef[1];
	/* 紀錄在圖片中的所有點 */
	double t_step = MIN(1/abs(coef[0]), 1/abs(coef[1]));        //最小步長
	double t = 0.0;
	Point2i p = prior;

	while (checkValidPt( p, Size2i(imgWidth, imgHeight) )){
		pts.push_back(p);
		t += t_step;
		p.x = prior.x + static_cast<int>(ceil(coef[0]*t) );
		p.y = prior.y + static_cast<int>(ceil(coef[1]*t) );		
	}
	reverse(pts.begin(), pts.end());
	t = 0.0;
	t_step *= -1;

	while (checkValidPt( p, Size2i(imgWidth, imgHeight) )){
		if(t!=0.0)
			pts.push_back(p);
		t += t_step;
		p.x = prior.x + static_cast<int>(ceil(coef[0]*t) );
		p.y = prior.y + static_cast<int>(ceil(coef[1]*t) );
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
	generateHypotheses(lines, prior);

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
	minMaxIdx(dist, minVal, NULL, minIdx);

	Line symmetryAxis = lines[*minIdx].clone();
	return symmetryAxis;
}

void LightingEstimation::generateHypotheses( vector<LightingEstimation::Line> &lines, Point2i prior )
{
	/*To do*/
}