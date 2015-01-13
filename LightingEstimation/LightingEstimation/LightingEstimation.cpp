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
	return p.x>=0 && p.y>=0 && p.x<imgSize.width && p.y<imgSize.height;
}

LightingEstimation::Line& LightingEstimation::Line::clone()
{
	Line *l = new Line();
	for(int i=0;i<pts.size();i++){
		l->pts.push_back(pts[i]);
	}
	l->_coef[0] = _coef[0];
	l->_coef[1] = _coef[1];
	l->_prior = _prior;
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
	if(src.channels()!=1)
		cvtColor(src, src, CV_BGR2GRAY);
	GradientFilter gf;
	gf.init(src);
	return gf.optimize();
}

LightingEstimation::Line& LightingEstimation::detectBiSymmetry(Mat src, Point2i prior)
{
	_shading = makeShading(src);
	vector<LightingEstimation::Line> lines;
	vector<double> dist;

	generateHypotheses(lines, prior, _shading.cols, _shading.rows);

	for(int i=0;i<lines.size();i++){
		int numValidPt = 0;
		double d_sum = 0.0;
		for(int j=0;j<lines[i].pts.size();j++){
			/* 找出經過p的垂直線 */
			Point2i p = lines[i].pts[j];
			double coef[2]= {-1*lines[i]._coef[1], lines[i]._coef[0]};    // 方向向量(a,b) -> (-b,a)
			double t_step = MIN(1/coef[0], 1/coef[1]);        //最小步長

			Point2i z1 = p, z2 = p;
			Size imgSize(_shading.cols, _shading.rows);
			double d=-1.0;
			double t=0.0;
			const double th = 0.0000001;
			while(Line::checkValidPt(z1, imgSize) && Line::checkValidPt(z2, imgSize)){
				if(t==0.0){
					t += t_step;
					z1 = Point2i(p.x+coef[0]*t, p.y+coef[1]*t);
					z2 = Point2i(p.x+coef[0]*(-1*t), p.y+coef[1]*(-1*t));
					continue;
				}
				if(abs(_shading.at<double>(z1.y, z1.x)-_shading.at<double>(z2.y, z2.x)) < th){
					d = (z1.x-z2.x)*(z1.x-z2.x) + (z1.y-z2.y)*(z1.y-z2.y);
					numValidPt++;
					break;
				}
				t += t_step;
				z1 = Point2i(p.x+coef[0]*t, p.y+coef[1]*t);
				z2 = Point2i(p.x+coef[0]*(-1*t), p.y+coef[1]*(-1*t));
			}
			if(d != -1.0)
				d_sum += d;
		}
		if(numValidPt==0)    /* 若這條線上完全沒有對稱點，就讓dist值超大 */
			dist.push_back(1000000000.0);
		else
			dist.push_back(d_sum / numValidPt);
		system("cls");
		cout<<"#lines NO:\t"<<i+1<<"/"<<lines.size()<<endl;
	}
	int minIdx[2];
	double minVal;
	cv::minMaxIdx(dist, &minVal, NULL, minIdx);

	return lines[minIdx[1]].clone();
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
	if(src.channels()==1){
		des = Mat(src.rows, src.cols, CV_8UC3);
		Mat img[3] = {src, src, src};
		merge(img, 3, des);
	
	}
	else
		des = src.clone();

	for(int i=0;i<lines.size();i++){
		for(int j=0;j<lines[i].pts.size();j++)
			line(des, lines[i].pts[j], lines[i].pts[j], Scalar(0, 0, 200));
	}
}

Mat& LightingEstimation::getShading()
{
	Mat *output = new Mat();
	_shading.convertTo(*output, CV_8UC1, 255);
	return *output;
}