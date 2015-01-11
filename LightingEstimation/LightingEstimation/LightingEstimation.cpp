#include "LightingEstimation.h"
#include "GradientFilter.h"

using namespace cv;
using namespace std;

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
		for(int j=0;j<lines[i].pt.size();j++){
			/* 找出經過p的垂直線 */
			Point2i p = lines[i].pt[j];
			double coef[2]= {-1*lines[i].coef[1], lines[i].coef[0]};    // 方向向量(a,b) -> (-b,a)
			double t_step = MIN(1/coef[0], 1/coef[1]);        //最小步長

			Point2i z1 = p, z2 = p;
			Size imgSize(shading.cols, shading.rows);
			double d=10000000.0;
			double t=0.0;
			while(Line::checkValid(z1, imgSize) && Line::checkValid(z2, imgSize)){
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