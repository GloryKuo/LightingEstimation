#include"LightingEstimation_marker.h"
using namespace cv;

struct onMouseData
{
	Mat *img;
	std::vector<Point2f> srcPts;
	std::vector<Point2f> desPts;
};

static void onMouse( int event, int x, int y, int, void* d )
{
	static int count = 0;
	onMouseData *data = static_cast<onMouseData *>(d);
	if(event != EVENT_LBUTTONDOWN)
		return;
	
	if(count<4){
		data->srcPts[count++] = Point2f((float)x, (float)y);
		circle(*(data->img), Point2f((float)x, (float)y), 5, Scalar(0,0,255));
		imshow("img", *(data->img));

		std::cout<<"("<<data->srcPts[count-1].x<<","<<data->srcPts[count-1].y<<")";
		std::cout<<" --> ("<<data->desPts[count-1].x<<","<<data->desPts[count-1].y<<")";
		std::cout<<std::endl;
	}
	if(count==4){
		count++;
		Mat H = getPerspectiveTransform(data->srcPts, data->desPts);    //find homography
		LE_marker::getInstance().estimate(*(data->img), H);
		
	}
}

int main(void)
{
	Mat img = imread("../input/input23.jpg");
	imshow("img", img);

	onMouseData data;
	data.srcPts.resize(4);
	data.desPts.resize(4);

	data.img = &img;
	data.desPts[0] = Point2f(-40.0,  40.0);            //counter-clockwise
	data.desPts[1] = Point2f(-40.0, -40.0);
	data.desPts[2] = Point2f( 40.0, -40.0);
	data.desPts[3] = Point2f( 40.0,  40.0);

	setMouseCallback("img", onMouse, &data);


	waitKey();
	return 0;
}