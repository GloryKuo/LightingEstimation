#include "LightingEstimation_marker.h"
#include <fstream>
#include <time.h>

#define NUM_MARKER 2
#define MARKER_LEN 20.0
using namespace cv;
using namespace std;


float dsFactor = 1.0f;        //downsample ratio
double cost;
std::vector<double> output;
ofstream fout;

struct onMouseData
{
	Mat img, label;
	string refer_path;
	std::vector<Point2f> srcPts;
	std::vector<Point2f> desPts;
} data;

static void onMouse( int event, int x, int y, int, void* d )
{
	static int count = 0;
	onMouseData *data = static_cast<onMouseData *>(d);
	static Mat click_img = imread(data->refer_path);
	if(event != EVENT_LBUTTONDOWN)
		return;

	data->srcPts[0] = Point2f(151,289);
	data->srcPts[1] = Point2f(347,307);
	data->srcPts[2] = Point2f(353,503);
	data->srcPts[3] = Point2f(129,505);
	data->srcPts[4] = Point2f(135,175);
	data->srcPts[5] = Point2f(419,207);
	data->srcPts[6] = Point2f(347,307);
	data->srcPts[7] = Point2f(151,289);
	count += 8;
	
	if(count < NUM_MARKER*4){
		data->srcPts[count++] = Point2f((float)x*dsFactor, (float)y*dsFactor);
		circle(click_img, Point2f((float)x, (float)y), 5, Scalar(0,0,255));
		imshow("click", click_img);

		std::cout<<"("<<data->srcPts[count-1].x<<","<<data->srcPts[count-1].y<<")";
		std::cout<<" --> ("<<data->desPts[count-1].x<<","<<data->desPts[count-1].y<<")";
		std::cout<<std::endl;
	}
	if(count == NUM_MARKER*4){
		destroyAllWindows();
		count++;
		resize((data->img), (data->img), Size((int)(data->img.cols*dsFactor), (int)(data->img.rows*dsFactor)));
		
		std::vector<cv::Mat> Hs;
		for(int i=0;i<NUM_MARKER;i++){
			std::vector<cv::Point2f> src_pts, des_pts;
			for(int j=i*4;j<(i*4)+4; j++){
				src_pts.push_back(data->srcPts.at(j));
				des_pts.push_back(data->desPts.at(j));
			}
			Mat H = getPerspectiveTransform(src_pts, des_pts);    //find homography
			Hs.push_back(H);
		}
		LE_marker::getInstance().setHomoMatrix(Hs);
		LE_marker::getInstance().setInputImg(data->img);
		if(!data->label.empty())
			LE_marker::getInstance().setLabel(data->label);

		clock_t start, end;
		start = clock();
		cost = LE_marker::getInstance().estimate();
		end = clock();
		cout<<"execution time:\t"<<(end-start)<<" ms"<<endl;
		

		/* output file *////////////////////////////////////////////
		LE_marker::getInstance().outputData(output);
		
		/* for evaluation */
		ofstream fout_conf("../output/config_m");
		fout_conf<<NUM_MARKER<<endl;
		fout_conf<<output[0]<<endl;
		fout_conf<<output[1]<<endl;
		for(int s=0;s<Hs.size();s++)
			fout_conf<<output[s*3+2]<<" "<<output[s*3+3]<<" "<<output[s*3+4]<<endl;
		fout_conf<<output[2+(NUM_MARKER*3)]<<" "<<output[2+(NUM_MARKER*3)+1]<<" "<<output[2+(NUM_MARKER*3)+2]<<endl;
		fout_conf.close();

		fout<<"ambient = "<<output[0]<<endl;
		fout<<"diffuse = "<<output[1]<<endl;
		for(int s=0;s<Hs.size();s++)
			fout<<"normal "<< s+1 <<" = ("<<output[s*3+2]<<", "<<output[s*3+3]<<", "<<output[s*3+4]<<")"<<endl;
		fout<<"light position = ("<<output[2+(NUM_MARKER*3)]<<", "<<output[2+(NUM_MARKER*3)+1]<<", "<<output[2+(NUM_MARKER*3)+2]<<")"<<endl;
		fout<<"minimum cost = "<<cost<<endl;
		fout<<"================================="<<endl;
		fout<<"execution time:\t"<<static_cast<float>(end-start)/1000.0f<<" s"<<endl;


		/* screen */
		system("cls");
		cout<<"Finished!"<<endl;
		cout<<"execution time:\t"<<static_cast<float>(end-start)/1000.0f<<" s"<<endl;
		cout<<"================================="<<endl;
		cout<<"ambient = "<<output[0]<<endl;
		cout<<"diffuse = "<<output[1]<<endl;
		for(int s=0;s<Hs.size();s++)
			cout<<"normal "<< s+1 <<" = ("<<output[s*3+2]<<", "<<output[s*3+3]<<", "<<output[s*3+4]<<")"<<endl;
		cout<<"light position = ("<<output[2+(NUM_MARKER*3)]<<", "<<output[2+(NUM_MARKER*3)+1]<<", "<<output[2+(NUM_MARKER*3)+2]<<")"<<endl;
		cout<<"minimum cost = "<<cost<<endl;
	}
}

void loadConfig(string path);

int main(int argc, char* argv[])
{
	/**
		argv[1] config檔案路徑
	**/
	if(argc<=1){
		std::cout<<"bad config path."<<std::endl;
		return -1;
	}
	loadConfig(argv[1]);
	
	/*手點*/
	data.srcPts.resize(NUM_MARKER*4);
	data.desPts.resize(NUM_MARKER*4);
	float half_markerLen = MARKER_LEN/2;
	for(int i=0;i<NUM_MARKER;i++){
		data.desPts[i*4] = Point2f(-half_markerLen,  half_markerLen);            //clockwise
		data.desPts[i*4+1] = Point2f(half_markerLen, half_markerLen);
		data.desPts[i*4+2] = Point2f( half_markerLen, -half_markerLen);
		data.desPts[i*4+3] = Point2f( -half_markerLen,  -half_markerLen);
	}
	/***/


	LightingEstimation_marker::getInstance().init(NUM_MARKER, MARKER_LEN);
	/*手點*/
	setMouseCallback("click", onMouse, &data);
	/***/	

	waitKey();
	fout.close();
	
	return 0;
}

void loadConfig(string path)
{
	/**
		line 1 input檔案路徑
		line 2 reference檔案路徑
		line 3 output檔案名
		line 4 label檔案路徑 (非必要)
		line 5 initial guess  (非必要)
	**/

	ifstream fin(path);
	string input_path, refer_path;
	fin >> input_path >> refer_path;
	Mat img = imread(input_path);
	imshow("input", img);
	imshow("click", imread(refer_path));
	data.img = img.clone();
	data.refer_path = refer_path;

	string output_path;
	fin >> output_path;
	fout.open(output_path);
	fout << "name : "<<input_path<<endl;
	if(!fin.eof()){
		string label_path;
		fin >> label_path;
		data.label = imread(label_path).clone();
	}

	double initGuess[5] = {0.15, 0.6, 0.0, 0.0, 5.0};
	if(!fin.eof()){
		string tmp;
		for(int i=0;i<5;i++){
			fin>>tmp;
			initGuess[i] = stod(tmp);
		}
	}
	LE_marker::getInstance().setInitGuess(initGuess[0], initGuess[1],
			initGuess[2], initGuess[3], initGuess[4]);
	fout<<"initial guess: ["<<initGuess[0]<<", "<<initGuess[1]<<", ";
	fout<<initGuess[2]<<", "<<initGuess[3]<<", "<<initGuess[4]<<"]"<<endl;
		

	fin.close();
}