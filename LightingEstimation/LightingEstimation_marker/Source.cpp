#include "LightingEstimation_marker.h"
#include <fstream>
#include <time.h>
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
	
	if(count<4){
		data->srcPts[count++] = Point2f((float)x*dsFactor, (float)y*dsFactor);
		circle(click_img, Point2f((float)x, (float)y), 5, Scalar(0,0,255));
		imshow("click", click_img);

		std::cout<<"("<<data->srcPts[count-1].x<<","<<data->srcPts[count-1].y<<")";
		std::cout<<" --> ("<<data->desPts[count-1].x<<","<<data->desPts[count-1].y<<")";
		std::cout<<std::endl;
	}
	if(count==4){
		destroyAllWindows();
		count++;
		resize((data->img), (data->img), Size((int)(data->img.cols*dsFactor), (int)(data->img.rows*dsFactor)));
		
		Mat H = getPerspectiveTransform(data->srcPts, data->desPts);    //find homography
		std::vector<Mat> Hs;
		Hs.push_back(H);
		LE_marker::getInstance().setHomoMatrix(Hs);
		LE_marker::getInstance().setInputImg(data->img);
		//LE_marker::getInstance().setLabel(data->label);

		clock_t start, end;
		start = clock();
		cost = LE_marker::getInstance().estimate();
		end = clock();
		cout<<"execution time:\t"<<(end-start)<<" ms"<<endl;
		
		LE_marker::getInstance().outputData(output);
		fout<<"ambient = "<<output[0]<<endl;
		fout<<"diffuse = "<<output[1]<<endl;
		for(int s=0;s<Hs.size();s++)
			fout<<"normal "<< s+1 <<" = ("<<output[s*3+2]<<", "<<output[s*3+3]<<", "<<output[s*3+4]<<")"<<endl;
		fout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
		fout<<"minimum cost = "<<cost<<endl;
		fout<<"================================="<<endl;
		fout<<"execution time:\t"<<static_cast<float>(end-start)/1000.0f<<" s"<<endl;

		system("cls");
		cout<<"Finished!"<<endl;
		cout<<"execution time:\t"<<static_cast<float>(end-start)/1000.0f<<" s"<<endl;
		cout<<"================================="<<endl;
		cout<<"ambient = "<<output[0]<<endl;
		cout<<"diffuse = "<<output[1]<<endl;
		for(int s=0;s<Hs.size();s++)
			cout<<"normal "<< s+1 <<" = ("<<output[s*3+2]<<", "<<output[s*3+3]<<", "<<output[s*3+4]<<")"<<endl;
		cout<<"light position = ("<<output[5]<<", "<<output[6]<<", "<<output[7]<<")"<<endl;
		cout<<"minimum cost = "<<cost<<endl;
	}
}

void loadConfig(string path);

int main(int argc, char* argv[])
{
	/**
		argv[1] config檔案路徑
	**/
	loadConfig(argv[1]);
	
	/*手點*/
	data.srcPts.resize(4);
	data.desPts.resize(4);
	float half_markerLen = 20.0/2;
	data.desPts[0] = Point2f(-half_markerLen,  half_markerLen);            //clockwise
	data.desPts[1] = Point2f(half_markerLen, half_markerLen);
	data.desPts[2] = Point2f( half_markerLen, -half_markerLen);
	data.desPts[3] = Point2f( -half_markerLen,  -half_markerLen);
	/***/


	
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