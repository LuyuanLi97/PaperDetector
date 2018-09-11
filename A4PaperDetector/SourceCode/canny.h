#ifndef _CANNY_
#define _CANNY_
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "CImg.h"
using namespace cimg_library;
using namespace std;

//HoughPoint : angle, rho, voting value
struct HoughPoint {
	int x, y, val;
	HoughPoint(int _x, int _y, int _val)
		: x(_x), y(_y), val(_val) {}
};

struct Point {
	int x, y;
	Point(int _x, int _y)
		: x(_x), y(_y) {}
};

class canny {
private:
	string filePath;	//input file path 
	CImg<float> in;
	CImg<float> outSmooth;
	CImg<float> outGradient;
	CImg<float> outOrientation;
	CImg<float> outThreshold;
	CImg<float> outNMS;
	CImg<float> result;	//hough transform result
	vector<Point> crossPoints;

	void cannyDiscrete(CImg<float> in, float sigma, float threshold);
	void hough(const CImg<float>& img, float in_thresh, float out_thresh);
	CImg<float> get_RGBtoGray(CImg<float> inColor);
	CImg<float> get_binary(CImg<float> img);
	
	float get_UVtoX(int u, int v);
	float get_UVtoY(int u, int v);
	//float bilinearInterpolate(float x, float y, int c);
	float a, b, c, d, e, f, m, l;

public:
	CImg<float> HoughSpace;
	
	canny(string file);
	int cannyDisplay(float sigma = 1.5f, float threshold = 6.0f);
	void houghDisplay(float in_thresh = 200.0f, float out_thresh = 0.5f);
	void perspectiveTransform();
};

#endif
