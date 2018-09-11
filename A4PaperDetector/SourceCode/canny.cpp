#include <iostream>
#include <string>
#include <fstream>
#include "CImg.h"
#include "gauss_filter.h"
#include "non_maximum_suppression.h"
#include "canny.h"
using namespace cimg_library;
using namespace std;

bool sortHoughPoint(HoughPoint e1, HoughPoint e2) {
	return e1.val > e2.val;	
}

bool sortCornerPoints(Point p1, Point p2) {
	return (p1.x * p1.x + p1.y * p1.y) < (p2.x * p2.x + p2.y * p2.y);
}

canny::canny(string file) {
	filePath = file;
	//in.load_bmp(filePath.c_str());
	//in.assign(input);
}

void canny::cannyDiscrete(CImg<float> in, float sigma, float threshold) {
	const int nx = in._width;
	const int ny = in._height;

	/************ initialize memory ************/
	outGradient = in; outGradient.fill(0.0f);
	CImg<int> dirmax(outGradient);
	CImg<float> derivative[4];
	for(int i = 0; i < 4; i++) { derivative[i] = outGradient; }
	outOrientation = outGradient;
	outThreshold = outGradient;
	outNMS = outGradient;

	/************** smoothing the input image ******************/
	CImg<float> filter;
	gauss_filter(filter, sigma, 0);	//void gauss_filter (CImg<float>& filter, float sigma=1.0f, int deriv=0)
	outSmooth = in.get_convolve(filter).convolve(filter.get_transpose());


	/************ loop over all pixels in the interior **********************/
	float fct = 1.0 / (2.0*sqrt(2.0f));
	for (int y = 1; y < ny-1; y++) {
		for(int x = 1; x < nx-1; x++) {
			//***** compute directional derivatives (E,NE,N,SE) ****//
			float grad_E = (outSmooth(x+1,y  ) - outSmooth(x-1,y  ))*0.5; // E
			float grad_NE = (outSmooth(x+1,y-1) - outSmooth(x-1,y+1))*fct; // NE
			float grad_N = (outSmooth(x,  y-1) - outSmooth(x,  y+1))*0.5; // N
			float grad_SE = (outSmooth(x+1,y+1) - outSmooth(x-1,y-1))*fct; // SE

			//***** compute gradient magnitude *********//
			float grad_mag = grad_E*grad_E + grad_N*grad_N;
			outGradient(x,y) = grad_mag;

			//***** compute gradient orientation (continuous version)*******//
			float angle = 0.0f;
			if (grad_mag > 0.0f) { angle =  atan2(grad_N, grad_E); }
			if (angle < 0.0) angle += cimg::PI;
			outOrientation(x,y) = angle*255.0/cimg::PI + 0.5; // -> outOrientation

			//***** compute absolute derivations *******//
			derivative[0](x,y) = grad_E = fabs(grad_E);
			derivative[1](x,y) = grad_NE = fabs(grad_NE);
			derivative[2](x,y) = grad_N = fabs(grad_N);
			derivative[3](x,y) = grad_SE = fabs(grad_SE);

			//***** compute direction of max derivative //
			if ((grad_E>grad_NE) && (grad_E>grad_N) && (grad_E>grad_SE)) {
				dirmax(x,y) = 0; // E
			} else if ((grad_NE>grad_N) && (grad_NE>grad_SE)){
				dirmax(x,y) = 1; // NE
			} else if (grad_N>grad_SE) {
				dirmax(x,y) = 2; // N
			} else {
				dirmax(x,y) = 3; // SE
			}
			// one may compute the contiuous dominant direction computation...
			//outOrientation(x,y) = dirmax(x,y)*255.f/4;
		} } // for x,y

	// directing vectors (E, NE, N, SE)
	int dir_vector[4][2] = {{1,0}, {1,-1}, {0,-1}, {1,1}};
	// direction of max derivative of
	// current pixel and its two neighbouring pixel (in direction of dir)
	int dir, dir1, dir2;

	//***** thresholding and (canny) non-max-supression *//
	for (int y = 2; y < ny-2; y++) {
		for (int x = 2; x < nx-2; x++) {
			dir = dirmax(x,y);
			if (derivative[dir](x,y) < threshold) {
				outThreshold(x,y) = 0.0f;
				outNMS(x,y) = 0.0f;
			} else {
				outThreshold(x,y) = 255.0f;
				int dx = dir_vector[dir][0];
				int dy = dir_vector[dir][1];
				dir1 = dirmax(x + dx,y + dy);
				dir2 = dirmax(x - dx,y - dy);
				outNMS(x,y) = 255.f*
					((derivative[dir](x,y) > derivative[dir1](x + dx, y + dy)) &&
					(derivative[dir](x,y) >= derivative[dir2](x-dx,y-dy)));
			} // -> outThreshold, outNMS
		} } // for x, y...
}

int canny::cannyDisplay(float sigma, float threshold) {
	// image after non-max-suppression
	string infile = filePath;      // required input filename
	string outfileGradient = "Gradient_Pro.bmp";  // saving the (normalised) gradient to file?
	string outfileNMS = "Edge_Output.bmp";       // saving the binary canny edges to file?

	// canny parameters
	//float sigma = 1.5f;
	//float threshold = 6.0f;

	//***** read image *****************//
	CImg<float> inColor(infile.c_str());
	CImg<float> in = get_RGBtoGray(inColor); // ensure greyscale img!
	//in.display("in");
	const int widthIn = in._width;
	const int heightIn = in._height;
	if ( widthIn == 0 || heightIn == 0 ) {
		cerr << "Error when loading input image." << endl;
		return -1;
	}

	//***** declare output images ******//
	//CImg<float> outS, outG, outO, outT, outNMS;

	//***** apply Canny filter *********//
	cannyDiscrete(in, sigma, threshold);

	//***** display output images ******//
	char  header[100];
	sprintf(header, "gaussian smoothed image: sigma = %f", sigma);
	//outSmooth.display(header);
	float maxgrad = 0.0f;
	cimg_forXY(outGradient,x,y) { maxgrad = std::max(maxgrad, outGradient(x,y)); }
	std::cout << "normalising [0.." << maxgrad << "] to [0..255]" << std::endl;
	sprintf(header, "gradient magnitude [0..%f]", maxgrad);
	//outGradient.display(header);
	//outOrientation.display("orientation map");
	sprintf(header, "thresholded with %f", threshold);
	//outThreshold.display(header);
	//outNMS.display("non-maximum suppression");


	//***** write output images ********//
	if (outfileGradient.length()>0) {
		std::cout << "saving gradient to " << outfileGradient << std::endl;
		outGradient *= (255.f/maxgrad);
		outGradient.save(outfileGradient.c_str());
	}
	if (outfileNMS.length()>0) {
		std::cout << "saving gradient to " << outfileNMS << std::endl;
		outNMS.save(outfileNMS.c_str());
	}


	return 0;
}

void canny::hough(const CImg<float>& img, float in_thresh, float out_thresh) {	
	// init result *******************************************************
	const int WIDTH = img.width();
	const int HEIGHT = img.height();
	CImg<float> originImg(filePath.c_str());
	result = originImg;
	
	// init hough space *******************************************************
	const float DIAGONAL = sqrt(WIDTH*WIDTH+HEIGHT*HEIGHT);
	const int OFFSET_N = (int)DIAGONAL; 
	//hough space : angle * rho   
	HoughSpace.assign (360, OFFSET_N);
	HoughSpace.fill (0.0f);
	
	//calculate hough space (voting) *****************************************
	//rectangular coordinate => polar coordinate
	cimg_forXY(img, x, y) {
		if (img(x, y) < in_thresh) continue;
		cimg_forX(HoughSpace, angle) {
			double theta = 1.0 * angle * cimg::PI / 180.0;
			int rho = (int)(x*cos(theta) + y*sin(theta));
			if (rho >= 0 && rho < HoughSpace.height()) {
				HoughSpace(angle % 360, rho)++;
			}
		}
	} 
	
	printf ("Voting done\n");
	//HoughSpace.display ("HoughSpace");
	
	// choose peak clusters in hough space ************************************
	float maxvote = HoughSpace(0, 0);
	for (int i = 0; i < 360 * OFFSET_N; i++) maxvote = max(maxvote, HoughSpace[i]);
	//nonmax: stores dots in several brightest clusters
	TVectorOfPairs nonmax;
	non_maximum_suppression(HoughSpace, nonmax, out_thresh*maxvote, 4);
	printf ("Suppression done: %d lines found\n", nonmax.size());
	
	//get houghDots representing four edges of the a4 paper
	if (nonmax.size() > 4) {
		vector<HoughPoint> hp;
		for (int i = 0; i < nonmax.size(); i++) {
			hp.push_back(HoughPoint(nonmax[i].first, nonmax[i].second, HoughSpace(nonmax[i].first, nonmax[i].second)));
		}	
		sort(hp.begin(), hp.end(), sortHoughPoint);
		nonmax.clear();
		for (int i = 0, j = 0; i < hp.size(), j < 4; i++) {
			double theta = 1.0 * (hp[i].x) * cimg::PI / 180.0;
			if (abs(theta - 0.0f) > 0.000001) {
				nonmax.push_back(make_pair(hp[i].x, hp[i].y));
				j++;
			}  
		}
	}
	
	//polar coordinate => rectangular coordinate
	//store (m, b) in imgLines
	//get m : m = imgLines[i].first;
	//get b : b = imgLines[i].second;
	std::vector<std::pair<double,double> > imgLines;	
	for (int i = 0; i < nonmax.size(); i++) {
		double theta = 1.0 * (nonmax[i].first ) * cimg::PI / 180.0;
		double m = -cos(theta) / sin(theta);
		double b = 1.0 * nonmax[i].second / sin(theta);
		imgLines.push_back(make_pair(m, b));
		cout << "line : m = " << m << " b = " << b 
			 << " value = " << HoughSpace(nonmax[i].first, nonmax[i].second) 
			 << " theta = " << theta << endl;
	}
	
	
	for (int i = 0; i < imgLines.size(); i++) {
		for (int j = i + 1; j < imgLines.size(); j++) {
			double m0 = imgLines[i].first;  
            double m1 = imgLines[j].first;  
            double b0 = imgLines[i].second;  
            double b1 = imgLines[j].second;  
            double x = (b1 - b0) / (m0 - m1);  
            double y = (m0*b1 - m1*b0) / (m0 - m1);  
            if (x >= 0 && x < result.width() && y >= 0 && y < result.height()) {  
                crossPoints.push_back(Point(x, y));  
            }  
		}
	}
	
	// draw lines *************************************************
	for (int i = 0; i < imgLines.size(); i++) {
		const int ymin = 1;  
	    const int ymax = result.height() - 1;  
	    const int x0 = (double)(ymin - imgLines[i].second) / imgLines[i].first;  
	    const int x1 = (double)(ymax - imgLines[i].second) / imgLines[i].first;  
	  
	    const int xmin = 1;  
	    const int xmax = result.width() - 1;  
	    const int y0 = (double)(xmin*imgLines[i].first + imgLines[i].second);  
	    const int y1 = (double)(xmax*imgLines[i].first + imgLines[i].second);  
	    //cout << xmin << ' ' << y0 << ' ' << xmax << ' ' << y1 << endl;
	  
	    const float color[] = { 255, 255, 0 };  
	  	
		if (abs(imgLines[i].first) > 1) {  
	        result.draw_line(x0, ymin, x1, ymax, color);  
	    }  
	    else {  
	        result.draw_line(xmin, y0, xmax, y1, color);  
	    }  
	}
	
	//draw corner points
	for (int i = 0; i < crossPoints.size(); ++i) {  
	    const double color[] = { 255, 255, 0 };  
	    result.draw_circle(crossPoints[i].x, crossPoints[i].y, 15, color);  
	}  
}

void canny::houghDisplay(float in_thresh, float out_thresh) {
	string outfile="Hough_Output.bmp";
	// pre-scaling/rotation operations
	float rotate = 0.0f;
	float zoom   = 1.0f;

	// load image and ensure greyscale img!
	CImg<float> input = outThreshold.get_channel(0);

	//input.display ("Input Image");

	// do the transform 
	hough (input, in_thresh, out_thresh);

	//result.display();

	if (result.size()>0) result.save (outfile.c_str());
}

void canny::perspectiveTransform() {
	sort(crossPoints.begin(), crossPoints.end(), sortCornerPoints);
	
	
	if (crossPoints[0].x > crossPoints[1].x) {
		int x0 = crossPoints[0].x;
		int y0 = crossPoints[0].y;
		
		int x1 = crossPoints[1].x;
		int y1 = crossPoints[1].y;
		
		int x2 = crossPoints[2].x;
		int y2 = crossPoints[2].y;
		
		int x3 = crossPoints[3].x;
		int y3 = crossPoints[3].y;
		
		//top left
		crossPoints[0].x = x1;
		crossPoints[0].y = y1;
		//top right
		crossPoints[1].x = x0;
		crossPoints[1].y = y0;
		//bottom left
		crossPoints[2].x = x3;
		crossPoints[2].y = y3;
		//bottom right
		crossPoints[3].x = x2;
		crossPoints[3].y = y2;
	}
	
	ofstream outFile;
	outFile.open("corner.csv", ios::app);
	outFile << "x" << ',' << "y" << endl;
	for (vector<Point>::iterator iter = crossPoints.begin(); iter != crossPoints.end(); iter++) {
		outFile << iter->x << ',' << iter->y << endl;
		cout << "Point: x = " << iter->x << " y = " << iter->y << endl;
	}
	//outFile << " " << ',' << " " << endl;
	outFile.close();
	
	//top left
	int x0 = crossPoints[0].x;
	int y0 = crossPoints[0].y;
	//top right
	int x1 = crossPoints[1].x;
	int y1 = crossPoints[1].y;
	//bottom left
	int x2 = crossPoints[2].x;
	int y2 = crossPoints[2].y;
	//bottom right
	int x3 = crossPoints[3].x;
	int y3 = crossPoints[3].y;
	
	const float W = 410, H = 594;
	const float u0 = 0, v0 = 0, // top-left
				u1 = W - 1, v1 = 0, // top-right
				u2 = 0, v2 = H - 1, // bottom-left
				u3 = W - 1, v3 = H - 1; // bottom-right 
				
	Eigen::MatrixXf UV(8, 1);
	Eigen::MatrixXf parameters = Eigen::MatrixXf::Constant(8, 1, 0);
	Eigen::MatrixXf A(8, 8);
	UV << u0, v0, u1, v1, u2, v2, u3, v3;
	A << x0, y0, 1, 0,  0,  0, -u0*x0, -u0*y0,
		 0,  0,  0, x0, y0, 1, -v0*x0, -v0*y0,
		 x1, y1, 1, 0,  0,  0, -u1*x1, -u1*y1,
		 0,  0,  0, x1, y1, 1, -v1*x1, -v1*y1,
		 x2, y2, 1, 0,  0,  0, -u2*x2, -u2*y2,
		 0,  0,  0, x2, y2, 1, -v2*x2, -v2*y2,
		 x3, y3, 1, 0,  0,  0, -u3*x3, -u3*y3,
		 0,  0,  0, x3, y3, 1, -v3*x3, -v3*y3;
	parameters = A.inverse() * UV;
	a = parameters(0, 0);
	b = parameters(1, 0);
	c = parameters(2, 0);
	d = parameters(3, 0);
	e = parameters(4, 0);
	f = parameters(5, 0);
	m = parameters(6, 0);
	l = parameters(7, 0);
	
	CImg<float> srcColor(filePath.c_str());
	CImg<float> src = get_RGBtoGray(srcColor);
	//src.display();
	CImg<float> dest(W, H, 1, 3, 0);
	cimg_forXYC(dest, u, v, c) { // c indicates color channels
		float x = get_UVtoX(u, v);
		float y = get_UVtoY(u, v);
		if(x >= 0 && y >= 0 && x + 1 < src.width() && y + 1 < src.height()) {
			int i = floor(x), j = floor(y);
			float a = x - i, b = y - j;
			dest(u, v, c) = (1 - a)*(1 - b)*src(i, j, c) + a*(1 - b)*src(i + 1, j, c)
					+ (1 - a)*b*src(i, j + 1, c) + a*b*src(i + 1, j + 1, c);
		}
	}
	
	int dotIndex = filePath.find(".");
	string filename = filePath.substr(0, dotIndex);
	string outfile = ".\\result\\" + filename + "_result.bmp";
	dest.save(outfile.c_str());
	//dest.display();
	CImg<float> binary = get_binary(dest);
	//binary.display();
}

float canny::get_UVtoY(int u, int v) {
	return ((c-u)*(v*m-d)-(f-v)*(u*m-a))/((u*l-b)*(v*m-d)-(v*l-e)*(u*m-a));
}

float canny::get_UVtoX(int u, int v) {
	return ((c-u)*(v*l-e)-(f-v)*(u*l-b))/((u*m-a)*(v*l-e)-(v*m-d)*(u*l-b));
}


CImg<float> canny::get_RGBtoGray(CImg<float> inColor) {
	CImg<float> grayscaled = inColor;
    cimg_forXY(inColor, x, y) {
        int r = inColor(x, y, 0);
        int g = inColor(x, y, 1);
        int b = inColor(x, y, 2);
        double newValue = (r * 0.2126 + g * 0.7152 + b * 0.0722);
        grayscaled(x, y, 0) = newValue;
        grayscaled(x, y, 1) = newValue;
        grayscaled(x, y, 2) = newValue;
    }
    return grayscaled;
}

CImg<float> canny::get_binary(CImg<float> img) {
	int cropMargin = 5;
	CImg<float> cropped = CImg<float>(img._width - cropMargin * 2, img._height - cropMargin * 2, 1, 1, 0);
	cimg_forXY(cropped, x, y) {
		int x0 = x + cropMargin;
		int y0 = y + cropMargin;
		cropped(x, y) = img(x0, y0, 0);
	}
	//cropped.display();
	
	CImg<float> binary = CImg<float>(cropped._width, cropped._height, 1, 1, 0);
	int row = 5;
	int col = 3;
	int subH = binary._height / row;
	int subW = binary._width / col;
	
	//for every sub picture
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int x0 = subW * j;
			int y0 = subH * i;
			
			//ostu
			int Na = 0, Nb = 0;
			int suma = 0, sumb = 0;
			float Ua = 0, Ub = 0;
			float sigma;
			float maxSigma = 0;
			int maxTh;
			int maxDiff = 0;
			
			for (int Th = 0; Th <= 255; Th++) {
				//for every pixel in sub picture
				for (int x = x0; x < x0 + subW; x++) {
					for (int y = y0; y < y0 + subH; y++) {
						if (cropped(x, y) <= Th) {
							Na++;
							suma += cropped(x, y);
						}
						else {
							Nb++;
							sumb += cropped(x, y, 0);
						}
					}
				}
				Ua = (Na == 0) ? 0.0f : suma/Na;
				Ub = (Nb == 0) ? 0.0f : sumb/Nb;
				
				sigma = Na * Nb * (Ua - Ub) * (Ua - Ub);
				int tmpDiff = abs(Ua - Ub);
				
				if (sigma > maxSigma) {
					maxSigma = sigma;
					maxTh = Th;
					maxDiff = tmpDiff;
				}
				
				Na = 0; Nb = 0; 
				suma = 0; sumb = 0;
			}
			//cout << "row, col = " << i << " " << j << endl;
			//cout << "max sigma = " << maxSigma << endl;
			//cout << "diff = " << maxDiff << endl;
			
			if (maxDiff <= 20) {
				for (int x = x0; x < x0 + subW; x++) {
					for (int y = y0; y < y0 + subH; y++) {
						cropped(x, y) = 0;
					}
				}
			}
			else {
				for (int x = x0; x < x0 + subW; x++) {
					for (int y = y0; y < y0 + subH; y++) {
						cropped(x, y) = cropped(x, y) > maxTh ? 0 : 255;
					}
				}
			}
		}
	} 
	cimg_forXY(cropped, x, y) {
		if (cropped(x, y) != 0 && cropped(x, y) != 255) {
			cropped(x, y) = 0;
		}
	}
	
	int dotIndex = filePath.find(".");
	string filename = filePath.substr(0, dotIndex);
	string outfile = ".\\binary\\" + filename + ".bmp";
	cropped.save(outfile.c_str());
	return cropped;
} 
