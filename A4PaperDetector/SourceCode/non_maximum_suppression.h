
#ifndef NON_MAXIMUM_SUPPRESSION_H
#define NON_MAXIMUM_SUPPRESSION_H

#include <vector>
#include "CImg.h"
using namespace cimg_library;
using namespace std;

/** a vector of pixel coordinates.
* Usage:
*    unsigned i;
*    int x, y;
*    TVectorOfPairs nonmax;
*    nonmax.push_back (make_pair(x,y));   // adding new pixel coordinates:
*    angle = nonmax[i].first;                 // get angle-coordinate of i-th pixel
*    rho = nonmax[i].second;                // get rho-coordinate of i-th pixel
*/
typedef std::vector<std::pair<int,int> > TVectorOfPairs;

/** apply non-maximum suppression
* \param input: some float image
* \param nonmax: a list of (x,y)-tuple of maxima
* \param thresh: ignore those with too small response
* \param halfwidth: halfwidth of the neighbourhood size
*/
void non_maximum_suppression (CImg<float>& img, TVectorOfPairs& nonmax,
							  float thresh, int halfwidth) 
{
	nonmax.clear();
	cimg_forXY(img, x, y) {
        int val = img(x, y);
        if (val < thresh) {
            img(x, y) = 0;
        }
        else {
            bool is_new_corner = true;
            for (int i = 0; i < nonmax.size(); i++) {
                if (sqrt((nonmax[i].first - x) * (nonmax[i].first - x) +
                    (nonmax[i].second - y) * (nonmax[i].second - y)) < 20) {
                    is_new_corner = false;
                    // compare with the other value in this cluster
                    if (val > img(nonmax[i].first, nonmax[i].second)) {
                    	std::pair<int,int> brighter = make_pair(x, y);
                        nonmax[i] = brighter; // update
                        break;
                    }
                }
            }
            if (is_new_corner) nonmax.push_back(make_pair(x,y));
        }
    } 
}

#endif /* NON_MAXIMUM_SUPPRESSION_H */
