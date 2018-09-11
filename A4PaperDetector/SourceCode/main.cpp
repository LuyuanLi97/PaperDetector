#include <iostream>
#include <string>
#include <io.h>
#include <vector>
#include "canny.h"
using namespace std;

void getFiles(string path, vector<string>& files);

int main() {
	vector<string> files;
	getFiles("test", files);
	int size = files.size();  
	cout << size << endl;
	for (int i = 0; i < size; i++) {  
	    cout << files[i].c_str() << endl;
	    canny paper1(files[i]);
		paper1.cannyDisplay(1.5f, 6.0f);
		paper1.houghDisplay(200.0f, 0.5f);
		paper1.perspectiveTransform();
	}

	return 0;
}

void getFiles(string path, vector<string>& files) {   
    long   hFile   =   0;   
    struct _finddata_t fileinfo;  
    string p;  
    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1) {  
        do  {  
            if((fileinfo.attrib &  _A_SUBDIR))  {  
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
                    getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
            }  
            else  {  
                files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
            }  
        }while(_findnext(hFile, &fileinfo)  == 0);  
        _findclose(hFile);  
    }  
}
