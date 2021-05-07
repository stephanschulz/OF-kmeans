#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp {
public:
	void setup();
	void update();
	void draw();
    void keyReleased(int key);
    
    void makeNewRandom();
    void makeNewGrid();
    void makeNewGridWithCenters(vector<cv::Point2f> labels);
    
};
